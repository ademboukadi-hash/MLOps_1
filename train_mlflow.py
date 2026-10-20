# ===== 1. Importer les bibliothèques =====
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if os.name == "nt":
    VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
else:
    VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import pandas as pd  # lire et manipuler le CSV
import mlflow  # module principal de MLflow
import mlflow.sklearn  # partie de MLflow dédiée à scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===== 2. Charger les données (une seule fois) =====
df = pd.read_csv("churn_data.csv")  # le tableau de 500 clients
y = df["churn"]  # la réponse à prédire : 1 = part, 0 = reste


# ===== 3. Fonction : préparer les données pour une liste de colonnes =====
def preparer(colonnes):
    # transforme le texte 'abonnement' en colonnes 0/1 (si présente)
    X = pd.get_dummies(df[colonnes], drop_first=True)
    # 80 % apprentissage / 20 % test, même découpage à chaque fois
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ===== 4. Fonction : lancer UN essai et l'enregistrer dans MLflow =====
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):  # ouvre un run au nom de l'essai
        model = RandomForestClassifier(
            n_estimators=n_estimators,  # nombre d'arbres
            max_depth=max_depth,  # profondeur maximale des arbres
            random_state=42  # graine : résultat reproductible
        )
        model.fit(X_train, y_train)  # entraînement
        pred = model.predict(X_test)  # prédictions sur les 100 clients de test

        # --- Paramètres : les réglages de cet essai ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))  # colonnes utilisées

        # --- Métriques : les résultats de cet essai ---
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # --- Artifact : le modèle sauvegardé dans ce run ---
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )

        print(f"{nom} -> accuracy : {round(acc, 3)} | f1 : {round(f1, 3)}")


# ===== 5. Programme principal =====
mlflow.set_experiment("churn-prediction")

TOUT = ["age", "salaire", "abonnement", "anciennete", "consommation"]

# Essai 1 : 100 arbres, profondeur 5, toutes les colonnes
lancer_essai("essai-1", 100, 5, TOUT)

# Essai 2 : modèle plus simple (50 arbres, profondeur 3)
lancer_essai("essai-2", 50, 3, TOUT)

# Essai 3 : sans la colonne salaire
lancer_essai("essai-3", 100, 5, ["age", "abonnement", "anciennete", "consommation"])