# ===== 1. Importer les bibliothèques =====
import pandas as pd # lire et manipuler le fichier CSV
import mlflow # le module principal de MLflow
import mlflow.sklearn # la partie de MLflow dédiée à scikit-learn
from sklearn.ensemble import RandomForestClassifier # notre algorithme
from sklearn.model_selection import train_test_split # découper les données
from sklearn.metrics import accuracy_score # mesurer la performance
# ===== 2. Charger et préparer les données (comme au TP1) =====
df = pd.read_csv("churn_data.csv") # lit le fichier : un tableau de 500 clients
# get_dummies transforme la colonne texte 'abonnement' en colonnes 0/1,
# car le modèle ne sait travailler qu'avec des nombres
X = pd.get_dummies(
df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
drop_first=True) # X = les 5 informations sur chaque client
y = df["churn"] # y = la réponse à prédire (1 = part, 0 = reste)
# 80 % des clients pour apprendre, 20 % pour tester (random_state=42 : même
# découpage à chaque exécution, donc résultats reproductibles)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
# ===== 3. Choisir l'expérience MLflow =====
# Regroupe tous nos essais sous un même nom. Créée automatiquement si absente.
mlflow.set_experiment("churn-prediction")
# ===== 4. Ouvrir un run : tout ce qui est dans ce bloc lui appartient =====
with mlflow.start_run(run_name="essai-1"): # essai-1 = le nom affiché dans MLflow
    # Créer le modèle avec ses réglages (les paramètres)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train) # entraînement : le modèle apprend
    # Prédire sur les 100 clients de test, puis comparer aux vraies réponses
    acc = accuracy_score(y_test, model.predict(X_test))
    # --- Enregistrer dans MLflow ---
    mlflow.log_param("n_estimators", 100) # paramètre : nombre d'arbres
    mlflow.log_param("max_depth", 5) # paramètre : profondeur maximale
    mlflow.log_metric("accuracy", acc) # métrique : le résultat obtenu
    # Sauvegarder le modèle lui-même (artifact). 'name' = nom du dossier du modèle.
    # serialization_format="cloudpickle" : voir la remarque sous le code.
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    print("Accuracy :", acc) # affichage dans le terminal