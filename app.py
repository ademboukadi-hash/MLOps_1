import os
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")  # Le titre s'affiche dans la page /docs

# ===== 3. Lire la configuration dans les variables d'environnement =====
# Rien n'est écrit « en dur » : le même code marche sur le poste, dans un
# conteneur seul, ou dans Docker Compose. Seules les variables changent.
# os.getenv("NOM", "valeur") : valeur de la variable, ou valeur par défaut si elle n'existe pas
PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
CHAMPION_URI = "models:/churn-model@champion"
MODEL_URI = os.getenv("MODEL_URI", CHAMPION_URI)

# ===== 4. Charger le champion UNE SEULE FOIS, au démarrage =====
# MODEL_URI peut être une adresse du Registry (models:/...) ou un simple dossier (model)
mlflow.set_tracking_uri(TRACKING_URI)
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except MlflowException as load_error:
    if MODEL_URI != CHAMPION_URI:
        raise

    champion = MlflowClient().get_model_version_by_alias("churn-model", "champion")
    model_id = champion.source.removeprefix("models:/")
    local_artifacts = next(
        (
            path
            for path in (PROJECT_ROOT / "mlruns").glob(
                f"*/models/{model_id}/artifacts"
            )
            if (path / "MLmodel").is_file()
        ),
        None,
    )
    if local_artifacts is None:
        raise load_error
    model = mlflow.sklearn.load_model(str(local_artifacts))

print(f"Modèle chargé depuis : {MODEL_URI}")


# ===== 5. Décrire les données attendues (validées automatiquement) =====
class Client(BaseModel):
    age: int  # Nombre entier
    salaire: float  # Nombre décimal
    abonnement: Literal["Basique", "Standard", "Premium"]  # 3 valeurs possibles
    anciennete: float
    consommation: float


# ===== 6. Route /health : vérifier que l'API est en vie =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 7. Route /predict : recevoir un client, renvoyer la prédiction =====
@app.post("/predict")
def predict(client: Client):
    # a) Transformer le client reçu en tableau d'une ligne
    ligne = pd.DataFrame([client.model_dump()])

    # b) Transformer le texte « abonnement » en colonnes 0/1, comme à l'entraînement
    ligne = pd.get_dummies(ligne)

    # c) Garder les colonnes du modèle, dans le bon ordre ; les absentes valent 0
    ligne = ligne.reindex(columns=model.feature_names_in_, fill_value=0)

    # d) Prédire : 1 = part, 0 = reste
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": (
            "Le client va partir" if resultat == 1 else "Le client reste"
        ),
    }