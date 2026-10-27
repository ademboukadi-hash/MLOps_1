# ===== 1. Importer les bibliothèques =====
from typing import Literal  # pour limiter les valeurs possibles d'un champ
import mlflow  # charger le modèle depuis le Registry
import mlflow.sklearn
import pandas as pd  # fabriquer le tableau attendu par le modèle
from fastapi import FastAPI  # le cadre de travail de l'API
from pydantic import BaseModel  # décrire et valider les données reçues

# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")  # le titre s'affiche dans la page /docs

# ===== 3. Charger le champion UNE SEULE FOIS, au démarrage =====
# (le recharger à chaque requête rendrait l'API très lente)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Décrire les données attendues (validées automatiquement) =====
class Client(BaseModel):
    age: int  # nombre entier
    salaire: float  # nombre décimal
    abonnement: Literal["Basique", "Standard", "Premium"]  
    anciennete: float
    consommation: float


# ===== 5. Route /health : vérifier que l'API est en vie =====
@app.get("/health")  # GET : on lit, on n'envoie rien
def health():
    return {"status": "ok"}


# ===== 6. Route /predict : recevoir un client, renvoyer la prédiction =====
@app.post("/predict")  # POST : on envoie un client
def predict(client: Client):  # FastAPI vérifie le JSON reçu avec Client
    # a) transformer le client reçu en tableau d'une ligne
    ligne = pd.DataFrame([client.model_dump()])
    # b) transformer le texte « abonnement » en colonnes 0/1, 
    ligne = pd.get_dummies(ligne)
    # c) garder les colonnes du modèle, dans le bon ordre ;
    ligne = ligne.reindex(columns=model.feature_names_in_, fill_value=0)
    # d) prédire : 1 = part, 0 = reste
    resultat = int(model.predict(ligne)[0])
    return {
        "churn": resultat,
        "message": (
            "Le client va partir" if resultat == 1 else "Le client reste"
        ),
    }