from pathlib import Path

import mlflow
import mlflow.sklearn  # pour recharger un modèle scikit-learn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# Même base que le TP3
PROJECT_ROOT = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")

# 1. Charger le champion PAR SON ALIAS : on ne connaît ni la version ni le run
CHAMPION_URI = "models:/churn-model@champion"
try:
    model = mlflow.sklearn.load_model(CHAMPION_URI)
except MlflowException as alias_error:
    # Les artefacts peuvent avoir été copiés avec le projet alors que leur
    # MLflow metadata garde un chemin absolu de l'ancien ordinateur.
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
        raise alias_error
    model = mlflow.sklearn.load_model(str(local_artifacts))

print("Modele charge :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))

# 2. Un nouveau client (valeurs au choix). Les colonnes d'abonnement sont
# déjà transformées en 0/1 : ici abonnement Standard
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])

# 3. Garder uniquement les colonnes connues du modèle, dans le bon ordre
client = client[list(model.feature_names_in_)]

# 4. Prédire : 1 = le client va partir, 0 = il reste
print("Le client va-t-il partir ? ->", "OUI" if model.predict(client)[0] == 1 else "NON")