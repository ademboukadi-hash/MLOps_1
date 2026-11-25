# ===== 1. Importer les bibliothèques =====
import os
import shutil
import tempfile
from pathlib import Path

import mlflow  # Le module principal de MLflow
import mlflow.sklearn  # Les fonctions dédiées à scikit-learn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# ===== 2. Se connecter à la base du Registry (celle du TP4) =====
PROJECT_ROOT = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")

# ===== 3. Charger le champion depuis le Registry =====
CHAMPION_URI = "models:/churn-model@champion"
try:
    model = mlflow.sklearn.load_model(CHAMPION_URI)
except MlflowException as load_error:
    # L'artefact peut avoir été déplacé avec le projet, alors que le chemin
    # absolu enregistré par MLflow appartient à un autre ordinateur.
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

# ===== 4. L'écrire dans un dossier ordinaire, « model » =====
# serialization_format="cloudpickle" : même format qu'aux TP3 et TP4
# (le format par défaut de MLflow 3 est refusé au chargement).
# Ce dossier sera copié dans l'image Docker : l'image n'aura plus besoin du Registry.
OUTPUT_DIR = PROJECT_ROOT / "model"
with tempfile.TemporaryDirectory(prefix=".model-export-", dir=PROJECT_ROOT) as temp_dir:
    staged_model = Path(temp_dir) / "model"
    mlflow.sklearn.save_model(
        model,
        str(staged_model),
        serialization_format="cloudpickle",
    )

    backup_dir = None
    if OUTPUT_DIR.exists():
        backup_dir = Path(
            tempfile.mkdtemp(prefix=".model-backup-", dir=PROJECT_ROOT)
        )
        backup_dir.rmdir()
        os.replace(OUTPUT_DIR, backup_dir)

    try:
        os.replace(staged_model, OUTPUT_DIR)
    except OSError:
        if backup_dir is not None:
            os.replace(backup_dir, OUTPUT_DIR)
        raise

    if backup_dir is not None:
        shutil.rmtree(backup_dir)

print(f"Champion exporté dans le dossier {OUTPUT_DIR}/")