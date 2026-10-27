from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")


def train_and_log_model() -> str:
    df = pd.read_csv(PROJECT_ROOT / "churn_data.csv")
    X = pd.get_dummies(
        df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
        drop_first=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["churn"], test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    with mlflow.start_run(run_name="champion-recovery") as run:
        mlflow.log_params({"n_estimators": 100, "max_depth": 5})
        mlflow.log_metrics(
            {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
        mlflow.sklearn.log_model(model, name="model")

    return run.info.run_id


def main() -> None:
    runs = mlflow.search_runs(
        experiment_names=["churn-prediction"],
        order_by=["metrics.f1 DESC"],
    )

    model_uri = None
    for _, run in runs.iterrows():
        candidate_uri = f"runs:/{run['run_id']}/model"
        try:
            mlflow.sklearn.load_model(candidate_uri)
        except MlflowException as exc:
            print(f"Skipping unavailable model for run {run['run_id']}: {exc}")
        else:
            model_uri = candidate_uri
            print(
                "Using existing run:",
                run.get("tags.mlflow.runName", run["run_id"]),
            )
            break

    if model_uri is None:
        print("No usable logged model found; training a replacement.")
        run_id = train_and_log_model()
        model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(model_uri, "churn-model")
    client = MlflowClient()
    client.set_registered_model_alias(
        "churn-model", "champion", registered.version
    )
    print(
        f"Champion alias -> churn-model version {registered.version} "
        f"(source: {model_uri})"
    )


if __name__ == "__main__":
    main()
