# Churn Prediction Project

This project is a small machine learning exercise for predicting customer churn from synthetic data. It demonstrates a lightweight MLOps workflow: generate data, train a model, and save the trained artifact for later use.

## Objective

The goal is to predict whether a customer is likely to leave a service based on a few features such as age, salary, subscription plan, seniority, and usage level.

## Data

The dataset is generated in `generate_data.py` with synthetic customer records. Features include:

- `age`
- `salaire`
- `abonnement`
- `anciennete`
- `consommation`

The target variable is:

- `churn` = 1 if the customer is likely to leave, otherwise 0

The churn rule used in this project is intentionally simple:

- `consommation < 30`
- and `anciennete < 2`

This combination is marked as churn.

## Files in the project

- `generate_data.py` — creates the synthetic dataset and saves it as `churn_data.csv`
- `churn_data.csv` — generated dataset
- `churn_model.ipynb` — notebook used to train and evaluate the model
- `model.pkl` — serialized trained model
- `log.txt` — execution notes/log output
- `REDMI.md` — project documentation
- `README.md` — project entry point documentation

## Setup

Use a Python environment with the dependencies below installed:

```bash
pip install pandas numpy scikit-learn
```

## Run the workflow

Generate the dataset:

```bash
python generate_data.py
```

Open and run the notebook:

```bash
jupyter notebook churn_model.ipynb
```

## Model

The project uses a `RandomForestClassifier` from `scikit-learn` to classify customers as churners or non-churners.

## Notes

This is a demo project designed to practice ML pipeline basics and reproducible experimentation rather than a production-grade MLOps system.
