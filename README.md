# Fraud Detection — Production ML System

A production-style fraud detection service built as part of an ML engineering portfolio. Demonstrates the full MLOps loop: data validation → feature pipeline → training → evaluation → online serving → monitoring.

## Stack

Python 3.12 · Poetry · LightGBM · scikit-learn · pydantic · MLflow · FastAPI · Evidently · Docker · GitHub Actions

## Project structure

```
src/fraud_detection/
    data/        # data loading + pydantic schema validation
    features/    # shared feature pipeline (train + serve — no skew)
    training/    # config-driven training pipeline + MLflow tracking
    evaluation/  # PR-AUC panel, cost-based threshold, fairness check
    serving/     # FastAPI scoring API (POST /score)
    monitoring/  # operational metrics, drift detection, prediction logging
tests/           # unit + data + model behavioural tests
config/          # config.yaml — all parameters in one place
notebooks/       # EDA only
```

## Setup

```bash
# Install Python 3.12 and Poetry, then:
poetry install
```

## Data

Download the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place the files at:

```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

## Usage

```bash
# Train (coming in M3)
poetry run python -m fraud_detection.training.train

# Serve (coming in M5)
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## Build milestones

- [x] M1 — Scaffold: repo structure, CI, Dockerfile, data loading + schema validation
- [ ] M2 — Feature pipeline + EDA notebook
- [ ] M3 — Training pipeline + MLflow + model registry
- [ ] M4 — Evaluation panel + cost-based threshold + fairness check
- [ ] M5 — FastAPI serving + monitoring + prediction logging
