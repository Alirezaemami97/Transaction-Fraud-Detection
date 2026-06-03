# Fraud Detection — Production ML System

A production-style fraud detection service built as an ML engineering portfolio project. Demonstrates the full MLOps loop end-to-end: data validation → shared feature pipeline → config-driven training → evaluation panel → online serving → prediction logging.

The system can train a model with one command and score a live transaction over HTTP. Every component is packaged, tested, typed, and containerised.

## Results

Evaluated on the IEEE-CIS dataset using a **time-based split** (train on first 60 days, test on remaining ~120 days) — no random split leakage.

| Metric | Value |
|---|---|
| PR-AUC (primary) | 0.483 |
| ROC-AUC | 0.852 |
| Cost-optimal threshold | 0.408 |
| LightGBM vs Isolation Forest | 0.483 vs 0.115 — supervised wins |

PR-AUC is the right metric for fraud: the dataset is ~3.5% fraud, so a model that always predicts "not fraud" achieves 96.5% accuracy but is completely useless. A random classifier scores ~0.035 PR-AUC; this model scores 0.483 — 14× better than random on genuinely future data.

The decision threshold (0.408) was chosen to minimise `10 × false_positives + 100 × false_negatives` — reflecting the asymmetric business cost of missing fraud vs. blocking a legitimate customer.

## Architecture

```
raw data → data layer → feature pipeline → training → evaluation → model registry
                              │                                          │
                              └──────────────── serving ────────────────┘
                                               (FastAPI)
                                                  │
                                            prediction log
```

The **feature pipeline is shared** between training and serving — the exact same `build_features()` function runs in both paths. This is the core guarantee against training-serving skew: if you change the feature logic, it changes everywhere at once.

## Stack

Python 3.12 · Poetry · LightGBM · scikit-learn (Isolation Forest) · Pydantic · MLflow · FastAPI · Docker · GitHub Actions

## Project structure

```
alireza-fraud-detection/
├── config/
│   └── config.yaml              # all run parameters in one place
├── src/fraud_detection/
│   ├── config.py                # pydantic-typed config wrapper
│   ├── data/                    # load_raw() + pydantic schema validation
│   ├── features/                # build_features() — shared by train + serve
│   ├── training/                # train.py — config-driven, MLflow-tracked
│   ├── evaluation/              # panel.py — PR-AUC, cost threshold, fairness
│   ├── serving/                 # FastAPI app: POST /score, GET /health
│   └── monitoring/              # JSONL prediction archive
├── tests/                       # 23 passing tests
├── notebooks/
│   └── 01_eda.ipynb             # EDA only — no production code
├── Dockerfile
└── .github/workflows/ci.yml    # ruff + mypy + pytest on every push
```

## Local setup

### Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation) 2.x
- Git

### 1. Clone the repo

```bash
git clone https://github.com/Alirezaemami97/Transaction-Fraud-Detection.git
cd Transaction-Fraud-Detection
```

### 2. Install dependencies

```bash
poetry install
```

This creates a virtualenv and installs all production and dev dependencies from `poetry.lock`.

### 3. Get the data

Download the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) from Kaggle (free account required) and place both files here:

```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

The `data/` directory is gitignored — it never gets committed.

**Kaggle CLI (fastest):**

```bash
# Set your token (from kaggle.com → Settings → API)
export KAGGLE_API_TOKEN='{"username":"...","key":"..."}'

poetry run kaggle competitions download -c ieee-fraud-detection -p data/raw/
unzip data/raw/ieee-fraud-detection.zip -d data/raw/
```

### 4. Verify CI passes

```bash
poetry run ruff check .
poetry run mypy src/
poetry run pytest tests/ -q
```

All 23 tests should pass with no ruff or mypy errors.

## Usage

### Train

```bash
poetry run python -m fraud_detection.training.train
```

Loads the raw data, applies the time-based split, trains LightGBM with all params from `config/config.yaml`, logs the run to MLflow, and registers the model under the name `fraud-lgbm`.

Training takes ~3–5 minutes on a laptop (500 trees, ~220k training rows).

### Inspect the MLflow run

```bash
poetry run mlflow ui
# open http://localhost:5000
```

The `fraud-detection` experiment will contain the run with logged params, PR-AUC, ROC-AUC, and the model artifact.

### Evaluate

```bash
poetry run python -m fraud_detection.evaluation.evaluate
```

Loads the registered model, runs the full evaluation panel, and prints:
- PR-AUC / ROC-AUC
- Cost-optimal threshold (minimises `fp_cost × FP + fn_cost × FN`)
- False positive rate and false negative rate per `ProductCD` segment (fairness check)
- LightGBM vs Isolation Forest PR-AUC comparison

### Serve (local)

```bash
poetry run uvicorn fraud_detection.serving.app:app --host 0.0.0.0 --port 8000
```

The app loads the registered MLflow model at startup and exposes two endpoints.

**Health check:**

```bash
curl http://localhost:8000/health
# {"status":"ok","model_name":"fraud-lgbm"}
```

**Score a transaction:**

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"TransactionAmt": 150.0, "ProductCD": "W", "TransactionDT": 86400}}'
# {"fraud_score": 0.082, "decision": "LEGIT", "threshold": 0.408}
```

Every scored transaction is appended to `logs/predictions.jsonl` for later label joining.

### Serve (Docker)

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

> **Note:** The Docker container needs access to the MLflow model registry. For local use, mount the `mlruns/` directory: `docker run -p 8000:8000 -v $(pwd)/mlruns:/app/mlruns fraud-detection`

## Configuration

All parameters live in `config/config.yaml` — nothing is hard-coded in the source:

```yaml
training:
  lgbm:
    n_estimators: 500
    learning_rate: 0.05
    num_leaves: 63
    scale_pos_weight: 28   # handles ~3.5% fraud rate imbalance

evaluation:
  fp_cost: 10              # cost of blocking a legitimate customer
  fn_cost: 100             # cost of missing a fraud

serving:
  decision_threshold: 0.408  # cost-optimal threshold from evaluation
```

To retrain with different hyperparameters, edit `config.yaml` and re-run `train.py`. The new run will be tracked in MLflow alongside the previous one.

## Key design decisions

**Time-based split, not random.** The data has a time axis. A random split leaks future transaction patterns into training and inflates every metric. Training on the first 60 days and testing on the remaining ~120 days reflects how the model will actually be used.

**Shared feature pipeline.** `build_features()` in `src/fraud_detection/features/pipeline.py` is imported by both `train.py` and `app.py`. There is no separate "preprocessing for training" and "preprocessing for serving" — one function, always in sync.

**Cost-based threshold.** The default 0.5 decision threshold is arbitrary. The evaluation step sweeps all thresholds and picks the one that minimises `fp_cost × FP + fn_cost × FN`. With `fn_cost = 10 × fp_cost`, the optimal threshold (0.408) is below 0.5 — the model accepts more false alarms to catch more fraud.

**LightGBM `category` dtype.** Categorical columns are cast to pandas `category` dtype before training. LightGBM handles these natively with its own split algorithm — better than one-hot encoding for high-cardinality columns like email domains, and no encoding step needed at serving time.

**`create_app()` factory for testability.** The FastAPI app is built by a factory function that accepts an injectable model and config. Tests pass a `_FakeModel` stub directly — no MLflow dependency, no patching, deterministic results.

## Milestones

- [x] M1 — Scaffold: repo structure, CI, Dockerfile, data loading + schema validation
- [x] M2 — Feature pipeline + EDA notebook
- [x] M3 — Training pipeline + MLflow + model registry
- [x] M4 — Evaluation panel + cost-based threshold + fairness check + Isolation Forest
- [x] M5 — FastAPI serving + prediction logging + Dockerfile wired
