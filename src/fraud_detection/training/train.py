"""Training entry point.

Run with:
    poetry run python -m fraud_detection.training.train
"""

import json
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.config import Config, load_config
from fraud_detection.data.loader import load_raw
from fraud_detection.features.pipeline import build_features, extract_target

logger = logging.getLogger(__name__)


def time_based_split(
    df: pd.DataFrame,
    cutoff_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train, test) on the TransactionDT time axis.

    All rows with TransactionDT < cutoff_days * 86400 go to train.
    Everything on or after goes to test.
    """
    cutoff_seconds = cutoff_days * 86_400
    train = df[df["TransactionDT"] < cutoff_seconds].copy()
    test = df[df["TransactionDT"] >= cutoff_seconds].copy()
    logger.info(
        "Time split: train=%d rows, test=%d rows (cutoff day %d)",
        len(train),
        len(test),
        cutoff_days,
    )
    return train, test


def run_training(config: Config, repo_root: Path) -> str:
    """Load data, train model, log to MLflow, register. Returns the run ID."""
    # ── 1. Load raw data ─────────────────────────────────────────────────────
    df, stats = load_raw(
        transaction_path=repo_root / config.data.transaction_path,
        identity_path=repo_root / config.data.identity_path,
    )
    logger.info("Loaded %d rows, fraud rate %.2f%%", stats.n_rows, stats.fraud_rate * 100)

    # ── 2. Time-based split ───────────────────────────────────────────────────
    train_df, test_df = time_based_split(df, config.data.train_cutoff_days)

    X_train = build_features(train_df, config.features)
    y_train = extract_target(train_df, config.data.target_column)

    X_test = build_features(test_df, config.features)
    y_test = extract_target(test_df, config.data.target_column)

    # ── 3. Train LightGBM ─────────────────────────────────────────────────────
    lgbm_params = config.training.lgbm.model_dump()
    model = lgb.LGBMClassifier(**lgbm_params, random_state=config.training.random_seed)

    logger.info("Fitting LightGBM with params: %s", lgbm_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(period=100)],
    )

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    logger.info("PR-AUC=%.4f  ROC-AUC=%.4f", pr_auc, roc_auc)

    # ── 5. Log to MLflow ─────────────────────────────────────────────────────
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "cutoff_days": config.data.train_cutoff_days,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                **lgbm_params,
                "random_seed": config.training.random_seed,
            }
        )
        mlflow.log_metrics({"pr_auc": pr_auc, "roc_auc": roc_auc})

        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name=config.mlflow.model_name,
        )

        # Log feature schema so the serving app can reconstruct full-width
        # feature vectors from partial transaction payloads.
        schema = {
            "feature_names": X_train.columns.tolist(),
            "cat_features": X_train.select_dtypes("category").columns.tolist(),
        }
        schema_file = Path("feature_schema.json")
        schema_file.write_text(json.dumps(schema))
        mlflow.log_artifact(str(schema_file), artifact_path="schema")
        schema_file.unlink()

        run_id: str = run.info.run_id
        logger.info(
            "MLflow run %s complete — model registered as '%s'", run_id, config.mlflow.model_name
        )

    return run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    repo_root = Path(__file__).parents[3]  # training/ → fraud_detection/ → src/ → repo root
    cfg = load_config(repo_root / "config/config.yaml")
    run_training(cfg, repo_root)
