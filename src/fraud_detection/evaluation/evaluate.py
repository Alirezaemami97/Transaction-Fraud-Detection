"""Evaluation entry point.

Loads the registered LightGBM model from MLflow, runs the full evaluation
panel, and prints a human-readable report.

Run with:
    poetry run python -m fraud_detection.evaluation.evaluate
"""

import logging
from pathlib import Path

import mlflow.lightgbm
import numpy as np
import numpy.typing as npt

from fraud_detection.config import Config, load_config
from fraud_detection.data.loader import load_raw
from fraud_detection.evaluation.panel import (
    compare_isolation_forest,
    compute_metrics,
    fairness_check,
    find_cost_threshold,
)
from fraud_detection.features.pipeline import build_features, extract_target
from fraud_detection.training.train import time_based_split

logger = logging.getLogger(__name__)


def run_evaluation(config: Config, repo_root: Path) -> None:
    # ── 1. Load data and split (same as training) ─────────────────────────────
    df, stats = load_raw(
        transaction_path=repo_root / config.data.transaction_path,
        identity_path=repo_root / config.data.identity_path,
    )
    train_df, test_df = time_based_split(df, config.data.train_cutoff_days)

    X_train = build_features(train_df, config.features)
    X_test = build_features(test_df, config.features)
    y_test = extract_target(test_df, config.data.target_column)

    # ── 2. Load registered model from MLflow ──────────────────────────────────
    model_uri = f"models:/{config.mlflow.model_name}/latest"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.lightgbm.load_model(model_uri)
    y_prob: npt.NDArray[np.float64] = model.predict_proba(X_test)[:, 1]

    # ── 3. Metric panel ────────────────────────────────────────────────────────
    panel = compute_metrics(y_test, y_prob)

    # ── 4. Cost-based threshold ────────────────────────────────────────────────
    cost_result = find_cost_threshold(
        y_test,
        y_prob,
        fp_cost=config.evaluation.fp_cost,
        fn_cost=config.evaluation.fn_cost,
    )

    # ── 5. Fairness check (on ProductCD — a proxy for transaction type) ────────
    fairness = fairness_check(
        df=test_df.reset_index(drop=True),
        y_true=y_test.reset_index(drop=True),
        y_prob=y_prob,
        threshold=cost_result.threshold,
        segment_col="ProductCD",
    )

    # ── 6. Isolation Forest comparison ────────────────────────────────────────
    comparison = compare_isolation_forest(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        lgbm_pr_auc=panel.pr_auc,
        random_seed=config.training.random_seed,
    )

    # ── 7. Print report ────────────────────────────────────────────────────────
    _print_report(panel, cost_result, fairness, comparison, config)


def _print_report(
    panel: object,
    cost_result: object,
    fairness: object,
    comparison: object,
    config: Config,
) -> None:
    from fraud_detection.evaluation.panel import (
        ComparisonResult,
        CostResult,
        FairnessResult,
        MetricPanel,
    )

    assert isinstance(panel, MetricPanel)
    assert isinstance(cost_result, CostResult)
    assert isinstance(fairness, FairnessResult)
    assert isinstance(comparison, ComparisonResult)

    sep = "─" * 60
    print(f"\n{sep}")
    print("FRAUD DETECTION — EVALUATION REPORT")
    print(sep)

    print("\n[1] METRIC PANEL")
    print(f"    PR-AUC   : {panel.pr_auc:.4f}  (primary — use for imbalanced data)")
    print(f"    ROC-AUC  : {panel.roc_auc:.4f}")

    print("\n[2] COST-BASED THRESHOLD")
    print(f"    fp_cost={config.evaluation.fp_cost}  fn_cost={config.evaluation.fn_cost}")
    print(f"    Optimal threshold : {cost_result.threshold:.4f}")
    print(f"    False positives   : {cost_result.false_positives:,}")
    print(f"    False negatives   : {cost_result.false_negatives:,}")
    print(f"    Total cost        : {cost_result.total_cost:,.0f}")

    print("\n[3] FAIRNESS CHECK  (segment: ProductCD)")
    print(f"    {'Segment':<12}  {'FPR':>8}  {'FNR':>8}")
    for seg in sorted(fairness.fpr_by_segment):
        fpr = fairness.fpr_by_segment[seg]
        fnr = fairness.fnr_by_segment[seg]
        print(f"    {seg:<12}  {fpr:>8.4f}  {fnr:>8.4f}")

    print("\n[4] MODEL COMPARISON")
    print(f"    {comparison.recommendation}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    repo_root = Path(__file__).parents[3]
    cfg = load_config(repo_root / "config/config.yaml")
    run_evaluation(cfg, repo_root)
