"""Evaluation panel for the fraud detection model.

Four functions — each answers one business question:
  compute_metrics()       → how good is the model overall?
  find_cost_threshold()   → what threshold minimises expected cost?
  fairness_check()        → does the model harm one segment more than others?
  compare_isolation_forest() → is supervised better than unsupervised here?
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class MetricPanel:
    pr_auc: float
    roc_auc: float
    # Full curve stored so callers can plot without recomputing
    precisions: npt.NDArray[np.float64]
    recalls: npt.NDArray[np.float64]
    thresholds: npt.NDArray[np.float64]


@dataclass
class CostResult:
    threshold: float
    total_cost: float
    false_positives: int
    false_negatives: int


@dataclass
class FairnessResult:
    segment_col: str
    # Per-segment: false positive rate and false negative rate
    fpr_by_segment: dict[str, float] = field(default_factory=dict)
    fnr_by_segment: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    lgbm_pr_auc: float
    iso_forest_pr_auc: float
    recommendation: str


# ── Functions ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_prob: npt.NDArray[np.float64]) -> MetricPanel:
    """Compute the headline metric panel for a binary classifier.

    PR-AUC is the primary metric for imbalanced fraud data.
    ROC-AUC is included for familiarity but is misleading on its own
    when the positive class is rare.
    """
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    logger.info("PR-AUC=%.4f  ROC-AUC=%.4f", pr_auc, roc_auc)
    return MetricPanel(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precisions=precisions,
        recalls=recalls,
        thresholds=thresholds,
    )


def find_cost_threshold(
    y_true: pd.Series,
    y_prob: npt.NDArray[np.float64],
    fp_cost: float,
    fn_cost: float,
) -> CostResult:
    """Find the decision threshold that minimises total business cost.

    At each threshold t:
        predicted_positive = y_prob >= t
        FP = predicted fraud that was actually legit  (annoyed customer)
        FN = predicted legit that was actually fraud  (financial loss)
        total_cost = fp_cost * FP + fn_cost * FN

    The optimal threshold is the t that minimises total_cost.
    It will be lower than 0.5 when fn_cost > fp_cost (i.e. missing fraud
    is more expensive than a false alarm), which is typical for fraud.
    """
    _, _, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold = thresholds[0]
    best_cost = float("inf")
    best_fp = 0
    best_fn = 0

    y_true_arr = np.array(y_true)

    for t in thresholds:
        predicted = (y_prob >= t).astype(int)
        fp = int(((predicted == 1) & (y_true_arr == 0)).sum())
        fn = int(((predicted == 0) & (y_true_arr == 1)).sum())
        cost = fp_cost * fp + fn_cost * fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)
            best_fp = fp
            best_fn = fn

    logger.info(
        "Cost-optimal threshold=%.4f  cost=%.0f  FP=%d  FN=%d",
        best_threshold,
        best_cost,
        best_fp,
        best_fn,
    )
    return CostResult(
        threshold=best_threshold,
        total_cost=best_cost,
        false_positives=best_fp,
        false_negatives=best_fn,
    )


def fairness_check(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_prob: npt.NDArray[np.float64],
    threshold: float,
    segment_col: str,
) -> FairnessResult:
    """Compute false positive rate and false negative rate per segment.

    FPR = FP / (FP + TN): fraction of legit customers wrongly flagged.
    FNR = FN / (FN + TP): fraction of fraudsters the model misses.

    A fair model has similar FPR/FNR across segments. Large gaps indicate
    the model is systematically harming (or under-protecting) one group.
    """
    predicted = (y_prob >= threshold).astype(int)
    result = FairnessResult(segment_col=segment_col)

    for segment in df[segment_col].dropna().unique():
        mask = df[segment_col] == segment
        yt = np.array(y_true)[mask]
        yp = predicted[mask]

        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        seg_key = str(segment)
        result.fpr_by_segment[seg_key] = round(fpr, 4)
        result.fnr_by_segment[seg_key] = round(fnr, 4)

    logger.info("Fairness check on '%s': %d segments", segment_col, len(result.fpr_by_segment))
    return result


def compare_isolation_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    lgbm_pr_auc: float,
    random_seed: int = 42,
) -> ComparisonResult:
    """Train an Isolation Forest and compare its PR-AUC to LightGBM's.

    Isolation Forest is unsupervised — it never sees the fraud labels during
    training. It flags transactions that are anomalous relative to the bulk
    of the data. This makes it useful for novel fraud that doesn't match
    historical patterns, but noisier for known patterns.

    Scores are negated (more negative = more anomalous in sklearn) then
    rescaled to [0,1] so they are comparable to LightGBM probabilities.
    """
    # Isolation Forest cannot handle category dtype — use numeric columns only
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    X_train_num = X_train[num_cols].fillna(-999)
    X_test_num = X_test[num_cols].fillna(-999)

    iso = IsolationForest(n_estimators=100, random_state=random_seed, n_jobs=-1)
    iso.fit(X_train_num)

    # decision_function: higher = more normal. Negate so higher = more anomalous.
    raw_scores = -iso.decision_function(X_test_num)

    # Rescale to [0, 1] for interpretability
    min_s, max_s = raw_scores.min(), raw_scores.max()
    iso_scores = (raw_scores - min_s) / (max_s - min_s) if max_s > min_s else raw_scores

    iso_pr_auc = average_precision_score(y_test, iso_scores)

    if lgbm_pr_auc >= iso_pr_auc:
        rec = (
            f"LightGBM (PR-AUC={lgbm_pr_auc:.4f}) outperforms "
            f"Isolation Forest (PR-AUC={iso_pr_auc:.4f}). "
            "Use LightGBM: supervised learning leverages known fraud labels effectively."
        )
    else:
        rec = (
            f"Isolation Forest (PR-AUC={iso_pr_auc:.4f}) outperforms "
            f"LightGBM (PR-AUC={lgbm_pr_auc:.4f}). "
            "Consider unsupervised scoring for novel fraud patterns."
        )

    logger.info("LightGBM PR-AUC=%.4f vs IsoForest PR-AUC=%.4f", lgbm_pr_auc, iso_pr_auc)
    return ComparisonResult(
        lgbm_pr_auc=lgbm_pr_auc,
        iso_forest_pr_auc=iso_pr_auc,
        recommendation=rec,
    )
