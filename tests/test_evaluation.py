"""Tests for the evaluation panel functions."""

import numpy as np
import pandas as pd

from fraud_detection.evaluation.panel import fairness_check, find_cost_threshold


def _binary_case() -> tuple[pd.Series, np.ndarray]:
    """10 samples with overlapping scores to force a real cost trade-off.

    Fraud scores: 0.9, 0.4, 0.95, 0.8, 0.35
    Legit scores: 0.1, 0.3, 0.6, 0.15, 0.05

    The overlap (e.g. 0.35 fraud vs 0.3 legit, 0.6 legit vs 0.4 fraud) means
    no threshold achieves zero cost — there is always a FP/FN trade-off.
    """
    y_true = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.15, 0.05, 0.9, 0.4, 0.95, 0.8, 0.35])
    return y_true, y_prob


def test_cost_threshold_fn_heavy_lower_than_fp_heavy() -> None:
    """When fn is expensive, optimal threshold must be lower than when fp is expensive.

    High fn_cost → lower threshold (catch more fraud, accept more false alarms).
    High fp_cost → higher threshold (avoid false alarms, accept missing some fraud).
    """
    y_true, y_prob = _binary_case()
    result_fn_heavy = find_cost_threshold(y_true, y_prob, fp_cost=1, fn_cost=100)
    result_fp_heavy = find_cost_threshold(y_true, y_prob, fp_cost=100, fn_cost=1)
    assert result_fn_heavy.threshold <= result_fp_heavy.threshold


def test_cost_result_counts_are_non_negative() -> None:
    y_true, y_prob = _binary_case()
    result = find_cost_threshold(y_true, y_prob, fp_cost=10, fn_cost=100)
    assert result.false_positives >= 0
    assert result.false_negatives >= 0
    assert result.total_cost >= 0


def test_fairness_check_returns_all_segments() -> None:
    """Every unique value in segment_col should appear in the result."""
    df = pd.DataFrame({"ProductCD": ["W", "W", "H", "H", "C", "C"]})
    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
    result = fairness_check(df, y_true, y_prob, threshold=0.5, segment_col="ProductCD")
    assert set(result.fpr_by_segment.keys()) == {"W", "H", "C"}
    assert set(result.fnr_by_segment.keys()) == {"W", "H", "C"}


def test_fairness_rates_are_in_unit_interval() -> None:
    """FPR and FNR must be valid rates in [0, 1]."""
    df = pd.DataFrame({"ProductCD": ["W", "W", "H", "H"]})
    y_true = pd.Series([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.6])
    result = fairness_check(df, y_true, y_prob, threshold=0.5, segment_col="ProductCD")
    for seg in result.fpr_by_segment:
        assert 0.0 <= result.fpr_by_segment[seg] <= 1.0
        assert 0.0 <= result.fnr_by_segment[seg] <= 1.0
