"""Behavioural tests for the training pipeline.

These tests use a tiny synthetic DataFrame — no real data required.
They verify that the model outputs are well-formed and directionally correct,
not that specific metric values are achieved.
"""

import pandas as pd
import pytest

from fraud_detection.config import (
    FeaturesConfig,
)
from fraud_detection.features.pipeline import build_features, extract_target
from fraud_detection.training.train import time_based_split

# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_synthetic_df(n_legit: int = 80, n_fraud: int = 20) -> pd.DataFrame:
    """Build a minimal DataFrame that mirrors the IEEE-CIS structure.

    Fraud rows have high TransactionAmt; legit rows have low amounts.
    This gives LightGBM an obvious signal to learn so even a tiny model
    trained on synthetic data will produce directionally correct scores.
    """
    import numpy as np

    rng = np.random.default_rng(42)

    legit = pd.DataFrame({
        "TransactionDT": rng.integers(0, 60 * 86_400, n_legit),
        "TransactionAmt": rng.uniform(1, 50, n_legit),
        "ProductCD": rng.choice(["W", "H", "C"], n_legit),
        "isFraud": [0] * n_legit,
    })
    fraud = pd.DataFrame({
        "TransactionDT": rng.integers(0, 60 * 86_400, n_fraud),
        "TransactionAmt": rng.uniform(5_000, 10_000, n_fraud),
        "ProductCD": rng.choice(["W", "H", "C"], n_fraud),
        "isFraud": [1] * n_fraud,
    })
    return pd.concat([legit, fraud], ignore_index=True)


_FEATURES_CFG = FeaturesConfig(numeric_fill_value=-999, cat_fill_value="missing")


# ── time_based_split tests ────────────────────────────────────────────────────

def test_time_split_is_exhaustive() -> None:
    """Train + test together must cover every row in the input."""
    df = _make_synthetic_df()
    train, test = time_based_split(df, cutoff_days=30)
    assert len(train) + len(test) == len(df)


def test_time_split_no_future_leakage() -> None:
    """Train set must contain no rows at or after the cutoff."""
    df = _make_synthetic_df()
    cutoff_seconds = 30 * 86_400
    train, _ = time_based_split(df, cutoff_days=30)
    assert (train["TransactionDT"] >= cutoff_seconds).sum() == 0


def test_time_split_test_contains_only_post_cutoff() -> None:
    """Test set must contain no rows before the cutoff."""
    df = _make_synthetic_df()
    cutoff_seconds = 30 * 86_400
    _, test = time_based_split(df, cutoff_days=30)
    assert (test["TransactionDT"] < cutoff_seconds).sum() == 0


# ── LightGBM behavioural tests ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model() -> object:
    """Train a tiny LightGBM on synthetic data. Shared across tests in this module."""
    import lightgbm as lgb

    df = _make_synthetic_df(n_legit=200, n_fraud=50)
    X = build_features(df, _FEATURES_CFG)
    y = extract_target(df)

    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=15, random_state=42)
    model.fit(X, y)
    return model


def test_scores_are_probabilities(trained_model: object) -> None:
    """All predicted probabilities must be in [0, 1]."""
    import lightgbm as lgb
    import numpy as np

    model: lgb.LGBMClassifier = trained_model  # type: ignore[assignment]
    df = _make_synthetic_df()
    X = build_features(df, _FEATURES_CFG)
    probs = model.predict_proba(X)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_high_amount_scores_higher_than_low_amount(trained_model: object) -> None:
    """A transaction with a suspiciously high amount should score higher than a tiny one."""
    import lightgbm as lgb
    import pandas as pd

    model: lgb.LGBMClassifier = trained_model  # type: ignore[assignment]

    legit_like = pd.DataFrame(
        {"TransactionDT": [1000], "TransactionAmt": [10.0], "ProductCD": ["W"]}
    )
    fraud_like = pd.DataFrame(
        {"TransactionDT": [1000], "TransactionAmt": [9000.0], "ProductCD": ["W"]}
    )

    score_legit = model.predict_proba(build_features(legit_like, _FEATURES_CFG))[:, 1][0]
    score_fraud = model.predict_proba(build_features(fraud_like, _FEATURES_CFG))[:, 1][0]

    assert score_fraud > score_legit, (
        f"Expected fraud-like row to score higher: fraud={score_fraud:.4f}, legit={score_legit:.4f}"
    )
