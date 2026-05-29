import math

import pandas as pd

from fraud_detection.config import FeaturesConfig
from fraud_detection.features.pipeline import build_features, extract_target

FILL_NUM = -999.0
FILL_CAT = "missing"

_CONFIG = FeaturesConfig(numeric_fill_value=FILL_NUM, cat_fill_value=FILL_CAT)


def _make_raw_df() -> pd.DataFrame:
    """Minimal raw DataFrame that mirrors the IEEE-CIS structure."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionAmt": [10.0, None, 30.0],
            "ProductCD": ["W", None, "C"],
            "isFraud": [0, 1, 0],
        }
    )


def test_build_features_drops_id_and_target() -> None:
    features = build_features(_make_raw_df(), _CONFIG)
    assert "TransactionID" not in features.columns
    assert "isFraud" not in features.columns


def test_build_features_fills_missing_numeric() -> None:
    features = build_features(_make_raw_df(), _CONFIG)
    assert math.isclose(features.loc[1, "TransactionAmt"], FILL_NUM)


def test_build_features_fills_missing_categorical() -> None:
    features = build_features(_make_raw_df(), _CONFIG)
    assert features.loc[1, "ProductCD"] == FILL_CAT


def test_build_features_categoricals_have_category_dtype() -> None:
    features = build_features(_make_raw_df(), _CONFIG)
    assert features["ProductCD"].dtype.name == "category"


def test_extract_target_returns_int_series() -> None:
    target = extract_target(_make_raw_df())
    assert target.dtype == int
    assert list(target) == [0, 1, 0]
