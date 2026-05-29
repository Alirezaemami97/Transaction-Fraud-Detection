import pandas as pd
import pytest

from fraud_detection.data.schema import validate_schema


def _make_valid_df(**overrides: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "TransactionID": [1, 2, 3],
        "TransactionDT": [100, 200, 300],
        "TransactionAmt": [10.0, 20.0, 30.0],
        "ProductCD": ["W", "H", "C"],
        "isFraud": [0, 0, 1],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_schema_passes_on_valid_data() -> None:
    stats = validate_schema(_make_valid_df())
    assert stats.n_rows == 3
    assert stats.n_fraud == 1
    assert abs(stats.fraud_rate - 1 / 3) < 1e-6


def test_validate_schema_fails_on_missing_column() -> None:
    df = _make_valid_df()
    df = df.drop(columns=["TransactionAmt"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_validate_schema_zero_fraud_rate() -> None:
    stats = validate_schema(_make_valid_df(isFraud=[0, 0, 0]))
    assert stats.fraud_rate == 0.0
    assert stats.n_fraud == 0
