import pandas as pd
from pydantic import BaseModel, field_validator

# Columns the rest of the system depends on — absent or wrong-typed means fail fast.
REQUIRED_COLUMNS: dict[str, str] = {
    "TransactionID": "int64",
    "TransactionDT": "int64",   # seconds elapsed from a reference point, not a timestamp
    "TransactionAmt": "float64",
    "ProductCD": "object",
    "isFraud": "int64",
}


class DatasetStats(BaseModel):
    """Summary stats produced after successful schema validation."""

    n_rows: int
    n_fraud: int
    fraud_rate: float
    missing_rate: float

    @field_validator("fraud_rate", "missing_rate")
    @classmethod
    def must_be_fraction(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Expected a value in [0, 1], got {v}")
        return v


def validate_schema(df: pd.DataFrame) -> DatasetStats:
    """Validate that *df* contains the required columns and return summary stats.

    Raises ValueError if any required column is missing.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return DatasetStats(
        n_rows=len(df),
        n_fraud=int(df["isFraud"].sum()),
        fraud_rate=float(df["isFraud"].mean()),
        missing_rate=float(df.isnull().mean().mean()),
    )
