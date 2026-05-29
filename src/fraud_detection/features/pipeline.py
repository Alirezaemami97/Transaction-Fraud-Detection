import pandas as pd

from fraud_detection.config import FeaturesConfig

# TransactionID is a row key, not a signal. isFraud is the label — serving never has it.
_DROP_COLUMNS = ["TransactionID", "isFraud"]


def build_features(df: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Transform a raw (merged) DataFrame into a model-ready feature matrix.

    Called by BOTH training and serving — identical code path is the guarantee
    against training-serving skew.

    Args:
        df: Raw merged DataFrame (transactions left-joined with identity).
        config: FeaturesConfig with fill sentinels.

    Returns:
        Feature matrix with no identifier or target columns.
    """
    feature_df = df.drop(columns=[c for c in _DROP_COLUMNS if c in df.columns])

    num_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Fill missing values with sentinels defined in config — never hardcode here.
    feature_df = feature_df.copy()
    feature_df[num_cols] = feature_df[num_cols].fillna(config.numeric_fill_value)
    feature_df[cat_cols] = feature_df[cat_cols].fillna(config.cat_fill_value)

    # category dtype lets LightGBM use its native categorical split logic,
    # which is better than one-hot for high-cardinality columns like email domain.
    for col in cat_cols:
        feature_df[col] = feature_df[col].astype("category")

    return feature_df


def extract_target(df: pd.DataFrame, target_col: str = "isFraud") -> pd.Series:
    """Extract the binary target column as an integer Series."""
    return df[target_col].astype(int)
