import logging
from pathlib import Path

import pandas as pd

from fraud_detection.data.schema import DatasetStats, validate_schema

logger = logging.getLogger(__name__)


def load_raw(
    transaction_path: str | Path,
    identity_path: str | Path,
) -> tuple[pd.DataFrame, DatasetStats]:
    """Load and join the raw IEEE-CIS CSVs, then validate the schema.

    Returns the merged DataFrame and a DatasetStats summary.
    Raises ValueError if schema validation fails.
    """
    transaction_path = Path(transaction_path)
    identity_path = Path(identity_path)

    logger.info("Reading transactions from %s", transaction_path)
    transactions = pd.read_csv(transaction_path)
    logger.info("Transactions: %d rows, %d columns", *transactions.shape)

    logger.info("Reading identity data from %s", identity_path)
    identity = pd.read_csv(identity_path)
    logger.info("Identity: %d rows, %d columns", *identity.shape)

    # Left join: every transaction is kept; identity columns are NaN when absent.
    df = transactions.merge(identity, on="TransactionID", how="left")
    logger.info("Merged: %d rows, %d columns", *df.shape)

    stats = validate_schema(df)
    logger.info(
        "Schema OK — %d rows | fraud rate %.2f%% | missing %.2f%%",
        stats.n_rows,
        stats.fraud_rate * 100,
        stats.missing_rate * 100,
    )

    return df, stats
