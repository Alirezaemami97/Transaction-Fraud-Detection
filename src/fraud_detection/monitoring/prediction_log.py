"""Prediction archive — append one JSONL record per scored transaction.

Every score is logged so that when fraud labels arrive (chargebacks, manual
review outcomes) they can be joined against this archive to measure real-world
model performance after deployment.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOGGED_FIELDS = {"TransactionAmt", "ProductCD", "TransactionDT"}


def log_prediction(
    transaction: dict[str, Any],
    score: float,
    decision: str,
    log_path: Path,
) -> None:
    """Append one record to the prediction archive at log_path."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "score": round(score, 6),
        "decision": decision,
        # Log a small subset of fields for traceability without storing PII
        **{k: transaction[k] for k in _LOGGED_FIELDS if k in transaction},
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.debug("Logged prediction: score=%.4f decision=%s", score, decision)
