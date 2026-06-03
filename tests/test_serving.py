"""Tests for the FastAPI serving app."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.config import (
    Config,
    DataConfig,
    EvaluationConfig,
    FeaturesConfig,
    LGBMConfig,
    MLflowConfig,
    ServingConfig,
    TrainingConfig,
)
from fraud_detection.serving.app import create_app

# ── Fake model ────────────────────────────────────────────────────────────────

class _FakeModel:
    """Minimal scorer stub that returns a fixed probability."""

    def __init__(self, score: float = 0.3) -> None:
        self._score = score

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.array([[1 - self._score, self._score]] * len(X))


def _make_config(threshold: float = 0.5) -> Config:
    return Config(
        data=DataConfig(
            transaction_path="data/raw/train_transaction.csv",
            identity_path="data/raw/train_identity.csv",
            train_cutoff_days=60,
            target_column="isFraud",
        ),
        features=FeaturesConfig(numeric_fill_value=-999, cat_fill_value="missing"),
        training=TrainingConfig(
            random_seed=42,
            test_size=0.2,
            lgbm=LGBMConfig(
                n_estimators=10,
                learning_rate=0.05,
                num_leaves=15,
                scale_pos_weight=28,
            ),
        ),
        evaluation=EvaluationConfig(fp_cost=10, fn_cost=100),
        mlflow=MLflowConfig(experiment_name="test", model_name="test-model"),
        serving=ServingConfig(
            host="0.0.0.0",
            port=8000,
            model_stage="None",
            decision_threshold=threshold,
        ),
    )


_SAMPLE_TRANSACTION: dict[str, Any] = {
    "TransactionAmt": 150.0,
    "ProductCD": "W",
    "TransactionDT": 100_000,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    test_app = create_app(
        config=_make_config(threshold=0.5),
        model=_FakeModel(score=0.3),
        log_path=tmp_path / "predictions.jsonl",
    )
    return TestClient(test_app)


@pytest.fixture
def high_score_client(tmp_path: Path) -> TestClient:
    test_app = create_app(
        config=_make_config(threshold=0.5),
        model=_FakeModel(score=0.8),
        log_path=tmp_path / "predictions.jsonl",
    )
    return TestClient(test_app)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client: TestClient) -> None:
    with client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_score_returns_legit_when_low_score(client: TestClient) -> None:
    """Score 0.3 < threshold 0.5 → LEGIT."""
    with client:
        resp = client.post("/score", json={"transaction": _SAMPLE_TRANSACTION})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "LEGIT"
    assert 0.0 <= body["fraud_score"] <= 1.0


def test_score_returns_fraud_when_high_score(high_score_client: TestClient) -> None:
    """Score 0.8 >= threshold 0.5 → FRAUD."""
    with high_score_client:
        resp = high_score_client.post("/score", json={"transaction": _SAMPLE_TRANSACTION})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "FRAUD"


def test_score_includes_threshold_in_response(client: TestClient) -> None:
    with client:
        resp = client.post("/score", json={"transaction": _SAMPLE_TRANSACTION})
    assert resp.json()["threshold"] == 0.5


def test_prediction_is_logged(tmp_path: Path) -> None:
    """Scoring a transaction should write one line to the prediction log."""
    import json

    log_file = tmp_path / "predictions.jsonl"
    test_app = create_app(
        config=_make_config(),
        model=_FakeModel(score=0.3),
        log_path=log_file,
    )
    with TestClient(test_app) as c:
        c.post("/score", json={"transaction": _SAMPLE_TRANSACTION})

    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "ts" in record
    assert "score" in record
    assert record["decision"] in ("FRAUD", "LEGIT")


def test_timing_header_present(client: TestClient) -> None:
    """Every response should carry an X-Response-Time-Ms header."""
    with client:
        resp = client.get("/health")
    assert "X-Response-Time-Ms" in resp.headers
