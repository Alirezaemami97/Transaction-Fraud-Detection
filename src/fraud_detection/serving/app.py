"""FastAPI serving app.

Production startup:
    poetry run uvicorn fraud_detection.serving.app:app --host 0.0.0.0 --port 8000

Docker:
    docker run -p 8000:8000 fraud-detection
"""

import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from fraud_detection.config import Config, load_config
from fraud_detection.features.pipeline import build_features
from fraud_detection.monitoring.prediction_log import log_prediction
from fraud_detection.serving.schemas import HealthResponse, ScoreRequest, ScoreResponse

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("logs/predictions.jsonl")


class _TimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Response-Time-Ms header and logs latency for every request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
        logger.info("%.1fms  %s %s", ms, request.method, request.url.path)
        return response


def create_app(
    config: Config | None = None,
    model: object | None = None,
    log_path: Path = _DEFAULT_LOG_PATH,
) -> FastAPI:
    """Build the FastAPI application.

    Pass config + model explicitly in tests to skip MLflow. In production,
    leave both as None and they are loaded from disk/registry at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if config is not None and model is not None:
            app.state.config = config
            app.state.model = model
        else:
            repo_root = Path(__file__).parents[3]
            cfg = load_config(repo_root / "config/config.yaml")
            app.state.config = cfg
            app.state.model = mlflow.lightgbm.load_model(
                f"models:/{cfg.mlflow.model_name}/latest"
            )
        app.state.log_path = log_path
        yield

    _app = FastAPI(
        title="Fraud Detection API",
        version="1.0.0",
        lifespan=lifespan,
    )
    _app.add_middleware(_TimingMiddleware)

    @_app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Liveness check — confirms the app is up and which model is loaded."""
        return HealthResponse(
            status="ok",
            model_name=request.app.state.config.mlflow.model_name,
        )

    @_app.post("/score", response_model=ScoreResponse)
    def score(body: ScoreRequest, request: Request) -> ScoreResponse:
        """Score a single transaction.

        Accepts any transaction fields; missing values are filled with
        sentinels by build_features() — the same logic used at training time.
        """
        cfg: Config = request.app.state.config

        # Build a one-row DataFrame from the transaction dict
        df = pd.DataFrame([body.transaction])
        X = build_features(df, cfg.features)

        fraud_score: float = float(
            request.app.state.model.predict_proba(X)[:, 1][0]
        )
        threshold: float = cfg.serving.decision_threshold
        decision = "FRAUD" if fraud_score >= threshold else "LEGIT"

        log_prediction(
            transaction=body.transaction,
            score=fraud_score,
            decision=decision,
            log_path=request.app.state.log_path,
        )

        return ScoreResponse(
            fraud_score=fraud_score,
            decision=decision,  # type: ignore[arg-type]
            threshold=threshold,
        )

    return _app


# Production instance — loaded by uvicorn / Docker
app = create_app()
