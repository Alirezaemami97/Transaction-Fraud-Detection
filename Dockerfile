# --- Build stage: install dependencies ---
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install poetry==2.4.1

COPY pyproject.toml poetry.lock ./

# Install only production deps (no dev tools) into an in-project venv
RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-interaction --no-root

# --- Runtime stage: lean final image ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the built venv from the builder stage
COPY --from=builder /app/.venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source and config
COPY src/ src/
COPY config/ config/

# Make the fraud_detection package importable without a pip install step.
# The venv already has all deps; adding src/ to PYTHONPATH is enough.
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "fraud_detection.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
