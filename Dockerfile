# -----------------------------------------------------------------------
# Stage 1 — builder
# Installs Python dependencies in an isolated layer so the final image
# doesn't need build tooling (gcc, pip cache, etc.)
# -----------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------
# Stage 2 — runtime (lean, production image)
# -----------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security best-practice
RUN groupadd --system appgroup \
 && useradd  --system --gid appgroup --home /app appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Application source
COPY src/api/    ./src/api/
COPY src/pipeline/feature_engineering.py ./src/pipeline/
COPY artifacts/  ./artifacts/

# Make Python aware of the src package paths
ENV PYTHONPATH="/app/src/api:/app/src/pipeline"
ENV ARTIFACTS_DIR="/app/artifacts"
ENV MODEL_VERSION="1.0.0"

# Drop to non-root
USER appuser

EXPOSE 8000

# Healthcheck so Docker / Compose can track liveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]