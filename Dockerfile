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

# LightGBM runtime dependency (provides libgomp.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Non-root user for security best-practice
RUN groupadd --system appgroup \
 && useradd  --system --gid appgroup --home /app appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Application source
COPY main.py ./main.py
COPY data_ingestion.py ./data_ingestion.py
COPY feature_engineering.py ./feature_engineering.py
COPY pipeline/ ./pipeline/

# Make Python aware of the current repo layout
ENV PYTHONPATH="/app:/app/pipeline"
ENV ARTIFACTS_DIR="/app/artifacts"
ENV MODEL_VERSION="1.0.0"

RUN mkdir -p /app/artifacts

# Drop to non-root
USER appuser

EXPOSE 8000

# Healthcheck so Docker / Compose can track liveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]