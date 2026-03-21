"""
Prediction REST API
--------------------
FastAPI application that exposes the trained LightGBM customer-satisfaction
classifier via a versioned REST API with:
  - POST /api/v1/predict        — single-record inference
  - POST /api/v1/predict/batch  — batch inference
  - GET  /api/v1/health         — liveness probe
  - GET  /api/v1/model/info     — model metadata
  - GET  /metrics               — Prometheus metrics (scraped by Prometheus)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
API_VERSION   = "v1"

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Prediction request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)
PREDICTION_DISTRIBUTION = Counter(
    "prediction_class_total",
    "Distribution of predicted satisfaction levels",
    ["predicted_class"],
)
MODEL_PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Confidence score of predictions",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
ACTIVE_REQUESTS = Gauge("api_active_requests", "Number of active requests being processed")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Input schema for a single customer record."""
    age: int = Field(..., ge=18, le=100, example=28)
    gender: str = Field(..., example="Female")
    purchase_frequency: str = Field(..., example="weekly")
    preferred_category: str = Field(..., example="Electronics")
    average_order_value: float = Field(..., ge=0, example=120.5)
    browsing_time_minutes: float = Field(..., ge=0, example=45.0)
    add_to_cart_not_purchased: int = Field(..., ge=0, example=3)
    product_reviews_count: int = Field(..., ge=0, example=12)
    cart_abandonment_rate: float = Field(..., ge=0.0, le=1.0, example=0.35)
    ai_usage_frequency: str = Field(..., example="often")
    chatbot_usage: int = Field(..., ge=0, le=10, example=5)
    recommendation_usage: int = Field(..., ge=0, le=10, example=7)
    personalization_usage: int = Field(..., ge=0, le=10, example=6)
    trust_ai: float = Field(..., ge=0.0, le=1.0, example=0.72)
    perceived_usefulness: float = Field(..., ge=0.0, le=1.0, example=0.80)
    privacy_concern: float = Field(..., ge=0.0, le=1.0, example=0.30)

    @field_validator("purchase_frequency")
    @classmethod
    def validate_purchase_frequency(cls, v):
        allowed = {"daily", "weekly", "monthly", "rarely"}
        if v.lower() not in allowed:
            raise ValueError(f"purchase_frequency must be one of {allowed}")
        return v.lower()

    @field_validator("ai_usage_frequency")
    @classmethod
    def validate_ai_usage_frequency(cls, v):
        allowed = {"always", "often", "sometimes", "rarely", "never"}
        if v.lower() not in allowed:
            raise ValueError(f"ai_usage_frequency must be one of {allowed}")
        return v.lower()


class PredictionResponse(BaseModel):
    predicted_satisfaction: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class BatchPredictionRequest(BaseModel):
    records: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

class ModelState:
    model        = None
    preprocessor = None
    label_encoder = None
    model_version = "unknown"


state = ModelState()


def load_artifacts() -> None:
    """Load model and preprocessing artifacts from disk."""
    try:
        state.model         = joblib.load(ARTIFACTS_DIR / "model.joblib")
        state.preprocessor  = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
        state.label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
        state.model_version = os.getenv("MODEL_VERSION", "1.0.0")
        logger.info("Artifacts loaded successfully. Model version: %s", state.model_version)
    except Exception as exc:
        logger.error("Failed to load artifacts: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the feature engineering from the training pipeline."""
    df = df.copy()
    import numpy as np
    df["ai_usage_score"] = (
        df["chatbot_usage"] + df["recommendation_usage"] + df["personalization_usage"]
    ) / 3.0
    df["trust_index"] = df["trust_ai"] + df["perceived_usefulness"] - df["privacy_concern"]
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    ).astype(str)
    df["spending_tier"] = pd.qcut(
        df["average_order_value"],
        q=5,
        labels=["very_low", "low", "medium", "high", "very_high"],
        duplicates="drop",
    ).astype(str)
    df["engagement_score"] = (
        df["browsing_time_minutes"] * np.log1p(df["product_reviews_count"])
    )
    df["high_abandonment"] = (df["cart_abandonment_rate"] > 0.5).astype(int)
    return df


def predict_single(features: CustomerFeatures) -> PredictionResponse:
    df = pd.DataFrame([features.model_dump()])
    df = _engineer_features(df)
    X  = state.preprocessor.transform(df)

    proba = state.model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    classes  = state.label_encoder.classes_   # e.g. ["High", "Low"]

    confidence = float(proba[pred_idx])
    predicted_class = classes[pred_idx]
    prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    PREDICTION_DISTRIBUTION.labels(predicted_class=predicted_class).inc()
    MODEL_PREDICTION_CONFIDENCE.observe(confidence)

    return PredictionResponse(
        predicted_satisfaction=predicted_class,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        model_version=state.model_version,
    )


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Customer Satisfaction Prediction API",
    description=(
        "Predicts customer satisfaction / trust toward AI systems "
        "in online shopping environments. DDM501 Group 4."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(f"/api/{API_VERSION}/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        api_version=API_VERSION,
    )


@app.get(f"/api/{API_VERSION}/model/info", tags=["System"])
async def model_info():
    return {
        "model_version":   state.model_version,
        "model_type":      type(state.model).__name__,
        "api_version":     API_VERSION,
        "target_variable": "satisfaction_level",
        "output_classes":  list(state.label_encoder.classes_) if state.label_encoder else [],
    }


@app.post(
    f"/api/{API_VERSION}/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Predict satisfaction for a single customer",
)
async def predict(features: CustomerFeatures):
    """
    Given customer demographic, shopping behaviour, and AI-perception features,
    returns a predicted satisfaction level (High / Low) with confidence score.
    """
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        result = predict_single(features)
        REQUEST_COUNT.labels(endpoint="predict", status="success").inc()
        return result
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint="predict").observe(elapsed)
        ACTIVE_REQUESTS.dec()


@app.post(
    f"/api/{API_VERSION}/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Batch predict satisfaction for multiple customers",
)
async def predict_batch(request: BatchPredictionRequest):
    """Process up to 1000 records in a single call."""
    if len(request.records) > 1000:
        raise HTTPException(status_code=400, detail="Batch size must not exceed 1000 records.")

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        predictions = [predict_single(r) for r in request.records]
        elapsed_ms  = (time.perf_counter() - start) * 1000
        REQUEST_COUNT.labels(endpoint="predict_batch", status="success").inc()
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            processing_time_ms=round(elapsed_ms, 2),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="predict_batch", status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)