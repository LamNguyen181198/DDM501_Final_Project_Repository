"""Prediction API for the AI in Retail satisfaction model."""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import httpx
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_ingestion import (
    AGE_GROUP_CATEGORIES,
    COUNTRY_CATEGORIES,
    EDUCATION_CATEGORIES,
    GENDER_CATEGORIES,
    REGION_CATEGORIES,
    SALARY_CATEGORIES,
)
from feature_engineering import create_domain_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://evidently:8001")
API_VERSION = "v1"

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
ACTIVE_REQUESTS = Gauge(
    "api_active_requests", "Number of active requests being processed"
)

CUSTOMER_FEATURES_EXAMPLE = {
    "country": "INDIA",
    "online_consumer": "YES",
    "age_group": "Gen X",
    "annual_salary_band": "Medium High",
    "gender": "Female",
    "education": "Masters' Degree",
    "payment_method_card": "YES",
    "living_region": "Metropolitan",
    "online_service_preference": "YES",
    "ai_endorsement": "YES",
    "ai_privacy_no_trust": "NO",
    "ai_enhance_experience": "YES",
    "ai_tool_chatbots": "YES",
    "ai_tool_virtual_assistant": "YES",
    "ai_tool_voice_photo_search": "NO",
    "payment_method_cod": "NO",
    "payment_method_ewallet": "YES",
    "product_category_appliances": "NO",
    "product_category_electronics": "YES",
    "product_category_groceries": "NO",
    "product_category_personal_care": "YES",
    "product_category_clothing": "NO",
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """Input schema for a single retail AI satisfaction record."""

    model_config = ConfigDict(json_schema_extra={"example": CUSTOMER_FEATURES_EXAMPLE})

    country: str = Field(...)
    online_consumer: str = Field(...)
    age_group: str = Field(...)
    annual_salary_band: str = Field(...)
    gender: str = Field(...)
    education: str = Field(...)
    payment_method_card: str = Field(...)
    living_region: str = Field(...)
    online_service_preference: str = Field(...)
    ai_endorsement: str = Field(...)
    ai_privacy_no_trust: str = Field(...)
    ai_enhance_experience: str = Field(...)
    ai_tool_chatbots: str = Field(...)
    ai_tool_virtual_assistant: str = Field(...)
    ai_tool_voice_photo_search: str = Field(...)
    payment_method_cod: str = Field(...)
    payment_method_ewallet: str = Field(...)
    product_category_appliances: str = Field(...)
    product_category_electronics: str = Field(...)
    product_category_groceries: str = Field(...)
    product_category_personal_care: str = Field(...)
    product_category_clothing: str = Field(...)

    @field_validator(
        "online_consumer",
        "payment_method_card",
        "online_service_preference",
        "ai_endorsement",
        "ai_privacy_no_trust",
        "ai_enhance_experience",
        "ai_tool_chatbots",
        "ai_tool_virtual_assistant",
        "ai_tool_voice_photo_search",
        "payment_method_cod",
        "payment_method_ewallet",
        "product_category_appliances",
        "product_category_electronics",
        "product_category_groceries",
        "product_category_personal_care",
        "product_category_clothing",
    )
    @classmethod
    def validate_yes_no(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed = {"YES", "NO"}
        if normalized not in allowed:
            raise ValueError(f"value must be one of {allowed}")
        return normalized

    @field_validator("country")
    @classmethod
    def validate_country(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in COUNTRY_CATEGORIES:
            raise ValueError(f"country must be one of {COUNTRY_CATEGORIES}")
        return normalized

    @field_validator("age_group")
    @classmethod
    def validate_age_group(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in AGE_GROUP_CATEGORIES:
            raise ValueError(f"age_group must be one of {AGE_GROUP_CATEGORIES}")
        return normalized

    @field_validator("annual_salary_band")
    @classmethod
    def validate_annual_salary_band(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in SALARY_CATEGORIES:
            raise ValueError(f"annual_salary_band must be one of {SALARY_CATEGORIES}")
        return normalized

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in GENDER_CATEGORIES:
            raise ValueError(f"gender must be one of {GENDER_CATEGORIES}")
        return normalized

    @field_validator("education")
    @classmethod
    def validate_education(cls, value: str) -> str:
        normalized = value.strip().replace("’", "'")
        if normalized not in EDUCATION_CATEGORIES:
            raise ValueError(f"education must be one of {EDUCATION_CATEGORIES}")
        return normalized

    @field_validator("living_region")
    @classmethod
    def validate_living_region(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in REGION_CATEGORIES:
            raise ValueError(f"living_region must be one of {REGION_CATEGORIES}")
        return normalized


class PredictionResponse(BaseModel):
    predicted_satisfaction: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "records": [CUSTOMER_FEATURES_EXAMPLE, CUSTOMER_FEATURES_EXAMPLE],
            }
        }
    )

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
    model = None
    preprocessor = None
    label_encoder = None
    model_version = "unknown"


state = ModelState()


def load_artifacts() -> None:
    """Load model and preprocessing artifacts from disk."""
    try:
        state.model = joblib.load(ARTIFACTS_DIR / "model.joblib")
        state.preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
        state.label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
        state.model_version = os.getenv("MODEL_VERSION", "1.0.0")
        logger.info(
            "Artifacts loaded successfully. Model version: %s", state.model_version
        )
    except Exception as exc:
        logger.error("Failed to load artifacts: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reuse the training-time feature engineering for inference."""
    return create_domain_features(df)


def _capture_for_drift(features: CustomerFeatures) -> None:
    """Fire-and-forget: send prediction record to Evidently for drift monitoring."""
    try:
        payload = features.model_dump()
        # Add derived features so Evidently sees the same columns as training
        df = pd.DataFrame([payload])
        df = create_domain_features(df)
        first_row = df.iloc[0].to_dict()
        with httpx.Client(timeout=2.0) as client:
            client.post(f"{EVIDENTLY_URL}/capture", json=first_row)
    except Exception as exc:
        logger.debug("Evidently capture skipped: %s", exc)


def predict_single(features: CustomerFeatures) -> PredictionResponse:
    df = pd.DataFrame([features.model_dump()])
    df = _engineer_features(df)
    X = state.preprocessor.transform(df)

    proba = state.model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    classes = state.label_encoder.classes_  # e.g. ["High", "Low"]

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
    title="AI Retail Satisfaction Prediction API",
    description=(
        "Predicts whether a retail customer is satisfied or unsatisfied with "
        "AI-assisted online shopping experiences."
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


def _get_server_config() -> tuple[str, int]:
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    return host, port


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


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
        "model_version": state.model_version,
        "model_type": type(state.model).__name__,
        "api_version": API_VERSION,
        "target_variable": "satisfaction_level",
        "output_classes": (
            list(state.label_encoder.classes_) if state.label_encoder else []
        ),
    }


@app.post(
    f"/api/{API_VERSION}/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Predict AI retail satisfaction for a single customer",
)
async def predict(features: CustomerFeatures, background_tasks: BackgroundTasks):
    """
    Given a retail customer's AI shopping profile, return a predicted
    satisfaction label with confidence score.
    """
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        result = predict_single(features)
        REQUEST_COUNT.labels(endpoint="predict", status="success").inc()
        # Send record to Evidently drift service asynchronously
        background_tasks.add_task(_capture_for_drift, features)
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
        raise HTTPException(
            status_code=400, detail="Batch size must not exceed 1000 records."
        )

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        predictions = [predict_single(r) for r in request.records]
        elapsed_ms = (time.perf_counter() - start) * 1000
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
    host, port = _get_server_config()
    access_host = "127.0.0.1" if host == "0.0.0.0" else host
    logger.info("Starting API server on http://%s:%s", access_host, port)
    logger.info("Open API docs at http://%s:%s/docs", access_host, port)
    uvicorn.run(app, host=host, port=port, reload=False)
