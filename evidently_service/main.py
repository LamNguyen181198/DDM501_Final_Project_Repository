"""
Evidently AI Drift Detection Service
======================================
Captures live prediction data, compares it against a reference dataset,
and exposes drift metrics to Prometheus for alerting and Grafana dashboards.

Endpoints
---------
POST /capture          – submit a new prediction record for drift monitoring
POST /analyze          – run drift analysis on buffered data vs reference
GET  /report           – return the latest HTML drift report
GET  /metrics          – Prometheus exposition (scrape target for Prometheus)
GET  /health           – liveness probe
GET  /status           – buffer statistics

Dataset features (aligned with pipeline/feature_engineer.py):
  Numeric  : ai_tool_usage_count, payment_method_count, product_category_count,
             ai_readiness_score, digital_payment_preference
  Ordinal  : age_group, annual_salary_band, education
  Nominal  : country, gender, living_region, online_consumer,
             payment_method_card, payment_method_cod, payment_method_ewallet,
             online_service_preference, ai_endorsement, ai_privacy_no_trust,
             ai_enhance_experience, ai_tool_chatbots,
             ai_tool_virtual_assistant, ai_tool_voice_photo_search,
             product_category_appliances, product_category_electronics,
             product_category_groceries, product_category_personal_care,
             product_category_clothing
"""

import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
try:
    from evidently import ColumnMapping  # evidently <0.4.x
except ImportError:
    try:
        from evidently.pipeline.column_mapping import ColumnMapping  # evidently 0.4-0.6
    except ImportError:
        from evidently.legacy.pipeline.column_mapping import ColumnMapping  # evidently 0.7+
try:
    from evidently.metric_presets import DataDriftPreset, DataQualityPreset
    from evidently.report import Report
except ImportError:
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset  # evidently 0.7+
    from evidently.legacy.report import Report  # evidently 0.7+
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUFFER_MAX = 1000          # max records kept in the rolling window
MIN_RECORDS_FOR_ANALYSIS = 50  # minimum before running Evidently
REFERENCE_PATH = Path("/app/reference_data/reference.parquet")
REPORT_PATH = Path("/app/reports/drift_report.html")

# Feature column names (must match pipeline/feature_engineer.py)
NUMERIC_FEATURES = [
    "ai_tool_usage_count",
    "payment_method_count",
    "product_category_count",
    "ai_readiness_score",
    "digital_payment_preference",
]
CATEGORICAL_FEATURES = [
    "country",
    "gender",
    "living_region",
    "online_consumer",
    "age_group",
    "annual_salary_band",
    "education",
    "payment_method_card",
    "payment_method_cod",
    "payment_method_ewallet",
    "online_service_preference",
    "ai_endorsement",
    "ai_privacy_no_trust",
    "ai_enhance_experience",
    "ai_tool_chatbots",
    "ai_tool_virtual_assistant",
    "ai_tool_voice_photo_search",
    "product_category_appliances",
    "product_category_electronics",
    "product_category_groceries",
    "product_category_personal_care",
    "product_category_clothing",
]
ALL_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
DRIFT_DETECTED = Gauge(
    "evidently_data_drift_detected",
    "1 if data drift is detected in the current window, 0 otherwise",
)
DRIFT_SCORE = Gauge(
    "evidently_drift_score",
    "Overall dataset drift share (fraction of drifted features)",
)
FEATURE_DRIFT = Gauge(
    "evidently_feature_drift",
    "Per-feature drift score (p-value or stat distance)",
    ["feature"],
)
MISSING_VALUES_RATIO = Gauge(
    "evidently_missing_values_ratio",
    "Fraction of missing values in current window",
)
BUFFER_SIZE = Gauge(
    "evidently_buffer_size",
    "Number of records currently in the drift-detection buffer",
)
CAPTURE_COUNTER = Counter(
    "evidently_captures_total",
    "Total number of prediction records captured",
)
ANALYSIS_COUNTER = Counter(
    "evidently_analysis_runs_total",
    "Total number of drift analysis runs completed",
)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_data_buffer: deque[dict] = deque(maxlen=BUFFER_MAX)
_reference_df: Optional[pd.DataFrame] = None
_last_report_html: str = "<p>No report generated yet.</p>"
_last_drift_detected: bool = False
_last_drift_score: float = 0.0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PredictionRecord(BaseModel):
    """A single prediction record captured from the main API."""

    # Derived numeric features (from feature_engineer.py)
    ai_tool_usage_count: float = Field(..., ge=0, le=5)
    payment_method_count: float = Field(..., ge=0, le=3)
    product_category_count: float = Field(..., ge=0, le=5)
    ai_readiness_score: float = Field(..., ge=0, le=5)
    digital_payment_preference: float = Field(..., ge=0, le=1)

    # Ordinal / categorical
    country: str
    gender: str
    living_region: str
    online_consumer: str
    age_group: str
    annual_salary_band: str
    education: str

    # Binary yes/no columns
    payment_method_card: str
    payment_method_cod: str
    payment_method_ewallet: str
    online_service_preference: str
    ai_endorsement: str
    ai_privacy_no_trust: str
    ai_enhance_experience: str
    ai_tool_chatbots: str
    ai_tool_virtual_assistant: str
    ai_tool_voice_photo_search: str
    product_category_appliances: str
    product_category_electronics: str
    product_category_groceries: str
    product_category_personal_care: str
    product_category_clothing: str


class AnalysisResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    drifted_features: list[str]
    n_records_analyzed: int
    missing_values_ratio: float


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Evidently Drift Detection Service",
    description="Data drift monitoring for the customer satisfaction model",
    version="1.0.0",
)


@app.on_event("startup")
def _load_reference() -> None:
    """Load reference dataset on startup (if available)."""
    global _reference_df
    if REFERENCE_PATH.exists():
        _reference_df = pd.read_parquet(REFERENCE_PATH)
        logger.info(
            "Reference dataset loaded: %d rows, %d columns.",
            len(_reference_df),
            len(_reference_df.columns),
        )
    else:
        logger.warning(
            "Reference dataset not found at %s. "
            "Upload it via POST /reference or mount the volume.",
            REFERENCE_PATH,
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "healthy", "buffer_size": len(_data_buffer)}


@app.get("/status")
def status():
    with _lock:
        n = len(_data_buffer)
    return {
        "buffer_size": n,
        "reference_loaded": _reference_df is not None,
        "last_drift_detected": _last_drift_detected,
        "last_drift_score": _last_drift_score,
        "min_records_for_analysis": MIN_RECORDS_FOR_ANALYSIS,
    }


@app.post("/capture", status_code=202)
def capture(record: PredictionRecord):
    """Buffer a new prediction record for drift analysis."""
    with _lock:
        _data_buffer.append(record.model_dump())
        BUFFER_SIZE.set(len(_data_buffer))
    CAPTURE_COUNTER.inc()
    return {"status": "captured", "buffer_size": len(_data_buffer)}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze():
    """Run Evidently drift analysis on the current buffer vs reference data."""
    global _last_report_html, _last_drift_detected, _last_drift_score

    if _reference_df is None:
        raise HTTPException(
            status_code=503,
            detail="Reference dataset not loaded. Mount reference_data volume.",
        )

    with _lock:
        current_records = list(_data_buffer)

    if len(current_records) < MIN_RECORDS_FOR_ANALYSIS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_RECORDS_FOR_ANALYSIS} records, "
            f"have {len(current_records)}.",
        )

    current_df = pd.DataFrame(current_records)

    # Align columns
    ref_cols = [c for c in ALL_COLUMNS if c in _reference_df.columns]
    cur_cols = [c for c in ref_cols if c in current_df.columns]

    ref = _reference_df[cur_cols].copy()
    cur = current_df[cur_cols].copy()

    column_mapping = ColumnMapping(
        numerical_features=[c for c in NUMERIC_FEATURES if c in cur_cols],
        categorical_features=[c for c in CATEGORICAL_FEATURES if c in cur_cols],
    )

    # Run Evidently report
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    # Save HTML report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    with open(REPORT_PATH) as fh:
        _last_report_html = fh.read()

    # Extract metrics from the report dict
    report_dict: dict[str, Any] = report.as_dict()
    drift_metrics = _extract_drift_metrics(report_dict, cur_cols)

    # Update Prometheus gauges
    _last_drift_detected = drift_metrics["drift_detected"]
    _last_drift_score = drift_metrics["drift_score"]
    DRIFT_DETECTED.set(1 if _last_drift_detected else 0)
    DRIFT_SCORE.set(_last_drift_score)
    for feat, score in drift_metrics["feature_scores"].items():
        FEATURE_DRIFT.labels(feature=feat).set(score)
    MISSING_VALUES_RATIO.set(drift_metrics["missing_ratio"])
    ANALYSIS_COUNTER.inc()

    logger.info(
        "Drift analysis complete — detected=%s, score=%.4f, drifted_features=%s",
        _last_drift_detected,
        _last_drift_score,
        drift_metrics["drifted_features"],
    )

    return AnalysisResponse(
        drift_detected=drift_metrics["drift_detected"],
        drift_score=drift_metrics["drift_score"],
        drifted_features=drift_metrics["drifted_features"],
        n_records_analyzed=len(current_records),
        missing_values_ratio=drift_metrics["missing_ratio"],
    )


@app.get("/report", response_class=HTMLResponse)
def get_report():
    """Return the latest HTML drift report."""
    return HTMLResponse(content=_last_report_html, status_code=200)


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Reference data upload
# ---------------------------------------------------------------------------


@app.post("/reference")
async def upload_reference(data: list[dict]):
    """Accept reference dataset as JSON array and save to disk."""
    global _reference_df
    df = pd.DataFrame(data)
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REFERENCE_PATH, index=False)
    _reference_df = df
    logger.info("Reference dataset updated — %d rows.", len(df))
    return {"status": "ok", "rows": len(df)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_drift_metrics(report_dict: dict, columns: list[str]) -> dict:
    """Parse Evidently report dict to extract dataset-level drift metrics."""
    drift_detected = False
    drift_score = 0.0
    feature_scores: dict[str, float] = {}
    drifted_features: list[str] = []
    missing_ratio = 0.0

    try:
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            metric_id = metric.get("metric", "")

            # DatasetDriftMetric
            if "DatasetDriftMetric" in metric_id:
                drift_detected = result.get("dataset_drift", False)
                drift_score = result.get("drift_share", 0.0)

            # ColumnDriftMetric per feature
            if "ColumnDriftMetric" in metric_id:
                col = result.get("column_name", "")
                score = result.get("drift_score", 0.0)
                if col:
                    feature_scores[col] = float(score)
                    if result.get("drift_detected", False):
                        drifted_features.append(col)

            # DatasetMissingValuesMetric
            if "DatasetMissingValuesMetric" in metric_id:
                missing_ratio = result.get(
                    "current", {}
                ).get("share_of_missing_values", 0.0)

    except Exception as exc:  # pragma: no cover
        logger.warning("Could not parse Evidently report: %s", exc)

    return {
        "drift_detected": drift_detected,
        "drift_score": float(drift_score),
        "feature_scores": feature_scores,
        "drifted_features": drifted_features,
        "missing_ratio": float(missing_ratio),
    }
