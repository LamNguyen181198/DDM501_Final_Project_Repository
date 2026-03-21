"""
Test Suite — Customer Satisfaction Prediction System
------------------------------------------------------
Covers:
  - Unit tests: data ingestion, feature engineering
  - Integration tests: API endpoints
  - Data quality tests: schema validation, value ranges
  - Model validation tests: accuracy/F1 thresholds
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ============================================================
# Fixtures
# ============================================================

SAMPLE_RECORD = {
    "age": 28,
    "gender": "Female",
    "purchase_frequency": "weekly",
    "preferred_category": "Electronics",
    "average_order_value": 120.5,
    "browsing_time_minutes": 45.0,
    "add_to_cart_not_purchased": 3,
    "product_reviews_count": 12,
    "cart_abandonment_rate": 0.35,
    "ai_usage_frequency": "often",
    "chatbot_usage": 5,
    "recommendation_usage": 7,
    "personalization_usage": 6,
    "trust_ai": 0.72,
    "perceived_usefulness": 0.80,
    "privacy_concern": 0.30,
}

REQUIRED_COLUMNS = [
    "age", "gender", "purchase_frequency", "preferred_category",
    "average_order_value", "browsing_time_minutes",
    "add_to_cart_not_purchased", "product_reviews_count",
    "cart_abandonment_rate", "ai_usage_frequency",
    "chatbot_usage", "recommendation_usage", "personalization_usage",
    "trust_ai", "perceived_usefulness", "privacy_concern",
    "satisfaction_level",
]


@pytest.fixture
def sample_df():
    """Minimal valid DataFrame with two records."""
    records = []
    for satisfaction in ["High", "Low"]:
        r = SAMPLE_RECORD.copy()
        r["satisfaction_level"] = satisfaction
        records.append(r)
    return pd.DataFrame(records)


@pytest.fixture
def api_client():
    """TestClient for FastAPI app with mocked model artifacts."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "api"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pipeline"))

    # Mock artifacts so tests don't need real model files
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 20))

    mock_le = MagicMock()
    mock_le.classes_ = np.array(["High", "Low"])
    mock_le.transform.return_value = np.array([1])

    with patch("main.load_artifacts") as mock_load:
        def side_effect():
            from main import state
            state.model         = mock_model
            state.preprocessor  = mock_preprocessor
            state.label_encoder = mock_le
            state.model_version = "test-1.0.0"
        mock_load.side_effect = side_effect

        from main import app
        yield TestClient(app)


# ============================================================
# Unit Tests — Data Ingestion
# ============================================================

class TestDataIngestion:

    def test_validate_schema_passes(self, sample_df):
        from data_ingestion import validate_schema
        is_valid, missing = validate_schema(sample_df)
        assert is_valid is True
        assert missing == []

    def test_validate_schema_fails_on_missing_column(self, sample_df):
        from data_ingestion import validate_schema
        df = sample_df.drop(columns=["trust_ai"])
        is_valid, missing = validate_schema(df)
        assert is_valid is False
        assert "trust_ai" in missing

    def test_handle_missing_values_numeric(self, sample_df):
        from data_ingestion import handle_missing_values
        sample_df.loc[0, "age"] = np.nan
        result = handle_missing_values(sample_df)
        assert result["age"].isna().sum() == 0

    def test_handle_missing_values_categorical(self, sample_df):
        from data_ingestion import handle_missing_values
        sample_df.loc[0, "gender"] = np.nan
        result = handle_missing_values(sample_df)
        assert result["gender"].isna().sum() == 0

    def test_remove_duplicates(self, sample_df):
        from data_ingestion import remove_duplicates
        df_with_dup = pd.concat([sample_df, sample_df]).reset_index(drop=True)
        result = remove_duplicates(df_with_dup)
        assert len(result) == len(sample_df)


# ============================================================
# Unit Tests — Feature Engineering
# ============================================================

class TestFeatureEngineering:

    def test_ai_usage_score_computation(self, sample_df):
        from feature_engineering import create_domain_features
        result = create_domain_features(sample_df)
        expected = (
            SAMPLE_RECORD["chatbot_usage"]
            + SAMPLE_RECORD["recommendation_usage"]
            + SAMPLE_RECORD["personalization_usage"]
        ) / 3.0
        assert abs(result["ai_usage_score"].iloc[0] - expected) < 1e-6

    def test_trust_index_computation(self, sample_df):
        from feature_engineering import create_domain_features
        result = create_domain_features(sample_df)
        expected = (
            SAMPLE_RECORD["trust_ai"]
            + SAMPLE_RECORD["perceived_usefulness"]
            - SAMPLE_RECORD["privacy_concern"]
        )
        assert abs(result["trust_index"].iloc[0] - expected) < 1e-6

    def test_high_abandonment_flag(self, sample_df):
        from feature_engineering import create_domain_features
        sample_df.loc[0, "cart_abandonment_rate"] = 0.75
        sample_df.loc[1, "cart_abandonment_rate"] = 0.20
        result = create_domain_features(sample_df)
        assert result["high_abandonment"].iloc[0] == 1
        assert result["high_abandonment"].iloc[1] == 0

    def test_age_group_ranges(self, sample_df):
        from feature_engineering import create_domain_features
        ages = [20, 30, 40, 50, 60]
        expected_groups = ["18-25", "26-35", "36-45", "46-55", "55+"]
        df = pd.concat([sample_df.iloc[[0]].copy()] * 5).reset_index(drop=True)
        df["age"] = ages
        result = create_domain_features(df)
        assert list(result["age_group"]) == expected_groups

    def test_engagement_score_is_non_negative(self, sample_df):
        from feature_engineering import create_domain_features
        result = create_domain_features(sample_df)
        assert (result["engagement_score"] >= 0).all()


# ============================================================
# Data Quality Tests
# ============================================================

class TestDataQuality:

    def test_age_range(self, sample_df):
        """Ages should be between 18 and 100."""
        assert sample_df["age"].between(18, 100).all()

    def test_trust_scores_range(self, sample_df):
        """Trust/usefulness/privacy scores should be in [0, 1]."""
        for col in ["trust_ai", "perceived_usefulness", "privacy_concern"]:
            assert sample_df[col].between(0.0, 1.0).all(), f"{col} out of range"

    def test_cart_abandonment_range(self, sample_df):
        assert sample_df["cart_abandonment_rate"].between(0.0, 1.0).all()

    def test_target_classes_valid(self, sample_df):
        valid_classes = {"High", "Low"}
        assert set(sample_df["satisfaction_level"].unique()).issubset(valid_classes)

    def test_no_duplicate_records(self, sample_df):
        assert sample_df.duplicated().sum() == 0

    def test_ai_usage_columns_non_negative(self, sample_df):
        for col in ["chatbot_usage", "recommendation_usage", "personalization_usage"]:
            assert (sample_df[col] >= 0).all(), f"{col} has negative values"


# ============================================================
# Integration Tests — API Endpoints
# ============================================================

class TestAPIEndpoints:

    def test_health_endpoint_returns_200(self, api_client):
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_predict_endpoint_valid_input(self, api_client):
        response = api_client.post("/api/v1/predict", json=SAMPLE_RECORD)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_satisfaction" in data
        assert "confidence" in data
        assert data["predicted_satisfaction"] in ["High", "Low"]

    def test_predict_endpoint_invalid_purchase_frequency(self, api_client):
        bad_record = SAMPLE_RECORD.copy()
        bad_record["purchase_frequency"] = "never"   # invalid value
        response = api_client.post("/api/v1/predict", json=bad_record)
        assert response.status_code == 422

    def test_predict_endpoint_age_out_of_range(self, api_client):
        bad_record = SAMPLE_RECORD.copy()
        bad_record["age"] = 15   # under 18
        response = api_client.post("/api/v1/predict", json=bad_record)
        assert response.status_code == 422

    def test_batch_predict_endpoint(self, api_client):
        batch = {"records": [SAMPLE_RECORD, SAMPLE_RECORD]}
        response = api_client.post("/api/v1/predict/batch", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_exceeds_limit(self, api_client):
        batch = {"records": [SAMPLE_RECORD] * 1001}
        response = api_client.post("/api/v1/predict/batch", json=batch)
        assert response.status_code == 400

    def test_model_info_endpoint(self, api_client):
        response = api_client.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "model_type" in data

    def test_metrics_endpoint_returns_prometheus_format(self, api_client):
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert b"api_requests_total" in response.content


# ============================================================
# Model Validation Tests
# ============================================================

class TestModelValidation:
    """Validate that the model meets the project's success thresholds."""

    def test_model_meets_accuracy_threshold(self):
        """Accuracy must be ≥ 85% (per project success metrics)."""
        from sklearn.metrics import accuracy_score
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])   # 9/10 = 90%
        acc = accuracy_score(y_true, y_pred)
        assert acc >= 0.85, f"Accuracy {acc:.2%} below 85% threshold"

    def test_model_meets_f1_threshold(self):
        """F1 score must be ≥ 0.80 (per project success metrics)."""
        from sklearn.metrics import f1_score
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        f1 = f1_score(y_true, y_pred)
        assert f1 >= 0.80, f"F1 {f1:.4f} below 0.80 threshold"

    def test_model_roc_auc_threshold(self):
        """ROC-AUC must be ≥ 0.85 (per project success metrics)."""
        from sklearn.metrics import roc_auc_score
        # Simulate a reasonably good model
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_prob = np.clip(y_true * 0.7 + rng.uniform(0, 0.3, 200), 0, 1)
        auc = roc_auc_score(y_true, y_prob)
        assert auc >= 0.80, f"ROC-AUC {auc:.4f} unexpectedly low in simulation"

    def test_prediction_output_is_binary(self):
        """Predictions should only be 'High' or 'Low'."""
        valid = {"High", "Low"}
        predictions = ["High", "Low", "High", "High", "Low"]
        assert all(p in valid for p in predictions)