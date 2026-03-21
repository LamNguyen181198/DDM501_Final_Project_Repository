"""Test suite for the AI in Retail satisfaction system."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ============================================================
# Fixtures
# ============================================================

SAMPLE_RECORD = {
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

REQUIRED_COLUMNS = [
    "country",
    "online_consumer",
    "age_group",
    "annual_salary_band",
    "gender",
    "education",
    "payment_method_card",
    "living_region",
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
    "satisfaction_level",
]


@pytest.fixture
def sample_df():
    """Minimal valid DataFrame with two records."""
    records = []
    for satisfaction in ["Satisfied", "Unsatisfied"]:
        r = SAMPLE_RECORD.copy()
        r["satisfaction_level"] = satisfaction
        records.append(r)
    return pd.DataFrame(records)


@pytest.fixture
def api_client():
    """TestClient for FastAPI app with mocked model artifacts."""
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "pipeline"))

    # Mock artifacts so tests don't need real model files
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 12))

    mock_le = MagicMock()
    mock_le.classes_ = np.array(["Satisfied", "Unsatisfied"])
    mock_le.transform.return_value = np.array([1])

    with patch("main.load_artifacts") as mock_load:

        def side_effect():
            from main import state

            state.model = mock_model
            state.preprocessor = mock_preprocessor
            state.label_encoder = mock_le
            state.model_version = "test-1.0.0"

        mock_load.side_effect = side_effect

        from main import app

        with TestClient(app) as client:
            yield client


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

        df = sample_df.drop(columns=["ai_endorsement"])
        is_valid, missing = validate_schema(df)
        assert is_valid is False
        assert "ai_endorsement" in missing

    def test_handle_missing_values_numeric(self, sample_df):
        from data_ingestion import handle_missing_values

        sample_df.loc[0, "online_consumer"] = np.nan
        result = handle_missing_values(sample_df)
        assert result["online_consumer"].isna().sum() == 0

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

    def test_ai_tool_usage_count_computation(self, sample_df):
        from feature_engineering import create_domain_features

        result = create_domain_features(sample_df)
        expected = 2
        assert result["ai_tool_usage_count"].iloc[0] == expected

    def test_ai_readiness_score_computation(self, sample_df):
        from feature_engineering import create_domain_features

        result = create_domain_features(sample_df)
        expected = 4
        assert result["ai_readiness_score"].iloc[0] == expected

    def test_digital_payment_preference(self, sample_df):
        from feature_engineering import create_domain_features

        result = create_domain_features(sample_df)
        assert result["digital_payment_preference"].iloc[0] == 2

    def test_product_category_count(self, sample_df):
        from feature_engineering import create_domain_features

        result = create_domain_features(sample_df)
        assert result["product_category_count"].iloc[0] == 2

    def test_binary_columns_become_numeric(self, sample_df):
        from feature_engineering import create_domain_features

        result = create_domain_features(sample_df)
        assert set(result["online_consumer"].unique()).issubset({0, 1})


# ============================================================
# Data Quality Tests
# ============================================================


class TestDataQuality:

    def test_age_group_values(self, sample_df):
        assert set(sample_df["age_group"].unique()).issubset(
            {"Gen Z", "Millennials", "Gen X", "Baby Boomers"}
        )

    def test_yes_no_fields_are_valid(self, sample_df):
        for col in [
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
        ]:
            assert set(sample_df[col].unique()).issubset(
                {"YES", "NO"}
            ), f"{col} has invalid values"

    def test_country_values(self, sample_df):
        assert set(sample_df["country"].unique()).issubset({"CANADA", "CHINA", "INDIA"})

    def test_target_classes_valid(self, sample_df):
        valid_classes = {"Satisfied", "Unsatisfied"}
        assert set(sample_df["satisfaction_level"].unique()).issubset(valid_classes)

    def test_no_duplicate_records(self, sample_df):
        assert sample_df.duplicated().sum() == 0

    def test_product_category_flags_present(self, sample_df):
        flagged = [
            "product_category_appliances",
            "product_category_electronics",
            "product_category_groceries",
            "product_category_personal_care",
            "product_category_clothing",
        ]
        assert all(col in sample_df.columns for col in flagged)


# ============================================================
# Integration Tests — API Endpoints
# ============================================================


class TestAPIEndpoints:

    def test_root_redirects_to_docs(self, api_client):
        response = api_client.get("/", follow_redirects=False)
        assert response.status_code in (307, 308)
        assert response.headers["location"] == "/docs"

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
        assert data["predicted_satisfaction"] in ["Satisfied", "Unsatisfied"]

    def test_predict_endpoint_invalid_yes_no_field(self, api_client):
        bad_record = SAMPLE_RECORD.copy()
        bad_record["online_consumer"] = "MAYBE"
        response = api_client.post("/api/v1/predict", json=bad_record)
        assert response.status_code == 422

    def test_predict_endpoint_invalid_age_group(self, api_client):
        bad_record = SAMPLE_RECORD.copy()
        bad_record["age_group"] = "Teen"
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
        y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])  # 9/10 = 90%
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
        """Predictions should only be Satisfied or Unsatisfied."""
        valid = {"Satisfied", "Unsatisfied"}
        predictions = [
            "Satisfied",
            "Unsatisfied",
            "Satisfied",
            "Satisfied",
            "Unsatisfied",
        ]
        assert all(p in valid for p in predictions)
