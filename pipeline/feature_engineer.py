"""Feature engineering for the AI in Retail Kaggle dataset."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder

from data_ingestion import (
    AGE_GROUP_CATEGORIES,
    EDUCATION_CATEGORIES,
    SALARY_CATEGORIES,
    TARGET_COLUMN,
    YES_NO_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")

AI_TOOL_COLUMNS = [
    "ai_tool_chatbots",
    "ai_tool_virtual_assistant",
    "ai_tool_voice_photo_search",
]
PAYMENT_METHOD_COLUMNS = [
    "payment_method_card",
    "payment_method_cod",
    "payment_method_ewallet",
]
PRODUCT_CATEGORY_COLUMNS = [
    "product_category_appliances",
    "product_category_electronics",
    "product_category_groceries",
    "product_category_personal_care",
    "product_category_clothing",
]
DERIVED_NUMERIC_FEATURES = [
    "ai_tool_usage_count",
    "payment_method_count",
    "product_category_count",
    "ai_readiness_score",
    "digital_payment_preference",
]
NUMERIC_FEATURES = YES_NO_COLUMNS + DERIVED_NUMERIC_FEATURES
ORDINAL_FEATURES = ["age_group", "annual_salary_band", "education"]
ORDINAL_CATEGORIES = [AGE_GROUP_CATEGORIES, SALARY_CATEGORIES, EDUCATION_CATEGORIES]
NOMINAL_FEATURES = ["country", "gender", "living_region"]


def _yes_no_to_int(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    return normalized.map({"YES": 1, "NO": 0}).fillna(0).astype(int)


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready numeric features while preserving categorical columns."""
    df = df.copy()

    for column in YES_NO_COLUMNS:
        df[column] = _yes_no_to_int(df[column])

    df["ai_tool_usage_count"] = df[AI_TOOL_COLUMNS].sum(axis=1)
    df["payment_method_count"] = df[PAYMENT_METHOD_COLUMNS].sum(axis=1)
    df["product_category_count"] = df[PRODUCT_CATEGORY_COLUMNS].sum(axis=1)
    df["ai_readiness_score"] = (
        df["online_consumer"]
        + df["online_service_preference"]
        + df["ai_endorsement"]
        + df["ai_enhance_experience"]
        - df["ai_privacy_no_trust"]
    )
    df["digital_payment_preference"] = df["payment_method_card"] + df["payment_method_ewallet"]

    logger.info(
        "Domain features created: ai_tool_usage_count, payment_method_count, "
        "product_category_count, ai_readiness_score, digital_payment_preference."
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that handles all feature types."""
    numeric_transformer = MinMaxScaler()

    ordinal_transformer = OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    nominal_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
            ("nom", nominal_transformer, NOMINAL_FEATURES),
        ],
        remainder="drop",
    )


def encode_target(y: pd.Series) -> np.ndarray:
    """Encode the Satisfied / Unsatisfied target to integer labels."""
    le = LabelEncoder()
    encoded = le.fit_transform(y)
    logger.info("Target classes: %s → %s", list(le.classes_), list(range(len(le.classes_))))
    return encoded, le


# ---------------------------------------------------------------------------
# Step 3 – end-to-end feature engineering runner
# ---------------------------------------------------------------------------

def run_feature_engineering(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Applies domain feature creation + preprocessing.
    Fits on train, transforms all three splits.
    Saves preprocessor artifact.

    Returns:
        X_train, X_val, X_test (np.ndarray)
        y_train, y_val, y_test (np.ndarray)
        preprocessor, label_encoder
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Domain features
    train_df = create_domain_features(train_df)
    val_df   = create_domain_features(val_df)
    test_df  = create_domain_features(test_df)

    # Split X / y
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val   = val_df.drop(columns=[TARGET_COLUMN])
    y_val   = val_df[TARGET_COLUMN]
    X_test  = test_df.drop(columns=[TARGET_COLUMN])
    y_test  = test_df[TARGET_COLUMN]

    # Target encoding (fit on train only)
    y_train_enc, le = encode_target(y_train)
    y_val_enc       = le.transform(y_val)
    y_test_enc      = le.transform(y_test)

    # Preprocessor
    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)
    X_test_enc  = preprocessor.transform(X_test)

    # Persist artifacts
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(le,           ARTIFACTS_DIR / "label_encoder.joblib")
    logger.info("Preprocessor and label encoder saved to %s.", ARTIFACTS_DIR)

    logger.info(
        "Feature matrix shapes → train %s | val %s | test %s",
        X_train_enc.shape, X_val_enc.shape, X_test_enc.shape,
    )
    return (
        X_train_enc, X_val_enc, X_test_enc,
        y_train_enc, y_val_enc, y_test_enc,
        preprocessor, le,
    )


if __name__ == "__main__":
    from data_ingestion import run_ingestion, split_data

    cleaned = run_ingestion()
    train_df, val_df, test_df = split_data(cleaned)
    run_feature_engineering(train_df, val_df, test_df)
    logger.info("Feature engineering complete.")