"""
Feature Engineering Module
---------------------------
Transforms raw, cleaned data into ML-ready features aligned with the
Customer Satisfaction / Trust Toward AI in Online Shopping project.

Key engineered features:
  - ai_usage_score   : composite AI usage intensity
  - trust_index      : net trust signal
  - age_group        : ordinal age bucketing
  - spending_tier    : order-value bucket
  - engagement_score : browsing + review activity
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
TARGET_COLUMN = "satisfaction_level"


# ---------------------------------------------------------------------------
# Step 1 – domain-specific feature creation
# ---------------------------------------------------------------------------

def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the DataFrame in-place (returns copy)."""
    df = df.copy()

    # AI Usage Score: average utilisation across three AI channels
    df["ai_usage_score"] = (
        df["chatbot_usage"]
        + df["recommendation_usage"]
        + df["personalization_usage"]
    ) / 3.0

    # Trust Index: net trust signal (usefulness minus privacy worry)
    df["trust_index"] = (
        df["trust_ai"]
        + df["perceived_usefulness"]
        - df["privacy_concern"]
    )

    # Age group (ordinal buckets)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    ).astype(str)

    # Spending tier (order value quintiles)
    df["spending_tier"] = pd.qcut(
        df["average_order_value"],
        q=5,
        labels=["very_low", "low", "medium", "high", "very_high"],
        duplicates="drop",
    ).astype(str)

    # Engagement score: normalised browsing × log(reviews + 1)
    df["engagement_score"] = (
        df["browsing_time_minutes"] * np.log1p(df["product_reviews_count"])
    )

    # Cart abandonment severity flag
    df["high_abandonment"] = (df["cart_abandonment_rate"] > 0.5).astype(int)

    logger.info("Domain features created: ai_usage_score, trust_index, age_group, "
                "spending_tier, engagement_score, high_abandonment.")
    return df


# ---------------------------------------------------------------------------
# Step 2 – sklearn preprocessing pipeline
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age",
    "average_order_value",
    "browsing_time_minutes",
    "add_to_cart_not_purchased",
    "product_reviews_count",
    "cart_abandonment_rate",
    "chatbot_usage",
    "recommendation_usage",
    "personalization_usage",
    "trust_ai",
    "perceived_usefulness",
    "privacy_concern",
    "ai_usage_score",
    "trust_index",
    "engagement_score",
]

ORDINAL_FEATURES = [
    "purchase_frequency",  # daily / weekly / monthly / rarely
    "ai_usage_frequency",  # always / often / sometimes / rarely / never
    "age_group",
    "spending_tier",
]

ORDINAL_CATEGORIES = [
    ["rarely", "monthly", "weekly", "daily"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["18-25", "26-35", "36-45", "46-55", "55+"],
    ["very_low", "low", "medium", "high", "very_high"],
]

NOMINAL_FEATURES = [
    "gender",
    "preferred_category",
]

BINARY_FEATURES = ["high_abandonment"]


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that handles all feature types."""
    numeric_transformer = MinMaxScaler()

    ordinal_transformer = OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    nominal_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",     numeric_transformer,  NUMERIC_FEATURES),
            ("ord",     ordinal_transformer,  ORDINAL_FEATURES),
            ("nom",     nominal_transformer,  NOMINAL_FEATURES),
            ("binary",  "passthrough",        BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def encode_target(y: pd.Series) -> np.ndarray:
    """Encode 'High'/'Low' target to 1/0."""
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