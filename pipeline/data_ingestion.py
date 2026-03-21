"""
Data Ingestion Module
---------------------
Loads the 'Online Shopping and AI Usage Dataset' from Kaggle (or local CSV),
validates schema, handles missing values, removes duplicates, and persists
a clean version for downstream processing.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition — every column and its expected dtype
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: dict[str, str] = {
    "age": "int64",
    "gender": "object",
    "purchase_frequency": "object",         # shopping behavior
    "preferred_category": "object",
    "average_order_value": "float64",
    "browsing_time_minutes": "float64",
    "add_to_cart_not_purchased": "int64",
    "product_reviews_count": "int64",
    "cart_abandonment_rate": "float64",
    "ai_usage_frequency": "object",          # AI usage behavior
    "chatbot_usage": "int64",
    "recommendation_usage": "int64",
    "personalization_usage": "int64",
    "trust_ai": "float64",                   # trust/perception
    "perceived_usefulness": "float64",
    "privacy_concern": "float64",
    "satisfaction_level": "object",          # target — "High" / "Low"
}

TARGET_COLUMN = "satisfaction_level"
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_PATH = DATA_DIR / "raw" / "online_shopping_ai.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "cleaned.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV from *path* and return a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it from Kaggle: 'online-shopping-and-ai-usage-dataset'."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d cols from %s", *df.shape, path)
    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """
    Verify that all required columns are present.
    Returns (is_valid, list_of_missing_columns).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.warning("Schema validation failed — missing columns: %s", missing)
        return False, missing
    logger.info("Schema validation passed.")
    return True, []


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values.
    - Numeric columns → median imputation
    - Categorical columns → mode imputation
    """
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed numeric '%s' with median %.4f", col, median_val)

    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Imputed categorical '%s' with mode '%s'", col, mode_val)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows.", removed)
    return df


def run_ingestion(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Full ingestion pipeline:
      load → validate → clean → persist.
    Returns the cleaned DataFrame.
    """
    df = load_raw_data(path)

    is_valid, missing = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = handle_missing_values(df)
    df = remove_duplicates(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info("Cleaned dataset saved to %s (%d rows).", PROCESSED_PATH, len(df))
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split.
    Returns (train_df, val_df, test_df).
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_relative, stratify=y_train_val, random_state=random_state,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df   = pd.concat([X_val,   y_val],   axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)

    logger.info(
        "Split → train %d | val %d | test %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    cleaned = run_ingestion()
    train, val, test = split_data(cleaned)
    train.to_csv(DATA_DIR / "processed" / "train.csv", index=False)
    val.to_csv(DATA_DIR / "processed" / "val.csv", index=False)
    test.to_csv(DATA_DIR / "processed" / "test.csv", index=False)
    logger.info("Data ingestion complete.")