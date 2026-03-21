"""Data ingestion for the AI in Retail Kaggle dataset."""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_DATASET = "pooriamst/online-shopping"
TARGET_COLUMN = "satisfaction_level"
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_PATH = DATA_DIR / "raw" / "online_shopping_ai.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "cleaned.csv"
CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

AGE_GROUP_CATEGORIES = ["Gen Z", "Millennials", "Gen X", "Baby Boomers"]
SALARY_CATEGORIES = ["Low", "Medium", "Medium High", "High"]
EDUCATION_CATEGORIES = [
    "Highschool Graduate",
    "University Graduate",
    "Masters' Degree",
    "Doctorate Degree",
]
COUNTRY_CATEGORIES = ["CANADA", "CHINA", "INDIA"]
GENDER_CATEGORIES = ["Female", "Male", "Prefer not to say"]
REGION_CATEGORIES = ["Metropolitan", "Suburban Areas", "Rural Areas"]
YES_NO_COLUMNS = [
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
]
REQUIRED_COLUMNS: dict[str, str] = {
    "country": "object",
    "online_consumer": "object",
    "age_group": "object",
    "annual_salary_band": "object",
    "gender": "object",
    "education": "object",
    "payment_method_card": "object",
    "living_region": "object",
    "online_service_preference": "object",
    "ai_endorsement": "object",
    "ai_privacy_no_trust": "object",
    "ai_enhance_experience": "object",
    "ai_tool_chatbots": "object",
    "ai_tool_virtual_assistant": "object",
    "ai_tool_voice_photo_search": "object",
    "payment_method_cod": "object",
    "payment_method_ewallet": "object",
    "product_category_appliances": "object",
    "product_category_electronics": "object",
    "product_category_groceries": "object",
    "product_category_personal_care": "object",
    "product_category_clothing": "object",
    TARGET_COLUMN: "object",
}
SOURCE_COLUMN_MAP = {
    "country": "country",
    "online_consumer": "online_consumer",
    "age": "age_group",
    "annual_salary": "annual_salary_band",
    "gender": "gender",
    "education": "education",
    "payment_method_credit_debit": "payment_method_card",
    "living_region": "living_region",
    "online_service_preference": "online_service_preference",
    "ai_endorsement": "ai_endorsement",
    "ai_privacy_no_trust": "ai_privacy_no_trust",
    "ai_enhance_experience": "ai_enhance_experience",
    "ai_satisfication": TARGET_COLUMN,
    "ai_tools_used_chatbots": "ai_tool_chatbots",
    "ai_tools_used_virtual_assistant": "ai_tool_virtual_assistant",
    "ai_tools_used_voice_photo_search": "ai_tool_voice_photo_search",
    "payment_method_cod": "payment_method_cod",
    "payment_method_ewallet": "payment_method_ewallet",
    "product_category_appliances": "product_category_appliances",
    "product_category_electronics": "product_category_electronics",
    "product_category_groceries": "product_category_groceries",
    "product_category_personal_care": "product_category_personal_care",
    "product_category_clothing": "product_category_clothing",
}


def _find_first_csv(directory: Path) -> Optional[Path]:
    csv_files = sorted(directory.rglob("*.csv"))
    return csv_files[0] if csv_files else None


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower()).strip("_")


def _normalize_text_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    return value.strip().replace("’", "'")


def _normalize_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns=lambda col: _normalize_column_name(str(col)))
    normalized = normalized.rename(columns=SOURCE_COLUMN_MAP)
    for column in normalized.select_dtypes(include="object").columns:
        normalized[column] = normalized[column].map(_normalize_text_value)
    return normalized


def download_raw_data_from_kaggle(path: Path = RAW_PATH) -> Optional[Path]:
    """Download the Kaggle dataset when the raw CSV is not present locally."""
    dataset_slug = os.getenv("KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET).strip()

    if dataset_slug == "owner/dataset-slug":
        raise ValueError(
            "KAGGLE_DATASET is still set to the README placeholder 'owner/dataset-slug'. "
            "Replace it with a real Kaggle dataset slug such as 'username/dataset-name'."
        )

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Automatic Kaggle download requires 'kagglehub'. "
            "Install requirements.txt before running the pipeline."
        ) from exc

    logger.info("Dataset missing locally. Downloading from Kaggle dataset '%s'.", dataset_slug)
    try:
        download_dir = Path(kagglehub.dataset_download(dataset_slug))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Kaggle dataset '{dataset_slug}'. "
            "Check that the slug is real, your Kaggle credentials are configured, "
            "and your account has permission to access the dataset."
        ) from exc

    source_csv = _find_first_csv(download_dir)
    if source_csv is None:
        raise FileNotFoundError(f"No CSV file was found in downloaded Kaggle dataset: {download_dir}")

    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, path)
    logger.info("Copied Kaggle dataset from %s to %s", source_csv, path)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV from *path* and return a DataFrame."""
    if not path.exists():
        download_raw_data_from_kaggle(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place the CSV there manually, or set KAGGLE_DATASET=<owner/dataset> "
            "to download it automatically from Kaggle."
        )
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding)
            df = _normalize_raw_dataset(df)
            logger.info(
                "Loaded %d rows × %d cols from %s using encoding %s",
                *df.shape,
                path,
                encoding,
            )
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        getattr(last_error, "encoding", "utf-8"),
        getattr(last_error, "object", b""),
        getattr(last_error, "start", 0),
        getattr(last_error, "end", 0),
        (
            f"Unable to decode CSV at {path}. Tried encodings: {', '.join(CSV_ENCODINGS)}. "
            "Verify that the downloaded dataset is a text CSV compatible with pandas."
        ),
    )


def validate_schema(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """
    Verify that all required columns are present.
    Returns (is_valid, list_of_missing_columns).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.warning("Schema validation failed — missing columns: %s", missing)
        logger.warning("Found columns in dataset: %s", list(df.columns))
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
        dataset_slug = os.getenv("KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET)
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"The current dataset{f' ({dataset_slug})' if dataset_slug else ''} does not match the training schema."
        )

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