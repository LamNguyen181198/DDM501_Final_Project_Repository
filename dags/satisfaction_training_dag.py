"""
Airflow DAG — Customer Satisfaction ML Training Pipeline
=========================================================
Orchestrates the full training pipeline for the AI retail satisfaction
prediction system (DDM501 Final Project — Group 4).

Schedule: Weekly on Monday at 02:00 UTC
Tasks
  1. ingest_data        – download + clean raw data (run_ingestion)
  2. split_data         – stratified train/val/test split
  3. feature_eng        – build preprocessor + encode target
  4. train_and_track    – train 3 models, log all runs to MLflow
  5. register_model     – promote best run to MLflow Model Registry
  6. export_model       – save model.joblib locally for the API
  7. notify             – log final summary to Airflow task log
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Ensure the project's pipeline package is importable inside Airflow workers
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DATA_DIR = _REPO_ROOT / "data"
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"

DEFAULT_ARGS = {
    "owner": "ddm501-group4",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _ingest_data(**context):
    """Download raw data from Kaggle and run ingestion pipeline."""
    from data_ingestion import run_ingestion

    logger.info("Starting data ingestion …")
    cleaned_df = run_ingestion(path=_DATA_DIR / "raw" / "online_shopping_ai.csv")
    n_rows = len(cleaned_df)
    logger.info("Ingestion complete — %d rows after cleaning.", n_rows)

    # Push shape metadata via XCom for downstream tasks
    context["ti"].xcom_push(key="n_rows", value=n_rows)
    context["ti"].xcom_push(key="columns", value=list(cleaned_df.columns))
    return n_rows


def _split_data(**context):
    """Stratified 70/10/20 split; persist splits to parquet for later tasks."""
    import pandas as pd
    from data_ingestion import run_ingestion, split_data

    logger.info("Loading cleaned data and splitting …")
    cleaned_df = run_ingestion(path=_DATA_DIR / "raw" / "online_shopping_ai.csv")
    train_df, val_df, test_df = split_data(cleaned_df)

    # Persist splits so feature-engineering and training tasks can reload them
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(_ARTIFACTS_DIR / "train_split.parquet", index=False)
    val_df.to_parquet(_ARTIFACTS_DIR / "val_split.parquet", index=False)
    test_df.to_parquet(_ARTIFACTS_DIR / "test_split.parquet", index=False)

    sizes = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    logger.info("Split sizes: %s", sizes)
    context["ti"].xcom_push(key="split_sizes", value=sizes)
    return sizes


def _feature_engineering(**context):
    """Build the ColumnTransformer preprocessor and encode target labels."""
    import pandas as pd
    from feature_engineer import run_feature_engineering

    logger.info("Running feature engineering …")
    train_df = pd.read_parquet(_ARTIFACTS_DIR / "train_split.parquet")
    val_df = pd.read_parquet(_ARTIFACTS_DIR / "val_split.parquet")
    test_df = pd.read_parquet(_ARTIFACTS_DIR / "test_split.parquet")

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, le = (
        run_feature_engineering(train_df, val_df, test_df)
    )

    # Persist arrays so the training task can load them without recomputing
    import numpy as np

    np.save(_ARTIFACTS_DIR / "X_train.npy", X_train)
    np.save(_ARTIFACTS_DIR / "X_val.npy", X_val)
    np.save(_ARTIFACTS_DIR / "X_test.npy", X_test)
    np.save(_ARTIFACTS_DIR / "y_train.npy", y_train)
    np.save(_ARTIFACTS_DIR / "y_val.npy", y_val)
    np.save(_ARTIFACTS_DIR / "y_test.npy", y_test)
    joblib.dump(le, _ARTIFACTS_DIR / "label_encoder.joblib")

    logger.info(
        "Feature engineering done — X_train shape: %s", X_train.shape
    )
    context["ti"].xcom_push(key="feature_shape", value=list(X_train.shape))
    context["ti"].xcom_push(key="classes", value=list(le.classes_))
    return list(X_train.shape)


def _train_and_track(**context):
    """Train all candidate models and log runs to MLflow."""
    import numpy as np
    from pipeline.train import train_and_track

    logger.info("Loading feature arrays …")
    X_train = np.load(_ARTIFACTS_DIR / "X_train.npy")
    X_val = np.load(_ARTIFACTS_DIR / "X_val.npy")
    X_test = np.load(_ARTIFACTS_DIR / "X_test.npy")
    y_train = np.load(_ARTIFACTS_DIR / "y_train.npy")
    y_val = np.load(_ARTIFACTS_DIR / "y_val.npy")
    y_test = np.load(_ARTIFACTS_DIR / "y_test.npy")
    le = joblib.load(_ARTIFACTS_DIR / "label_encoder.joblib")

    logger.info("Starting MLflow experiment runs …")
    best_run_id, best_model_name = train_and_track(
        X_train, y_train, X_val, y_val, X_test, y_test, list(le.classes_)
    )

    logger.info(
        "Best model: %s  run_id=%s", best_model_name, best_run_id
    )
    context["ti"].xcom_push(key="best_run_id", value=best_run_id)
    context["ti"].xcom_push(key="best_model_name", value=best_model_name)
    return best_run_id


def _register_model(**context):
    """Register the best MLflow run in the Model Registry."""
    from pipeline.train import register_best_model

    best_run_id = context["ti"].xcom_pull(
        task_ids="train_and_track", key="best_run_id"
    )
    best_model_name = context["ti"].xcom_pull(
        task_ids="train_and_track", key="best_model_name"
    )
    logger.info(
        "Registering run %s (model=%s) …", best_run_id, best_model_name
    )
    register_best_model(best_run_id, best_model_name)
    context["ti"].xcom_push(key="registered_run_id", value=best_run_id)


def _export_model(**context):
    """Download the best model artifact and save it as model.joblib."""
    from pipeline.train import save_best_model_locally

    best_run_id = context["ti"].xcom_pull(
        task_ids="train_and_track", key="best_run_id"
    )
    best_model_name = context["ti"].xcom_pull(
        task_ids="train_and_track", key="best_model_name"
    )
    logger.info("Exporting model (run_id=%s) to artifacts/ …", best_run_id)
    save_best_model_locally(best_run_id, best_model_name)
    logger.info("model.joblib written to %s", _ARTIFACTS_DIR / "model.joblib")


def _notify(**context):
    """Log a final summary message — extend to Slack/email as needed."""
    best_run_id = context["ti"].xcom_pull(
        task_ids="train_and_track", key="best_run_id"
    )
    best_model_name = context["ti"].xcom_pull(
        task_ids="train_and_track", key="best_model_name"
    )
    split_sizes = context["ti"].xcom_pull(
        task_ids="split_data", key="split_sizes"
    )
    feature_shape = context["ti"].xcom_pull(
        task_ids="feature_engineering", key="feature_shape"
    )
    logger.info(
        "=== Training pipeline complete ===\n"
        "  Best model   : %s\n"
        "  MLflow run   : %s\n"
        "  Split sizes  : %s\n"
        "  Feature shape: %s\n",
        best_model_name,
        best_run_id,
        split_sizes,
        feature_shape,
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="satisfaction_model_training",
    description="Weekly retraining pipeline for AI retail satisfaction model",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="0 2 * * 1",  # Every Monday at 02:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["ddm501", "ml-training", "satisfaction"],
) as dag:

    t1 = PythonOperator(
        task_id="ingest_data",
        python_callable=_ingest_data,
    )

    t2 = PythonOperator(
        task_id="split_data",
        python_callable=_split_data,
    )

    t3 = PythonOperator(
        task_id="feature_engineering",
        python_callable=_feature_engineering,
    )

    t4 = PythonOperator(
        task_id="train_and_track",
        python_callable=_train_and_track,
    )

    t5 = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    t6 = PythonOperator(
        task_id="export_model",
        python_callable=_export_model,
    )

    t7 = PythonOperator(
        task_id="notify",
        python_callable=_notify,
    )

    # Pipeline order: ingest → split → features → train → register → export → notify
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
