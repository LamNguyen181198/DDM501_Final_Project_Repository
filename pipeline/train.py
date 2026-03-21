"""
Model Training & Experiment Tracking
--------------------------------------
Trains three candidate classifiers, performs hyperparameter tuning,
tracks every experiment with MLflow, and promotes the best model to
the MLflow Model Registry.

Models evaluated:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. LightGBM (primary model per architecture design)
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_ingestion import run_ingestion, split_data
from feature_engineer import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR      = Path("data")
ARTIFACTS_DIR = Path("artifacts")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "ai_retail_satisfaction"

MODELS = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"C": 1.0, "solver": "lbfgs"},
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
    },
    "lightgbm": {
        "model": LGBMClassifier(random_state=42, verbose=-1),
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
        },
    },
}


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_track(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> tuple[str, str]:
    """
    Train all models, log to MLflow, return run_id of the best model.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_run_id  = None
    best_f1      = -1.0
    best_model_name = None

    for name, config in MODELS.items():
        logger.info("Training model: %s", name)
        model  = config["model"]
        params = config["params"]
        model.set_params(**params)

        with mlflow.start_run(run_name=name) as run:
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", name)
            mlflow.log_param("dataset", "pooriamst/online-shopping")

            # ---- Cross-validation on training set ----
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
            mlflow.log_metric("cv_f1_mean", cv_f1.mean())
            mlflow.log_metric("cv_f1_std",  cv_f1.std())
            logger.info("CV F1: %.4f ± %.4f", cv_f1.mean(), cv_f1.std())

            # ---- Full training ----
            model.fit(X_train, y_train)

            # ---- Validation metrics ----
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            logger.info("Validation metrics: %s", val_metrics)

            # ---- Test metrics ----
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1]
            test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            logger.info("Test metrics: %s", test_metrics)

            # ---- Classification report as artifact ----
            report = classification_report(
                y_test,
                y_test_pred,
                labels=list(range(len(class_names))),
                target_names=class_names,
            )
            report_path = ARTIFACTS_DIR / f"{name}_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report)
            mlflow.log_artifact(str(report_path))

            # ---- Log model to MLflow ----
            if name == "lightgbm":
                mlflow.lightgbm.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")

            # ---- Track best ----
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_run_id = run.info.run_id
                best_model_name = name

    logger.info("Best model: %s (val F1=%.4f, run_id=%s)", best_model_name, best_f1, best_run_id)
    return best_run_id, best_model_name


def register_best_model(run_id: str, model_name: str) -> None:
    """Register the best model in the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, "ai_retail_satisfaction_model")
    logger.info(
        "Registered model '%s' version %s from run %s.",
        registered.name, registered.version, run_id,
    )
    # Use alias instead of stage: stages are deprecated in newer MLflow.
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name="ai_retail_satisfaction_model",
        alias="production",
        version=registered.version,
    )
    logger.info("Model alias 'production' now points to version %s.", registered.version)


def save_best_model_locally(run_id: str, model_name: str) -> None:
    """Download the production model artifact for Docker-based serving."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_uri = f"runs:/{run_id}/model"
    if model_name == "lightgbm":
        model = mlflow.lightgbm.load_model(model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri)

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    logger.info("Best model saved locally to %s.", ARTIFACTS_DIR / "model.joblib")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cleaned = run_ingestion()
    train_df, val_df, test_df = split_data(cleaned)
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocessor, le,
    ) = run_feature_engineering(train_df, val_df, test_df)

    best_run_id, best_model_name = train_and_track(
        X_train, y_train, X_val, y_val, X_test, y_test, list(le.classes_)
    )
    register_best_model(best_run_id, best_model_name)
    save_best_model_locally(best_run_id, best_model_name)
    logger.info("Model training pipeline complete.")