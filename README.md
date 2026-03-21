# DDM501 Final Project Repository

AI retail satisfaction prediction project built around the Kaggle dataset
`pooriamst/online-shopping`.

The repository includes:

- a training pipeline in `pipeline/`
- experiment tracking with MLflow
- a FastAPI inference service in `main.py`
- Prometheus and Grafana support through Docker Compose
- test coverage in `tests/`

## Repository Structure

- `pipeline/` - data ingestion, feature engineering, and model training
- `main.py` - FastAPI prediction API
- `docker-compose.yml` - API, MLflow, Prometheus, and Grafana stack
- `Dockerfile` - API container build
- `tests/` - unit and integration tests
- `.github/workflows/ci_cd.yml` - CI pipeline

## Prerequisites

- Python 3.11
- PowerShell on Windows
- Docker Desktop if you want to run the container stack
- Kaggle credentials configured locally if you want automatic dataset download

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

If VS Code opens a terminal that is not using the virtual environment, run commands
with the project interpreter directly:

```powershell
.\venv\Scripts\python.exe main.py
```

Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Requirement

The project now targets the Kaggle dataset:

```text
pooriamst/online-shopping
```

The ingestion pipeline automatically normalizes the raw Kaggle columns into the
internal training schema.

Normalized feature set used by the model:

- `country`
- `online_consumer`
- `age_group`
- `annual_salary_band`
- `gender`
- `education`
- `payment_method_card`
- `living_region`
- `online_service_preference`
- `ai_endorsement`
- `ai_privacy_no_trust`
- `ai_enhance_experience`
- `ai_tool_chatbots`
- `ai_tool_virtual_assistant`
- `ai_tool_voice_photo_search`
- `payment_method_cod`
- `payment_method_ewallet`
- `product_category_appliances`
- `product_category_electronics`
- `product_category_groceries`
- `product_category_personal_care`
- `product_category_clothing`
- `satisfaction_level`

The raw file is expected at:

```text
data/raw/online_shopping_ai.csv
```

You have two ways to provide it.

### Option 1: Place the CSV manually

Copy a compatible CSV file to:

```text
data/raw/online_shopping_ai.csv
```

### Option 2: Download from Kaggle automatically

The code uses `pooriamst/online-shopping` by default. In most cases you do not
need to set an environment variable.

Optional override:

```powershell
$env:KAGGLE_DATASET="pooriamst/online-shopping"
```

Then run training. The ingestion pipeline will:

1. download the dataset with `kagglehub`
2. copy the first CSV it finds into `data/raw/online_shopping_ai.csv`
3. continue ingestion and training

If Kaggle returns `403 Client Error: Forbidden`, it usually means one of these:

1. you left the placeholder value unchanged
2. your Kaggle API credentials are not configured locally
3. the dataset requires access or consent your account does not have

## Run MLflow Locally

Start MLflow in a separate terminal from the repo root:

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

The training pipeline uses:

```text
MLFLOW_TRACKING_URI=http://localhost:5000
```

You can override it if needed.

## Train the Model

From the repo root:

```powershell
python pipeline/train.py
```

This pipeline will:

1. download `pooriamst/online-shopping` if the raw CSV is missing
2. normalize the raw Kaggle schema to the internal model schema
3. split the data into train, validation, and test sets
4. train Logistic Regression, Random Forest, and LightGBM models
5. log metrics and artifacts to MLflow
6. save the best model and preprocessing artifacts to `artifacts/`

If training succeeds, these artifacts are created:

```text
artifacts/model.joblib
artifacts/preprocessor.joblib
artifacts/label_encoder.joblib
```

## Run the API Locally

After training has generated artifacts, start the API:

```powershell
.\venv\Scripts\activate
python main.py
```

If the terminal is still not using the virtual environment, run:

```powershell
.\venv\Scripts\python.exe main.py
```

Available endpoints:

- `GET /api/v1/health`
- `GET /api/v1/model/info`
- `POST /api/v1/predict`
- `POST /api/v1/predict/batch`
- `GET /metrics`

The API now accepts the normalized retail features that correspond to the
Kaggle dataset, for example:

```json
{
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
	"product_category_clothing": "NO"
}
```

Local URLs:

- API docs: `http://localhost:8000/docs`
- Health endpoint: `http://localhost:8000/api/v1/health`
- Metrics endpoint: `http://localhost:8000/metrics`

## Run Tests

Run the test suite from the repo root:

```powershell
python -m pytest tests/test_system.py -q
```

## Run with Docker Compose

The API container expects trained artifacts in the local `artifacts/` folder.

Recommended flow:

1. train locally first
2. confirm `artifacts/` contains the model files
3. start the stack

Start the stack:

```powershell
docker compose up -d --build
```

Services:

- API: `http://localhost:8000`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Stop the stack:

```powershell
docker compose down
```

## Current Limitation

The codebase is now aligned to `pooriamst/online-shopping`.

If you switch to a different dataset, you will need to update:

1. the ingestion normalization in `pipeline/data_ingestion.py`
2. the feature engineering logic in `pipeline/feature_engineer.py`
3. the API request schema in `main.py`
