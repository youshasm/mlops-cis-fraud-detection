# IEEE CIS Fraud Detection MLOps Pipeline

Production-grade MLOps project for the IEEE CIS Fraud Detection dataset using Kubeflow Pipelines v2, XGBoost, LightGBM, scikit-learn, SHAP, Prometheus, FastAPI, Docker, and GitHub Actions.

Container-first setup is supported via Docker Compose. You can run everything in isolated containers without installing Python packages on your host.

## Project Overview

This repository implements an end-to-end fraud detection workflow:

- Data ingestion and validation
- Preprocessing and feature engineering
- Model training and evaluation
- SHAP explainability
- Drift simulation and retraining strategy analysis
- Kubeflow pipeline orchestration
- Prometheus monitoring and GitHub Actions CI/CD

## Repository Layout

```text
fraud-detection-mlops/
├── .github/workflows/ci_cd.yml
├── api/
│   ├── __init__.py
│   └── main.py
├── components/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   ├── test_identity.csv
│   └── sample_submission.csv
├── drift/
│   ├── __init__.py
│   └── drift_simulation.py
├── monitoring/
│   ├── alert_rules.yml
│   └── prometheus.yml
├── outputs/
│   ├── data/
│   ├── models/
│   ├── plots/
│   └── pipeline.yaml
├── pipeline/
│   ├── __init__.py
│   └── kubeflow_pipeline.py
├── retraining/
│   ├── __init__.py
│   └── retraining_strategy.py
├── tests/
│   └── test_components.py
├── Dockerfile
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- Kubeflow Pipelines v2 running locally or remotely
- Docker Desktop
- Docker Compose v2 (`docker compose`)
- IEEE CIS Fraud Detection CSV files in the `data/` folder

## Quick Start (Docker-Only, Isolated)

1. Run the full data-to-model workflow in one container:

```bash
docker compose up --build mlops-run
```

2. Start inference API and Prometheus:

```bash
docker compose up --build api prometheus
```

3. Verify services:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

4. Open Prometheus UI:

```text
http://localhost:9090
```

## Installation (Optional Local Development)

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If your environment does not already have the optional packages installed, install them before running the SHAP, LightGBM, or imbalanced-learn steps.

## Compose Services

The `docker-compose.yml` file defines two service groups.

- `pipeline` profile services:
	- `mlops-run` (recommended, runs all component scripts sequentially)
	- `ingest`, `validate`, `preprocess`, `feature_engineer`, `train`, `evaluate`, `explainability` (advanced, step-by-step containers)
- always-on services: `api`, `prometheus`

Compose project name is set to `fraud-detection-mlops` in `docker-compose.yml`.

All batch containers write outputs to `outputs/` in the project folder.

## Execution Model (Important)

There are two separate execution paths:

1. Docker Compose component flow
- Runs Python scripts directly from `components/`.
- No Kubeflow required.
- Good for local isolated development.

2. Kubeflow DSL pipeline flow
- Uses `@dsl.component` and `@dsl.pipeline` in `pipeline/kubeflow_pipeline.py`.
- Requires your Kubernetes/Kubeflow cluster.
- Compile with:

```bash
python pipeline/kubeflow_pipeline.py --output-path outputs/pipeline.yaml
```

These two flows are complementary, but they do not run automatically one before the other.

## End-to-End Workflow

Run the scripts in this order:

1. Ingest raw tables

```bash
python components/data_ingestion.py --transaction-path data/train_transaction.csv --identity-path data/train_identity.csv --output-path outputs/data/raw_data.csv
```

2. Validate the merged dataset

```bash
python components/data_validation.py --input-path outputs/data/raw_data.csv --output-path outputs/validation_report.txt
```

3. Preprocess the data

```bash
python components/preprocessing.py --input-path outputs/data/raw_data.csv --output-path outputs/data/processed_data.csv
```

4. Engineer features

```bash
python components/feature_engineering.py --input-path outputs/data/processed_data.csv --output-path outputs/data/featured_data.csv
```

5. Train models

```bash
python components/train.py --input-path outputs/data/featured_data.csv --model-dir outputs/models
```

6. Evaluate the trained models

```bash
python components/evaluate.py --input-path outputs/data/featured_data.csv --model-dir outputs/models --report-path outputs/evaluation_report.txt --plot-dir outputs/plots
```

7. Generate SHAP explainability outputs

```bash
python components/explainability.py --input-path outputs/data/featured_data.csv --model-path outputs/models/best_model.pkl --output-dir outputs/plots/shap
```

## Kubeflow Pipeline (DSL)

Yes, DSL components and pipeline are implemented in `pipeline/kubeflow_pipeline.py`.

Compile the pipeline package:

```bash
python pipeline/kubeflow_pipeline.py --output-path outputs/pipeline.yaml
```

The DSL pipeline uses:

- `@dsl.pipeline` and `@dsl.component` from KFP v2
- Artifact passing between steps
- Retry enabled on each component
- A conditional deploy step gated on AUC-ROC > 0.85
- A persistent volume claim for shared artifacts
- Target namespace `fraud-detection`

## Monitoring and API

Start the FastAPI service:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /health`
- `POST /predict`
- `GET /metrics`

Prometheus scrapes the `/metrics` endpoint using `monitoring/prometheus.yml`, and alert rules are defined in `monitoring/alert_rules.yml`.

## Drift Simulation

Run the drift experiment:

```bash
python drift/drift_simulation.py --input-path outputs/data/featured_data.csv
```

This splits the data by `TransactionDT`, injects label drift on high-value transactions, and prints a before/after comparison table.

## Retraining Strategy

Run the retraining simulation:

```bash
python retraining/retraining_strategy.py --input-path outputs/data/featured_data.csv
```

This compares a simple periodic retraining approach against the hybrid strategy that retrains weekly and triggers immediately when recall falls below 0.80.

## Docker

Build the image:

```bash
docker build -t fraud-detection-mlops .
```

The Dockerfile provides a multi-stage build for training and a lightweight FastAPI inference stage with a healthcheck.

Run the full containerized stack with Docker Compose:

```bash
docker compose up --build api prometheus
```

Run the pipeline stages as containers:

```bash
docker compose --profile pipeline up --build ingest validate preprocess feature_engineer train evaluate explainability
```

Recommended simplified command:

```bash
docker compose up --build mlops-run
```

The Compose file uses the same training image for all batch steps and a separate inference image for the API.

Useful container commands:

```bash
# Start only monitoring and inference
docker compose up --build api prometheus

# Run one full batch flow
docker compose up --build mlops-run

# Run one batch stage (advanced)
docker compose --profile pipeline run --rm train

# Stop and remove running services
docker compose down
```

## CI/CD

The GitHub Actions workflow in `.github/workflows/ci_cd.yml` includes:

- CI: flake8, pytest, and schema validation
- Build: Docker image build and push to GHCR
- CD: Kubeflow pipeline trigger via KFP v2 client
- Intelligent trigger: webhook-driven pipeline retriggering from monitoring alerts

## Outputs

Generated artifacts are written under `outputs/`:

- `outputs/data/raw_data.csv`
- `outputs/data/processed_data.csv`
- `outputs/data/featured_data.csv`
- `outputs/models/*.pkl`
- `outputs/evaluation_report.txt`
- `outputs/validation_report.txt`
- `outputs/plots/`
- `outputs/plots/shap/`
- `outputs/pipeline.yaml`

## Troubleshooting

- If `best_model.pkl` is missing, run the pipeline profile at least through the `train` service.
- If API starts but `/predict` fails, confirm feature columns match the trained model input schema.
- If Prometheus has no targets, verify the API container is healthy and `http://api:8000/metrics` is reachable from inside the Compose network.
- If Kubeflow submission fails in CI/CD, verify `KUBEFLOW_HOST` and `KUBEFLOW_TOKEN` repository secrets.

## Notes

- All scripts expose command-line arguments through `argparse`.
- File I/O and model operations are wrapped with error handling.
- Several optional dependencies are loaded dynamically so the scripts remain runnable even in lean environments.
- The included tests are smoke tests and are intended to validate the main helper functions quickly.
