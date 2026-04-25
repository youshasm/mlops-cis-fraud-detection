# Project: IEEE CIS Fraud Detection — MLOps Pipeline

## Context
This is a production-grade MLOps assignment for a fraud detection system using the
IEEE CIS Fraud Detection dataset (train_transaction.csv + train_identity.csv).
Kubeflow is running locally on Docker Desktop using kfp v2.

## Tech Stack
- Python 3.10+
- kfp v2 (Kubeflow Pipelines SDK) with DSL
- XGBoost, LightGBM, scikit-learn
- imbalanced-learn (SMOTE)
- SHAP for explainability
- Prometheus + Grafana for monitoring
- GitHub Actions for CI/CD
- Docker for containerization
- FastAPI for inference API
- Pandas, NumPy, Matplotlib, Seaborn

## Project Structure
fraud-detection-mlops/
├── .github/
│   ├── copilot-instructions.md
│   └── workflows/
│       └── ci_cd.yml
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── pipeline/
│   └── kubeflow_pipeline.py
├── monitoring/
│   ├── prometheus.yml
│   └── alert_rules.yml
├── drift/
│   └── drift_simulation.py
├── retraining/
│   └── retraining_strategy.py
├── outputs/
│   ├── data/
│   ├── models/
│   └── plots/
│       └── shap/
├── Dockerfile
└── requirements.txt

## Coding Standards
- Every function must have clear docstrings
- Use argparse for all configs — no hardcoded paths
- Use try/except on all file I/O and model operations
- Print a clear header banner at the start of each script
- Use print("="*50) as section separators throughout
- All outputs saved to outputs/ directory (create if not exists)
- All plots saved as PNG files
- All models saved as .pkl files
- Each script must be runnable independently from the command line

## Kubeflow DSL Rules
- Always use kfp v2 DSL syntax
- Use @dsl.pipeline decorator for the pipeline function
- Use @dsl.component decorator for each step
- Use Input and Output artifacts for passing data between components
- Add retry=2 on every component
- Use dsl.Condition for conditional deployment (deploy only if AUC-ROC > 0.85)
- Use a PersistentVolumeClaim for artifact sharing between steps
- Target namespace: fraud-detection
- Always compile pipeline to outputs/pipeline.yaml using compiler.Compiler().compile()
- Never use kfp v1 syntax (no ContainerOp, no dsl.VolumeOp old style)

## Task Requirements

### Task 1 — Kubeflow Pipeline (pipeline/kubeflow_pipeline.py)
- 7 sequential components: ingest → validate → preprocess → feature_engineer → train → evaluate → conditional_deploy
- retry=2 on each component
- Conditional deployment using dsl.Condition: deploy only if AUC-ROC > 0.85
- Persistent volume for artifact storage
- Namespace: fraud-detection
- Compile to outputs/pipeline.yaml

### Task 2 — Data Challenges
- Drop columns with >50% missing values
- Fill numerical nulls with median, categorical with mode
- Apply KNN imputer on top 10 important numerical columns
- Target encoding on high-cardinality columns: card1, card2, addr1, addr2, P_emaildomain, R_emaildomain
- Compare SMOTE vs class_weight strategies and print a comparison table

### Task 3 — Models (train.py)
Train and evaluate 3 models:
1. XGBoost
2. LightGBM
3. Hybrid: RandomForest + SelectFromModel feature selection
Metrics for each: Precision, Recall, F1, AUC-ROC, Confusion Matrix

### Task 4 — Cost-Sensitive Learning (train.py)
- Train standard vs cost-sensitive version of each model
- XGBoost: use scale_pos_weight
- LightGBM: use is_unbalance or class_weight
- RandomForest: use class_weight
- Print business impact table: fraud loss vs false alarm cost comparison

### Task 5 — CI/CD (.github/workflows/ci_cd.yml)
4 stages:
1. CI: trigger on push and pull_request → flake8, pytest, schema validation
2. Build: Docker image for training + FastAPI inference → push to GHCR
3. CD: trigger Kubeflow pipeline run using compiled pipeline.yaml via kfp v2 client
4. Intelligent Trigger: webhook from Prometheus alertmanager re-triggers pipeline on recall < 0.80 or drift exceeded

### Task 6 — Monitoring
- FastAPI inference API exposes /metrics endpoint using prometheus_client
- Track: request rate, latency, error rate, CPU/memory, fraud recall, false positive rate, confidence distribution
- Alert rules: recall < 0.80 (critical), drift > threshold (warning), latency > 500ms (warning)

### Task 7 — Drift Simulation (drift/drift_simulation.py)
- Split by TransactionDT: earlier 70% = train, later 30% = test
- Inject new fraud patterns by flipping labels on high-value transactions in test split
- Print before/after drift comparison table of model performance metrics

### Task 8 — Retraining Strategy (retraining/retraining_strategy.py)
- Hybrid: retrain weekly + trigger immediately when recall < 0.80
- Compare hybrid vs simple periodic retraining
- Print comparison table: stability, compute cost, performance improvement

### Task 9 — Explainability (explainability.py)
- Use SHAP TreeExplainer on best performing model
- Generate: summary plot, waterfall plot, force plot for one fraud case
- Print top 10 features with SHAP values
- Save all plots to outputs/plots/shap/

## Output Locations
| Output | Path |
|---|---|
| Raw merged data | outputs/data/raw_data.csv |
| Processed data | outputs/data/processed_data.csv |
| Featured data | outputs/data/featured_data.csv |
| All models | outputs/models/*.pkl |
| Evaluation report | outputs/evaluation_report.txt |
| Validation report | outputs/validation_report.txt |
| Confusion matrix plots | outputs/plots/ |
| SHAP plots | outputs/plots/shap/ |
| Compiled pipeline | outputs/pipeline.yaml |