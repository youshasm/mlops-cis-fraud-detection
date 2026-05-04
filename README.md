# IEEE CIS Fraud Detection MLOps Pipeline

End-to-end fraud detection MLOps project using Kubeflow Pipelines v2, XGBoost/LightGBM/RandomForest, SHAP, Prometheus, Grafana, and GitHub Actions.

This README is focused on:
- Kubeflow execution (primary path)
- GitHub Actions with self-hosted runner (`.\\run.cmd`)
- Prometheus and Grafana setup

## 1) Prerequisites

- Windows with PowerShell
- Python 3.10+
- Kubernetes cluster (Docker Desktop Kubernetes is OK)
- Kubeflow Pipelines installed and accessible
- `kubectl` configured
- Dataset files inside `data/`:
  - `train_transaction.csv`
  - `train_identity.csv`
  - `test_transaction.csv`
  - `test_identity.csv`
  - `sample_submission.csv`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 2) Verify Project Structure

Expected important folders/files:

```text
.github/workflows/ci_cd.yml
components/
pipeline/kubeflow_pipeline.py
monitoring/prometheus.yml
monitoring/alert_rules.yml
monitoring/alertmanager.yml
monitoring/grafana/
data/
outputs/
```

## 3) Kubeflow Pipeline: Complete Step-by-Step

### Step 3.1: Ensure Kubeflow is reachable

```bash
kubectl get pods -n kubeflow
```

If using local UI access:

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Open: `http://127.0.0.1:8080`

### Step 3.2: Create/verify shared PVC

```bash
kubectl apply -f pipeline/fraud-detection-pvc.yaml
kubectl get pvc -n kubeflow
```

### Step 3.3: Copy CSV files to PVC (required by pipeline pods)

Use your helper pod (already used in your setup):

```bash
kubectl exec -n kubeflow pvc-copy-helper -- sh -c "mkdir -p /mnt/shared/data"
kubectl cp "data/train_transaction.csv" kubeflow/pvc-copy-helper:/mnt/shared/data/
kubectl cp "data/train_identity.csv" kubeflow/pvc-copy-helper:/mnt/shared/data/
kubectl cp "data/test_transaction.csv" kubeflow/pvc-copy-helper:/mnt/shared/data/
kubectl cp "data/test_identity.csv" kubeflow/pvc-copy-helper:/mnt/shared/data/
kubectl cp "data/sample_submission.csv" kubeflow/pvc-copy-helper:/mnt/shared/data/
kubectl exec -n kubeflow pvc-copy-helper -- sh -c "ls -lh /mnt/shared/data"
```

### Step 3.4: Compile pipeline YAML

```bash
python pipeline/kubeflow_pipeline.py --output-path outputs/pipeline.yaml
```

### Step 3.5: Submit a run

Use your submission helper script:

```bash
python submit_fixed_run.py
```

### Step 3.6: Monitor run status

```bash
python check_run_status.py
kubectl get pods -n kubeflow | Select-String "fraud-detection-mlops|NAME"
```

Pipeline steps executed sequentially:
1. ingest
2. validate
3. preprocess
4. feature_engineer
5. train
6. evaluate
7. conditional_deploy (only if AUC-ROC > 0.85)

### Step 3.7: Clean old failed pods (optional)

```bash
$failedPods = kubectl get pods -n kubeflow --no-headers | Where-Object { $_ -match 'fraud-detection-mlops' -and ($_ -match '\\sError\\s' -or $_ -match '\\sOOMKilled\\s') } | ForEach-Object { ($_ -split '\\s+')[0] }
$failedPods | ForEach-Object { kubectl delete pod -n kubeflow $_ --ignore-not-found=true }
```

## 4) GitHub Actions: Self-Hosted Runner (`.\\run.cmd`)

### Step 4.1: Create required repository secrets

In GitHub repo: **Settings > Secrets and variables > Actions > New repository secret**

Add at least:
- `KUBEFLOW_HOST` = `http://127.0.0.1:8080` (or your cluster endpoint)
- `KUBEFLOW_NAMESPACE` = `fraud-detection`
- `GHCR_PAT` = GitHub Personal Access Token with package push permissions (if workflow pushes to GHCR)
- `GITHUB_TOKEN` is auto-provided by GitHub Actions (no manual secret needed)

Optional for alert-based retraining:
- `ALERT_WEBHOOK_URL`

### Step 4.2: Create and register the self-hosted runner

In GitHub repo: **Settings > Actions > Runners > New self-hosted runner**

Download and configure runner from GitHub instructions, then start it from runner folder:

```powershell
.\run.cmd
```

Keep this terminal open while jobs run.

### Step 4.3: Verify labels in workflow

Open `.github/workflows/ci_cd.yml` and confirm `runs-on` matches your runner labels.

Example:

```yaml
runs-on: [self-hosted, windows]
```

### Step 4.4: Trigger workflow

Option A: push to branch configured in workflow trigger.

```bash
git add .
git commit -m "Trigger CI/CD"
git push origin main
```

Option B: run manually from GitHub UI (**Actions > workflow > Run workflow**).

### Step 4.5: Watch job logs

- GitHub UI: **Actions > selected run > job > step logs**
- Runner terminal (`.\\run.cmd`) shows real-time execution on your machine

### Step 4.6: Confirm CD stage submitted Kubeflow run

After CD job success:

```bash
python check_run_status.py
```

Also verify in Kubeflow UI: `http://127.0.0.1:8080`

## 5) Prometheus (Docker)

Only monitoring services use Docker here.

### Step 5.1: Start Prometheus

```bash
docker run -d --name prometheus -p 9090:9090 -v "${PWD}/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:latest --config.file=/etc/prometheus/prometheus.yml
```

Open:
- `http://localhost:9090`
- `http://localhost:9090/targets`
- `http://localhost:9090/alerts`

### Step 5.2: Reload config/rules after edits

```bash
curl -X POST http://localhost:9090/-/reload
```

### Step 5.3: Useful queries

```promql
model_fraud_recall
model_false_positive_rate
rate(http_requests_total{status=~"5.."}[5m])
histogram_quantile(0.95, http_request_duration_seconds_bucket)
```

## 6) Grafana (Docker)

### Step 6.1: Start Grafana with provisioning

```bash
docker run -d --name grafana -p 3000:3000 -v "${PWD}/monitoring/grafana/provisioning:/etc/grafana/provisioning" -v "${PWD}/monitoring/grafana/dashboards:/var/lib/grafana/dashboards" grafana/grafana:latest
```

Open: `http://localhost:3000`

Default login:
- Username: `admin`
- Password: `admin`

### Step 6.2: Verify dashboards

Expected dashboards:
- `system_health.json`
- `model_performance.json`
- `data_drift.json`

### Step 6.3: Verify datasource

Datasource provisioning file:
- `monitoring/grafana/provisioning/datasources/prometheus.yml`

Dashboard provisioning file:
- `monitoring/grafana/provisioning/dashboards/dashboards.yml`

If datasource is not auto-loaded, add Prometheus manually with URL:
- `http://host.docker.internal:9090`

## 7) GitHub Actions + Monitoring Integration (Optional)

- Prometheus evaluates alerts from `monitoring/alert_rules.yml`
- Alertmanager routes events based on `monitoring/alertmanager.yml`
- Alert bridge can send webhook/repository_dispatch to trigger retraining workflow

## 8) Outputs

Main generated outputs:
- `outputs/data/raw_data.csv`
- `outputs/data/processed_data.csv`
- `outputs/data/featured_data.csv`
- `outputs/models/*.pkl`
- `outputs/evaluation_report.txt`
- `outputs/validation_report.txt`
- `outputs/plots/`
- `outputs/plots/shap/`
- `outputs/pipeline.yaml`

## 9) Quick Troubleshooting

- Kubeflow run stuck at ingest: verify CSV files exist in `/mnt/shared/data` inside PVC.
- Frequent OOMKilled pods: reduce memory usage in ingest/preprocess or increase cluster resources.
- GitHub self-hosted job not picked: ensure runner terminal with `.\\run.cmd` is online and labels match workflow.
- Prometheus shows down target: verify endpoint path `/metrics` and network reachability.
- Grafana empty panels: check Prometheus datasource URL and time range.
