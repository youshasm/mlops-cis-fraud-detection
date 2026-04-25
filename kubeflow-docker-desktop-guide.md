# Kubeflow on Docker Desktop — Complete Setup Guide

> **Tested on:** Windows 11, Docker Desktop with WSL 2, Kubernetes v1.34.1, KFP v2.5.0  
> **Time required:** ~45 minutes (mostly image download time)

---

## Why This Guide Exists

Setting up Kubeflow locally has several hidden pitfalls that will waste hours if you don't know about them upfront:

| Pitfall | What Happens |
|---|---|
| Using KFP version < 2.5.0 | Images point to dead `gcr.io` links → `ImagePullBackOff` forever |
| Using `$VAR` syntax in Windows CMD | Variable not expanded → wrong version fetched from GitHub |
| Not allocating enough RAM | Pods stuck in `Pending` indefinitely |
| Not waiting for MySQL before other pods | `CrashLoopBackOff` race conditions |

This guide avoids all of them.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11, macOS, Linux | Windows 11 with WSL 2 |
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 6+ cores |
| Disk | 50 GB free | 80 GB free |
| Docker Desktop | v4.x+ | Latest |

---

## Part 1 — System Configuration

### 1.1 Configure WSL 2 Resources (Windows Only)

Docker Desktop on Windows uses WSL 2 to manage resources. You must configure limits manually via a `.wslconfig` file — Docker Desktop's resource sliders have no effect on WSL 2 backends.

Open Notepad or VS Code and create/edit:

```
C:\Users\<YourUsername>\.wslconfig
```

Add the following (adjust based on your total RAM):

```ini
[wsl2]
memory=12GB       # Use ~60-70% of your total RAM
processors=4      # Number of CPU cores to allocate
swap=4GB          # Swap space
```

**RAM allocation guide:**

| Your Total RAM | Set `memory=` |
|---|---|
| 8 GB | 5GB (tight, may struggle) |
| 16 GB | 10GB |
| 32 GB | 20GB |
| 64 GB | 40GB |

Then restart WSL:

```powershell
wsl --shutdown
```

Restart Docker Desktop after this.

---

### 1.2 Enable Kubernetes in Docker Desktop

1. Open Docker Desktop
2. Click the **gear icon** (Settings)
3. Navigate to **Kubernetes** in the left sidebar
4. Select **Kubeadm** as the cluster provisioning method

   > ⚠️ **Do NOT use Kind** for Kubeflow. Kind uses `containerd` as its default runtime, which requires manifest modifications to work with standard Kubeflow images. Kubeadm works out of the box.

5. Check **Enable Kubernetes**
6. Click **Apply & Restart**
7. Wait for the green Kubernetes indicator in the bottom-left of Docker Desktop

### 1.3 Verify the Cluster is Ready

```bash
kubectl cluster-info
kubectl get nodes
```

Expected output:

```
NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   Xm    v1.34.x
```

If the node shows `Ready` you are good to proceed.

---

## Part 2 — Deploy Kubeflow Pipelines

### ⚠️ Critical: Version Selection

**Always use KFP version 2.5.0 or higher.**

Here is why:

- KFP ≤ 2.4.x → images hosted on `gcr.io` (Google Container Registry)
- Google deprecated the public `gcr.io` registry in 2023
- All image pulls from `gcr.io` now return 404 → `ImagePullBackOff`
- KFP 2.5.0+ → images migrated to `ghcr.io` (GitHub Container Registry) ✅

The correct images look like this:

```
ghcr.io/kubeflow/kfp-api-server:2.5.0        ✅ Works
ghcr.io/kubeflow/kfp-frontend:2.5.0          ✅ Works
gcr.io/ml-pipeline/frontend:2.3.0            ❌ 404 Dead
```

---

### 2.1 Set the Pipeline Version

**PowerShell:**
```powershell
$env:PIPELINE_VERSION="2.5.0"
```

**CMD:**
```cmd
set PIPELINE_VERSION=2.5.0
```

> ⚠️ **Windows CMD vs PowerShell variable syntax:**
> - CMD uses `%PIPELINE_VERSION%`
> - PowerShell uses `$env:PIPELINE_VERSION`
> - Linux/Mac Bash uses `$PIPELINE_VERSION`
>
> Mixing these up is the #1 cause of deployment failures. When in doubt, just hardcode the version number directly in the commands below.

---

### 2.2 Apply Cluster-Scoped Resources

**PowerShell:**
```powershell
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$env:PIPELINE_VERSION"
```

**CMD:**
```cmd
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=%PIPELINE_VERSION%"
```

**Or hardcoded (safest):**
```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.5.0"
```

Expected output includes:
```
namespace/kubeflow created
customresourcedefinition.apiextensions.k8s.io/applications.app.k8s.io created
customresourcedefinition.apiextensions.k8s.io/workflows.argoproj.io created
...
clusterrolebinding.rbac.authorization.k8s.io/kubeflow-pipelines-cache-deployer-clusterrolebinding created
```

---

### 2.3 Wait for CRDs to be Established

```bash
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
```

Expected output:
```
customresourcedefinition.apiextensions.k8s.io/applications.app.k8s.io condition met
```

---

### 2.4 Deploy Kubeflow Pipelines Platform

**PowerShell:**
```powershell
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$env:PIPELINE_VERSION"
```

**CMD:**
```cmd
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=%PIPELINE_VERSION%"
```

**Or hardcoded:**
```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.5.0"
```

This creates 14 deployments, 8 services, persistent volumes for MySQL and MinIO, and all RBAC resources.

---

## Part 3 — Fix the MinIO Image (Required for KFP 2.5.0)

Even in KFP 2.5.0, the MinIO image was not migrated and still points to a dead `gcr.io` link:

```
gcr.io/ml-pipeline/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance  ❌
```

You must patch it manually to use the official Docker Hub image:

**PowerShell (single line, recommended):**
```powershell
kubectl set image deployment/minio minio=minio/minio:RELEASE.2019-08-14T20-37-41Z -n kubeflow
```

**PowerShell (multi-line using backtick):**
```powershell
kubectl set image deployment/minio `
    minio=minio/minio:RELEASE.2019-08-14T20-37-41Z `
    -n kubeflow
```

**Bash:**
```bash
kubectl set image deployment/minio minio=minio/minio:RELEASE.2019-08-14T20-37-41Z -n kubeflow
```

> ⚠️ In PowerShell, `\` is not a line-continuation character. Use a one-liner or backtick `` ` ``.

Expected output:
```
deployment.apps/minio image updated
```

> **Why this matters:** `ml-pipeline` (the core API server) connects to MinIO on startup to create its artifact bucket. If MinIO fails to start, `ml-pipeline` crashes, which causes `ml-pipeline-persistenceagent` and `ml-pipeline-scheduledworkflow` to crash in a chain reaction.

---

## Part 4 — Wait for Pods to Start

### 4.1 Watch Pod Status

```bash
kubectl get pods -n kubeflow --watch
```

Image downloads take **10–20 minutes** on the first run depending on your internet speed. This is normal.

Pod lifecycle:
```
Pending → ContainerCreating → Running ✅
```

Some pods will show `CrashLoopBackOff` early — this is expected due to startup race conditions (e.g., `metadata-grpc` starting before MySQL is ready). Kubernetes will retry them automatically.

### 4.2 Expected Final State

Run this when things settle down:

```bash
kubectl get pods -n kubeflow
```

All 14 pods should show `1/1 Running`:

```
cache-deployer-deployment-xxx         1/1     Running
cache-server-xxx                      1/1     Running
metadata-envoy-deployment-xxx         1/1     Running
metadata-grpc-deployment-xxx          1/1     Running   ← may show 4 restarts, normal
metadata-writer-xxx                   1/1     Running
minio-xxx                             1/1     Running
ml-pipeline-xxx                       1/1     Running
ml-pipeline-persistenceagent-xxx      1/1     Running
ml-pipeline-scheduledworkflow-xxx     1/1     Running
ml-pipeline-ui-xxx                    1/1     Running
ml-pipeline-viewer-crd-xxx            1/1     Running
ml-pipeline-visualizationserver-xxx   1/1     Running
mysql-xxx                             1/1     Running
workflow-controller-xxx               1/1     Running
```

---

## Part 5 — Handle Startup Race Conditions

Due to the order in which Kubernetes starts pods, some will crash and need a manual restart after their dependencies are ready. This is the standard sequence:

```
MySQL starts
    ↓ (metadata-grpc was crashing while waiting for this)
metadata-grpc becomes healthy
    ↓
MinIO starts (after image patch)
    ↓ (ml-pipeline was crashing while waiting for this)
ml-pipeline becomes healthy
    ↓ (persistenceagent was crashing while waiting for this)
All pods healthy ✅
```

If after 10 minutes any of these pods are still in `CrashLoopBackOff`, restart them manually:

```bash
kubectl rollout restart deployment/ml-pipeline -n kubeflow
kubectl rollout restart deployment/ml-pipeline-scheduledworkflow -n kubeflow
kubectl rollout restart deployment/ml-pipeline-persistenceagent -n kubeflow
```

---

## Part 6 — Access the Kubeflow UI

```bash
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
```

Keep this terminal open. Open your browser and navigate to:

```
http://localhost:8080
```

You should see the Kubeflow Pipelines dashboard.

> **Note:** Every time you restart Docker Desktop, you need to re-run the port-forward command. The cluster and pods persist, but port-forwarding does not.

---

## Part 7 — Teardown & Cleanup

To remove Kubeflow Pipelines completely:

```bash
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.5.0"
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.5.0"
```

To reset the entire Kubernetes cluster (nuclear option):

- Docker Desktop → Settings → Kubernetes → **Reset Kubernetes Cluster**

---

## Troubleshooting

### Pod stuck in `ImagePullBackOff`

Check which image is failing:
```bash
kubectl describe pod <pod-name> -n kubeflow | grep "Image:"
```

If it shows `gcr.io/ml-pipeline/...` the image is dead. Use `kubectl set image` to patch it to a working registry equivalent.

### Pod stuck in `Pending`

Usually a resource issue. Check:
```bash
kubectl describe pod <pod-name> -n kubeflow | grep -A5 "Events:"
```

If it says `Insufficient memory`, increase your `.wslconfig` memory allocation and restart WSL.

### `metadata-grpc` keeps restarting

This is a MySQL race condition. Check if MySQL is ready first:
```bash
kubectl get pod -n kubeflow -l app=mysql
```

Wait for `1/1 Running`, then the `metadata-grpc` pod will self-heal within a few retry cycles.

### `ml-pipeline` keeps crashing

Check the logs:
```bash
kubectl logs -n kubeflow deployment/ml-pipeline 2>&1 | Select-Object -First 40
```

If you see `connection refused` to MinIO, make sure you applied the MinIO image patch in Part 3, then restart:
```bash
kubectl rollout restart deployment/ml-pipeline -n kubeflow
```

### Port-forward stops working

The port-forward process may have been killed. Simply re-run:
```bash
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
```

---

## Quick Reference — Full Setup Commands

Copy-paste this entire block in PowerShell for a clean install:

```powershell
# Step 1: Verify cluster
kubectl get nodes

# Step 2: Deploy cluster-scoped resources
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.5.0"

# Step 3: Wait for CRDs
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# Step 4: Deploy platform
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.5.0"

# Step 5: Fix MinIO image (REQUIRED)
kubectl set image deployment/minio minio=minio/minio:RELEASE.2019-08-14T20-37-41Z -n kubeflow

# Step 6: Wait for all pods (10-20 min)
kubectl get pods -n kubeflow --watch

# Step 7: If ml-pipeline keeps crashing after MinIO is Running, restart it
kubectl rollout restart deployment/ml-pipeline -n kubeflow
kubectl rollout restart deployment/ml-pipeline-scheduledworkflow -n kubeflow
kubectl rollout restart deployment/ml-pipeline-persistenceagent -n kubeflow

# Step 8: Access UI
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
# Open http://localhost:8080
```

---

## Version Reference

| KFP Version | Image Registry | Status |
|---|---|---|
| ≤ 2.4.x | `gcr.io/ml-pipeline/` | ❌ Dead — do not use |
| **2.5.0** | `ghcr.io/kubeflow/` | ✅ Working — minimum recommended |
| **2.14.x** | `ghcr.io/kubeflow/` | ✅ Working |
| **2.15.x** | `ghcr.io/kubeflow/` | ✅ Latest stable (uses SeaweedFS instead of MinIO by default) |

> For local development, **2.5.0 is the sweet spot** — stable, well-documented, and simpler than 2.15.x which has additional migration requirements.

---

*Guide written based on real deployment experience on Windows 11 + Docker Desktop + WSL 2 + Kubernetes v1.34.1*
