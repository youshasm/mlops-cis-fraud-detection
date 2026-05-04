"""FastAPI inference service for the fraud detection model.

The service exposes /predict, /metrics, and /health endpoints and tracks basic
Prometheus metrics for request volume, latency, errors, confidence, recall, and
false positive rate.
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

try:  # pragma: no cover - runtime fallback for environments without the package
    prometheus_client = importlib.import_module("prometheus_client")
    CONTENT_TYPE_LATEST = prometheus_client.CONTENT_TYPE_LATEST
    Counter = prometheus_client.Counter
    Gauge = prometheus_client.Gauge
    Histogram = prometheus_client.Histogram
    generate_latest = prometheus_client.generate_latest
except Exception:  # pragma: no cover - runtime fallback
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class _Metric:
        """Minimal no-op metric used when prometheus_client is unavailable."""

        def inc(self, *_args, **_kwargs) -> None:
            pass

        def observe(self, *_args, **_kwargs) -> None:
            pass

        def set(self, *_args, **_kwargs) -> None:
            pass

    def Counter(*_args, **_kwargs):
        return _Metric()

    def Gauge(*_args, **_kwargs):
        return _Metric()

    def Histogram(*_args, **_kwargs):
        return _Metric()

    def generate_latest() -> bytes:
        return b""


MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("outputs", "models", "best_model.pkl"))

app = FastAPI(title="IEEE CIS Fraud Detection API", version="1.0.0")
process = psutil.Process(os.getpid())
model = None

REQUEST_COUNT = Counter("fraud_api_requests_total", "Total inference requests")
ERROR_COUNT = Counter("fraud_api_errors_total", "Total inference errors")
LATENCY = Histogram("fraud_api_request_latency_seconds", "Inference latency")
CONFIDENCE = Histogram(
    "fraud_api_prediction_confidence", "Prediction confidence distribution", buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
)
RECALL_GAUGE = Gauge("fraud_recall", "Observed fraud recall from feedback")
FALSE_POSITIVE_RATE = Gauge("false_positive_rate", "Observed false positive rate from feedback")
PRECISION_GAUGE = Gauge("fraud_precision", "Observed fraud precision from feedback")
DETECTION_RATE = Gauge("fraud_detection_rate", "Observed fraction of positive fraud predictions")
DRIFT_GAUGE = Gauge("data_drift_score", "Current data drift score")
MISSING_RATIO_GAUGE = Gauge("input_missing_ratio", "Share of missing input values")
ANOMALY_COUNT_GAUGE = Gauge("input_anomaly_count", "Input anomaly count from validation")
CPU_GAUGE = Gauge("cpu_usage_percent", "CPU utilization percentage")
MEMORY_GAUGE = Gauge("memory_usage_percent", "Memory utilization percentage")


@app.on_event("startup")
def load_model() -> None:
    """Load the trained model when the application starts."""
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("=" * 50)
            print(f"Loaded model from {MODEL_PATH}")
            print("=" * 50)
        except Exception as exc:  # pragma: no cover - runtime safety
            raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {exc}") from exc
    else:
        print(f"Model path not found: {MODEL_PATH}")


@app.get("/health")
def health() -> Dict[str, str]:
    """Return the service health status."""
    return {"status": "ok", "model_loaded": str(model is not None)}


@app.post("/predict")
def predict(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one or more fraud records."""
    global model
    REQUEST_COUNT.inc()
    start_time = time.perf_counter()
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        frame = pd.DataFrame(records)

        metadata_fields = [
            "_data_drift_score",
            "_input_missing_ratio",
            "_input_anomaly_count",
        ]
        for field in metadata_fields:
            if field in frame.columns and not frame[field].empty:
                value = pd.to_numeric(frame[field], errors="coerce").dropna()
                if not value.empty:
                    if field == "_data_drift_score":
                        DRIFT_GAUGE.set(float(value.iloc[0]))
                    elif field == "_input_missing_ratio":
                        MISSING_RATIO_GAUGE.set(float(value.iloc[0]))
                    elif field == "_input_anomaly_count":
                        ANOMALY_COUNT_GAUGE.set(float(value.iloc[0]))

        helper_columns = [column for column in frame.columns if column.startswith("_")]
        if helper_columns:
            frame = frame.drop(columns=helper_columns)

        labels = None
        if "isFraud" in frame.columns:
            labels = frame["isFraud"].astype(int).to_numpy()
            frame = frame.drop(columns=["isFraud"])

        predictions = model.predict(frame)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)
            confidences = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] > 1 else probabilities.ravel()
        else:
            confidences = np.asarray(predictions, dtype=float)
        for confidence in confidences:
            CONFIDENCE.observe(float(confidence))

        detection_rate = float(np.asarray(predictions).mean()) if len(predictions) else 0.0
        DETECTION_RATE.set(detection_rate)

        if labels is not None:
            actual = labels
            predicted = np.asarray(predictions, dtype=int)
            tp = int(((predicted == 1) & (actual == 1)).sum())
            fp = int(((predicted == 1) & (actual == 0)).sum())
            fn = int(((predicted == 0) & (actual == 1)).sum())
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            false_positive_rate = fp / max(int((actual == 0).sum()), 1)
            RECALL_GAUGE.set(recall)
            PRECISION_GAUGE.set(precision)
            FALSE_POSITIVE_RATE.set(false_positive_rate)

        CPU_GAUGE.set(process.cpu_percent(interval=None))
        MEMORY_GAUGE.set(process.memory_percent())
        return {
            "predictions": np.asarray(predictions).tolist(),
            "confidence": np.asarray(confidences).tolist(),
        }
    except HTTPException:
        ERROR_COUNT.inc()
        raise
    except Exception as exc:  # pragma: no cover - runtime safety
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - start_time)


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
