"""Model evaluation for the IEEE CIS Fraud Detection dataset.

The script loads every trained model artifact, evaluates each model on the same
holdout split, prints fraud-class metrics, and saves a text report plus
confusion matrix plots.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BANNER = "=" * 50


def print_banner(title: str) -> None:
    """Print a consistent section banner."""
    print(BANNER)
    print(title)
    print(BANNER)


def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load the featured dataset."""
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to load featured data: {exc}") from exc


def load_models(model_dir: str) -> Dict[str, object]:
    """Load all pickle model artifacts from disk."""
    models: Dict[str, object] = {}
    for path in glob.glob(os.path.join(model_dir, "*.pkl")):
        model_name = os.path.splitext(os.path.basename(path))[0]
        try:
            models[model_name] = joblib.load(path)
        except Exception as exc:  # pragma: no cover - runtime safety
            raise RuntimeError(f"Failed to load model {path}: {exc}") from exc
    return models


def predict_probabilities(model: object, features: pd.DataFrame) -> np.ndarray:
    """Return fraud probabilities from a fitted model."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return 1 / (1 + np.exp(-scores))
    predictions = model.predict(features)
    return np.asarray(predictions, dtype=float)


def build_report_lines(model_name: str, y_true: pd.Series, probabilities: np.ndarray) -> List[str]:
    """Create the report section for one model."""
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return [
        f"Model: {model_name}",
        f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}",
        f"Recall: {recall_score(y_true, predictions, zero_division=0):.4f}",
        f"F1: {f1_score(y_true, predictions, zero_division=0):.4f}",
        f"AUC-ROC: {roc_auc_score(y_true, probabilities):.4f}",
        f"Fraud class precision: {precision_score(y_true, predictions, pos_label=1, zero_division=0):.4f}",
        f"Fraud class recall: {recall_score(y_true, predictions, pos_label=1, zero_division=0):.4f}",
        f"Fraud class F1: {f1_score(y_true, predictions, pos_label=1, zero_division=0):.4f}",
        f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
    ]


def save_confusion_matrix_plot(model_name: str, y_true: pd.Series, predictions: np.ndarray, output_dir: str) -> None:
    """Save a confusion matrix plot for a model."""
    ensure_directory(output_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    try:
        fig.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"), dpi=300)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to save confusion matrix plot for {model_name}: {exc}") from exc
    finally:
        plt.close(fig)


def save_report(report: str, output_path: str) -> None:
    """Write the evaluation report to disk."""
    ensure_directory(os.path.dirname(output_path))
    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(report)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to save evaluation report: {exc}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fraud detection models.")
    parser.add_argument(
        "--input-path",
        default=os.path.join("outputs", "data", "featured_data.csv"),
        help="Path to featured data",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join("outputs", "models"),
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--report-path",
        default=os.path.join("outputs", "evaluation_report.txt"),
        help="Where to save the evaluation report",
    )
    parser.add_argument(
        "--plot-dir",
        default=os.path.join("outputs", "plots"),
        help="Directory for confusion matrix plots",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    return parser.parse_args()


def main() -> int:
    """Evaluate all trained models and save a report."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Model Evaluation")
    try:
        frame = load_data(args.input_path)
        if "isFraud" not in frame.columns:
            raise KeyError("isFraud column is required for evaluation")

        y = frame["isFraud"].astype(int)
        X = frame.drop(columns=["isFraud"])
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )

        models = load_models(args.model_dir)
        if not models:
            raise FileNotFoundError(f"No model artifacts found in {args.model_dir}")

        report_sections: List[str] = ["IEEE CIS Fraud Detection - Evaluation Report", BANNER]
        all_metrics: List[Dict[str, float]] = []

        for model_name, model in sorted(models.items()):
            if model_name == "training_summary":
                continue
            probabilities = predict_probabilities(model, X_test)
            predictions = (probabilities >= 0.5).astype(int)
            metrics_lines = build_report_lines(model_name, y_test, probabilities)
            report_sections.extend(metrics_lines)
            report_sections.append("")
            save_confusion_matrix_plot(model_name, y_test, predictions, args.plot_dir)
            all_metrics.append(
                {
                    "model": model_name,
                    "precision": precision_score(y_test, predictions, zero_division=0),
                    "recall": recall_score(y_test, predictions, zero_division=0),
                    "f1": f1_score(y_test, predictions, zero_division=0),
                    "auc_roc": roc_auc_score(y_test, probabilities),
                }
            )

        metrics_frame = pd.DataFrame(all_metrics).sort_values("auc_roc", ascending=False)
        report_sections.append("Model comparison summary:")
        report_sections.append(metrics_frame.to_string(index=False))
        report_text = "\n".join(report_sections)
        print(report_text)
        save_report(report_text, args.report_path)
        print(f"Saved evaluation report to: {args.report_path}")
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
