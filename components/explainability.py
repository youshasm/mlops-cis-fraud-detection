"""SHAP explainability for the IEEE CIS Fraud Detection dataset.

This script loads the best model artifact, computes SHAP values for a fraud
example, and saves summary, waterfall, and force plots under outputs/plots/shap/.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import importlib

shap = None
try:  # pragma: no cover - runtime fallback when optional dependency is unavailable
    shap = importlib.import_module("shap")
except Exception:  # pragma: no cover - runtime fallback
    shap = None


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


def load_model(path: str) -> object:
    """Load a fitted model artifact."""
    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to load model: {exc}") from exc


def get_model_components(model: object) -> Tuple[object, object, object | None]:
    """Return preprocessing, selector, and final estimator components."""
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        selector = model.named_steps.get("selector")
        estimator = model.named_steps.get("model")
        return preprocessor, estimator, selector
    return None, model, None


def build_transformed_matrix(model: object, features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Transform features into the representation used by the model."""
    preprocessor, estimator, selector = get_model_components(model)
    if preprocessor is None:
        matrix = features.to_numpy()
        return matrix, list(features.columns)

    transformed = preprocessor.transform(features)
    feature_names = list(preprocessor.get_feature_names_out())

    if selector is not None:
        selected_mask = selector.get_support()
        transformed = selector.transform(transformed)
        feature_names = [name for name, keep in zip(feature_names, selected_mask) if keep]

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed), feature_names


def save_matplotlib_plot(path: str) -> None:
    """Persist the active matplotlib figure."""
    try:
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to save plot {path}: {exc}") from exc
    finally:
        plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate SHAP explainability plots.")
    parser.add_argument(
        "--input-path",
        default=os.path.join("outputs", "data", "featured_data.csv"),
        help="Path to featured data",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join("outputs", "models", "best_model.pkl"),
        help="Path to the best trained model",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "plots", "shap"),
        help="Where to save SHAP plots",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Maximum number of rows to use for explanation plots",
    )
    return parser.parse_args()


def main() -> int:
    """Generate SHAP plots for the best model."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Explainability")
    try:
        if shap is None:
            raise RuntimeError("SHAP is not installed in the current environment")

        frame = load_data(args.input_path)
        if "isFraud" not in frame.columns:
            raise KeyError("isFraud column is required for explainability")

        model = load_model(args.model_path)
        y = frame["isFraud"].astype(int)
        X = frame.drop(columns=["isFraud"])
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        fraud_rows = X_test[y_test == 1]
        if fraud_rows.empty:
            fraud_rows = X_test.head(1)

        sample = fraud_rows.head(1)
        background = X_test.sample(n=min(args.sample_size, len(X_test)), random_state=42)
        transformed_sample, feature_names = build_transformed_matrix(model, sample)
        transformed_background, _ = build_transformed_matrix(model, background)

        preprocessor, estimator, selector = get_model_components(model)
        if hasattr(estimator, "predict_proba"):
            explainer = shap.TreeExplainer(estimator, data=transformed_background)
        else:
            explainer = shap.TreeExplainer(estimator)

        shap_values = explainer.shap_values(transformed_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)

        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[-1]

        ensure_directory(args.output_dir)

        summary_sample, _ = build_transformed_matrix(model, X_test.sample(n=min(args.sample_size, len(X_test)), random_state=42))
        summary_values = explainer.shap_values(summary_sample)
        if isinstance(summary_values, list):
            summary_values = summary_values[-1]
        summary_values = np.asarray(summary_values)

        shap.summary_plot(summary_values, summary_sample, feature_names=feature_names, show=False)
        save_matplotlib_plot(os.path.join(args.output_dir, "shap_summary.png"))

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=expected_value,
                data=transformed_sample[0],
                feature_names=feature_names,
            ),
            show=False,
        )
        save_matplotlib_plot(os.path.join(args.output_dir, "shap_waterfall.png"))

        shap.force_plot(
            expected_value,
            shap_values[0],
            transformed_sample[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        save_matplotlib_plot(os.path.join(args.output_dir, "shap_force.png"))

        contributions = pd.Series(shap_values[0], index=feature_names).sort_values(key=lambda s: s.abs(), ascending=False)
        print("Top 10 features by SHAP contribution for one fraud case:")
        print(contributions.head(10).to_string())
        print(f"Saved SHAP plots to: {args.output_dir}")
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
