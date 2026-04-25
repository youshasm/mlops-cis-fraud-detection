"""Concept drift simulation for the fraud detection dataset.

The script sorts data by TransactionDT, trains on the earliest 70%, evaluates on
both the original and drift-injected late split, and prints a before/after
comparison table.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BANNER = "=" * 50


def print_banner(title: str) -> None:
    """Print a consistent section banner."""
    print(BANNER)
    print(title)
    print(BANNER)


def load_data(path: str) -> pd.DataFrame:
    """Load featured data for drift simulation."""
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to load data: {exc}") from exc


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    """Create a preprocessing plus random forest model pipeline."""
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )


def split_by_transaction_time(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split earlier 70% and later 30% based on TransactionDT ordering."""
    if "TransactionDT" not in frame.columns:
        raise KeyError("TransactionDT column is required for drift simulation")
    ordered = frame.sort_values("TransactionDT").reset_index(drop=True)
    split_index = int(len(ordered) * 0.7)
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def inject_drift(test_frame: pd.DataFrame, value_column: str = "TransactionAmt", flip_fraction: float = 0.25) -> pd.DataFrame:
    """Flip a fraction of labels on high-value transactions to simulate drift."""
    if "isFraud" not in test_frame.columns:
        raise KeyError("isFraud column is required for drift injection")
    if value_column not in test_frame.columns:
        return test_frame.copy()

    drifted = test_frame.copy()
    threshold = drifted[value_column].quantile(0.9)
    high_value_indices = drifted.index[drifted[value_column] >= threshold].tolist()
    flip_count = max(int(len(high_value_indices) * flip_fraction), 1) if high_value_indices else 0
    for index in high_value_indices[:flip_count]:
        drifted.at[index, "isFraud"] = 1 - int(drifted.at[index, "isFraud"])
    return drifted


def evaluate_model(model: Pipeline, test_frame: pd.DataFrame) -> Dict[str, float]:
    """Compute fraud detection metrics for a holdout frame."""
    y_true = test_frame["isFraud"].astype(int)
    features = test_frame.drop(columns=["isFraud"])
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "auc_roc": roc_auc_score(y_true, probabilities),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simulate concept drift in fraud detection.")
    parser.add_argument(
        "--input-path",
        default=os.path.join("outputs", "data", "featured_data.csv"),
        help="Path to featured data",
    )
    parser.add_argument(
        "--flip-fraction",
        type=float,
        default=0.25,
        help="Fraction of high-value transactions to flip",
    )
    return parser.parse_args()


def main() -> int:
    """Run the drift simulation and print a comparison table."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Drift Simulation")
    try:
        frame = load_data(args.input_path)
        train_frame, test_frame = split_by_transaction_time(frame)
        drifted_test_frame = inject_drift(test_frame, flip_fraction=args.flip_fraction)

        y_train = train_frame["isFraud"].astype(int)
        X_train = train_frame.drop(columns=["isFraud"])
        pipeline = build_pipeline(X_train)
        pipeline.fit(X_train, y_train)

        before_metrics = evaluate_model(pipeline, test_frame)
        after_metrics = evaluate_model(pipeline, drifted_test_frame)

        comparison = pd.DataFrame(
            {
                "metric": ["precision", "recall", "f1", "auc_roc"],
                "before_drift": [before_metrics[key] for key in ["precision", "recall", "f1", "auc_roc"]],
                "after_drift": [after_metrics[key] for key in ["precision", "recall", "f1", "auc_roc"]],
            }
        )
        print("Before/after drift comparison table:")
        print(comparison.to_string(index=False))
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
