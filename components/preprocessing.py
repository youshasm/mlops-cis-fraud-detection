"""Preprocessing for the IEEE CIS Fraud Detection dataset.

This script drops high-missingness columns, imputes remaining missing values,
applies KNN imputation to the most informative numeric features, and compares
SMOTE against class-weighted learning on a numeric subset.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

try:  # pragma: no cover - runtime fallback when optional dependency is unavailable
    SMOTE = importlib.import_module("imblearn.over_sampling").SMOTE
except Exception:  # pragma: no cover - runtime fallback
    class SMOTE:  # type: ignore[no-redef]
        """Fallback sampler that returns the input unchanged."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit_resample(self, features, target):
            return features, target


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
    """Load the raw merged dataset."""
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to load raw data: {exc}") from exc


def drop_sparse_columns(frame: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with more than threshold missing values."""
    missing_share = frame.isna().mean()
    sparse_columns = missing_share[missing_share > threshold].index.tolist()
    if sparse_columns:
        print(f"Dropping columns with >{threshold:.0%} missing values: {sparse_columns}")
    return frame.drop(columns=sparse_columns)


def fill_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric columns with median and categorical columns with mode."""
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in result.columns if column not in numeric_columns]

    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")
        result[numeric_columns] = numeric_imputer.fit_transform(result[numeric_columns])

    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        result[categorical_columns] = categorical_imputer.fit_transform(result[categorical_columns])

    return result


def knn_impute_important_numeric(frame: pd.DataFrame, target_column: str = "isFraud") -> pd.DataFrame:
    """Apply KNN imputation to the top 10 informative numeric columns.

    The numeric columns are ranked by mutual information with the target.
    """
    if target_column not in frame.columns:
        return frame

    result = frame.copy()
    numeric_columns = [
        column for column in result.select_dtypes(include=[np.number]).columns if column != target_column
    ]
    if not numeric_columns:
        return result

    target = result[target_column].astype(int)
    numeric_frame = result[numeric_columns].copy()

    ranked_scores = mutual_info_classif(numeric_frame.fillna(numeric_frame.median()), target, random_state=42)
    ranked = pd.Series(ranked_scores, index=numeric_columns).sort_values(ascending=False)
    top_columns = ranked.head(min(10, len(ranked))).index.tolist()
    if not top_columns:
        return result

    print(f"Top numeric columns selected for KNN imputation: {top_columns}")
    knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
    result[top_columns] = knn_imputer.fit_transform(result[top_columns])
    return result


def compare_imbalance_strategies(frame: pd.DataFrame, target_column: str = "isFraud") -> pd.DataFrame:
    """Compare SMOTE with class-weighted logistic regression on numeric features."""
    if target_column not in frame.columns:
        return pd.DataFrame()

    numeric_frame = frame.select_dtypes(include=[np.number]).copy()
    if target_column not in numeric_frame.columns:
        return pd.DataFrame()

    X = numeric_frame.drop(columns=[target_column])
    y = numeric_frame[target_column].astype(int)

    if X.empty or y.nunique() < 2:
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    baseline_predictions = baseline_pipeline.predict(X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    smote_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, solver="liblinear")),
        ]
    )
    smote_pipeline.fit(X_resampled, y_resampled)
    smote_predictions = smote_pipeline.predict(X_test)

    comparison = pd.DataFrame(
        {
            "strategy": ["class_weight", "SMOTE"],
            "precision": [
                precision_score(y_test, baseline_predictions, zero_division=0),
                precision_score(y_test, smote_predictions, zero_division=0),
            ],
            "recall": [
                recall_score(y_test, baseline_predictions, zero_division=0),
                recall_score(y_test, smote_predictions, zero_division=0),
            ],
            "f1": [
                f1_score(y_test, baseline_predictions, zero_division=0),
                f1_score(y_test, smote_predictions, zero_division=0),
            ],
        }
    )
    return comparison


def save_data(frame: pd.DataFrame, output_path: str) -> None:
    """Save processed data to disk."""
    ensure_directory(os.path.dirname(output_path))
    try:
        frame.to_csv(output_path, index=False)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to save processed data: {exc}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess the merged fraud dataset.")
    parser.add_argument(
        "--input-path",
        default=os.path.join("outputs", "data", "raw_data.csv"),
        help="Path to raw merged data",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join("outputs", "data", "processed_data.csv"),
        help="Where to save processed data",
    )
    return parser.parse_args()


def main() -> int:
    """Run preprocessing end to end."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Preprocessing")
    try:
        frame = load_data(args.input_path)
        frame = drop_sparse_columns(frame, threshold=0.5)
        frame = fill_missing_values(frame)
        frame = knn_impute_important_numeric(frame, target_column="isFraud")
        comparison = compare_imbalance_strategies(frame, target_column="isFraud")
        if not comparison.empty:
            print("\nSMOTE vs class_weight comparison:")
            print(comparison.to_string(index=False))
        save_data(frame, args.output_path)
        print(f"Saved processed data to: {args.output_path}")
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
