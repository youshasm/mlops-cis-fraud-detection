"""Feature engineering for the IEEE CIS Fraud Detection dataset.

The script adds target encoding for high-cardinality columns, derives temporal
features from TransactionDT, scales numeric columns, and saves the featured
dataset for model training.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BANNER = "=" * 50
HIGH_CARDINALITY_COLUMNS = [
    "card1",
    "card2",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
]


def print_banner(title: str) -> None:
    """Print a consistent section banner."""
    print(BANNER)
    print(title)
    print(BANNER)


def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load the processed dataset."""
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to load processed data: {exc}") from exc


def target_encode(frame: pd.DataFrame, columns: List[str], target_column: str = "isFraud") -> pd.DataFrame:
    """Apply smoothed target encoding to selected columns."""
    result = frame.copy()
    if target_column not in result.columns:
        return result

    global_mean = result[target_column].mean()
    smoothing = 20.0

    for column in columns:
        if column not in result.columns:
            continue
        encoded_name = f"{column}_target_encoded"
        series = result[column].astype(str).fillna("missing")
        stats = result.groupby(series)[target_column].agg(["mean", "count"])
        smooth_values = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        result[encoded_name] = series.map(smooth_values).fillna(global_mean)
    return result


def create_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create hour and day features from TransactionDT."""
    result = frame.copy()
    if "TransactionDT" not in result.columns:
        return result
    transaction_dt = pd.to_numeric(result["TransactionDT"], errors="coerce").fillna(0)
    result["transaction_hour"] = ((transaction_dt // 3600) % 24).astype(int)
    result["transaction_day"] = (transaction_dt // 86400).astype(int)
    return result


def scale_numeric_features(frame: pd.DataFrame, target_column: str = "isFraud") -> pd.DataFrame:
    """Scale numeric features using StandardScaler."""
    result = frame.copy()
    numeric_columns = [
        column for column in result.select_dtypes(include=[np.number]).columns if column != target_column
    ]
    if not numeric_columns:
        return result

    scaler = StandardScaler()
    result[numeric_columns] = scaler.fit_transform(result[numeric_columns])
    return result


def save_data(frame: pd.DataFrame, output_path: str) -> None:
    """Persist the featured dataset to disk."""
    ensure_directory(os.path.dirname(output_path))
    try:
        frame.to_csv(output_path, index=False)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to save featured data: {exc}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Feature engineer the fraud dataset.")
    parser.add_argument(
        "--input-path",
        default=os.path.join("outputs", "data", "processed_data.csv"),
        help="Path to processed data",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join("outputs", "data", "featured_data.csv"),
        help="Where to save featured data",
    )
    return parser.parse_args()


def main() -> int:
    """Run feature engineering end to end."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Feature Engineering")
    try:
        frame = load_data(args.input_path)
        frame = target_encode(frame, HIGH_CARDINALITY_COLUMNS, target_column="isFraud")
        frame = create_time_features(frame)
        frame = scale_numeric_features(frame, target_column="isFraud")
        print("Featured columns added:")
        print([column for column in frame.columns if column.endswith("_target_encoded") or column in ["transaction_hour", "transaction_day"]])
        save_data(frame, args.output_path)
        print(f"Saved featured data to: {args.output_path}")
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
