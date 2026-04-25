"""Retraining strategy simulation for fraud detection.

The script compares a simple periodic retraining policy against a hybrid policy
that retrains weekly and also reacts immediately when recall drops below 0.80.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BANNER = "=" * 50


@dataclass
class StrategyResult:
    """Container for retraining simulation outcomes."""

    strategy: str
    stability: float
    compute_cost: float
    performance_improvement: float


def print_banner(title: str) -> None:
    """Print a consistent section banner."""
    print(BANNER)
    print(title)
    print(BANNER)


def load_data(path: str) -> pd.DataFrame:
    """Load the featured dataset."""
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


def build_time_windows(frame: pd.DataFrame, window_size: int) -> List[pd.DataFrame]:
    """Split the frame into contiguous time windows sorted by TransactionDT."""
    ordered = frame.sort_values("TransactionDT").reset_index(drop=True)
    windows = []
    for start in range(0, len(ordered), window_size):
        windows.append(ordered.iloc[start : start + window_size].copy())
    return windows


def simulate_periodic_retraining(windows: List[pd.DataFrame], retrain_every: int) -> StrategyResult:
    """Simulate simple periodic retraining."""
    recalls: List[float] = []
    retrain_count = 0
    train_buffer = windows[0].copy()
    for index, window in enumerate(windows[1:], start=1):
        if index % retrain_every == 0:
            train_buffer = pd.concat([train_buffer, window], ignore_index=True)
            retrain_count += 1
        model = build_pipeline(train_buffer.drop(columns=["isFraud"]))
        model.fit(train_buffer.drop(columns=["isFraud"]), train_buffer["isFraud"].astype(int))
        predictions = model.predict(window.drop(columns=["isFraud"]))
        recalls.append(recall_score(window["isFraud"].astype(int), predictions, zero_division=0))
        train_buffer = pd.concat([train_buffer, window], ignore_index=True)
    baseline_recall = np.mean(recalls) if recalls else 0.0
    stability = 1.0 / (np.std(recalls) + 1e-6) if recalls else 0.0
    return StrategyResult(
        strategy="simple_periodic",
        stability=float(stability),
        compute_cost=float(retrain_count),
        performance_improvement=float(baseline_recall),
    )


def simulate_hybrid_retraining(windows: List[pd.DataFrame], recall_threshold: float, retrain_every: int) -> StrategyResult:
    """Simulate weekly retraining plus immediate recall-triggered retraining."""
    recalls: List[float] = []
    retrain_count = 0
    train_buffer = windows[0].copy()
    for index, window in enumerate(windows[1:], start=1):
        should_retrain = index % retrain_every == 0
        model = build_pipeline(train_buffer.drop(columns=["isFraud"]))
        model.fit(train_buffer.drop(columns=["isFraud"]), train_buffer["isFraud"].astype(int))
        predictions = model.predict(window.drop(columns=["isFraud"]))
        window_recall = recall_score(window["isFraud"].astype(int), predictions, zero_division=0)
        recalls.append(window_recall)
        if should_retrain or window_recall < recall_threshold:
            train_buffer = pd.concat([train_buffer, window], ignore_index=True)
            retrain_count += 1
        else:
            train_buffer = pd.concat([train_buffer, window], ignore_index=True)
    baseline_recall = np.mean(recalls) if recalls else 0.0
    stability = 1.0 / (np.std(recalls) + 1e-6) if recalls else 0.0
    return StrategyResult(
        strategy="hybrid",
        stability=float(stability),
        compute_cost=float(retrain_count),
        performance_improvement=float(baseline_recall),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare fraud retraining strategies.")
    parser.add_argument(
        "--input-path",
        default=os.path.join("outputs", "data", "featured_data.csv"),
        help="Path to featured data",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5000,
        help="Rows per simulation window",
    )
    parser.add_argument(
        "--retrain-every",
        type=int,
        default=7,
        help="Retrain interval in windows",
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=0.80,
        help="Recall threshold for immediate retraining",
    )
    return parser.parse_args()


def main() -> int:
    """Run the retraining strategy comparison."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Retraining Strategy")
    try:
        frame = load_data(args.input_path)
        if "isFraud" not in frame.columns or "TransactionDT" not in frame.columns:
            raise KeyError("isFraud and TransactionDT are required for retraining simulation")

        windows = build_time_windows(frame, args.window_size)
        if len(windows) < 2:
            raise ValueError("Not enough time windows to simulate retraining")

        periodic = simulate_periodic_retraining(windows, args.retrain_every)
        hybrid = simulate_hybrid_retraining(windows, args.recall_threshold, args.retrain_every)

        comparison = pd.DataFrame(
            [
                {
                    "strategy": periodic.strategy,
                    "stability": periodic.stability,
                    "compute_cost": periodic.compute_cost,
                    "performance_improvement": periodic.performance_improvement,
                },
                {
                    "strategy": hybrid.strategy,
                    "stability": hybrid.stability,
                    "compute_cost": hybrid.compute_cost,
                    "performance_improvement": hybrid.performance_improvement,
                },
            ]
        )
        print("Retraining strategy comparison table:")
        print(comparison.to_string(index=False))
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
