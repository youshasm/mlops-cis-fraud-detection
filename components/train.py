"""Model training for the IEEE CIS Fraud Detection dataset.

This script trains three model families - XGBoost, LightGBM, and a hybrid
RandomForest feature-selection pipeline - in both standard and cost-sensitive
variants. All models are saved to outputs/models/ as pickle files.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

try:  # pragma: no cover - runtime fallback when optional dependency is unavailable
    LGBMClassifier = importlib.import_module("lightgbm").LGBMClassifier
except Exception:  # pragma: no cover - runtime fallback
    LGBMClassifier = None


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


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical columns."""
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_model_pipelines(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    """Construct standard and cost-sensitive training pipelines."""
    preprocessor = build_preprocessor(X_train)
    positive_count = max(int((y_train == 1).sum()), 1)
    negative_count = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = negative_count / positive_count

    def build_lightgbm_model(class_weight: object | None = None) -> object:
        """Create a LightGBM estimator or a random forest fallback."""
        if LGBMClassifier is None:
            return RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight if class_weight is not None else None,
            )
        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight,
        )

    xgb_standard = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    xgb_cost_sensitive = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    lgb_standard = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", build_lightgbm_model()),
        ]
    )
    lgb_cost_sensitive = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", build_lightgbm_model(class_weight="balanced")),
        ]
    )

    rf_selector = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf_selector_cost = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
    )
    rf_model_cost = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    rf_standard = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("selector", SelectFromModel(rf_selector, threshold="median")),
            ("model", rf_model),
        ]
    )
    rf_cost_sensitive = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("selector", SelectFromModel(rf_selector_cost, threshold="median")),
            ("model", rf_model_cost),
        ]
    )

    return {
        "xgboost_standard": xgb_standard,
        "xgboost_cost_sensitive": xgb_cost_sensitive,
        "lightgbm_standard": lgb_standard,
        "lightgbm_cost_sensitive": lgb_cost_sensitive,
        "random_forest_standard": rf_standard,
        "random_forest_cost_sensitive": rf_cost_sensitive,
    }


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "auc_roc": roc_auc_score(y_true, probabilities),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def compute_business_cost(confusion: Dict[str, float], fraud_loss: float, false_alarm_cost: float) -> float:
    """Estimate business cost from false negatives and false positives."""
    return confusion["fn"] * fraud_loss + confusion["fp"] * false_alarm_cost


def save_model(model: Pipeline, path: str) -> None:
    """Persist a fitted model pipeline using joblib."""
    ensure_directory(os.path.dirname(path))
    try:
        joblib.dump(model, path)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to save model to {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--input-path",
        default=os.path.join("outputs", "data", "featured_data.csv"),
        help="Path to featured data",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join("outputs", "models"),
        help="Directory to save trained models",
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
        help="Random seed for splitting and estimators",
    )
    parser.add_argument(
        "--fraud-loss",
        type=float,
        default=200.0,
        help="Estimated loss per missed fraud case",
    )
    parser.add_argument(
        "--false-alarm-cost",
        type=float,
        default=5.0,
        help="Estimated cost per false positive alert",
    )
    return parser.parse_args()


def main() -> int:
    """Train all model variants and save the best model."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Model Training")
    try:
        frame = load_data(args.input_path)
        if "isFraud" not in frame.columns:
            raise KeyError("isFraud column is required for training")

        y = frame["isFraud"].astype(int)
        X = frame.drop(columns=["isFraud"])
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )

        pipelines = build_model_pipelines(X_train, y_train)
        records: List[Dict[str, float]] = []
        fitted_models: Dict[str, Pipeline] = {}

        for name, pipeline in pipelines.items():
            print(f"Training {name}...")
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate_predictions(y_test, probabilities)
            cost = compute_business_cost(metrics, args.fraud_loss, args.false_alarm_cost)
            metrics.update({"model": name, "business_cost": cost})
            records.append(metrics)
            fitted_models[name] = pipeline
            save_model(pipeline, os.path.join(args.model_dir, f"{name}.pkl"))

        results = pd.DataFrame(records)
        results["combined_score"] = results["auc_roc"] + results["f1"]
        best_row = results.sort_values(["auc_roc", "f1"], ascending=False).iloc[0]
        best_name = str(best_row["model"])
        best_model = fitted_models[best_name]
        save_model(best_model, os.path.join(args.model_dir, "best_model.pkl"))

        print("\nModel performance summary:")
        print(results[["model", "precision", "recall", "f1", "auc_roc", "business_cost"]].to_string(index=False))

        business_table = results[["model", "business_cost"]].sort_values("business_cost")
        print("\nBusiness impact table:")
        print(business_table.to_string(index=False))

        summary_path = os.path.join(args.model_dir, "training_summary.json")
        ensure_directory(args.model_dir)
        try:
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(results.to_dict(orient="records"), handle, indent=2)
        except Exception as exc:  # pragma: no cover - runtime safety
            raise RuntimeError(f"Failed to save training summary: {exc}") from exc

        print(f"Best model saved to: {os.path.join(args.model_dir, 'best_model.pkl')}")
        print(f"Training summary saved to: {summary_path}")
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
