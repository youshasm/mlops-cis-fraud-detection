"""Data ingestion for the IEEE CIS Fraud Detection dataset.

This script loads the transaction and identity tables, merges them on
TransactionID, prints a compact profiling summary, and saves the merged data
under outputs/data/raw_data.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import pandas as pd


BANNER = "=" * 50


def print_banner(title: str) -> None:
    """Print a consistent section banner."""
    print(BANNER)
    print(title)
    print(BANNER)


def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_and_merge(transaction_path: str, identity_path: str) -> pd.DataFrame:
    """Load the raw tables and merge them on TransactionID."""
    try:
        transactions = pd.read_csv(transaction_path)
        identities = pd.read_csv(identity_path)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to read input files: {exc}") from exc

    if "TransactionID" not in transactions.columns:
        raise KeyError("TransactionID not found in transaction file")
    if "TransactionID" not in identities.columns:
        raise KeyError("TransactionID not found in identity file")

    merged = transactions.merge(identities, on="TransactionID", how="left")
    return merged


def describe_frame(frame: pd.DataFrame) -> None:
    """Print shape, dtype summary, and the first five rows."""
    print(f"Shape: {frame.shape}")
    print("\nDtypes summary:")
    print(frame.dtypes.astype(str).value_counts())
    print("\nFirst 5 rows:")
    print(frame.head())


def save_frame(frame: pd.DataFrame, output_path: str) -> None:
    """Persist the merged data to CSV."""
    ensure_directory(os.path.dirname(output_path))
    try:
        frame.to_csv(output_path, index=False)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to save merged data: {exc}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Merge fraud detection tables.")
    parser.add_argument(
        "--transaction-path",
        default=os.path.join("data", "train_transaction.csv"),
        help="Path to train_transaction.csv",
    )
    parser.add_argument(
        "--identity-path",
        default=os.path.join("data", "train_identity.csv"),
        help="Path to train_identity.csv",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join("outputs", "data", "raw_data.csv"),
        help="Where to save the merged dataset",
    )
    return parser.parse_args()


def main() -> int:
    """Run the ingestion workflow."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Data Ingestion")
    try:
        merged = load_and_merge(args.transaction_path, args.identity_path)
        describe_frame(merged)
        save_frame(merged, args.output_path)
        print(f"Saved merged data to: {args.output_path}")
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
