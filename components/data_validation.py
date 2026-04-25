"""Data validation for the IEEE CIS Fraud Detection dataset.

The script validates the merged raw dataset, reports schema details, missing
value percentages, and class balance for isFraud, then saves a validation
report to outputs/validation_report.txt.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd


BANNER = "=" * 50


EXPECTED_COLUMNS = ["TransactionID", "TransactionDT", "isFraud"]


def print_banner(title: str) -> None:
    """Print a consistent section banner."""
    print(BANNER)
    print(title)
    print(BANNER)


def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load the merged raw dataset."""
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to load raw data: {exc}") from exc


def build_report(frame: pd.DataFrame) -> str:
    """Create a textual validation report."""
    lines: List[str] = []
    lines.append("IEEE CIS Fraud Detection - Validation Report")
    lines.append(BANNER)
    lines.append(f"Shape: {frame.shape}")
    lines.append("")
    lines.append("Schema overview:")
    lines.append(frame.dtypes.astype(str).to_string())
    lines.append("")

    missing_pct = (frame.isna().mean() * 100).sort_values(ascending=False)
    flagged = missing_pct[missing_pct > 50]
    lines.append("Missing value percentages:")
    lines.append(missing_pct.to_string())
    lines.append("")
    lines.append("Columns with >50% missing values:")
    if flagged.empty:
        lines.append("None")
    else:
        lines.append(flagged.to_string())
    lines.append("")

    lines.append("Expected columns check:")
    for column in EXPECTED_COLUMNS:
        status = "present" if column in frame.columns else "missing"
        lines.append(f"- {column}: {status}")
    lines.append("")

    if "isFraud" in frame.columns:
        class_distribution = frame["isFraud"].value_counts(dropna=False).sort_index()
        class_percent = frame["isFraud"].value_counts(normalize=True, dropna=False).sort_index() * 100
        lines.append("Class distribution for isFraud:")
        for label in class_distribution.index:
            lines.append(
                f"- Class {label}: count={class_distribution[label]}, share={class_percent[label]:.2f}%"
            )
    else:
        lines.append("Class distribution for isFraud: unavailable")

    lines.append("")
    lines.append(f"Duplicate rows: {frame.duplicated().sum()}")
    return "\n".join(lines)


def save_report(report: str, output_path: str) -> None:
    """Write the report to disk."""
    ensure_directory(os.path.dirname(output_path))
    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(report)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError(f"Failed to save validation report: {exc}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate the merged fraud dataset.")
    parser.add_argument(
        "--input-path",
        default=os.path.join("outputs", "data", "raw_data.csv"),
        help="Path to the merged raw data",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join("outputs", "validation_report.txt"),
        help="Where to save the validation report",
    )
    return parser.parse_args()


def main() -> int:
    """Run the validation workflow."""
    args = parse_args()
    print_banner("IEEE CIS Fraud Detection - Data Validation")
    try:
        frame = load_data(args.input_path)
        report = build_report(frame)
        print(report)
        save_report(report, args.output_path)
        print(f"Saved validation report to: {args.output_path}")
        return 0
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
