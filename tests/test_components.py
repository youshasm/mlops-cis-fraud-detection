"""Smoke tests for the fraud detection project components."""

from __future__ import annotations

import pandas as pd

from components.data_ingestion import load_and_merge
from components.data_validation import build_report
from components.feature_engineering import create_time_features, target_encode


def test_load_and_merge(tmp_path) -> None:
    """The ingestion helper should merge the two tables on TransactionID."""
    transaction_path = tmp_path / "transactions.csv"
    identity_path = tmp_path / "identity.csv"
    pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [100, 200],
            "isFraud": [0, 1],
        }
    ).to_csv(transaction_path, index=False)
    pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "DeviceType": ["desktop", "mobile"],
        }
    ).to_csv(identity_path, index=False)

    merged = load_and_merge(str(transaction_path), str(identity_path))

    assert merged.shape == (2, 4)
    assert "DeviceType" in merged.columns


def test_build_report_flags_missing_columns() -> None:
    """The validation report should include missingness and class balance."""
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "mostly_missing": [None, None, 1],
        }
    )

    report = build_report(frame)

    assert "Columns with >50% missing values:" in report
    assert "mostly_missing" in report
    assert "Class distribution for isFraud:" in report


def test_feature_engineering_adds_expected_columns() -> None:
    """Feature engineering should add temporal and encoded features."""
    frame = pd.DataFrame(
        {
            "TransactionDT": [3600, 7200],
            "card1": [100, 200],
            "card2": [10, 20],
            "addr1": [1, 1],
            "addr2": [2, 2],
            "P_emaildomain": ["a.com", "b.com"],
            "R_emaildomain": ["a.com", "a.com"],
            "isFraud": [0, 1],
        }
    )

    encoded = target_encode(frame, ["card1", "P_emaildomain"], target_column="isFraud")
    featured = create_time_features(encoded)

    assert "card1_target_encoded" in featured.columns
    assert "P_emaildomain_target_encoded" in featured.columns
    assert "transaction_hour" in featured.columns
    assert "transaction_day" in featured.columns
