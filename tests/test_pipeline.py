
import pandas as pd
import pytest
import src.pipeline as pipeline_module

from src.pipeline import validate_transactions, deduplicate_transactions


def test_validate_transactions():

    df = pd.DataFrame({
        "account_number": ["AC100", "AC101", "AC102"],
        "raw_txn_date": ["2026-08-01", "2026-08-01", "2026-08-01"],
        "debit": [1000, 1200, None],
        "credit": [None, 1200, 500],
        "ref_no": ["REF001", "REF002", "REF003"]
    })

    valid_df, quarantine_df = validate_transactions(df)

    # Two valid records and one invalid record
    assert len(valid_df) == 2
    assert len(quarantine_df) == 1

    # Verify the invalid record has a reason code
    assert "reason_code" in quarantine_df.columns


def test_deduplicate_transactions():

    df = pd.DataFrame({
        "account_number": ["AC100", "AC100", "AC101"],
        "txn_date": ["2026-08-01", "2026-08-01", "2026-08-02"],
        "txn_time": ["10:00:00", "10:00:00", "11:00:00"],
        "description": ["ATM", "ATM", "UPI"],
        "ref_no": ["REF001", "REF001", "REF002"],
        "debit": [1000, 1000, 500],
        "credit": [None, None, None],
        "balance": [9000, 9000, 8500],
        "account_type": ["SAVINGS", "SAVINGS", "SAVINGS"]
    })

    clean_df, duplicates_df = deduplicate_transactions(df)

    assert len(clean_df) == 2
    assert len(duplicates_df) == 1

    assert clean_df["ref_no"].nunique() == 2


def test_upsert_inserts_and_updates(tmp_path, monkeypatch):

    # Use a temporary output directory for testing
    monkeypatch.setattr(
        pipeline_module,
        "OUTPUT_DIR",
        tmp_path
    )

    # First incoming transaction
    incoming_df = pd.DataFrame({
        "account_number": ["AC100"],
        "ref_no": ["REF001"],
        "balance": [9000],
        "debit": [1000],
        "credit": [None],
        "description": ["ATM Withdrawal"],
        "account_type": ["SAVINGS"]
    })

    # Test 1: Insert a new transaction
    first_result = pipeline_module.upsert_transactions(
        incoming_df
    )

    assert len(first_result) == 1
    assert first_result.iloc[0]["balance"] == 9000

    # Verify new transaction audit output
    new_records = pd.read_csv(
        tmp_path / "new_transactions.csv"
    )

    assert len(new_records) == 1

    # Test 2: Update an existing transaction
    updated_df = incoming_df.copy()

    updated_df.loc[0, "balance"] = 8500

    second_result = pipeline_module.upsert_transactions(
        updated_df
    )

    assert len(second_result) == 1
    assert second_result.iloc[0]["balance"] == 8500

    # Verify update audit output
    updated_records = pd.read_csv(
        tmp_path / "updated_transactions.csv"
    )

    assert len(updated_records) == 1
    assert updated_records.iloc[0]["balance"] == 8500


def test_upsert_idempotency(tmp_path, monkeypatch):

    monkeypatch.setattr(
        pipeline_module,
        "OUTPUT_DIR",
        tmp_path
    )

    incoming_df = pd.DataFrame({
        "account_number": ["AC200"],
        "ref_no": ["REF200"],
        "balance": [15000],
        "debit": [500],
        "credit": [None],
        "description": ["UPI Payment"],
        "account_type": ["SAVINGS"]
    })

    # First execution
    first_result = pipeline_module.upsert_transactions(
        incoming_df
    )

    # Second execution with identical data
    second_result = pipeline_module.upsert_transactions(
        incoming_df
    )

    # Record count must remain unchanged
    assert len(first_result) == 1
    assert len(second_result) == 1

    # No new transactions on second execution
    new_records = pd.read_csv(
        tmp_path / "new_transactions.csv"
    )

    assert len(new_records) == 0

    # No unnecessary updates
    updated_records = pd.read_csv(
        tmp_path / "updated_transactions.csv"
    )

    assert len(updated_records) == 0

