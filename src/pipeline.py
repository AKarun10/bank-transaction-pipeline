
import pandas as pd
from pathlib import Path
import logging
import json
import hashlib


# ==========================================
# LOGGING CONFIGURATION
# ==========================================

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

logger = logging.getLogger(__name__)


def setup_logging():

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                LOG_DIR / "pipeline.log",
                encoding="utf-8"
            ),
            logging.StreamHandler()
        ],
        force=True
    )

# ==========================================
# INCREMENTAL PROCESSING HELPERS
# ==========================================

def get_file_hash(file_path):

    file_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def load_processed_files():

    if PROCESSED_FILES_MANIFEST.exists():

        with open(PROCESSED_FILES_MANIFEST, "r") as file:
            return json.load(file)

    return {}


def save_processed_files(processed_files):

    with open(PROCESSED_FILES_MANIFEST, "w") as file:
        json.dump(processed_files, file, indent=4)


# --------------------------------------------------
# 1. DEFINE PROJECT DIRECTORIES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
PROCESSED_FILES_MANIFEST = BASE_DIR / "processed_files.json"


# --------------------------------------------------
# 2. DEFINE INPUT FILES
# --------------------------------------------------

INPUT_FILES = [
    "batch_2026-08-01.csv",
    "batch_2026-08-02.csv",
    "batch_2026-08-04_late_correction.csv"
]


# --------------------------------------------------
# 3. INGEST ALL FILES
# --------------------------------------------------

def ingest_files(file_names):

    dataframes = []

    for file_name in file_names:

        file_path = INPUT_DIR / file_name

        print(f"Reading file: {file_name}")

        df = pd.read_csv(file_path)

        # Track the original source file
        df["source_file"] = file_name

        dataframes.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_df


# --------------------------------------------------
# 4. NORMALIZE TRANSACTION DATA
# --------------------------------------------------

def normalize_transactions(df):

    df = df.copy()

    # Preserve original values for audit purposes
    df["raw_txn_date"] = df["txn_date"]
    df["raw_txn_time"] = df["txn_time"]

    # Standardize transaction date
    df["txn_date"] = pd.to_datetime(
        df["txn_date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Standardize transaction time
    df["txn_time"] = pd.to_datetime(
        df["txn_time"],
        format="mixed",
        errors="coerce"
    ).dt.strftime("%H:%M:%S")

    # Convert financial columns to numeric
    amount_columns = [
        "debit",
        "credit",
        "balance"
    ]

    for column in amount_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# --------------------------------------------------
# 5. VALIDATE TRANSACTION DATA
# --------------------------------------------------

def validate_transactions(df):

    df = df.copy()

    # Store validation reasons for every record
    validation_reasons = [[] for _ in range(len(df))]

    # RULE 1: Check invalid transaction dates
    invalid_dates = pd.to_datetime(
        df["raw_txn_date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    ).isna()

    for position, is_invalid in enumerate(invalid_dates):
        if is_invalid:
            validation_reasons[position].append(
                "INVALID_TRANSACTION_DATE"
            )

    # RULE 2: Both debit and credit are populated
    both_populated = (
        df["debit"].notna() &
        df["credit"].notna()
    )

    for position, is_invalid in enumerate(both_populated):
        if is_invalid:
            validation_reasons[position].append(
                "BOTH_DEBIT_CREDIT_POPULATED"
            )

    # RULE 3: Both debit and credit are missing
    both_missing = (
        df["debit"].isna() &
        df["credit"].isna()
    )

    for position, is_invalid in enumerate(both_missing):
        if is_invalid:
            validation_reasons[position].append(
                "MISSING_DEBIT_AND_CREDIT"
            )

    
    # RULE 4: Account number is missing
    missing_account = (
        df["account_number"].isna() |
        df["account_number"].astype(str).str.strip().eq("")
    )

    for position, is_invalid in enumerate(missing_account):
        if is_invalid:
            validation_reasons[position].append(
                "MISSING_ACCOUNT_NUMBER"
            )

    # RULE 5: Reference number is missing
    missing_reference = (
        df["ref_no"].isna() |
        df["ref_no"].astype(str).str.strip().eq("")
    )

    for position, is_invalid in enumerate(missing_reference):
        if is_invalid:
            validation_reasons[position].append(
                "MISSING_REFERENCE_NUMBER"
            )


    # Identify invalid records
    invalid_mask = pd.Series(
        [len(reasons) > 0 for reasons in validation_reasons],
        index=df.index
    )

    # Separate valid and invalid records
    valid_df = df.loc[~invalid_mask].copy()

    quarantine_df = df.loc[invalid_mask].copy()

    # Attach reason codes to quarantined records
    quarantine_df["reason_code"] = [
        "|".join(validation_reasons[position])
        for position in range(len(validation_reasons))
        if invalid_mask.iloc[position]
    ]

    return valid_df, quarantine_df


# --------------------------------------------------
# 6. DEDUPLICATE TRANSACTIONS
# --------------------------------------------------

def deduplicate_transactions(df):

    df = df.copy()

    # Define composite transaction identity
    dedup_columns = [
        "account_number",
        "txn_date",
        "txn_time",
        "description",
        "ref_no",
        "debit",
        "credit",
        "balance",
        "account_type"
    ]

    # Identify duplicate records
    duplicate_mask = df.duplicated(
        subset=dedup_columns,
        keep="first"
    )

    # Capture duplicate records for auditing
    duplicates_df = df.loc[duplicate_mask].copy()

    # Retain only the first occurrence
    clean_df = df.loc[~duplicate_mask].copy()

    # Calculate duplicate count
    duplicates_removed = len(duplicates_df)

    print("\n========== DEDUPLICATION SUMMARY ==========")

    print("Records before deduplication:", len(df))

    print("Duplicates removed:", duplicates_removed)

    print("Records after deduplication:", len(clean_df))

    return clean_df, duplicates_df


# ==========================================
# 7. UPSERT TRANSACTIONS
# ==========================================

def upsert_transactions(incoming_df):

    canonical_path = OUTPUT_DIR / "canonical_transactions.csv"

    key_columns = [
        "account_number",
        "ref_no"
    ]

    # Validate business keys
    if incoming_df[key_columns].isna().any().any():
        raise ValueError(
            "Account number or reference number is missing. "
            "Cannot perform upsert."
        )

    # Remove repeated business keys within the incoming batch.
    # Keep the latest occurrence.
    incoming_df = incoming_df.drop_duplicates(
        subset=key_columns,
        keep="last"
    ).copy()

    # Read existing canonical data
    if canonical_path.exists():
        existing_df = pd.read_csv(canonical_path)
    else:
        existing_df = pd.DataFrame(columns=incoming_df.columns)

    # First execution: initialize canonical dataset
    if existing_df.empty:

        canonical_df = incoming_df.copy()
        new_records = incoming_df.copy()
        updated_records = incoming_df.iloc[0:0].copy()

    else:

        # Avoid duplicate business keys in existing canonical data
        existing_df = existing_df.drop_duplicates(
            subset=key_columns,
            keep="last"
        )

        existing_indexed = existing_df.set_index(key_columns)
        incoming_indexed = incoming_df.set_index(key_columns)

        # Identify new transactions
        new_keys = incoming_indexed.index[
            ~incoming_indexed.index.isin(existing_indexed.index)
        ]

        new_records = incoming_indexed.loc[
            new_keys
        ].reset_index()

        # Identify existing transactions that have changed
        common_keys = existing_indexed.index.intersection(
            incoming_indexed.index
        )

        comparison_columns = [
            col for col in incoming_df.columns
            if col in existing_df.columns
            and col not in key_columns
            and col not in [
                "source_file",
                "raw_txn_date",
                "raw_txn_time"
            ]
        ]

        differences = existing_indexed.loc[
            common_keys, comparison_columns
        ].compare(
            incoming_indexed.loc[
                common_keys, comparison_columns
            ]
        )

        updated_keys = differences.index.unique()

        updated_records = incoming_indexed.loc[
            updated_keys
        ].reset_index()

        # Preserve existing records that are not being replaced
        unchanged_existing = existing_indexed.loc[
            ~existing_indexed.index.isin(incoming_indexed.index)
        ]

        # Apply inserts and updates
        canonical_df = pd.concat(
            [
                unchanged_existing.reset_index(),
                incoming_df
            ],
            ignore_index=True
        )

    # Save canonical dataset
    canonical_df.to_csv(
        canonical_path,
        index=False
    )

    # Save audit outputs
    new_records.to_csv(
        OUTPUT_DIR / "new_transactions.csv",
        index=False
    )

    updated_records.to_csv(
        OUTPUT_DIR / "updated_transactions.csv",
        index=False
    )

    print("\n========== UPSERT SUMMARY ==========")

    print("New transactions inserted:", len(new_records))

    print("Existing transactions updated:", len(updated_records))

    print("Final canonical records:", len(canonical_df))

    return canonical_df



# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    setup_logging()

    logger.info("======================================")
    logger.info("BANK TRANSACTION PIPELINE STARTED")
    logger.info("======================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    # 1. Ingestion
    
    # Load previously processed files
    processed_files = load_processed_files()

    # Calculate hashes for available input files
    current_file_hashes = {
        file_name: get_file_hash(INPUT_DIR / file_name)
        for file_name in INPUT_FILES
    }

    # Identify new or modified files
    new_files = [
        file_name
        for file_name, file_hash in current_file_hashes.items()
        if processed_files.get(file_name) != file_hash
    ]

    # Skip execution if nothing new is available
    if not new_files:
        logger.info("No new or modified files found. Skipping ingestion.")
        return

    logger.info("New or modified files found: %s", new_files)

    # Ingest only new or modified files
    transactions = ingest_files(new_files)


    logger.info(
        "Data ingestion completed. Records received: %s",
        len(transactions)
    )

    # 2. Normalization
    logger.info("Starting transaction normalization")

    transactions = normalize_transactions(transactions)

    logger.info("Transaction normalization completed")

    # 3. Data validation
    logger.info("Starting data quality validation")

    valid_transactions, quarantine_transactions = (
        validate_transactions(transactions)
    )

    logger.info(
        "Validation completed. Valid: %s | Quarantined: %s",
        len(valid_transactions),
        len(quarantine_transactions)
    )

    # 4. Deduplication
    logger.info("Starting duplicate removal")

    transactions_to_upsert, duplicates_removed_df = (
        deduplicate_transactions(valid_transactions)
    )

    logger.info(
        "Deduplication completed. Duplicates removed: %s",
        len(duplicates_removed_df)
    )

    # 5. Upsert
    logger.info("Starting transaction upsert")

    canonical_transactions = upsert_transactions(
        transactions_to_upsert
    )

    logger.info(
        "Upsert completed. Canonical records: %s",
        len(canonical_transactions)
    )


    # Save removed duplicates for audit
    duplicates_removed_df.to_csv(
        OUTPUT_DIR / "duplicates_removed.csv",
        index=False
    )

    # Save validation outputs
    valid_transactions.to_csv(
        OUTPUT_DIR / "valid_staging.csv",
        index=False
    )

    quarantine_transactions.to_csv(
        OUTPUT_DIR / "quarantine_transactions.csv",
        index=False
    )

    print("\n========== DATA QUALITY SUMMARY ==========")

    print("Total records received:", len(transactions))

    print("Valid records:", len(valid_transactions))

    print("Quarantined records:", len(quarantine_transactions))

    print("\nQuarantine Reason Breakdown:")

    print(
        quarantine_transactions["reason_code"].value_counts()
    )

    print("\nQuarantined Records:")

    print(
        quarantine_transactions[
            [
                "account_number",
                "ref_no",
                "debit",
                "credit",
                "reason_code",
                "source_file"
            ]
        ].to_string(index=False)
    )
    # Update processed files manifest after successful execution
    for file_name in new_files:
        processed_files[file_name] = current_file_hashes[file_name]

    save_processed_files(processed_files)

    logger.info("Processed files manifest updated successfully")

    logger.info("BANK TRANSACTION PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    try:
        main()

    except Exception:
        logger.exception(
            "BANK TRANSACTION PIPELINE FAILED"
        )
        raise

