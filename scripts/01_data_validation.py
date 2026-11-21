
from pathlib import Path
import pandas as pd
import logging
import sys

# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "Data" / "Raw"
LOG_DIR = PROJECT_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "data_validation.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("DataValidation")

# Required dataset files and their expected columns
REQUIRED_FILES = {
    "train.csv": [
        "id", "date", "store_nbr", "family", "sales", "onpromotion"
    ],
    "stores.csv": [
        "store_nbr", "city", "state", "type", "cluster"
    ],
    "oil.csv": [
        "date", "dcoilwtico"
    ],
    "holidays_events.csv": [
        "date", "type", "locale", "locale_name",
        "description", "transferred"
    ],
    "transactions.csv": [
        "date", "store_nbr", "transactions"
    ]
}


def validate_small_file(file_name, expected_columns):
    """Validate smaller CSV files."""

    file_path = RAW_DIR / file_name

    if not file_path.exists():
        logger.error(f"Missing file: {file_name}")
        return False

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            logger.error(f"{file_name} is empty")
            return False

        missing_columns = set(expected_columns) - set(df.columns)

        if missing_columns:
            logger.error(
                f"{file_name}: Missing columns {missing_columns}"
            )
            return False

        if "date" in df.columns:
            invalid_dates = pd.to_datetime(
                df["date"], errors="coerce"
            ).isna().sum()

            logger.info(
                f"{file_name}: Invalid dates = {invalid_dates}"
            )

        logger.info(
            f"{file_name}: Valid structure | Rows = {len(df)}"
        )

        return True

    except Exception as error:
        logger.exception(f"Error reading {file_name}: {error}")
        return False


def validate_train_file():
    """Validate the large training dataset in chunks."""

    file_path = RAW_DIR / "train.csv"

    if not file_path.exists():
        logger.error("Missing file: train.csv")
        return False

    try:
        total_rows = 0
        total_invalid_dates = 0
        total_negative_sales = 0
        required_columns = REQUIRED_FILES["train.csv"]

        for chunk in pd.read_csv(file_path, chunksize=250000):

            missing_columns = (
                set(required_columns) - set(chunk.columns)
            )

            if missing_columns:
                logger.error(
                    f"train.csv: Missing columns {missing_columns}"
                )
                return False

            total_rows += len(chunk)

            dates = pd.to_datetime(
                chunk["date"], errors="coerce"
            )

            total_invalid_dates += dates.isna().sum()

            sales = pd.to_numeric(
                chunk["sales"], errors="coerce"
            )

            total_negative_sales += (sales < 0).sum()

        if total_rows == 0:
            logger.error("train.csv contains no records")
            return False

        logger.info(f"Training rows: {total_rows}")
        logger.info(f"Invalid dates: {total_invalid_dates}")
        logger.info(f"Negative sales records: {total_negative_sales}")

        if total_invalid_dates > 0:
            logger.error("Training data contains invalid dates")
            return False

        if total_negative_sales > 0:
            logger.error("Training data contains negative sales")
            return False

        logger.info("train.csv validation successful")
        return True

    except Exception as error:
        logger.exception(f"Training data validation failed: {error}")
        return False


def run_validation():

    logger.info("Starting data validation...")

    results = {}

    for file_name, columns in REQUIRED_FILES.items():

        if file_name == "train.csv":
            continue

        results[file_name] = validate_small_file(
            file_name, columns
        )

    results["train.csv"] = validate_train_file()

    failed_files = [
        file_name
        for file_name, status in results.items()
        if not status
    ]

    if failed_files:
        logger.error(f"Validation failed: {failed_files}")
        logger.error("Pipeline execution should stop.")
        return False

    logger.info("All required datasets passed validation.")
    return True


if __name__ == "__main__":

    success = run_validation()

    if not success:
        sys.exit(1)

    print("\nDATA VALIDATION COMPLETED SUCCESSFULLY")
