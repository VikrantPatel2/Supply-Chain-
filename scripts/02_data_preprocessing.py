
from pathlib import Path
import pandas as pd
import logging

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "Data" / "Raw"
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
LOG_DIR = PROJECT_DIR / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "preprocessing.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("Preprocessing")


def clean_data(df, file_name):
    """Apply common and dataset-specific cleaning."""

    df.columns = df.columns.str.strip()

    # Clean date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )
        df = df.dropna(subset=["date"])

    # Remove unnecessary whitespace from text columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    # Training dataset cleaning
    if file_name == "train.csv":

        df["sales"] = pd.to_numeric(
            df["sales"], errors="coerce"
        )

        df["onpromotion"] = pd.to_numeric(
            df["onpromotion"], errors="coerce"
        )

        df = df.dropna(
            subset=["sales", "store_nbr", "family"]
        )

        # Missing promotion values mean no recorded promotion
        df["onpromotion"] = df["onpromotion"].fillna(0)

    # Transactions cleaning
    elif file_name == "transactions.csv":

        df["transactions"] = pd.to_numeric(
            df["transactions"], errors="coerce"
        )

        df = df.dropna(
            subset=["store_nbr", "transactions"]
        )

    # Store information
    elif file_name == "stores.csv":

        df["store_nbr"] = pd.to_numeric(
            df["store_nbr"], errors="coerce"
        )

        df = df.dropna(subset=["store_nbr"])

    # Oil price cleaning
    elif file_name == "oil.csv":

        df["dcoilwtico"] = pd.to_numeric(
            df["dcoilwtico"], errors="coerce"
        )

        # Keep missing oil prices for later time-series imputation

    return df


def preprocess_large_file(file_name, chunksize=250000):

    input_path = RAW_DIR / file_name
    output_path = PROCESSED_DIR / "train_clean_automated.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    total_rows = 0
    cleaned_rows = 0

    # Remove old automated output before creating a new one
    if output_path.exists():
        output_path.unlink()

    logger.info("Starting large dataset preprocessing...")

    for chunk in pd.read_csv(input_path, chunksize=chunksize):

        total_rows += len(chunk)

        chunk = clean_data(chunk, file_name)

        cleaned_rows += len(chunk)

        chunk.to_csv(
            output_path,
            mode="a",
            index=False,
            header=not output_path.exists()
        )

        logger.info(
            f"Processed {total_rows:,} input rows"
        )

    logger.info(f"Original rows: {total_rows:,}")
    logger.info(f"Cleaned rows: {cleaned_rows:,}")
    logger.info(f"Saved file: {output_path}")

    return output_path


def preprocess_small_file(file_name):

    input_path = RAW_DIR / file_name
    output_path = PROCESSED_DIR / (
        file_name.replace(".csv", "_clean_automated.csv")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    df = pd.read_csv(input_path)

    original_rows = len(df)

    df = clean_data(df, file_name)

    df.to_csv(output_path, index=False)

    logger.info(
        f"{file_name}: {original_rows:,} -> {len(df):,} rows"
    )

    logger.info(f"Saved: {output_path}")

    return output_path


def run_preprocessing():

    logger.info("Starting preprocessing pipeline...")

    # Process the large training dataset
    preprocess_large_file("train.csv")

    # Process smaller datasets
    small_files = [
        "stores.csv",
        "oil.csv",
        "transactions.csv",
        "holidays_events.csv"
    ]

    for file_name in small_files:
        preprocess_small_file(file_name)

    logger.info("All preprocessing tasks completed!")

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_preprocessing()
