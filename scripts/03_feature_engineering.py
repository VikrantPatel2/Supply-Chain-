
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "Data" / "Raw"
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
LOG_DIR = PROJECT_DIR / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "train_clean_automated.csv"
HOLIDAY_PATH = RAW_DIR / "holidays_events.csv"

OUTPUT_PATH = (
    PROCESSED_DIR / "sales_features_automated.csv"
)

# =====================================================
# 2. LOGGING CONFIGURATION
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "feature_engineering.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("FeatureEngineering")


# =====================================================
# 3. FEATURE ENGINEERING FUNCTION
# =====================================================

def run_feature_engineering():

    logger.info("Starting feature engineering...")

    # -------------------------------------------------
    # Check input files
    # -------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned training data not found: {INPUT_PATH}"
        )

    if not HOLIDAY_PATH.exists():
        raise FileNotFoundError(
            f"Holiday dataset not found: {HOLIDAY_PATH}"
        )

    # -------------------------------------------------
    # Load cleaned sales data
    # -------------------------------------------------

    logger.info("Loading cleaned sales dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"]
    )

    logger.info(f"Dataset loaded: {df.shape}")

    # -------------------------------------------------
    # Data type conversion
    # -------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="coerce"
    )

    df["store_nbr"] = pd.to_numeric(
        df["store_nbr"],
        errors="coerce"
    )

    df["onpromotion"] = pd.to_numeric(
        df["onpromotion"],
        errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=["date", "sales", "store_nbr", "family"]
    )

    # -------------------------------------------------
    # Sort data chronologically
    # -------------------------------------------------

    logger.info("Sorting sales data...")

    group_cols = ["store_nbr", "family"]

    df = df.sort_values(
        group_cols + ["date"]
    ).reset_index(drop=True)

    # =================================================
    # 4. DATE FEATURES
    # =================================================

    logger.info("Creating date features...")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # =================================================
    # 5. CYCLICAL DATE FEATURES
    # =================================================

    logger.info("Creating cyclical date features...")

    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # =================================================
    # 6. HOLIDAY FEATURES
    # =================================================

    logger.info("Creating holiday features...")

    holidays = pd.read_csv(HOLIDAY_PATH)

    holidays["date"] = pd.to_datetime(
        holidays["date"],
        errors="coerce"
    )

    holidays = holidays.dropna(
        subset=["date"]
    )

    # Count holiday event records for each date
    holiday_counts = (
        holidays.groupby("date")
        .size()
        .reset_index(name="holiday_event_count")
    )

    df = df.merge(
        holiday_counts,
        on="date",
        how="left"
    )

    df["holiday_event_count"] = (
        df["holiday_event_count"]
        .fillna(0)
        .astype(int)
    )

    df["has_holiday_event"] = (
        df["holiday_event_count"] > 0
    ).astype(int)

    # =================================================
    # 7. LAG FEATURES
    # =================================================

    logger.info("Creating sales lag features...")

    grouped_sales = df.groupby(
        group_cols,
        sort=False
    )["sales"]

    df["sales_lag_1"] = grouped_sales.shift(1)

    df["sales_lag_7"] = grouped_sales.shift(7)

    df["sales_lag_14"] = grouped_sales.shift(14)

    df["sales_lag_28"] = grouped_sales.shift(28)

    df["sales_lag_30"] = grouped_sales.shift(30)

    # =================================================
    # 8. ROLLING WINDOW FEATURES
    # =================================================

    logger.info("Creating rolling sales features...")

    # Shift by one day to avoid target leakage
    grouped_sales = df.groupby(
        group_cols,
        sort=False
    )["sales"]

    df["sales_rolling_mean_7"] = (
        grouped_sales.transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df["sales_rolling_mean_14"] = (
        grouped_sales.transform(
            lambda x: x.shift(1).rolling(14).mean()
        )
    )

    df["sales_rolling_mean_28"] = (
        grouped_sales.transform(
            lambda x: x.shift(1).rolling(28).mean()
        )
    )

    df["sales_rolling_mean_30"] = (
        grouped_sales.transform(
            lambda x: x.shift(1).rolling(30).mean()
        )
    )

    df["sales_rolling_std_7"] = (
        grouped_sales.transform(
            lambda x: x.shift(1).rolling(7).std()
        )
    )

    # =================================================
    # 9. MISSING VALUE HANDLING
    # =================================================

    logger.info("Handling missing lag and rolling values...")

    lag_columns = [
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "sales_lag_30",
        "sales_rolling_mean_7",
        "sales_rolling_mean_14",
        "sales_rolling_mean_28",
        "sales_rolling_mean_30",
        "sales_rolling_std_7"
    ]

    df[lag_columns] = (
        df[lag_columns].fillna(0)
    )

    # =================================================
    # 10. FINAL VALIDATION
    # =================================================

    logger.info("Validating engineered features...")

    required_features = [
        "has_holiday_event",
        "holiday_event_count",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "sales_lag_28",
        "sales_rolling_mean_14",
        "sales_rolling_mean_28"
    ]

    missing_features = [
        col for col in required_features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing engineered features: {missing_features}"
        )

    # =================================================
    # 11. SAVE FEATURE-ENGINEERED DATA
    # =================================================

    logger.info("Saving feature-engineered dataset...")

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info("Feature engineering completed!")
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Output saved: {OUTPUT_PATH}")

    print("\n======================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("======================================")

    print("Dataset shape:", df.shape)

    print("\nNew features created:")
    for feature in required_features:
        print(" -", feature)

    print("\nOutput file:", OUTPUT_PATH)


# =====================================================
# 12. SCRIPT EXECUTION
# =====================================================

if __name__ == "__main__":
    run_feature_engineering()
