
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from catboost import CatBoostRegressor
import logging

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "Data" / "Raw"
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

OUTPUT_PATH = REPORT_DIR / "16_day_demand_forecast.csv"

TRAIN_PATH = PROCESSED_DIR / "train_clean_automated.csv"
TEST_PATH = RAW_DIR / "test.csv"
HOLIDAY_PATH = RAW_DIR / "holidays_events.csv"

MODEL_PATH = MODEL_DIR / "optimized_catboost_model.cbm"

for folder in [REPORT_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "future_forecasting.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("FutureForecasting")


# ==========================================
# 3. DATE FEATURE FUNCTION
# ==========================================

def create_date_features(date):

    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.dayofweek,
        "day_of_year": date.dayofyear,
        "week_of_year": int(date.isocalendar().week),
        "quarter": date.quarter,
        "is_weekend": int(date.dayofweek >= 5),

        "month_sin": np.sin(
            2 * np.pi * (date.month - 1) / 12
        ),

        "month_cos": np.cos(
            2 * np.pi * (date.month - 1) / 12
        ),

        "day_sin": np.sin(
            2 * np.pi * date.dayofweek / 7
        ),

        "day_cos": np.cos(
            2 * np.pi * date.dayofweek / 7
        )
    }


# ==========================================
# 4. MAIN FORECASTING FUNCTION
# ==========================================

def run_future_forecasting():

    logger.info("Starting future demand forecasting...")

    # --------------------------------------
    # Check files
    # --------------------------------------

    for path in [
        TRAIN_PATH,
        TEST_PATH,
        HOLIDAY_PATH,
        MODEL_PATH
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------
    # Load historical sales
    # --------------------------------------

    logger.info("Loading historical sales data...")

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["date"]
    )

    train["date"] = pd.to_datetime(train["date"])
    train["sales"] = pd.to_numeric(
        train["sales"],
        errors="coerce"
    )

    train = train.dropna(
        subset=["date", "sales", "store_nbr", "family"]
    )

    train["store_nbr"] = train["store_nbr"].astype(int)
    train["family"] = train["family"].astype(str)

    train = train.sort_values("date")

    logger.info(f"Historical data shape: {train.shape}")

    # --------------------------------------
    # Load future test data
    # --------------------------------------

    logger.info("Loading future test data...")

    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["date"]
    )

    test["date"] = pd.to_datetime(test["date"])
    test["store_nbr"] = test["store_nbr"].astype(int)
    test["family"] = test["family"].astype(str)

    test["onpromotion"] = pd.to_numeric(
        test["onpromotion"],
        errors="coerce"
    ).fillna(0)

    test = test.sort_values(
        ["date", "store_nbr", "family"]
    ).reset_index(drop=True)

    logger.info(f"Future test data shape: {test.shape}")

    # --------------------------------------
    # Load holiday data
    # --------------------------------------

    logger.info("Loading holiday information...")

    holidays = pd.read_csv(
        HOLIDAY_PATH,
        parse_dates=["date"]
    )

    holidays["date"] = pd.to_datetime(
        holidays["date"],
        errors="coerce"
    )

    holidays = holidays.dropna(subset=["date"])

    holiday_counts = (
        holidays.groupby("date")
        .size()
        .to_dict()
    )

    # --------------------------------------
    # Load CatBoost model
    # --------------------------------------

    logger.info("Loading CatBoost model...")

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    feature_names = model.feature_names_

    expected_features = [
        "store_nbr",
        "family",
        "onpromotion",
        "has_holiday_event",
        "holiday_event_count",
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "quarter",
        "is_weekend",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "sales_rolling_mean_7",
        "sales_rolling_mean_14",
        "sales_rolling_mean_28"
    ]

    if feature_names != expected_features:
        raise ValueError(
            "Model feature order does not match "
            "the expected feature configuration."
        )

    logger.info("CatBoost model loaded successfully.")

    # --------------------------------------
    # Build historical sales lookup
    # --------------------------------------

    logger.info("Building historical sales history...")

    history = {}

    for (store, family), group in train.groupby(
        ["store_nbr", "family"]
    ):

        key = (int(store), str(family))

        history[key] = dict(
            zip(group["date"], group["sales"])
        )

    # --------------------------------------
    # Recursive future forecasting
    # --------------------------------------

    logger.info("Generating future predictions...")

    predictions = []

    future_dates = sorted(test["date"].unique())

    for current_date in future_dates:

        current_date = pd.Timestamp(current_date)

        daily_test = test[
            test["date"] == current_date
        ]

        feature_rows = []
        daily_keys = []

        for row in daily_test.itertuples(index=False):

            store = int(row.store_nbr)
            family = str(row.family)

            key = (store, family)

            if key not in history:
                raise ValueError(
                    f"No historical sales found for {key}"
                )

            sales_history = history[key]

            # Previous calendar-day sales
            lag_1 = sales_history.get(
                current_date - timedelta(days=1), 0
            )

            lag_7 = sales_history.get(
                current_date - timedelta(days=7), 0
            )

            lag_14 = sales_history.get(
                current_date - timedelta(days=14), 0
            )

            lag_28 = sales_history.get(
                current_date - timedelta(days=28), 0
            )

            # Rolling averages from previous days only
            previous_7 = [
                sales_history.get(
                    current_date - timedelta(days=i), 0
                )
                for i in range(1, 8)
            ]

            previous_14 = [
                sales_history.get(
                    current_date - timedelta(days=i), 0
                )
                for i in range(1, 15)
            ]

            previous_28 = [
                sales_history.get(
                    current_date - timedelta(days=i), 0
                )
                for i in range(1, 29)
            ]

            rolling_mean_7 = np.mean(previous_7)
            rolling_mean_14 = np.mean(previous_14)
            rolling_mean_28 = np.mean(previous_28)

            date_features = create_date_features(
                current_date
            )

            holiday_count = int(
                holiday_counts.get(current_date, 0)
            )

            features = {
                "store_nbr": store,
                "family": family,
                "onpromotion": float(row.onpromotion),

                "has_holiday_event": int(
                    holiday_count > 0
                ),

                "holiday_event_count": holiday_count,

                **date_features,

                "sales_lag_1": lag_1,
                "sales_lag_7": lag_7,
                "sales_lag_14": lag_14,
                "sales_lag_28": lag_28,

                "sales_rolling_mean_7": rolling_mean_7,
                "sales_rolling_mean_14": rolling_mean_14,
                "sales_rolling_mean_28": rolling_mean_28
            }

            feature_rows.append(features)
            daily_keys.append(key)

        # Ensure exact model feature order
        X_future = pd.DataFrame(
            feature_rows
        )[feature_names]

        # Predict all store-family combinations for this date
        daily_predictions = model.predict(X_future)

        # Store predictions and update history
        for key, predicted_sales in zip(
            daily_keys,
            daily_predictions
        ):

            predicted_sales = float(predicted_sales)

            predictions.append({
                "date": current_date,
                "store_nbr": key[0],
                "family": key[1],
                "predicted_sales": predicted_sales
            })

            # Use today's prediction for future lag features
            history[key][current_date] = predicted_sales

        logger.info(
            f"Forecast completed for {current_date.date()}"
        )

    # --------------------------------------
    # Save future forecast
    # --------------------------------------

    forecast_df = pd.DataFrame(predictions)

    forecast_df = forecast_df.sort_values(
        ["date", "store_nbr", "family"]
    ).reset_index(drop=True)

    if len(forecast_df) != len(test):
        raise ValueError(
            f"Prediction count mismatch: "
            f"{len(forecast_df)} predictions "
            f"for {len(test)} test rows."
        )

    forecast_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info("Future forecasting completed.")
    logger.info(f"Output saved: {OUTPUT_PATH}")

    print("\n========================================")
    print("16-DAY FUTURE FORECAST COMPLETED")
    print("========================================")

    print("Forecast shape:", forecast_df.shape)

    print(
        "Date range:",
        forecast_df["date"].min(),
        "to",
        forecast_df["date"].max()
    )

    print("\nForecast preview:")
    print(forecast_df.head())

    print("\nForecast summary:")
    print(forecast_df["predicted_sales"].describe())

    print("\nOutput file:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    run_future_forecasting()
