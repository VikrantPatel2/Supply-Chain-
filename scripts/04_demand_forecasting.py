
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from catboost import CatBoostRegressor

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
LOG_DIR = PROJECT_DIR / "logs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "forecasting.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("DemandForecasting")

# Input and output files
DATA_PATH = PROCESSED_DIR / "sales_features_automated.csv"
MODEL_PATH = MODEL_DIR / "optimized_catboost_model.cbm"
OUTPUT_PATH = REPORT_DIR / "automated_demand_forecast.csv"


def run_forecasting():

    logger.info("Starting demand forecasting...")

    # Check files
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"CatBoost model not found: {MODEL_PATH}")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    logger.info(f"Dataset loaded: {df.shape}")

    # Load trained model
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    logger.info("CatBoost model loaded successfully")

    # Obtain exact training feature names
    feature_names = model.feature_names_

    if not feature_names:
        raise ValueError(
            "Model does not contain feature names. "
            "Provide the exact training feature list."
        )

    # Check missing features
    missing_features = [
        col for col in feature_names
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # Select features in exact training order
    X = df[feature_names].copy()

    # Handle missing values only for numeric columns
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns

    X[numeric_columns] = X[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # CatBoost can handle numeric NaN values natively.
    # Categorical missing values need consistent handling.
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:
        X[col] = X[col].fillna("Unknown").astype(str)

    # Generate predictions
    predictions = model.predict(X)

    # Create forecast output
    forecast_df = df[
        ["date", "store_nbr", "family"]
    ].copy()

    forecast_df["predicted_sales"] = np.maximum(
        predictions, 0
    )

    # Save forecast
    forecast_df.to_csv(OUTPUT_PATH, index=False)

    logger.info("Forecast generation completed")
    logger.info(f"Forecast records: {len(forecast_df):,}")
    logger.info(f"Saved to: {OUTPUT_PATH}")

    print("\nDEMAND FORECASTING COMPLETED")
    print(f"Forecast records: {len(forecast_df):,}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nForecast preview:")
    print(forecast_df.head())


if __name__ == "__main__":
    run_forecasting()
