
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

FORECAST_PATH = REPORT_DIR / "16_day_demand_forecast.csv"

OUTPUT_PATH = REPORT_DIR / "inventory_inputs_16_day.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# LOGGING
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "inventory_inputs.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("InventoryInputs")


# =====================================
# MAIN FUNCTION
# =====================================

def prepare_inventory_inputs():

    logger.info("Preparing inventory planning inputs...")

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH}"
        )

    forecast = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"]
    )

    required_columns = [
        "date",
        "store_nbr",
        "family",
        "predicted_sales"
    ]

    missing = [
        col for col in required_columns
        if col not in forecast.columns
    ]

    if missing:
        raise ValueError(
            f"Missing forecast columns: {missing}"
        )

    forecast["predicted_sales"] = pd.to_numeric(
        forecast["predicted_sales"],
        errors="coerce"
    )

    forecast = forecast.dropna(
        subset=["date", "store_nbr", "family", "predicted_sales"]
    )

    # Demand cannot be negative for inventory planning
    forecast["predicted_sales"] = (
        forecast["predicted_sales"].clip(lower=0)
    )

    # =================================
    # AGGREGATE FORECAST BY STORE/FAMILY
    # =================================

    inventory = (
        forecast.groupby(
            ["store_nbr", "family"]
        )
        .agg(
            forecast_total_16_days=(
                "predicted_sales", "sum"
            ),
            average_daily_demand=(
                "predicted_sales", "mean"
            ),
            maximum_daily_demand=(
                "predicted_sales", "max"
            ),
            demand_std_16_days=(
                "predicted_sales", "std"
            )
        )
        .reset_index()
    )

    # =================================
    # BUSINESS INPUTS
    # =================================

    # These values must be supplied by the business.
    inventory["on_hand_units"] = np.nan
    inventory["on_order_units"] = np.nan
    inventory["backorder_units"] = 0

    # Initial planning assumptions.
    # Replace with actual business values.
    inventory["lead_time_days"] = 3
    inventory["review_period_days"] = 7
    inventory["service_level"] = 0.95

    # =================================
    # VALIDATION
    # =================================

    if inventory.duplicated(
        subset=["store_nbr", "family"]
    ).any():
        raise ValueError(
            "Duplicate store-family combinations found."
        )

    inventory["demand_std_16_days"] = (
        inventory["demand_std_16_days"].fillna(0)
    )

    # =================================
    # SAVE
    # =================================

    inventory.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info("Inventory input preparation completed.")

    print("\n================================")
    print("INVENTORY INPUTS CREATED")
    print("================================")

    print("Shape:", inventory.shape)
    print("\nPreview:")
    print(inventory.head())

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    prepare_inventory_inputs()
