
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

INPUT_PATH = REPORT_DIR / "inventory_inputs_16_day.csv"
OUTPUT_PATH = REPORT_DIR / "inventory_data_ready.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# LOGGING
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "inventory_data.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("InventoryData")


# =====================================
# MAIN FUNCTION
# =====================================

def generate_inventory_data():

    logger.info("Generating sample inventory data...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    # Reproducible sample data
    rng = np.random.default_rng(42)

    # Expected stock based on forecast demand
    avg_demand = df["average_daily_demand"].clip(lower=0)

    # Simulate stock on hand
    df["on_hand_units"] = np.maximum(
        0,
        np.round(
            avg_demand * rng.uniform(1, 15, len(df))
        )
    ).astype(int)

    # Simulate incoming purchase orders
    df["on_order_units"] = np.maximum(
        0,
        np.round(
            avg_demand * rng.uniform(0, 8, len(df))
        )
    ).astype(int)

    # Simulate pending backorders
    df["backorder_units"] = rng.poisson(
        lam=1,
        size=len(df)
    )

    # Keep initial planning assumptions
    df["lead_time_days"] = 3
    df["review_period_days"] = 7
    df["service_level"] = 0.95

    # =================================
    # VALIDATION
    # =================================

    inventory_columns = [
        "on_hand_units",
        "on_order_units",
        "backorder_units"
    ]

    if df[inventory_columns].isna().any().any():
        raise ValueError(
            "Missing inventory values detected."
        )

    if (df[inventory_columns] < 0).any().any():
        raise ValueError(
            "Negative inventory values detected."
        )

    # =================================
    # SAVE
    # =================================

    df.to_csv(OUTPUT_PATH, index=False)

    logger.info("Sample inventory data generated.")

    print("\n================================")
    print("SAMPLE INVENTORY DATA CREATED")
    print("================================")

    print("Shape:", df.shape)

    print("\nInventory preview:")
    print(
        df[
            [
                "store_nbr",
                "family",
                "on_hand_units",
                "on_order_units",
                "backorder_units"
            ]
        ].head()
    )

    print("\nMissing values:")
    print(df[inventory_columns].isna().sum())

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    generate_inventory_data()
