
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from statistics import NormalDist

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

INPUT_PATH = REPORT_DIR / "inventory_data_ready.csv"

OUTPUT_PATH = REPORT_DIR / "inventory_optimization_results.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "inventory_optimization.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("InventoryOptimization")


# ==========================================
# 3. MAIN FUNCTION
# ==========================================

def run_inventory_optimization():

    logger.info("Starting inventory optimization...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Inventory input not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    # ======================================
    # 4. VALIDATE REQUIRED COLUMNS
    # ======================================

    required_columns = [
        "store_nbr",
        "family",
        "average_daily_demand",
        "demand_std_16_days",
        "on_hand_units",
        "on_order_units",
        "backorder_units",
        "lead_time_days",
        "review_period_days",
        "service_level"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    numeric_columns = [
        "average_daily_demand",
        "demand_std_16_days",
        "on_hand_units",
        "on_order_units",
        "backorder_units",
        "lead_time_days",
        "review_period_days",
        "service_level"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or invalid numeric values detected."
        )

    if (df[numeric_columns] < 0).any().any():
        raise ValueError(
            "Negative values detected in inventory inputs."
        )

    if not df["service_level"].between(0, 1, inclusive="neither").all():
        raise ValueError(
            "Service level must be greater than 0 and less than 1."
        )

    if (df["lead_time_days"] <= 0).any():
        raise ValueError(
            "Lead time must be greater than zero."
        )

    # ======================================
    # 5. SERVICE LEVEL Z-SCORES
    # ======================================

    df["z_score"] = df["service_level"].apply(
        NormalDist().inv_cdf
    )

    # ======================================
    # 6. DEMAND PARAMETERS
    # ======================================

    df["average_daily_demand"] = (
        df["average_daily_demand"].clip(lower=0)
    )

    # The 16-day forecast standard deviation is
    # used as a provisional daily-demand variability
    # estimate for this demonstration.

    df["daily_demand_std_proxy"] = (
        df["demand_std_16_days"].clip(lower=0)
    )

    # ======================================
    # 7. SAFETY STOCK
    # ======================================

    df["safety_stock"] = (
        df["z_score"]
        * df["daily_demand_std_proxy"]
        * np.sqrt(df["lead_time_days"])
    )

    df["safety_stock"] = np.ceil(
        df["safety_stock"]
    ).astype(int)

    # ======================================
    # 8. REORDER POINT
    # ======================================

    df["reorder_point"] = np.ceil(
        (
            df["average_daily_demand"]
            * df["lead_time_days"]
        )
        + df["safety_stock"]
    ).astype(int)

    # ======================================
    # 9. TARGET INVENTORY LEVEL
    # ======================================

    df["target_stock_level"] = np.ceil(
        (
            df["average_daily_demand"]
            * (
                df["lead_time_days"]
                + df["review_period_days"]
            )
        )
        + df["safety_stock"]
    ).astype(int)

    # ======================================
    # 10. INVENTORY POSITION
    # ======================================

    df["inventory_position"] = (
        df["on_hand_units"]
        + df["on_order_units"]
        - df["backorder_units"]
    )

    # ======================================
    # 11. RECOMMENDED ORDER QUANTITY
    # ======================================

    df["recommended_order_quantity"] = np.ceil(
        (
            df["target_stock_level"]
            - df["inventory_position"]
        ).clip(lower=0)
    ).astype(int)

    # ======================================
    # 12. STOCKOUT RISK INDICATOR
    # ======================================

    df["stockout_risk"] = np.select(
        [
            df["on_hand_units"] <= df["backorder_units"],
            df["inventory_position"] <= df["reorder_point"],
        ],
        [
            "Critical",
            "High"
        ],
        default="Normal"
    )

    # ======================================
    # 13. REPLENISHMENT STATUS
    # ======================================

    df["replenishment_status"] = np.select(
        [
            df["on_hand_units"] <= df["backorder_units"],
            df["inventory_position"] <= df["reorder_point"],
            df["recommended_order_quantity"] > 0
        ],
        [
            "Urgent Replenishment",
            "Reorder Required",
            "Monitor Inventory"
        ],
        default="Sufficient Stock"
    )

    # ======================================
    # 14. VALIDATION
    # ======================================

    if (df["recommended_order_quantity"] < 0).any():
        raise ValueError(
            "Negative order quantity detected."
        )

    if df["recommended_order_quantity"].isna().any():
        raise ValueError(
            "Missing order quantity detected."
        )

    # ======================================
    # 15. SAVE RESULTS
    # ======================================

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info("Inventory optimization completed.")

    # ======================================
    # 16. SUMMARY
    # ======================================

    print("\n==========================================")
    print("INVENTORY OPTIMIZATION COMPLETED")
    print("==========================================")

    print("\nTotal records:", len(df))

    print("\nStockout risk summary:")
    print(df["stockout_risk"].value_counts())

    print("\nReplenishment status:")
    print(df["replenishment_status"].value_counts())

    print(
        "\nTotal recommended order quantity:",
        df["recommended_order_quantity"].sum()
    )

    print("\nTop 10 replenishment recommendations:")

    print(
        df.sort_values(
            "recommended_order_quantity",
            ascending=False
        )[
            [
                "store_nbr",
                "family",
                "on_hand_units",
                "inventory_position",
                "reorder_point",
                "safety_stock",
                "recommended_order_quantity",
                "stockout_risk",
                "replenishment_status"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nOutput saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    run_inventory_optimization()
