
from pathlib import Path
import pandas as pd
import logging

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

INPUT_PATH = REPORT_DIR / "inventory_optimization_results.csv"

STORE_OUTPUT = REPORT_DIR / "store_inventory_kpis.csv"
FAMILY_OUTPUT = REPORT_DIR / "product_family_inventory_kpis.csv"
SUMMARY_OUTPUT = REPORT_DIR / "inventory_business_summary.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "business_report.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("BusinessReport")


def generate_business_report():

    logger.info("Generating inventory business report...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    # --------------------------------
    # Overall business KPIs
    # --------------------------------

    total_records = len(df)

    critical_records = (
        df["stockout_risk"] == "Critical"
    ).sum()

    high_records = (
        df["stockout_risk"] == "High"
    ).sum()

    at_risk_records = critical_records + high_records

    urgent_records = (
        df["replenishment_status"] == "Urgent Replenishment"
    ).sum()

    reorder_records = (
        df["replenishment_status"] == "Reorder Required"
    ).sum()

    total_order_quantity = (
        df["recommended_order_quantity"].sum()
    )

    total_inventory_position = (
        df["inventory_position"].sum()
    )

    at_risk_percentage = (
        at_risk_records / total_records * 100
        if total_records else 0
    )

    # --------------------------------
    # Store-level KPIs
    # --------------------------------

    store_kpis = (
        df.groupby("store_nbr")
        .agg(
            product_combinations=("family", "count"),
            total_order_quantity=(
                "recommended_order_quantity", "sum"
            ),
            critical_records=(
                "stockout_risk",
                lambda x: (x == "Critical").sum()
            ),
            high_risk_records=(
                "stockout_risk",
                lambda x: (x == "High").sum()
            ),
            inventory_position=(
                "inventory_position", "sum"
            )
        )
        .reset_index()
    )

    store_kpis["at_risk_records"] = (
        store_kpis["critical_records"]
        + store_kpis["high_risk_records"]
    )

    store_kpis["at_risk_percentage"] = (
        store_kpis["at_risk_records"]
        / store_kpis["product_combinations"] * 100
    )

    store_kpis = store_kpis.sort_values(
        "total_order_quantity",
        ascending=False
    )

    # --------------------------------
    # Product family KPIs
    # --------------------------------

    family_kpis = (
        df.groupby("family")
        .agg(
            stores=("store_nbr", "nunique"),
            total_order_quantity=(
                "recommended_order_quantity", "sum"
            ),
            average_daily_demand=(
                "average_daily_demand", "mean"
            ),
            critical_records=(
                "stockout_risk",
                lambda x: (x == "Critical").sum()
            ),
            high_risk_records=(
                "stockout_risk",
                lambda x: (x == "High").sum()
            )
        )
        .reset_index()
    )

    family_kpis["at_risk_records"] = (
        family_kpis["critical_records"]
        + family_kpis["high_risk_records"]
    )

    family_kpis = family_kpis.sort_values(
        "total_order_quantity",
        ascending=False
    )

    # --------------------------------
    # Summary report
    # --------------------------------

    summary = pd.DataFrame({
        "metric": [
            "Total store-family combinations",
            "Critical inventory records",
            "High-risk inventory records",
            "Total at-risk records",
            "At-risk percentage",
            "Urgent replenishment records",
            "Reorder required records",
            "Total recommended order quantity",
            "Total inventory position"
        ],
        "value": [
            total_records,
            critical_records,
            high_records,
            at_risk_records,
            round(at_risk_percentage, 2),
            urgent_records,
            reorder_records,
            total_order_quantity,
            total_inventory_position
        ]
    })

    # --------------------------------
    # Save reports
    # --------------------------------

    store_kpis.to_csv(STORE_OUTPUT, index=False)
    family_kpis.to_csv(FAMILY_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    logger.info("Business reports generated successfully.")

    print("\n====================================")
    print("INVENTORY BUSINESS REPORT")
    print("====================================")

    print("\nOverall KPIs:")
    print(summary.to_string(index=False))

    print("\nTop 10 stores by order quantity:")
    print(store_kpis.head(10).to_string(index=False))

    print("\nTop 10 product families by order quantity:")
    print(family_kpis.head(10).to_string(index=False))

    print("\nGenerated files:")
    print(STORE_OUTPUT)
    print(FAMILY_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    generate_business_report()
