
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"


# ==================================================
# 2. POSTGRESQL DATABASE CONFIGURATION
# ==================================================

DB_CONFIG = {
    "host": "localhost",
    "database": "Supply_Chain_Forecasting",
    "user": "postgres",
    "password": "Vikrant123",
    "port": 5432
}


# ==================================================
# 3. GENERIC CSV TO POSTGRESQL LOADER
# ==================================================

def load_csv_to_postgres(
    connection,
    csv_file,
    table_name,
    columns,
    conflict_columns
):

    file_path = REPORT_DIR / csv_file

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    print(f"\nLoading file: {csv_file}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Map CSV column to PostgreSQL column
    if table_name == "supply_chain.forecast_results":
        df.rename(
            columns={"date": "forecast_date"},
            inplace=True
        )

    # Validate required columns
    missing_columns = [
        col for col in columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {csv_file}: {missing_columns}"
        )

    # Convert NaN to None for PostgreSQL
    df = df.astype(object).where(
        pd.notna(df),
        None
    )

    # Convert DataFrame to records
    records = [
        tuple(row)
        for row in df[columns].itertuples(
            index=False,
            name=None
        )
    ]

    if not records:
        print(f"No records found in {csv_file}")
        return

    # SQL column names
    column_names = ", ".join(columns)

    conflict_target = ", ".join(conflict_columns)

    # Columns to update on conflict
    update_columns = [
        col for col in columns
        if col not in conflict_columns
    ]

    if update_columns:

        update_clause = ", ".join(
            f"{col} = EXCLUDED.{col}"
            for col in update_columns
        )

        conflict_action = (
            f"DO UPDATE SET {update_clause}"
        )

    else:

        conflict_action = "DO NOTHING"

    # UPSERT query
    query = f"""
        INSERT INTO {table_name} ({column_names})
        VALUES %s
        ON CONFLICT ({conflict_target})
        {conflict_action};
    """

    # Execute batch insert
    with connection.cursor() as cursor:

        execute_values(
            cursor,
            query,
            records,
            page_size=1000
        )

    connection.commit()

    print(
        f"Successfully loaded {len(records):,} records "
        f"into {table_name}"
    )


# ==================================================
# 4. MAIN DATABASE LOADING PROCESS
# ==================================================

def main():

    connection = None

    try:

        print("=" * 60)
        print("SUPPLY CHAIN POSTGRESQL DATA LOADER")
        print("=" * 60)

        print("\nConnecting to PostgreSQL...")

        connection = psycopg2.connect(
            **DB_CONFIG
        )

        print("PostgreSQL connection successful!")

        # ------------------------------------------
        # A. FUTURE DEMAND FORECAST RESULTS
        # ------------------------------------------

        load_csv_to_postgres(
            connection,
            "16_day_demand_forecast.csv",
            "supply_chain.forecast_results",
            [
                "forecast_date",
                "store_nbr",
                "family",
                "predicted_sales"
            ],
            [
                "forecast_date",
                "store_nbr",
                "family"
            ]
        )

        # ------------------------------------------
        # B. INVENTORY OPTIMIZATION RESULTS
        # ------------------------------------------

        load_csv_to_postgres(
            connection,
            "inventory_optimization_results.csv",
            "supply_chain.inventory_optimization_results",
            [
                "store_nbr",
                "family",
                "forecast_total_16_days",
                "average_daily_demand",
                "maximum_daily_demand",
                "demand_std_16_days",
                "on_hand_units",
                "on_order_units",
                "backorder_units",
                "lead_time_days",
                "review_period_days",
                "service_level",
                "z_score",
                "daily_demand_std_proxy",
                "safety_stock",
                "reorder_point",
                "target_stock_level",
                "inventory_position",
                "recommended_order_quantity",
                "stockout_risk",
                "replenishment_status"
            ],
            [
                "store_nbr",
                "family"
            ]
        )

        # ------------------------------------------
        # C. STORE INVENTORY KPIs
        # ------------------------------------------

        load_csv_to_postgres(
            connection,
            "store_inventory_kpis.csv",
            "supply_chain.store_inventory_kpis",
            [
                "store_nbr",
                "product_combinations",
                "total_order_quantity",
                "critical_records",
                "high_risk_records",
                "inventory_position",
                "at_risk_records",
                "at_risk_percentage"
            ],
            [
                "store_nbr"
            ]
        )

        # ------------------------------------------
        # D. PRODUCT FAMILY INVENTORY KPIs
        # ------------------------------------------

        load_csv_to_postgres(
            connection,
            "product_family_inventory_kpis.csv",
            "supply_chain.product_family_inventory_kpis",
            [
                "family",
                "stores",
                "total_order_quantity",
                "average_daily_demand",
                "critical_records",
                "high_risk_records",
                "at_risk_records"
            ],
            [
                "family"
            ]
        )

        print("\n" + "=" * 60)
        print("ALL FOUR DATASETS LOADED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as error:

        if connection:
            connection.rollback()

        print("\nDATABASE LOADING FAILED!")
        print("Error:", error)

        raise

    finally:

        if connection:
            connection.close()

            print("\nPostgreSQL connection closed.")


# ==================================================
# 5. EXECUTION
# ==================================================

if __name__ == "__main__":
    main()
