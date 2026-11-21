import pandas as pd
from pathlib import Path

files = [
    "reports/16_day_demand_forecast.csv",
    "reports/inventory_optimization_results.csv",
    "reports/store_inventory_kpis.csv",
    "reports/product_family_inventory_kpis.csv"
]

for file in files:
    path = Path(file)

    print("\n" + "=" * 60)
    print("FILE:", file)

    df = pd.read_csv(path, nrows=5)

    print("COLUMNS:")
    print(df.columns.tolist())

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nSAMPLE:")
    print(df.head(2))