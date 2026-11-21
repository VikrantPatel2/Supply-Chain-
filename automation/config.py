
from pathlib import Path

# Project root directory
BASE_DIR = Path(
    r"C:\Users\vikra\Desktop\Supply Chain Algo"
)

# Main directories
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

# Model
MODEL_PATH = (
    MODEL_DIR / "optimized_catboost_model.cbm"
)

# Forecast reports
FORECAST_PATH = (
    REPORT_DIR / "16_day_demand_forecast.csv"
)

INVENTORY_PATH = (
    REPORT_DIR / "inventory_optimization_results.csv"
)

# Database configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "Supply_Chain_Forecasting"
DB_USER = "postgres"

# Create required directories
for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    REPORT_DIR,
    LOG_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

print("Project configuration loaded!")
