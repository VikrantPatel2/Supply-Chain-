
from pathlib import Path
import subprocess
import sys
import logging
import time
from datetime import datetime

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "master_pipeline.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("MasterPipeline")

# ==========================================
# PIPELINE STEPS
# ==========================================

PIPELINE_STEPS = [
    ("Data Validation", "01_data_validation.py"),
    ("Data Preprocessing", "02_data_preprocessing.py"),
    ("Feature Engineering", "03_feature_engineering.py"),
    ("Future Demand Forecasting", "05_future_demand_forecasting.py"),
    ("Inventory Input Preparation", "06_prepare_inventory_inputs.py"),
    ("Sample Inventory Generation", "07_generate_inventory_data.py"),
    ("Inventory Optimization", "08_inventory_optimization.py"),
    ("Business KPI Reporting", "09_inventory_business_report.py")
]

# ==========================================
# EXECUTION FUNCTION
# ==========================================

def run_step(step_name, script_name):

    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    logger.info("=" * 60)
    logger.info(f"STARTING: {step_name}")
    logger.info(f"Script: {script_name}")

    start_time = time.time()

    result = subprocess.run(
        [PYTHON, "-u", str(script_path)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    elapsed = round(time.time() - start_time, 2)

    if result.stdout:
        logger.info(result.stdout)

    if result.stderr:
        logger.info(result.stderr)

    if result.returncode != 0:
        logger.error(
            f"FAILED: {step_name} | Exit code: {result.returncode}"
        )

        raise RuntimeError(
            f"Pipeline stopped because {step_name} failed."
        )

    logger.info(
        f"COMPLETED: {step_name} | Time: {elapsed} seconds"
    )

    return elapsed


# ==========================================
# MAIN PIPELINE
# ==========================================

def run_pipeline():

    overall_start = time.time()

    completed_steps = []

    logger.info("=" * 60)
    logger.info("SUPPLY CHAIN FORECASTING PIPELINE STARTED")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("=" * 60)

    try:

        for step_name, script_name in PIPELINE_STEPS:

            elapsed = run_step(
                step_name,
                script_name
            )

            completed_steps.append({
                "step": step_name,
                "time_seconds": elapsed
            })

        total_time = round(
            time.time() - overall_start, 2
        )

        logger.info("=" * 60)
        logger.info("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {total_time} seconds")

        for step in completed_steps:
            logger.info(
                f"{step['step']}: {step['time_seconds']} seconds"
            )

        logger.info("=" * 60)

        print("\nSUPPLY CHAIN PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {total_time} seconds")

    except Exception:

        logger.exception("SUPPLY CHAIN PIPELINE FAILED")

        print("\nPIPELINE FAILED. Check logs/master_pipeline.log")

        raise


if __name__ == "__main__":
    run_pipeline()
