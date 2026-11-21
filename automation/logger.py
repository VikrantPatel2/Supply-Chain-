
import logging
from pathlib import Path

BASE_DIR = Path(
    r"C:\Users\vikra\Desktop\Supply Chain Algo"
)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),

    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],

    force=True
)

logger = logging.getLogger("SupplyChainPipeline")
