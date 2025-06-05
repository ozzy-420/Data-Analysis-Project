# config.py
from pathlib import Path
import logging

ROOT_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_FILE = OUTPUT_DIR / "app.log"

DATA_SOURCE_PATH = DATA_DIR / "UCI_Credit_Card.csv"
STAGE1_OUTPUT_DIR = OUTPUT_DIR / "stage1"
STAGE2_OUTPUT_DIR = OUTPUT_DIR / "stage2"
STAGE3_OUTPUT_DIR = OUTPUT_DIR / "stage3"

for directory in (OUTPUT_DIR,
                  STAGE1_OUTPUT_DIR,
                  STAGE2_OUTPUT_DIR,
                  STAGE3_OUTPUT_DIR):
    directory.mkdir(exist_ok=True)

TARGET_COLUMN = 'default.payment.next.month'
RANDOM_STATE = 42
LOGGING_LEVEL_CONSOLE = logging.INFO
LOGGING_LEVEL_FILE = logging.DEBUG

MAPPING = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {1: "Graduate", 2: "University", 3: "High",
                  4: "Other", 5: "Unknown", 0: "Unknown",
                  6: "Unknown"}
}

DEFAULT_PLOT_CONFIG = {
    "dpi": 300,
    "bbox_inches": "tight",
    "palette": "coolwarm"
}
