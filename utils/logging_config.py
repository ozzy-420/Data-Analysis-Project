import logging
import sys
from utils.config import LOGGING_LEVEL_CONSOLE, LOGGING_LEVEL_FILE, LOG_FILE

# Create a file handler
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(LOGGING_LEVEL_CONSOLE)

# Define a common formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL_FILE)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
