import logging
import sys

# Create a file handler
file_handler = logging.FileHandler('../app.log', mode='w')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)

# Define a common formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Example log messages
# logger.info("This is an info message.")
# logger.debug("This is a debug message.")
# logger.error("This is an error message.")