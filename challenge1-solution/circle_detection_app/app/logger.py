# app/logger.py

import logging
import os

# Ensure the logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure the logger
logger = logging.getLogger("circle_detection_app")
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

# Formatter for the logs
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# File handler to write logs to a file
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
file_handler.setLevel(logging.INFO)  # Set to INFO for general logs
file_handler.setFormatter(formatter)

# Stream handler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set to DEBUG for console
console_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
