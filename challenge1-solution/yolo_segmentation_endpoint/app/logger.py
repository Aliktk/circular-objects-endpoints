# app/logger.py

import logging
import os

# Ensure the logs directory exists
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure logger
logger = logging.getLogger("yolo_endpoint_logger")
logger.setLevel(logging.DEBUG)  # Capture all levels; filter via handlers

# Create file handler for INFO and higher
info_handler = logging.FileHandler(os.path.join(LOG_DIR, "app_info.log"))
info_handler.setLevel(logging.INFO)

# Create file handler for ERROR and higher
error_handler = logging.FileHandler(os.path.join(LOG_DIR, "app_error.log"))
error_handler.setLevel(logging.ERROR)

# Create console handler for debugging (optional)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add to handlers
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)
