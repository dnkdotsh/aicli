# aicli/logger.py

import logging
from logging.handlers import RotatingFileHandler
import sys

import config

def setup_logger():
    """Configures and returns a project-wide logger."""
    # Create logger object
    logger = logging.getLogger("aicli")
    logger.setLevel(logging.INFO)

    # Prevent propagation to the root logger to avoid duplicate messages
    logger.propagate = False

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler for user-facing info/warnings/errors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Rotating file handler for persistent logs
    # Rotates when the log reaches 1MB, keeping up to 5 backup logs.
    file_handler = RotatingFileHandler(
        config.ROTATING_LOG_FILE, maxBytes=1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger if they aren't already added
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# Singleton logger instance to be imported by other modules
log = setup_logger()
