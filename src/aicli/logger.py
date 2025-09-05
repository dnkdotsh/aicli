# aicli/logger.py
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 David

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import logging
import sys
from logging.handlers import RotatingFileHandler

from . import config


def setup_logger():
    """Configures and returns a project-wide logger."""
    logger = logging.getLogger("aicli")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # By the time this function is called, bootstrap.py has already ensured
    # the log directory exists, so we can safely create the file handler.
    try:
        # Rotating file handler for persistent logs
        # Rotates when the log reaches 1MB, keeping up to 5 backup logs.
        file_handler = RotatingFileHandler(
            config.ROTATING_LOG_FILE,
            maxBytes=1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
    except OSError as e:
        print(
            f"CRITICAL: Could not create log file {config.ROTATING_LOG_FILE}: {e}",
            file=sys.stderr,
        )
        # Fallback to just a console logger if file creation fails
        if not logger.handlers:
            logger.addHandler(console_handler)
        return logger

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# Singleton logger instance to be imported by other modules
log = setup_logger()
