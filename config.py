# aicli/config.py

import os
from pathlib import Path

# --- Main Configuration ---
# This is now the location for non-user-configurable constants.
# User-facing defaults are managed in settings.py

# Base directory for all application-generated files.
DATA_DIR = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'aicli'

# --- Log and Data Directories ---
LOG_DIRECTORY = DATA_DIR / "logs"
IMAGE_DIRECTORY = DATA_DIR / "images"

# --- Specific File Paths ---
IMAGE_LOG_FILE = LOG_DIRECTORY / "image_log.jsonl"
PERSISTENT_MEMORY_FILE = LOG_DIRECTORY / "persistent_memory.txt"
RAW_LOG_FILE = LOG_DIRECTORY / "raw.log"

# --- Chat History Configuration ---
# These values are related to application logic rather than user preference.
# A "turn" consists of one user message and one assistant response.
HISTORY_SUMMARY_THRESHOLD_TURNS = 12
HISTORY_SUMMARY_TRIM_TURNS = 6

# Token configuration for internal tasks.
SUMMARY_MAX_TOKENS = 2048

