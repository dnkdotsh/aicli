# aicli/config.py
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY;
# without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


from pathlib import Path

import platformdirs

APP_NAME = "aicli"
APP_AUTHOR = "DankASaurus"  # Optional, but good practice for platformdirs

# Base directory for user-specific configuration files.
# Uses OS-native paths:
# - Windows: C:\Users\<user>\AppData\Roaming\DankASaurus\aicli
# - macOS:   ~/Library/Application Support/aicli
# - Linux:   ~/.config/aicli (or XDG_CONFIG_HOME)
CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR))

# Base directory for all application-generated data files.
# Uses OS-native paths:
# - Windows: C:\Users\<user>\AppData\Local\DankASaurus\aicli
# - macOS:   ~/Library/Application Support/aicli
# - Linux:   ~/.local/share/aicli (or XDG_DATA_HOME)
DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))


# --- Log and Data Directories (under DATA_DIR) ---
LOG_DIRECTORY = DATA_DIR / "logs"
IMAGE_DIRECTORY = DATA_DIR / "images"
SESSIONS_DIRECTORY = DATA_DIR / "sessions"

# --- Config Directories (under CONFIG_DIR) ---
PERSONAS_DIRECTORY = CONFIG_DIR / "personas"

# --- Specific File Paths ---
SETTINGS_FILE = CONFIG_DIR / "settings.json"
DOTENV_FILE = CONFIG_DIR / ".env"
IMAGE_LOG_FILE = LOG_DIRECTORY / "image_log.jsonl"
PERSISTENT_MEMORY_FILE = LOG_DIRECTORY / "persistent_memory.txt"
RAW_LOG_FILE = LOG_DIRECTORY / "raw.log"
ROTATING_LOG_FILE = LOG_DIRECTORY / "aicli.log"

# --- Chat History Configuration ---
# These values are related to application logic rather than user preference.
# A "turn" consists of one user message and one assistant response.
HISTORY_SUMMARY_THRESHOLD_TURNS = 12
HISTORY_SUMMARY_TRIM_TURNS = 6

# The size in bytes at which to warn the user about large context on first prompt.
# 100 KB is a sensible threshold, representing a significant amount of text
# (approx. 20-25k tokens) that warrants a cost/usage warning.
LARGE_ATTACHMENT_THRESHOLD_BYTES = 100 * 1024  # 100 KB
