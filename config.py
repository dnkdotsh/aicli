# aicli/config.py

# Unified Command-Line AI Client
# Copyright (C) 2025 <name of author>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os

# --- Main Configuration ---

# Default engine can be 'openai' or 'gemini'
DEFAULT_ENGINE = 'gemini'

# Default models for each service
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5-nano"
DEFAULT_OPENAI_IMAGE_MODEL = "dall-e-3"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

# Models for automated helper tasks (renaming, memory, etc.)
DEFAULT_HELPER_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_HELPER_MODEL_GEMINI = "gemini-1.5-flash-latest"

# Token configuration
DEFAULT_MAX_TOKENS = 8192
SUMMARY_MAX_TOKENS = 2048 # Max tokens for generating a session summary

# Chat history configuration
HISTORY_SUMMARY_THRESHOLD_TURNS = 10
# Number of turns (1 turn = 1 user + 1 assistant message) from the beginning of a
# conversation to summarize when the threshold is met.
HISTORY_SUMMARY_TRIM_TURNS = 5


# --- Log Configuration ---

LOG_DIRECTORY = "logs"
IMAGE_DIRECTORY = "images"

# Ensure log paths are relative to the log directory
IMAGE_LOG_FILE = os.path.join(LOG_DIRECTORY, "image_log.jsonl")
PERSISTENT_MEMORY_FILE = os.path.join(LOG_DIRECTORY, "persistent_memory.txt") # Unified Memory
RAW_LOG_FILE = os.path.join(LOG_DIRECTORY, "raw.log")
