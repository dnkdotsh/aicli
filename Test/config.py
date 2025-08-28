# config.py
import os

# --- Main Configuration ---

# Default engine can be 'openai' or 'gemini'
DEFAULT_ENGINE = 'gemini'

# Default models for each service
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_IMAGE_MODEL = "dall-e-3"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

# Models for automated helper tasks (renaming, memory, etc.)
DEFAULT_HELPER_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_HELPER_MODEL_GEMINI = "gemini-1.5-flash-latest"

# Token configuration
DEFAULT_MAX_TOKENS = 8192

# Chat history configuration
HISTORY_SUMMARY_THRESHOLD_TURNS = 10  # NEW: Summarize history after this many turns
MAX_HISTORY_TURNS = 20


# --- Log Configuration ---

LOG_DIRECTORY = "logs"
IMAGE_DIRECTORY = "images"

# Ensure log paths are relative to the log directory
IMAGE_LOG_FILE = os.path.join(LOG_DIRECTORY, "image_log.jsonl")
PERSISTENT_MEMORY_FILE = os.path.join(LOG_DIRECTORY, "persistent_memory.txt") # Unified Memory
RAW_LOG_FILE = os.path.join(LOG_DIRECTORY, "raw.log")
