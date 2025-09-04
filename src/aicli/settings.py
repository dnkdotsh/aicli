# aicli/settings.py
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

import json
from pathlib import Path
from typing import Any

from . import config
from .logger import log


def _ensure_dir_exists(directory_path: Path) -> None:
    """Creates a directory if it does not already exist."""
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Failed to create directory at %s: %s", directory_path, e)
        # We raise an exception here so the caller can handle the error message.
        raise


def _get_default_settings() -> dict[str, Any]:
    """Returns a dictionary of the default application settings."""
    return {
        # --- General ---
        "default_engine": "gemini",
        "api_timeout": 60,
        # --- Models ---
        "default_gemini_model": "gemini-1.5-flash-latest",
        "default_openai_chat_model": "gpt-4o-mini",
        "default_openai_image_model": "dall-e-3",
        "helper_model_gemini": "gemini-1.5-flash-latest",
        "helper_model_openai": "gpt-4o-mini",
        # --- Behavior ---
        "stream": True,
        "memory_enabled": True,
        "default_max_tokens": 4096,
        "summary_max_tokens": 4096,
        "log_rename_max_tokens": 2048,
        "image_prompt_max_tokens": 2048,
    }


def _load_settings() -> dict[str, Any]:
    """Loads settings from the JSON file, merging them with defaults."""
    defaults = _get_default_settings()
    if not config.SETTINGS_FILE.exists():
        return defaults
    try:
        with open(config.SETTINGS_FILE, encoding="utf-8") as f:
            user_settings = json.load(f)
        defaults.update(user_settings)
        return defaults
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Could not load settings file: %s. Using defaults.", e)
        return defaults


def save_setting(key: str, value: str) -> tuple[bool, str]:
    """
    Saves a single setting to the JSON file after type conversion.
    Returns a tuple of (success_boolean, message_string).
    """
    default_settings = _get_default_settings()
    if key not in default_settings:
        return False, f"Unknown setting: '{key}'."

    current_settings = _load_settings()
    original_type = type(default_settings.get(key))
    converted_value: Any = value

    try:
        if original_type == bool:
            if value.lower() in ["true", "yes", "1"]:
                converted_value = True
            elif value.lower() in ["false", "no", "0"]:
                converted_value = False
            else:
                raise ValueError("Invalid boolean value. Use true/false.")
        elif original_type == int:
            converted_value = int(value)
    except ValueError as e:
        return False, f"Error: {e}"

    current_settings[key] = converted_value

    user_settings_to_save = {
        k: v for k, v in current_settings.items() if k in default_settings
    }

    try:
        _ensure_dir_exists(config.CONFIG_DIR)
        with open(config.SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(user_settings_to_save, f, indent=2)
        settings[key] = converted_value
        return True, f"Setting '{key}' updated to '{converted_value}'."
    except OSError as e:
        log.error("Failed to save settings: %s", e)
        return False, f"Error saving settings file: {e}"


settings: dict[str, Any] = _load_settings()
