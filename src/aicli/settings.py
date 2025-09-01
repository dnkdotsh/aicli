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

from . import config, utils
from .logger import log


def _get_default_settings():
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
    }


def _load_settings():
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


def save_setting(key: str, value: str):
    """Saves a single setting to the JSON file after type conversion."""
    if key not in _get_default_settings():
        print(f"{utils.SYSTEM_MSG}--> Unknown setting: '{key}'.{utils.RESET_COLOR}")
        return

    current_settings = _load_settings()

    # Attempt to convert value to a more appropriate type
    original_type = type(_get_default_settings().get(key))
    converted_value = value

    if original_type == bool:
        if value.lower() in ["true", "yes", "1"]:
            converted_value = True
        elif value.lower() in ["false", "no", "0"]:
            converted_value = False
        else:
            print(
                f"{utils.SYSTEM_MSG}--> Invalid boolean value: {value}. Use true/false.{utils.RESET_COLOR}"
            )
            return
    elif original_type == int:
        try:
            converted_value = int(value)
        except ValueError:
            print(
                f"{utils.SYSTEM_MSG}--> Invalid integer value: {value}.{utils.RESET_COLOR}"
            )
            return

    current_settings[key] = converted_value

    # Prune any settings that are no longer in the defaults
    default_keys = _get_default_settings().keys()
    user_settings_to_save = {
        k: v
        for k, v in current_settings.items()
        if k in default_keys and k != "version"
    }

    try:
        # Ensure the parent directory exists before writing
        utils.ensure_dir_exists(config.CONFIG_DIR)
        with open(config.SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(user_settings_to_save, f, indent=2)
        print(
            f"{utils.SYSTEM_MSG}--> Setting '{key}' updated to '{converted_value}'.{utils.RESET_COLOR}"
        )
        # Update runtime settings object as well
        settings[key] = converted_value
    except OSError as e:
        log.error("Failed to save settings: %s", e)


settings = _load_settings()
