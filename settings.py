# aicli/settings.py
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

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


import os
import json
import sys
from pathlib import Path

# --- Configuration File Path ---
# Adheres to the XDG Base Directory Specification for placing user-specific config files.
CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'aicli'
SETTINGS_FILE = CONFIG_DIR / 'settings.json'

# --- Default Settings ---
# This dictionary represents the fallback configuration if the settings file is missing or corrupted.
DEFAULT_SETTINGS = {
    'default_engine': 'gemini',
    'default_openai_chat_model': 'gpt-4o-mini',
    'default_openai_image_model': 'dall-e-3',
    'default_gemini_model': 'gemini-1.5-flash-latest',
    'default_max_tokens': 8192,
    'stream': True,
    'helper_model_openai': 'gpt-4o-mini',
    'helper_model_gemini': 'gemini-1.5-flash-latest',
    'memory_enabled': False,
    'api_timeout': 180,
}

def _get_settings() -> dict:
    """Loads settings from the JSON file, creating it with defaults if it doesn't exist."""
    if not SETTINGS_FILE.exists():
        print(f"--> Creating default settings file at: {SETTINGS_FILE}", file=sys.stderr)
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_SETTINGS, f, indent=4)
            return DEFAULT_SETTINGS
        except IOError as e:
            print(f"Fatal: Could not create settings file: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            user_settings = json.load(f)
            # Ensure all keys from default are present
            settings = DEFAULT_SETTINGS.copy()
            settings.update(user_settings)
            return settings
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read or parse settings file ({e}). Using default settings.", file=sys.stderr)
        return DEFAULT_SETTINGS

def save_setting(key: str, value) -> bool:
    """Saves a single setting to the JSON file."""
    if key not in DEFAULT_SETTINGS:
        print(f"--> Error: '{key}' is not a valid setting.", file=sys.stderr)
        return False

    # Convert value for specific types
    if key in ['stream', 'memory_enabled']:
        if str(value).lower() in ['true', 'on', 'yes', '1']:
            value = True
        elif str(value).lower() in ['false', 'off', 'no', '0']:
            value = False
    elif key in ['default_max_tokens', 'api_timeout']:
        try:
            value = int(value)
        except ValueError:
            print(f"--> Error: '{key}' requires an integer value.", file=sys.stderr)
            return False

    current_settings = _get_settings()
    current_settings[key] = value
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(current_settings, f, indent=4)
        print(f"--> Setting '{key}' updated to '{value}'.")
        return True
    except IOError as e:
        print(f"--> Error: Could not write to settings file: {e}", file=sys.stderr)
        return False

# Load settings on module import to be used application-wide.
settings = _get_settings()
