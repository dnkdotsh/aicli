# aicli/theme_manager.py
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

"""
Manages the loading and application of color themes.
"""

import json
from importlib import resources

from . import config
from .logger import log
from .settings import settings

USER_THEMES_DIR = config.CONFIG_DIR / "themes"


def _load_packaged_theme(name: str) -> dict:
    """Loads a theme from the internal package resources."""
    try:
        theme_files = resources.files("aicli.themes")
        with (
            resources.as_file(theme_files / f"{name}.json") as theme_path,
            open(theme_path, encoding="utf-8") as f,
        ):
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError, ModuleNotFoundError) as e:
        log.warning("Could not load packaged theme '%s': %s", name, e)
        return {}


def _load_user_theme(name: str) -> dict:
    """Loads a theme from the user's configuration directory."""
    if not USER_THEMES_DIR.exists():
        return {}
    theme_path = USER_THEMES_DIR / f"{name}.json"
    if not theme_path.is_file():
        return {}
    try:
        with open(theme_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Could not load user theme '%s': %s", name, e)
        return {}


def get_theme(name: str) -> dict:
    """
    Constructs a theme by loading the default and merging a specified
    theme on top of it.
    """
    # Always load the default theme as the base to ensure all keys are present.
    theme = _load_packaged_theme("default")
    if not theme:
        log.error("Critical: Default theme could not be loaded. Styling will fail.")
        return {}

    if name.lower() != "default":
        # User themes take precedence over packaged themes.
        custom_theme = _load_user_theme(name) or _load_packaged_theme(name)
        if custom_theme:
            theme.update(custom_theme)
        else:
            log.warning("Theme '%s' not found. Falling back to default.", name)

    return theme


def list_themes() -> dict[str, str]:
    """Lists all available themes from both packaged and user directories."""
    themes = {}

    # Load packaged themes first
    try:
        theme_files = resources.files("aicli.themes")
        for item in theme_files.iterdir():
            if item.name.endswith(".json"):
                theme_name = item.name[:-5]
                content = _load_packaged_theme(theme_name)
                themes[theme_name] = content.get("description", "No description.")
    except (ModuleNotFoundError, FileNotFoundError):
        log.warning("Could not list packaged themes.")

    # Load user themes, overwriting packaged themes with the same name
    if USER_THEMES_DIR.exists():
        for theme_path in USER_THEMES_DIR.glob("*.json"):
            theme_name = theme_path.stem
            content = _load_user_theme(theme_name)
            themes[theme_name] = content.get("description", "No description.")

    return dict(sorted(themes.items()))


def reload_theme() -> None:
    """Re-loads the active theme from settings into the global variable."""
    global ACTIVE_THEME
    ACTIVE_THEME = load_active_theme()


def load_active_theme() -> dict:
    """Loads the theme specified in the application settings."""
    active_theme_name = settings.get("active_theme", "default")
    return get_theme(active_theme_name)


# Load the active theme into a global variable for easy access by other modules.
ACTIVE_THEME = load_active_theme()
