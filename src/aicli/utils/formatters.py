# aicli/utils/formatters.py
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
Utility functions for formatting data for display, such as token counts,
byte sizes, and sanitizing strings. Also contains UI color constants.
"""

import re

from .. import theme_manager

# Load prompt colors from the active theme, with hardcoded defaults for resilience.
USER_PROMPT = theme_manager.ACTIVE_THEME.get("prompt_color_user", "\033[94m")
ASSISTANT_PROMPT = theme_manager.ACTIVE_THEME.get("prompt_color_assistant", "\033[92m")
SYSTEM_MSG = theme_manager.ACTIVE_THEME.get("prompt_color_system", "\033[93m")
DIRECTOR_PROMPT = theme_manager.ACTIVE_THEME.get("prompt_color_director", "\033[95m")
RESET_COLOR = "\033[0m"


def sanitize_filename(name: str) -> str:
    r"""Sanitizes a string to be a valid filename."""
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[-\s]+", "_", name)
    return name or "unnamed_log"


def format_token_string(token_dict: dict[str, int]) -> str:
    """Formats the token dictionary into a consistent string for the toolbar."""
    if not token_dict or not any(token_dict.values()):
        return ""

    p = token_dict.get("prompt", 0)
    c = token_dict.get("completion", 0)
    t = token_dict.get("total", 0)
    r = token_dict.get("reasoning", 0) or max(0, t - (p + c))
    return f"[P:{p}/C:{c}/R:{r}/T:{t}]"


def estimate_token_count(text: str) -> int:
    """
    Provides a simple, fast estimation of token count.
    This formula is a rough heuristic.
    """
    if not text:
        return 0

    # Heuristic: Code/structured text often has newlines. Prose usually doesn't.
    if "\n" in text:
        # For code, a character-based estimate is often better.
        # A divisor of 4 is a common, effective heuristic.
        return round(len(text) / 4)
    else:
        # For simple prose, word count is a more reliable estimate.
        return len(text.split())


def format_bytes(byte_count: int) -> str:
    """Converts a byte count to a human-readable string (KB, MB, etc.)."""
    if byte_count is None or byte_count == 0:
        return "0.00 B"
    power, n = 1024, 0
    power_labels = {0: "B", 1: "KB", 2: "MB", 3: "GB"}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}"


def clean_ai_response_text(engine_name: str, raw_response: str) -> str:
    """Strips any self-labels the AI might have added."""
    # This pattern matches "[TAG]", optional whitespace, an optional colon, and more optional whitespace.
    pattern = re.compile(
        r"^\s*\[" + re.escape(engine_name.capitalize()) + r"\]\s*:?\s*", re.IGNORECASE
    )
    return pattern.sub("", raw_response.lstrip())
