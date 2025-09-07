# aicli/chat_ui.py
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
This module contains the user interface components for managing interactive
chat sessions. It handles user input, command parsing, and displaying output,
delegating all business logic to a SessionManager instance.
"""

from __future__ import annotations

import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from . import commands, config, theme_manager
from . import settings as app_settings
from .logger import log
from .managers.multichat_manager import MultiChatSession
from .managers.session_manager import SessionManager
from .utils.formatters import (
    DIRECTOR_PROMPT,
    RESET_COLOR,
    SYSTEM_MSG,
    USER_PROMPT,
    estimate_token_count,
    format_token_string,
)
from .utils.message_builder import extract_text_from_message


class SingleChatUI:
    """Manages the lifecycle and I/O for an interactive single-chat session."""

    def __init__(self, session_manager: SessionManager, session_name: str | None):
        self.session = session_manager
        # This instance needs its own reference to the session_name for the run loop.
        self.session_name = session_name
        # Pass session_name to the manager for potential use in workflows
        # (e.g., image generation logging)
        self.session.session_name = session_name

    def _create_style_from_theme(self) -> Style:
        """Creates a prompt_toolkit Style object from the current theme."""
        try:
            return Style.from_dict(
                {
                    # Apply background to the entire toolbar container
                    "bottom-toolbar": theme_manager.ACTIVE_THEME.get(
                        "style_bottom_toolbar_background", ""
                    ),
                    "bottom-toolbar.separator": theme_manager.ACTIVE_THEME.get(
                        "style_bottom_toolbar_separator", ""
                    ),
                    "bottom-toolbar.tokens": theme_manager.ACTIVE_THEME.get(
                        "style_bottom_toolbar_tokens", ""
                    ),
                    "bottom-toolbar.io": theme_manager.ACTIVE_THEME.get(
                        "style_bottom_toolbar_io", ""
                    ),
                    "bottom-toolbar.model": theme_manager.ACTIVE_THEME.get(
                        "style_bottom_toolbar_model", ""
                    ),
                    "bottom-toolbar.persona": theme_manager.ACTIVE_THEME.get(
                        "style_bottom_toolbar_persona", ""
                    ),
                    "bottom-toolbar.live": theme_manager.ACTIVE_THEME.get(
                        "style_bottom_toolbar_live", ""
                    ),
                }
            )
        except (ValueError, TypeError) as e:
            print(
                f"{SYSTEM_MSG}--> Warning: Invalid theme style format detected: {e}{RESET_COLOR}"
            )
            log.warning("Invalid theme style format: %s", e)
            return Style.from_dict({})

    def _get_bottom_toolbar_content(self) -> Any | None:
        """Constructs the dynamic content for the prompt_toolkit bottom toolbar."""
        # --- Live Context Token Estimation ---
        base_context_tokens = 0
        full_system_prompt = self.session._assemble_full_system_prompt()
        if full_system_prompt:
            base_context_tokens += estimate_token_count(full_system_prompt)

        for message in self.session.state.history:
            text_content = extract_text_from_message(message)
            base_context_tokens += estimate_token_count(text_content)

        live_buffer_text = get_app().current_buffer.text
        live_buffer_tokens = estimate_token_count(live_buffer_text)
        total_live_tokens = base_context_tokens + live_buffer_tokens

        # --- Component Mapping ---
        component_map = {
            "tokens": (
                True,
                "class:bottom-toolbar.tokens",
                format_token_string(self.session.state.last_turn_tokens),
            ),
            "io": (
                app_settings.settings["toolbar_show_total_io"],
                "class:bottom-toolbar.io",
                f"Session I/O: {self.session.state.total_prompt_tokens}p / {self.session.state.total_completion_tokens}c",
            ),
            "model": (
                app_settings.settings["toolbar_show_model"],
                "class:bottom-toolbar.model",
                f"Model: {self.session.state.model}",
            ),
            "persona": (
                app_settings.settings["toolbar_show_persona"]
                and self.session.state.current_persona,
                "class:bottom-toolbar.persona",
                f"Persona: {self.session.state.current_persona.name if self.session.state.current_persona else ''}",
            ),
            "live": (
                app_settings.settings["toolbar_show_live_tokens"],
                "class:bottom-toolbar.live",
                f"Live Context: ~{total_live_tokens}t",
            ),
        }

        order = app_settings.settings["toolbar_priority_order"].split(",")
        width = shutil.get_terminal_size().columns

        styled_parts = []
        current_length = 0
        separator = app_settings.settings["toolbar_separator"]
        sep_len = len(separator)
        sep_style_str = "class:bottom-toolbar.separator"

        for key in order:
            if key not in component_map:
                continue

            is_enabled, style_class, text = component_map[key]
            if not is_enabled or not text:
                continue

            part_len = len(text)
            required_len = part_len + (sep_len if styled_parts else 0)

            if current_length + required_len > width:
                break

            if styled_parts:
                styled_parts.append((sep_style_str, separator))
            styled_parts.append((style_class, text))
            current_length += required_len

        return styled_parts

    def run(self) -> None:
        log_filename_base = (
            self.session_name
            or f"chat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.session.state.engine.name}"
        )
        log_filepath = config.CHATLOG_DIRECTORY / f"{log_filename_base}.jsonl"
        print(
            f"Starting interactive chat with {self.session.state.engine.name.capitalize()} ({self.session.state.model})."
        )
        print(
            f"Type '/help' for commands or '/exit' to end. Log file: {log_filepath.name}"
        )

        style = self._create_style_from_theme()
        prompt_session = PromptSession(
            history=InMemoryHistory(self.session.state.command_history), style=style
        )
        first_turn = not self.session.state.history

        try:
            while True:
                toolbar_content = (
                    self._get_bottom_toolbar_content
                    if app_settings.settings["toolbar_enabled"]
                    else None
                )
                prompt_message = f"\n{USER_PROMPT}You: {RESET_COLOR}"
                user_input = prompt_session.prompt(
                    ANSI(prompt_message),
                    bottom_toolbar=toolbar_content,
                    refresh_interval=0.5,
                ).strip()
                if not user_input:
                    sys.stdout.write("\x1b[1A\x1b[2K")
                    sys.stdout.flush()
                    continue
                if user_input.startswith("/"):
                    sys.stdout.write("\x1b[1A\x1b[2K")
                    sys.stdout.flush()
                    if self._handle_slash_command(user_input, prompt_session.history):
                        break

                    if self.session.state.ui_refresh_needed:
                        prompt_session.style = self._create_style_from_theme()
                        # Force prompt_toolkit to re-render so toolbar presence is recomputed
                        try:
                            get_app().invalidate()
                        except Exception as e:
                            log.warning("Could not invalidate prompt app: %s", e)
                        self.session.state.ui_refresh_needed = False
                    continue

                if self.session.take_turn(user_input, first_turn):
                    self._log_turn(log_filepath)

                first_turn = False
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted by user.")
        finally:
            print("\nSession ended.")
        self.session.cleanup(self.session_name, log_filepath)

    def _handle_slash_command(
        self, user_input: str, cli_history: InMemoryHistory
    ) -> bool:
        parts = user_input.strip().split()
        command_str, args = parts[0].lower(), parts[1:]
        handler = commands.COMMAND_MAP.get(command_str)
        if handler:
            if command_str == "/save":
                return handler(args, self.session, cli_history) or False
            result = handler(args, self.session)
            return result if isinstance(result, bool) else False
        print(
            f"{SYSTEM_MSG}--> Unknown command: {command_str}. Type /help.{RESET_COLOR}"
        )
        return False

    def _log_turn(self, log_filepath: Path) -> None:
        try:
            last_turn = self.session.state.history[-2:]
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": self.session.state.model,
                "prompt": last_turn[0],
                "response": last_turn[1],
            }
            with open(log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except (OSError, IndexError) as e:
            log.warning("Could not write to session log file: %s", e)


class MultiChatUI:
    """Manages the UI lifecycle and I/O for an interactive multi-chat session."""

    def __init__(
        self,
        session: MultiChatSession,
        session_name: str | None,
        initial_prompt: str | None,
    ):
        self.session = session
        self.session_name = session_name
        self.initial_prompt = initial_prompt

    def run(self) -> None:
        log_filename_base = (
            self.session_name
            or f"multichat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        log_filepath = config.CHATLOG_DIRECTORY / f"{log_filename_base}.jsonl"
        print(
            f"Starting interactive multi-chat. Primary: {self.session.primary_engine_name.capitalize()}. Log: {log_filepath.name}"
        )
        print("Type /help for commands or '/exit to end.")

        cli_history = InMemoryHistory(self.session.state.command_history)
        if self.initial_prompt:
            self.session.process_turn(
                self.initial_prompt, log_filepath, is_first_turn=True
            )

        try:
            while True:
                prompt_message = f"\n{DIRECTOR_PROMPT}Director> {RESET_COLOR}"
                user_input = prompt(ANSI(prompt_message), history=cli_history).strip()
                if not user_input:
                    sys.stdout.write("\x1b[1A\x1b[2K")
                    sys.stdout.flush()
                    continue
                if user_input.lstrip().startswith("/"):
                    sys.stdout.write("\x1b[1A\x1b[2K")
                    sys.stdout.flush()
                    if self._handle_slash_command(
                        user_input, cli_history, log_filepath
                    ):
                        break
                else:
                    self.session.process_turn(user_input, log_filepath)
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted.")
        finally:
            print("\nSession ended.")
            # Add cleanup logic here

    def _handle_slash_command(
        self, user_input: str, cli_history: InMemoryHistory, log_filepath: Path
    ) -> bool:
        parts = user_input.strip().split()
        command_str, args = parts[0].lower(), parts[1:]
        if command_str == "/ai":
            self.session.process_turn(user_input, log_filepath)
            return False
        handler = commands.MULTICHAT_COMMAND_MAP.get(command_str)
        if handler:
            return handler(args, self.session, cli_history) or False
        print(f"{SYSTEM_MSG}--> Unknown command: {command_str}.{RESET_COLOR}")
        return False
