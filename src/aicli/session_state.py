# aicli/session_state.py
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
This module defines the data structures (dataclasses) that hold the state
for single and multi-chat sessions. It contains no business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import personas as persona_manager
from .engine import AIEngine


@dataclass
class SessionState:
    """A dataclass to hold the state of an interactive chat session."""

    engine: AIEngine
    model: str
    system_prompt: str | None
    initial_system_prompt: str | None
    current_persona: persona_manager.Persona | None
    max_tokens: int | None
    memory_enabled: bool
    attachments: dict[Path, str] = field(default_factory=dict)
    attached_images: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    command_history: list[str] = field(default_factory=list)
    debug_active: bool = False
    stream_active: bool = True
    session_raw_logs: list[dict[str, Any]] = field(default_factory=list)
    exit_without_memory: bool = False
    force_quit: bool = False
    custom_log_rename: str | None = None
    # Token tracking for the session toolbar
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    last_turn_tokens: dict[str, Any] = field(default_factory=dict)
    # Flag to signal the UI loop to refresh styles
    ui_refresh_needed: bool = False


@dataclass
class MultiChatSessionState:
    """Holds the state for a multi-chat session."""

    openai_engine: AIEngine
    gemini_engine: AIEngine
    openai_model: str
    gemini_model: str
    max_tokens: int
    system_prompts: dict[str, str] = field(default_factory=dict)
    initial_image_data: list[dict[str, Any]] = field(default_factory=list)
    shared_history: list[dict[str, Any]] = field(default_factory=list)
    command_history: list[str] = field(default_factory=list)
    debug_active: bool = False
    session_raw_logs: list[dict[str, Any]] = field(default_factory=list)
    exit_without_memory: bool = False
    force_quit: bool = False
    custom_log_rename: str | None = None
