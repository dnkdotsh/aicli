# aicli/managers/session_manager.py
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
This module contains the SessionManager, the central controller for managing
the state and orchestration of a chat session, independent of the UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import api_client, config, prompts, workflows
from .. import personas as persona_manager
from .. import settings as app_settings
from ..engine import get_engine
from ..logger import log
from ..session_state import SessionState
from ..utils.config_loader import get_default_model_for_engine
from ..utils.file_processor import read_system_prompt
from ..utils.formatters import (
    ASSISTANT_PROMPT,
    RESET_COLOR,
    SYSTEM_MSG,
    format_token_string,
    sanitize_filename,
)
from ..utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
    extract_text_from_message,
)
from .context_manager import ContextManager


class SessionManager:
    """Encapsulates the state and orchestration logic for a single-chat session."""

    def __init__(
        self,
        engine_name: str,
        model: str | None = None,
        max_tokens: int | None = None,
        stream_active: bool = True,
        memory_enabled: bool = True,
        debug_active: bool = False,
        persona: persona_manager.Persona | None = None,
        system_prompt_arg: str | None = None,
        files_arg: list[str] | None = None,
        exclude_arg: list[str] | None = None,
    ):
        api_key = api_client.check_api_keys(engine_name)
        engine_instance = get_engine(engine_name, api_key)
        final_model = model or get_default_model_for_engine(engine_name)

        persona_attachment_path_strs = set(persona.attachments if persona else [])
        all_files_to_process = list(files_arg or [])
        all_files_to_process.extend(persona_attachment_path_strs)

        self.context_manager = ContextManager(
            files_arg=all_files_to_process,
            memory_enabled=memory_enabled,
            exclude_arg=exclude_arg,
        )

        initial_system_prompt = None
        if system_prompt_arg:
            initial_system_prompt = read_system_prompt(system_prompt_arg)
        elif persona and persona.system_prompt:
            initial_system_prompt = persona.system_prompt

        system_prompt_parts = []
        if initial_system_prompt:
            system_prompt_parts.append(initial_system_prompt)
        if self.context_manager.memory_content:
            system_prompt_parts.append(
                f"--- PERSISTENT MEMORY ---\n{self.context_manager.memory_content}"
            )
        final_system_prompt = (
            "\n\n".join(system_prompt_parts) if system_prompt_parts else None
        )

        persona_attachments_set = {
            p
            for p in self.context_manager.attachments
            if str(p) in persona_attachment_path_strs
        }

        self.state = SessionState(
            engine=engine_instance,
            model=final_model,
            system_prompt=final_system_prompt,
            initial_system_prompt=initial_system_prompt,
            current_persona=persona,
            max_tokens=max_tokens,
            memory_enabled=memory_enabled,
            attachments=self.context_manager.attachments,
            persona_attachments=persona_attachments_set,
            attached_images=self.context_manager.image_data,
            debug_active=debug_active,
            stream_active=stream_active,
        )
        self.image_workflow = workflows.ImageGenerationWorkflow(self)

    # --- Core Orchestration Methods ---

    def take_turn(self, user_input: str, first_turn: bool) -> bool:
        """
        Handles user input for a conversational turn. Returns True if a turn
        occurred that should be logged, False otherwise.
        """
        if self.image_workflow.img_prompt_crafting:
            should_generate, final_prompt = self.image_workflow.process_prompt_input(
                user_input
            )
            if should_generate and final_prompt:
                self.image_workflow._generate_image_from_session(final_prompt)
            return False

        user_msg = construct_user_message(
            self.state.engine.name,
            user_input,
            self.state.attached_images if first_turn else [],
        )
        messages = list(self.state.history) + [user_msg]
        srl_list = self.state.session_raw_logs if self.state.debug_active else None

        if self.state.stream_active:
            print(
                f"\n{ASSISTANT_PROMPT}Assistant: {RESET_COLOR}",
                end="",
                flush=True,
            )

        response, tokens = api_client.perform_chat_request(
            engine=self.state.engine,
            model=self.state.model,
            messages_or_contents=messages,
            system_prompt=self._assemble_full_system_prompt(),
            max_tokens=self.state.max_tokens,
            stream=self.state.stream_active,
            session_raw_logs=srl_list,
        )

        if not self.state.stream_active:
            print(
                f"\n{ASSISTANT_PROMPT}Assistant: {RESET_COLOR}{response}",
                end="",
            )

        self.state.last_turn_tokens = tokens
        self.state.total_prompt_tokens += tokens.get("prompt", 0)
        self.state.total_completion_tokens += tokens.get("completion", 0)

        if self.state.stream_active or not response.endswith("\n"):
            print()

        asst_msg = construct_assistant_message(self.state.engine.name, response)
        self.state.history.extend([user_msg, asst_msg])

        if len(self.state.history) >= config.HISTORY_SUMMARY_THRESHOLD_TURNS * 2:
            self._condense_chat_history()

        return True

    def cleanup(self, session_name: str | None, log_filepath: Path) -> None:
        """Performs cleanup tasks at the end of a session."""
        if (
            not self.state.force_quit
            and not self.state.exit_without_memory
            and log_filepath.exists()
            and self.state.history
        ):
            if self.state.memory_enabled:
                workflows.consolidate_memory(self)
            else:
                print(
                    f"{SYSTEM_MSG}--> Persistent memory not enabled, skipping update.{RESET_COLOR}"
                )

            if self.state.custom_log_rename:
                self._rename_log_file(log_filepath, self.state.custom_log_rename)
            elif not session_name:
                workflows.rename_log_with_ai(self, log_filepath)

        if self.state.debug_active:
            self._save_debug_log(log_filepath.stem)

    def handle_single_shot(self, prompt: str) -> None:
        """Executes a single, non-interactive chat request."""
        messages = [
            construct_user_message(
                self.state.engine.name, prompt, self.state.attached_images
            )
        ]
        response, tokens = api_client.perform_chat_request(
            self.state.engine,
            self.state.model,
            messages,
            self._assemble_full_system_prompt(),
            self.state.max_tokens,
            self.state.stream_active,
        )
        if not self.state.stream_active:
            print(response, end="")

        if not response.endswith("\n"):
            print()

        token_str = format_token_string(tokens)
        if token_str:
            print(f"{SYSTEM_MSG}{token_str}{RESET_COLOR}", file=sys.stderr)

    # --- Internal Helper Methods ---

    def _assemble_full_system_prompt(self) -> str | None:
        """Constructs the complete system prompt from state."""
        prompt_parts = []
        if self.state.system_prompt:
            prompt_parts.append(self.state.system_prompt)

        if self.state.attachments:
            attachment_texts = [
                f"--- FILE: {path.as_posix()} ---\n{content}"
                for path, content in self.state.attachments.items()
            ]
            prompt_parts.append(
                "--- ATTACHED FILES ---\n" + "\n\n".join(attachment_texts)
            )
        return "\n\n".join(prompt_parts) if prompt_parts else None

    def _condense_chat_history(self) -> None:
        """Summarizes the beginning of a long chat history to save tokens."""
        print(f"\n{SYSTEM_MSG}--> Condensing conversation history...{RESET_COLOR}")
        trim_count = config.HISTORY_SUMMARY_TRIM_TURNS * 2
        turns_to_summarize = self.state.history[:trim_count]
        remaining_history = self.state.history[trim_count:]

        log_content = self._get_history_for_helpers(turns_to_summarize)
        summary_prompt = prompts.HISTORY_SUMMARY_PROMPT.format(log_content=log_content)
        summary_text, _ = self._perform_helper_request(
            summary_prompt, app_settings.settings["summary_max_tokens"]
        )

        if summary_text:
            summary_message = construct_user_message(
                self.state.engine.name,
                f"[PREVIOUSLY DISCUSSED]:\n{summary_text.strip()}",
                [],
            )
            self.state.history = [summary_message] + remaining_history
            print(f"{SYSTEM_MSG}--> History condensed successfully.{RESET_COLOR}")

    def _perform_helper_request(
        self, prompt_text: str, max_tokens: int | None
    ) -> tuple[str | None, dict]:
        """Executes a request to a helper model for internal tasks."""
        helper_model_key = f"helper_model_{self.state.engine.name}"
        task_model = app_settings.settings[helper_model_key]
        messages = [construct_user_message(self.state.engine.name, prompt_text, [])]
        response, tokens = api_client.perform_chat_request(
            engine=self.state.engine,
            model=task_model,
            messages_or_contents=messages,
            system_prompt=None,
            max_tokens=max_tokens,
            stream=False,
        )
        if response and not response.startswith("API Error:"):
            return response, tokens
        log.warning("Helper request failed. Reason: %s", response or "No response")
        return None, {}

    def _get_history_for_helpers(
        self, history_override: list[dict] | None = None
    ) -> str:
        """Formats chat history into a simple string for helper model prompts."""
        history = (
            history_override if history_override is not None else self.state.history
        )
        return "\n".join(
            f"{msg.get('role', 'unknown')}: {extract_text_from_message(msg)}"
            for msg in history
        )

    def _rename_log_file(self, old_path: Path, new_name_base: str) -> None:
        """Renames a log file with a sanitized name."""
        new_path = old_path.with_name(f"{sanitize_filename(new_name_base)}.jsonl")
        try:
            old_path.rename(new_path)
            print(
                f"{SYSTEM_MSG}--> Session log renamed to: {new_path.name}{RESET_COLOR}"
            )
        except OSError as e:
            log.error("Failed to rename session log: %s", e)

    def _save_debug_log(self, log_filename_base: str) -> None:
        """Saves the raw API log if debug mode is active."""
        debug_filepath = config.LOG_DIRECTORY / f"debug_{log_filename_base}.jsonl"
        print(f"Saving debug log to: {debug_filepath}")
        try:
            with open(debug_filepath, "w", encoding="utf-8") as f:
                for entry in self.state.session_raw_logs:
                    f.write(json.dumps(entry) + "\n")
        except OSError as e:
            log.error("Could not save debug log file: %s", e)
