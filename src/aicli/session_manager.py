# aicli/session_manager.py
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
the state and business logic of a chat session, independent of the UI.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
from dataclasses import asdict, fields
from pathlib import Path

from . import api_client, config, prompts, workflows
from . import personas as persona_manager
from . import settings as app_settings
from .engine import get_engine
from .logger import log
from .prompts import CONTINUATION_PROMPT
from .session_state import MultiChatSessionState, SessionState
from .utils.config_loader import get_default_model_for_engine
from .utils.file_processor import process_files, read_system_prompt
from .utils.formatters import (
    ASSISTANT_PROMPT,
    RESET_COLOR,
    SYSTEM_MSG,
    clean_ai_response_text,
    format_bytes,
    format_token_string,
    sanitize_filename,
)
from .utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
    extract_text_from_message,
    translate_history,
)


class SessionManager:
    """Encapsulates the state and logic for a single-chat session."""

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

        memory_content, attachments, image_data = process_files(
            all_files_to_process, memory_enabled, exclude_arg
        )

        initial_system_prompt = None
        if system_prompt_arg:
            initial_system_prompt = read_system_prompt(system_prompt_arg)
        elif persona and persona.system_prompt:
            initial_system_prompt = persona.system_prompt

        system_prompt_parts = []
        if initial_system_prompt:
            system_prompt_parts.append(initial_system_prompt)
        if memory_content:
            system_prompt_parts.append(f"--- PERSISTENT MEMORY ---\n{memory_content}")
        final_system_prompt = (
            "\n\n".join(system_prompt_parts) if system_prompt_parts else None
        )

        persona_attachments_set = {
            p for p in attachments if str(p) in persona_attachment_path_strs
        }

        self.state = SessionState(
            engine=engine_instance,
            model=final_model,
            system_prompt=final_system_prompt,
            initial_system_prompt=initial_system_prompt,
            current_persona=persona,
            max_tokens=max_tokens,
            memory_enabled=memory_enabled,
            attachments=attachments,
            persona_attachments=persona_attachments_set,
            attached_images=image_data,
            debug_active=debug_active,
            stream_active=stream_active,
        )
        self.image_workflow = workflows.ImageGenerationWorkflow(self)

    def _assemble_full_system_prompt(self) -> str | None:
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
        print(f"\n{SYSTEM_MSG}--> Condensing conversation history...{RESET_COLOR}")
        trim_count = config.HISTORY_SUMMARY_TRIM_TURNS * 2
        turns_to_summarize = self.state.history[:trim_count]
        remaining_history = self.state.history[trim_count:]

        log_content = "\n".join(
            f"{msg.get('role', 'unknown')}: {extract_text_from_message(msg)}"
            for msg in turns_to_summarize
        )
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

    def handle_single_shot(self, prompt: str) -> None:
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

    def take_turn(self, user_input: str, first_turn: bool) -> bool:
        """
        Handles user input. Returns True if a conversational turn occurred
        that should be logged, False otherwise.
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

    def set_custom_log_rename(self, name: str) -> None:
        self.state.custom_log_rename = name

    def set_force_quit(self) -> None:
        self.state.force_quit = True

    def toggle_stream(self) -> None:
        self.state.stream_active = not self.state.stream_active
        status = "ENABLED" if self.state.stream_active else "DISABLED"
        print(f"{SYSTEM_MSG}--> Response streaming is now {status}.{RESET_COLOR}")

    def toggle_debug(self) -> None:
        self.state.debug_active = not self.state.debug_active
        status = "ENABLED" if self.state.debug_active else "DISABLED"
        print(
            f"{SYSTEM_MSG}--> Session-specific debug logging is now {status}.{RESET_COLOR}"
        )

    def view_memory(self) -> None:
        try:
            content = config.PERSISTENT_MEMORY_FILE.read_text(encoding="utf-8")
            print(f"{SYSTEM_MSG}--- Persistent Memory ---{RESET_COLOR}\n{content}")
        except FileNotFoundError:
            print(f"{SYSTEM_MSG}--> Persistent memory is currently empty.{RESET_COLOR}")
        except OSError as e:
            print(f"{SYSTEM_MSG}--> Error reading memory file: {e}{RESET_COLOR}")

    def consolidate_memory(self) -> None:
        if not self.state.history:
            print(
                f"{SYSTEM_MSG}--> Nothing to consolidate; history is empty.{RESET_COLOR}"
            )
            return
        workflows.consolidate_memory(self)

    def inject_memory(self, fact: str) -> None:
        workflows.inject_memory(self, fact)

    def _read_memory_file(self) -> str:
        try:
            return config.PERSISTENT_MEMORY_FILE.read_text(encoding="utf-8")
        except (OSError, FileNotFoundError):
            return ""

    def _write_memory_file(self, content: str) -> None:
        try:
            config.PERSISTENT_MEMORY_FILE.write_text(content.strip(), encoding="utf-8")
        except OSError as e:
            log.error("Failed to write to persistent memory file: %s", e)

    def set_max_tokens(self, tokens: int) -> None:
        self.state.max_tokens = tokens
        print(f"{SYSTEM_MSG}--> Max tokens set to: {tokens}.{RESET_COLOR}")

    def clear_history(self) -> None:
        self.state.history.clear()
        self.state.total_prompt_tokens = 0
        self.state.total_completion_tokens = 0
        self.state.last_turn_tokens = {}
        print(
            f"{SYSTEM_MSG}--> Conversation history and token counts have been cleared.{RESET_COLOR}"
        )

    def set_model(self, model_name: str) -> None:
        self.state.model = model_name
        print(f"{SYSTEM_MSG}--> Model set to: {model_name}.{RESET_COLOR}")

    def switch_engine(self, new_engine_name: str | None = None) -> None:
        if not new_engine_name:
            new_engine_name = (
                "gemini" if self.state.engine.name == "openai" else "openai"
            )
        if new_engine_name not in ["openai", "gemini"]:
            print(f"{SYSTEM_MSG}--> Unknown engine: {new_engine_name}.{RESET_COLOR}")
            return
        try:
            api_key = api_client.check_api_keys(new_engine_name)
            self.state.history = translate_history(self.state.history, new_engine_name)
            self.state.engine = get_engine(new_engine_name, api_key)
            self.state.model = get_default_model_for_engine(new_engine_name)
            print(
                f"{SYSTEM_MSG}--> Engine switched to {self.state.engine.name.capitalize()}. Model set to {self.state.model}.{RESET_COLOR}"
            )
        except api_client.MissingApiKeyError as e:
            print(f"{SYSTEM_MSG}--> Switch failed: {e}{RESET_COLOR}")

    def view_history(self) -> None:
        print(json.dumps(self.state.history, indent=2))

    def view_state(self) -> None:
        print(f"{SYSTEM_MSG}--- Session State ---{RESET_COLOR}")
        persona_name = (
            self.state.current_persona.name if self.state.current_persona else "None"
        )
        print(f"  Active Persona: {persona_name}")
        print(f"  Engine: {self.state.engine.name}, Model: {self.state.model}")
        print(f"  Max Tokens: {self.state.max_tokens or 'Default'}")
        print(f"  Streaming: {'On' if self.state.stream_active else 'Off'}")
        print(f"  Memory on Exit: {'On' if self.state.memory_enabled else 'Off'}")
        print(f"  Debug Logging: {'On' if self.state.debug_active else 'Off'}")
        print(
            f"  Total Session I/O: {self.state.total_prompt_tokens}p / {self.state.total_completion_tokens}c"
        )
        print(f"  System Prompt: {'Active' if self.state.system_prompt else 'None'}")
        if self.state.attachments:
            total_size = sum(
                p.stat().st_size for p in self.state.attachments if p.exists()
            )
            print(
                f"  Attached Text Files: {len(self.state.attachments)} ({format_bytes(total_size)})"
            )
        if self.state.attached_images:
            print(f"  Attached Images: {len(self.state.attached_images)}")
        if self.image_workflow.img_prompt_crafting:
            print("  Image Crafting: ACTIVE")
            if self.image_workflow.img_prompt:
                print(f"  Current Prompt: {self.image_workflow.img_prompt[:50]}...")
        if self.image_workflow.last_img_prompt:
            print(f"  Last Image Prompt: {self.image_workflow.last_img_prompt[:50]}...")

    def save(self, args: list[str], command_history: list[str]) -> bool:
        should_remember, should_stay = "--remember" in args, "--stay" in args
        filename_parts = [arg for arg in args if arg not in ("--remember", "--stay")]
        filename = " ".join(filename_parts)
        if not filename:
            print(f"{SYSTEM_MSG}--> Generating descriptive name...{RESET_COLOR}")
            filename, _ = self._perform_helper_request(
                prompts.LOG_RENAMING_PROMPT.format(
                    log_content=self._get_history_for_helpers()
                ),
                50,
            )
            if not filename:
                print(
                    f"{SYSTEM_MSG}--> Could not auto-generate a name. Save cancelled.{RESET_COLOR}"
                )
                return False
            filename = sanitize_filename(filename)

        self.state.command_history = command_history
        if self._save_session_to_file(filename):
            if not should_remember:
                self.state.exit_without_memory = True
            return not should_stay
        return False

    def load(self, filename: str) -> bool:
        if not filename.endswith(".json"):
            filename += ".json"
        filepath = config.SESSIONS_DIRECTORY / filename
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            if "engine_name" not in data or "model" not in data:
                log.warning(
                    "Loaded session file %s is missing required keys.", filepath
                )
                print(f"{SYSTEM_MSG}--> Invalid session file.{RESET_COLOR}")
                return False

            if data.get("session_type") == "multichat":
                log.warning(
                    "Attempted to load a multi-chat session in a single-chat context."
                )
                print(
                    f"{SYSTEM_MSG}--> Cannot load a multi-chat session here.{RESET_COLOR}"
                )
                return False

            engine_name = data.pop("engine_name")
            api_key = api_client.check_api_keys(engine_name)
            data["engine"] = get_engine(engine_name, api_key)

            if "attachments" in data:
                data["attachments"] = {
                    Path(k): v for k, v in data["attachments"].items()
                }

            if "persona_attachments" in data:
                data["persona_attachments"] = {
                    Path(p) for p in data["persona_attachments"]
                }

            persona_filename = data.get("current_persona")
            data["current_persona"] = (
                persona_manager.load_persona(persona_filename)
                if persona_filename and isinstance(persona_filename, str)
                else None
            )

            data["total_prompt_tokens"] = data.get("total_prompt_tokens", 0)
            data["total_completion_tokens"] = data.get("total_completion_tokens", 0)
            data["last_turn_tokens"] = data.get("last_turn_tokens", {})
            data["ui_refresh_needed"] = data.get("ui_refresh_needed", False)

            state_field_names = {f.name for f in fields(SessionState)}
            filtered_data = {k: v for k, v in data.items() if k in state_field_names}

            self.state = SessionState(**filtered_data)
            print(
                f"{SYSTEM_MSG}--> Session '{filepath.name}' loaded successfully.{RESET_COLOR}"
            )
            return True
        except (
            OSError,
            json.JSONDecodeError,
            api_client.MissingApiKeyError,
            TypeError,
        ) as e:
            log.error("Error in SessionManager.load: %s", e)
            print(f"{SYSTEM_MSG}--> Error loading session: {e}{RESET_COLOR}")
            return False

    def list_personas(self) -> None:
        personas = persona_manager.list_personas()
        if not personas:
            print(f"{SYSTEM_MSG}--> No personas found.{RESET_COLOR}")
            return
        print(f"{SYSTEM_MSG}--- Available Personas ---{RESET_COLOR}")
        for p in personas:
            print(f"  - {p.filename.replace('.json', '')}: {p.description}")

    def refresh_files(self, search_term: str | None) -> None:
        if not self.state.attachments:
            print(f"{SYSTEM_MSG}--> No files attached to refresh.{RESET_COLOR}")
            return

        paths_to_refresh = [
            p
            for p in self.state.attachments
            if not search_term or search_term in p.name
        ]
        if not paths_to_refresh:
            print(f"{SYSTEM_MSG}--> No files matching '{search_term}'.{RESET_COLOR}")
            return

        updated, removed = [], []
        for path in paths_to_refresh:
            try:
                self.state.attachments[path] = path.read_text(
                    encoding="utf-8", errors="ignore"
                )
                updated.append(path.name)
            except (OSError, FileNotFoundError):
                removed.append(path.name)
                del self.state.attachments[path]

        if updated:
            print(f"{SYSTEM_MSG}--> Refreshed: {', '.join(updated)}{RESET_COLOR}")
        if removed:
            print(
                f"{SYSTEM_MSG}--> Removed (not found): {', '.join(removed)}{RESET_COLOR}"
            )

    def list_files(self) -> None:
        if not self.state.attachments:
            print(f"{SYSTEM_MSG}--> No text files are attached.{RESET_COLOR}")
            return

        file_list = sorted(
            [
                (p, p.stat().st_size if p.exists() else 0)
                for p in self.state.attachments
            ],
            key=lambda i: i[1],
            reverse=True,
        )
        print(f"{SYSTEM_MSG}--- Attached Files ---{RESET_COLOR}")
        for path, size in file_list:
            print(f"  - {path.name} ({format_bytes(size)})")

    def attach_file(self, path_str: str) -> None:
        path = Path(path_str).resolve()
        if not path.exists():
            print(f"{SYSTEM_MSG}--> Error: Path not found: {path_str}{RESET_COLOR}")
            return
        if path in self.state.attachments:
            print(
                f"{SYSTEM_MSG}--> Error: File '{path.name}' is already attached.{RESET_COLOR}"
            )
            return

        _, new_attachments, _ = process_files([str(path)], False, [])
        if new_attachments:
            self.state.attachments.update(new_attachments)
            print(f"{SYSTEM_MSG}--> Attached content from: {path.name}{RESET_COLOR}")
        else:
            print(
                f"{SYSTEM_MSG}--> No readable text content found at path: {path.name}{RESET_COLOR}"
            )

    def detach_file(self, filename: str) -> None:
        path_to_remove = next(
            (p for p in self.state.attachments if p.name == filename), None
        )
        if path_to_remove:
            del self.state.attachments[path_to_remove]
            # Also remove from persona tracking if it was a persona file
            self.state.persona_attachments.discard(path_to_remove)
            print(f"{SYSTEM_MSG}--> Detached file: {path_to_remove.name}{RESET_COLOR}")
        else:
            print(f"{SYSTEM_MSG}--> No attached file named '{filename}'.{RESET_COLOR}")

    def switch_persona(self, name: str) -> None:
        # Remove attachments from the old persona, if any
        for path in self.state.persona_attachments:
            self.state.attachments.pop(path, None)
        self.state.persona_attachments.clear()

        if name.lower() == "clear":
            if not self.state.current_persona:
                print(f"{SYSTEM_MSG}--> No active persona to clear.{RESET_COLOR}")
                return
            self.state.system_prompt = self.state.initial_system_prompt
            self.state.current_persona = None
            print(f"{SYSTEM_MSG}--> Persona cleared.{RESET_COLOR}")
            self.state.history.append(
                construct_user_message(
                    self.state.engine.name, "[SYSTEM] Persona cleared.", []
                )
            )
            return

        new_persona = persona_manager.load_persona(name)
        if not new_persona:
            print(f"{SYSTEM_MSG}--> Persona '{name}' not found.{RESET_COLOR}")
            return

        if new_persona.engine and new_persona.engine != self.state.engine.name:
            print(
                f"{SYSTEM_MSG}--> Switching engine to {new_persona.engine} for persona '{new_persona.name}'...{RESET_COLOR}"
            )
            self.switch_engine(new_persona.engine)

        if new_persona.model:
            self.state.model = new_persona.model
        if new_persona.max_tokens is not None:
            self.state.max_tokens = new_persona.max_tokens
        if new_persona.stream is not None:
            self.state.stream_active = new_persona.stream

        # Add attachments from the new persona
        if new_persona.attachments:
            _, new_attachments, _ = process_files(
                new_persona.attachments, use_memory=False, exclusions=[]
            )
            self.state.attachments.update(new_attachments)
            self.state.persona_attachments.update(new_attachments.keys())
            print(
                f"{SYSTEM_MSG}--> Attached {len(new_attachments)} file(s) from persona.{RESET_COLOR}"
            )

        self.state.system_prompt = new_persona.system_prompt
        self.state.current_persona = new_persona
        print(f"{SYSTEM_MSG}--> Switched to persona: '{new_persona.name}'{RESET_COLOR}")
        self.state.history.append(
            construct_user_message(
                self.state.engine.name,
                f"[SYSTEM] Persona switched to '{new_persona.name}'.",
                [],
            )
        )

    def handle_image_command(self, args: list[str]) -> None:
        """Delegates image command handling to the dedicated workflow."""
        self.image_workflow.run(args)

    def _get_history_for_helpers(self) -> str:
        return "\n".join(
            f"{msg.get('role', 'unknown')}: {extract_text_from_message(msg)}"
            for msg in self.state.history
        )

    def _save_session_to_file(self, filename: str) -> bool:
        safe_name = sanitize_filename(filename.rsplit(".", 1)[0]) + ".json"
        filepath = config.SESSIONS_DIRECTORY / safe_name

        state_dict = asdict(self.state)
        state_dict["engine_name"] = self.state.engine.name
        del state_dict["engine"]
        state_dict["attachments"] = {
            str(k): v for k, v in self.state.attachments.items()
        }
        state_dict["persona_attachments"] = [
            str(p) for p in self.state.persona_attachments
        ]
        state_dict["current_persona"] = (
            self.state.current_persona.filename if self.state.current_persona else None
        )

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
            print(f"{SYSTEM_MSG}--> Session saved to: {filepath}{RESET_COLOR}")
            return True
        except (OSError, TypeError) as e:
            log.error("Failed to save session state: %s", e)
            print(f"{SYSTEM_MSG}--> Error saving session: {e}{RESET_COLOR}")
            return False

    def _rename_log_file(self, old_path: Path, new_name_base: str) -> None:
        new_path = old_path.with_name(f"{sanitize_filename(new_name_base)}.jsonl")
        try:
            old_path.rename(new_path)
            print(
                f"{SYSTEM_MSG}--> Session log renamed to: {new_path.name}{RESET_COLOR}"
            )
        except OSError as e:
            log.error("Failed to rename session log: %s", e)

    def _save_debug_log(self, log_filename_base: str) -> None:
        debug_filepath = config.LOG_DIRECTORY / f"debug_{log_filename_base}.jsonl"
        print(f"Saving debug log to: {debug_filepath}")
        try:
            with open(debug_filepath, "w", encoding="utf-8") as f:
                for entry in self.state.session_raw_logs:
                    f.write(json.dumps(entry) + "\n")
        except OSError as e:
            log.error("Could not save debug log file: %s", e)


class MultiChatSession:
    """Manages the state and logic for an interactive multi-chat session."""

    def __init__(
        self,
        initial_state: MultiChatSessionState,
    ):
        self.state = initial_state
        self.primary_engine_name = app_settings.settings["default_engine"]
        self.engines = {
            "openai": self.state.openai_engine,
            "gemini": self.state.gemini_engine,
        }
        self.models = {
            "openai": self.state.openai_model,
            "gemini": self.state.gemini_model,
        }
        self.engine_aliases = {
            "gpt": "openai",
            "openai": "openai",
            "gem": "gemini",
            "gemini": "gemini",
        }

    def _secondary_worker(self, engine, model, history, system_prompt, result_queue):
        srl_list = self.state.session_raw_logs if self.state.debug_active else None
        try:
            text, _ = api_client.perform_chat_request(
                engine,
                model,
                history,
                system_prompt,
                self.state.max_tokens,
                stream=True,
                print_stream=False,
                session_raw_logs=srl_list,
            )
            cleaned = clean_ai_response_text(engine.name, text)
            result_queue.put({"engine_name": engine.name, "text": cleaned})
        except Exception as e:
            log.error("Secondary worker failed: %s", e)
            result_queue.put({"engine_name": engine.name, "text": f"Error: {e}"})

    def process_turn(
        self, prompt_text: str, log_filepath: Path, is_first_turn: bool = False
    ) -> None:
        srl_list = self.state.session_raw_logs if self.state.debug_active else None
        if prompt_text.lower().lstrip().startswith("/ai"):
            parts = prompt_text.lstrip().split(" ", 2)
            if len(parts) < 2 or parts[1].lower() not in self.engine_aliases:
                print(f"{SYSTEM_MSG}--> Usage: /ai <gpt|gem> [prompt]{RESET_COLOR}")
                return
            target_engine_name = self.engine_aliases[parts[1].lower()]
            target_prompt = parts[2] if len(parts) > 2 else CONTINUATION_PROMPT
            user_msg_text = (
                f"Director to {target_engine_name.capitalize()}: {target_prompt}"
            )
            print(f"\n{SYSTEM_MSG}{user_msg_text}{RESET_COLOR}")
            user_msg = construct_user_message(
                target_engine_name,
                user_msg_text,
                self.state.initial_image_data if is_first_turn else [],
            )
            current_history = translate_history(
                self.state.shared_history + [user_msg], target_engine_name
            )
            engine, model = (
                self.engines[target_engine_name],
                self.models[target_engine_name],
            )
            print(
                f"\n{ASSISTANT_PROMPT}[{engine.name.capitalize()}]: {RESET_COLOR}",
                end="",
                flush=True,
            )
            raw_response, _ = api_client.perform_chat_request(
                engine,
                model,
                current_history,
                self.state.system_prompts[target_engine_name],
                self.state.max_tokens,
                stream=True,
                session_raw_logs=srl_list,
            )
            print()
            cleaned = clean_ai_response_text(engine.name, raw_response)
            asst_msg_text = f"[{engine.name.capitalize()}]: {cleaned}"
            asst_msg = construct_assistant_message("openai", asst_msg_text)
            self.state.shared_history.extend(
                [construct_user_message("openai", user_msg_text, []), asst_msg]
            )
            self._log_multichat_turn(log_filepath, self.state.shared_history[-2:])
        else:
            primary_engine = self.engines[self.primary_engine_name]
            secondary_engine = self.engines[
                "gemini" if self.primary_engine_name == "openai" else "openai"
            ]
            user_msg_text = f"Director to All: {prompt_text}"
            user_msg = construct_user_message(
                "openai",
                user_msg_text,
                self.state.initial_image_data if is_first_turn else [],
            )
            result_queue = queue.Queue()
            history_primary = translate_history(
                self.state.shared_history + [user_msg], primary_engine.name
            )
            history_secondary = translate_history(
                self.state.shared_history + [user_msg], secondary_engine.name
            )

            thread = threading.Thread(
                target=self._secondary_worker,
                args=(
                    secondary_engine,
                    self.models[secondary_engine.name],
                    history_secondary,
                    self.state.system_prompts[secondary_engine.name],
                    result_queue,
                ),
            )
            thread.start()

            print(
                f"\n{ASSISTANT_PROMPT}[{primary_engine.name.capitalize()}]: {RESET_COLOR}",
                end="",
                flush=True,
            )
            primary_raw, _ = api_client.perform_chat_request(
                primary_engine,
                self.models[primary_engine.name],
                history_primary,
                self.state.system_prompts[primary_engine.name],
                self.state.max_tokens,
                stream=True,
                session_raw_logs=srl_list,
            )
            print("\n")
            thread.join()
            secondary_result = result_queue.get()
            print(
                f"{ASSISTANT_PROMPT}[{secondary_result['engine_name'].capitalize()}]: {RESET_COLOR}{secondary_result['text']}"
            )

            primary_cleaned = clean_ai_response_text(primary_engine.name, primary_raw)
            primary_msg = construct_assistant_message(
                "openai", f"[{primary_engine.name.capitalize()}]: {primary_cleaned}"
            )
            secondary_msg = construct_assistant_message(
                "openai",
                f"[{secondary_result['engine_name'].capitalize()}]: {secondary_result['text']}",
            )

            first, second = (
                (primary_msg, secondary_msg)
                if self.primary_engine_name == "openai"
                else (secondary_msg, primary_msg)
            )
            self.state.shared_history.extend([user_msg, first, second])
            self._log_multichat_turn(log_filepath, self.state.shared_history[-3:])

    def _log_multichat_turn(
        self, log_filepath: Path, history_slice: list[dict]
    ) -> None:
        try:
            with open(log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps({"history_slice": history_slice}) + "\n")
        except OSError as e:
            log.warning("Could not write to multi-chat session log file: %s", e)
