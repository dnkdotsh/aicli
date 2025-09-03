# aicli/session_manager.py
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

from __future__ import annotations

import datetime
import json
import queue
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory

from . import api_client, commands, config, utils
from . import personas as persona_manager
from . import settings as app_settings
from .engine import AIEngine, get_engine
from .logger import log
from .prompts import (
    CONTINUATION_PROMPT,
    DIRECT_MEMORY_INJECTION_PROMPT,
    HISTORY_SUMMARY_PROMPT,
    LOG_RENAMING_PROMPT,
    MEMORY_INTEGRATION_PROMPT,
)


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
        final_model = model or utils.get_default_model_for_engine(engine_name)

        memory_content, attachments, image_data = utils.process_files(
            files_arg, memory_enabled, exclude_arg
        )

        initial_system_prompt = None
        if system_prompt_arg:
            initial_system_prompt = utils.read_system_prompt(system_prompt_arg)
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

        self.state = SessionState(
            engine=engine_instance,
            model=final_model,
            system_prompt=final_system_prompt,
            initial_system_prompt=initial_system_prompt,
            current_persona=persona,
            max_tokens=max_tokens,
            memory_enabled=memory_enabled,
            attachments=attachments,
            attached_images=image_data,
            debug_active=debug_active,
            stream_active=stream_active,
        )

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
        print(
            f"\n{utils.SYSTEM_MSG}--> Condensing conversation history...{utils.RESET_COLOR}"
        )
        trim_count = config.HISTORY_SUMMARY_TRIM_TURNS * 2
        turns_to_summarize = self.state.history[:trim_count]
        remaining_history = self.state.history[trim_count:]

        log_content = "\n".join(
            f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}"
            for msg in turns_to_summarize
        )
        summary_prompt = HISTORY_SUMMARY_PROMPT.format(log_content=log_content)
        summary_text, _ = self._perform_helper_request(
            summary_prompt, app_settings.settings["summary_max_tokens"]
        )

        if summary_text:
            summary_message = utils.construct_user_message(
                self.state.engine.name,
                f"[PREVIOUSLY DISCUSSED]:\n{summary_text.strip()}",
                [],
            )
            self.state.history = [summary_message] + remaining_history
            print(
                f"{utils.SYSTEM_MSG}--> History condensed successfully.{utils.RESET_COLOR}"
            )

    def _perform_helper_request(
        self, prompt_text: str, max_tokens: int | None
    ) -> tuple[str | None, dict]:
        helper_model_key = f"helper_model_{self.state.engine.name}"
        task_model = app_settings.settings[helper_model_key]
        messages = [
            utils.construct_user_message(self.state.engine.name, prompt_text, [])
        ]
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
            utils.construct_user_message(
                self.state.engine.name, prompt, self.state.attached_images
            )
        ]
        if self.state.stream_active:
            print(
                f"{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}",
                end="",
                flush=True,
            )
        response, tokens = api_client.perform_chat_request(
            self.state.engine,
            self.state.model,
            messages,
            self._assemble_full_system_prompt(),
            self.state.max_tokens,
            self.state.stream_active,
        )
        if not self.state.stream_active:
            print(
                f"{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{response}",
                end="",
            )
        print(utils.format_token_string(tokens))

    def take_turn(self, user_input: str, first_turn: bool) -> None:
        user_msg = utils.construct_user_message(
            self.state.engine.name,
            user_input,
            self.state.attached_images if first_turn else [],
        )
        messages = list(self.state.history) + [user_msg]
        srl_list = self.state.session_raw_logs if self.state.debug_active else None

        if self.state.stream_active:
            print(
                f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}",
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
                f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{response}",
                end="",
            )

        print(utils.format_token_string(tokens))
        asst_msg = utils.construct_assistant_message(self.state.engine.name, response)
        self.state.history.extend([user_msg, asst_msg])

        if len(self.state.history) >= config.HISTORY_SUMMARY_THRESHOLD_TURNS * 2:
            self._condense_chat_history()

    def cleanup(self, session_name: str | None, log_filepath: Path) -> None:
        if (
            not self.state.force_quit
            and not self.state.exit_without_memory
            and log_filepath.exists()
            and self.state.history
        ):
            if self.state.memory_enabled:
                self._consolidate_memory_from_session()
            else:
                print(
                    f"{utils.SYSTEM_MSG}--> Persistent memory not enabled, skipping update.{utils.RESET_COLOR}"
                )

            if self.state.custom_log_rename:
                self._rename_log_file(log_filepath, self.state.custom_log_rename)
            elif not session_name:
                self._rename_log_with_ai(log_filepath)

        if self.state.debug_active:
            self._save_debug_log(log_filepath.stem)

    def set_custom_log_rename(self, name: str) -> None:
        self.state.custom_log_rename = name

    def set_force_quit(self) -> None:
        self.state.force_quit = True

    def toggle_stream(self) -> None:
        self.state.stream_active = not self.state.stream_active
        status = "ENABLED" if self.state.stream_active else "DISABLED"
        print(
            f"{utils.SYSTEM_MSG}--> Response streaming is now {status}.{utils.RESET_COLOR}"
        )

    def toggle_debug(self) -> None:
        self.state.debug_active = not self.state.debug_active
        status = "ENABLED" if self.state.debug_active else "DISABLED"
        print(
            f"{utils.SYSTEM_MSG}--> Session-specific debug logging is now {status}.{utils.RESET_COLOR}"
        )

    def view_memory(self) -> None:
        try:
            content = config.PERSISTENT_MEMORY_FILE.read_text(encoding="utf-8")
            print(
                f"{utils.SYSTEM_MSG}--- Persistent Memory ---{utils.RESET_COLOR}\n{content}"
            )
        except FileNotFoundError:
            print(
                f"{utils.SYSTEM_MSG}--> Persistent memory is currently empty.{utils.RESET_COLOR}"
            )
        except OSError as e:
            print(
                f"{utils.SYSTEM_MSG}--> Error reading memory file: {e}{utils.RESET_COLOR}"
            )

    def consolidate_memory(self) -> None:
        if not self.state.history:
            print(
                f"{utils.SYSTEM_MSG}--> Nothing to consolidate; history is empty.{utils.RESET_COLOR}"
            )
            return
        self._consolidate_memory_from_session()

    def inject_memory(self, fact: str) -> None:
        print(
            f"{utils.SYSTEM_MSG}--> Injecting fact into persistent memory...{utils.RESET_COLOR}"
        )
        prompt_text = DIRECT_MEMORY_INJECTION_PROMPT.format(
            existing_ltm=self._read_memory_file(), new_fact=fact
        )
        updated_memory, _ = self._perform_helper_request(prompt_text, None)
        if updated_memory:
            self._write_memory_file(updated_memory)
            print(
                f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}"
            )

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

    def _consolidate_memory_from_session(self) -> None:
        print(
            f"{utils.SYSTEM_MSG}--> Updating persistent memory with session content...{utils.RESET_COLOR}"
        )
        prompt_text = MEMORY_INTEGRATION_PROMPT.format(
            existing_ltm=self._read_memory_file(),
            session_content=self._get_history_for_helpers(),
        )
        updated_memory, _ = self._perform_helper_request(prompt_text, None)
        if updated_memory:
            self._write_memory_file(updated_memory)
            print(
                f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}"
            )

    def set_max_tokens(self, tokens: int) -> None:
        self.state.max_tokens = tokens
        print(f"{utils.SYSTEM_MSG}--> Max tokens set to: {tokens}.{utils.RESET_COLOR}")

    def clear_history(self) -> None:
        self.state.history.clear()
        print(
            f"{utils.SYSTEM_MSG}--> Conversation history has been cleared.{utils.RESET_COLOR}"
        )

    def set_model(self, model_name: str) -> None:
        self.state.model = model_name
        print(f"{utils.SYSTEM_MSG}--> Model set to: {model_name}{utils.RESET_COLOR}")

    def switch_engine(self, new_engine_name: str | None = None) -> None:
        if not new_engine_name:
            new_engine_name = (
                "gemini" if self.state.engine.name == "openai" else "openai"
            )
        if new_engine_name not in ["openai", "gemini"]:
            print(
                f"{utils.SYSTEM_MSG}--> Unknown engine: {new_engine_name}.{utils.RESET_COLOR}"
            )
            return
        try:
            api_key = api_client.check_api_keys(new_engine_name)
            self.state.history = utils.translate_history(
                self.state.history, new_engine_name
            )
            self.state.engine = get_engine(new_engine_name, api_key)
            self.state.model = utils.get_default_model_for_engine(new_engine_name)
            print(
                f"{utils.SYSTEM_MSG}--> Engine switched to {self.state.engine.name.capitalize()}. Model set to {self.state.model}.{utils.RESET_COLOR}"
            )
        except api_client.MissingApiKeyError as e:
            print(f"{utils.SYSTEM_MSG}--> Switch failed: {e}{utils.RESET_COLOR}")

    def view_history(self) -> None:
        print(json.dumps(self.state.history, indent=2))

    def view_state(self) -> None:
        print(f"{utils.SYSTEM_MSG}--- Session State ---{utils.RESET_COLOR}")
        persona_name = (
            self.state.current_persona.name if self.state.current_persona else "None"
        )
        print(f"  Active Persona: {persona_name}")
        print(f"  Engine: {self.state.engine.name}, Model: {self.state.model}")
        print(f"  Max Tokens: {self.state.max_tokens or 'Default'}")
        print(f"  Streaming: {'On' if self.state.stream_active else 'Off'}")
        print(f"  Memory on Exit: {'On' if self.state.memory_enabled else 'Off'}")
        print(f"  Debug Logging: {'On' if self.state.debug_active else 'Off'}")
        print(f"  System Prompt: {'Active' if self.state.system_prompt else 'None'}")
        if self.state.attachments:
            total_size = sum(
                p.stat().st_size for p in self.state.attachments if p.exists()
            )
            print(
                f"  Attached Text Files: {len(self.state.attachments)} ({utils.format_bytes(total_size)})"
            )
        if self.state.attached_images:
            print(f"  Attached Images: {len(self.state.attached_images)}")

    def save(self, args: list[str], command_history: list[str]) -> bool:
        should_remember, should_stay = "--remember" in args, "--stay" in args
        filename_parts = [arg for arg in args if arg not in ("--remember", "--stay")]
        filename = " ".join(filename_parts)
        if not filename:
            print(
                f"{utils.SYSTEM_MSG}--> Generating descriptive name...{utils.RESET_COLOR}"
            )
            filename, _ = self._perform_helper_request(
                LOG_RENAMING_PROMPT.format(log_content=self._get_history_for_helpers()),
                50,
            )
            if not filename:
                print(
                    f"{utils.SYSTEM_MSG}--> Could not auto-generate a name. Save cancelled.{utils.RESET_COLOR}"
                )
                return False
            filename = utils.sanitize_filename(filename)

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
            new_state = commands.load_session_from_file(filepath)
            if new_state:
                self.state = new_state
                return True
            return False
        except Exception as e:
            log.error("Error in SessionManager.load: %s", e)
            print(
                f"{utils.SYSTEM_MSG}--> Error loading session: {e}{utils.RESET_COLOR}"
            )
            return False

    def list_personas(self) -> None:
        personas = persona_manager.list_personas()
        if not personas:
            print(f"{utils.SYSTEM_MSG}--> No personas found.{utils.RESET_COLOR}")
            return
        print(f"{utils.SYSTEM_MSG}--- Available Personas ---{utils.RESET_COLOR}")
        for p in personas:
            print(f"  - {p.filename.replace('.json', '')}: {p.description}")

    def refresh_files(self, search_term: str | None) -> None:
        if not self.state.attachments:
            print(
                f"{utils.SYSTEM_MSG}--> No files attached to refresh.{utils.RESET_COLOR}"
            )
            return

        paths_to_refresh = [
            p
            for p in self.state.attachments
            if not search_term or search_term in p.name
        ]
        if not paths_to_refresh:
            print(
                f"{utils.SYSTEM_MSG}--> No files matching '{search_term}'.{utils.RESET_COLOR}"
            )
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
            print(
                f"{utils.SYSTEM_MSG}--> Refreshed: {', '.join(updated)}{utils.RESET_COLOR}"
            )
        if removed:
            print(
                f"{utils.SYSTEM_MSG}--> Removed (not found): {', '.join(removed)}{utils.RESET_COLOR}"
            )

    def list_files(self) -> None:
        if not self.state.attachments:
            print(
                f"{utils.SYSTEM_MSG}--> No text files are attached.{utils.RESET_COLOR}"
            )
            return

        file_list = sorted(
            [
                (p, p.stat().st_size if p.exists() else 0)
                for p in self.state.attachments
            ],
            key=lambda i: i[1],
            reverse=True,
        )
        print(f"{utils.SYSTEM_MSG}--- Attached Files ---{utils.RESET_COLOR}")
        for path, size in file_list:
            print(f"  - {path.name} ({utils.format_bytes(size)})")

    def attach_file(self, path_str: str) -> None:
        path = Path(path_str).resolve()
        if not path.exists():
            print(
                f"{utils.SYSTEM_MSG}--> Error: Path not found: {path_str}{utils.RESET_COLOR}"
            )
            return
        if path in self.state.attachments:
            print(
                f"{utils.SYSTEM_MSG}--> Error: File '{path.name}' is already attached.{utils.RESET_COLOR}"
            )
            return

        _, new_attachments, _ = utils.process_files([str(path)], False, [])
        if new_attachments:
            self.state.attachments.update(new_attachments)
            print(
                f"{utils.SYSTEM_MSG}--> Attached content from: {path.name}{utils.RESET_COLOR}"
            )
        else:
            print(
                f"{utils.SYSTEM_MSG}--> No readable text content found at path: {path.name}{utils.RESET_COLOR}"
            )

    def detach_file(self, filename: str) -> None:
        path_to_remove = next(
            (p for p in self.state.attachments if p.name == filename), None
        )
        if path_to_remove:
            del self.state.attachments[path_to_remove]
            print(
                f"{utils.SYSTEM_MSG}--> Detached file: {path_to_remove.name}{utils.RESET_COLOR}"
            )
        else:
            print(
                f"{utils.SYSTEM_MSG}--> No attached file named '{filename}'.{utils.RESET_COLOR}"
            )

    def switch_persona(self, name: str) -> None:
        if name.lower() == "clear":
            if not self.state.current_persona:
                print(
                    f"{utils.SYSTEM_MSG}--> No active persona to clear.{utils.RESET_COLOR}"
                )
                return
            self.state.system_prompt = self.state.initial_system_prompt
            self.state.current_persona = None
            print(f"{utils.SYSTEM_MSG}--> Persona cleared.{utils.RESET_COLOR}")
            self.state.history.append(
                utils.construct_user_message(
                    self.state.engine.name, "[SYSTEM] Persona cleared.", []
                )
            )
            return

        new_persona = persona_manager.load_persona(name)
        if not new_persona:
            print(
                f"{utils.SYSTEM_MSG}--> Persona '{name}' not found.{utils.RESET_COLOR}"
            )
            return

        if new_persona.engine and new_persona.engine != self.state.engine.name:
            print(
                f"{utils.SYSTEM_MSG}--> Switching engine to {new_persona.engine} for persona '{new_persona.name}'...{utils.RESET_COLOR}"
            )
            self.switch_engine(new_persona.engine)

        if new_persona.model:
            self.state.model = new_persona.model
        if new_persona.max_tokens is not None:
            self.state.max_tokens = new_persona.max_tokens
        if new_persona.stream is not None:
            self.state.stream_active = new_persona.stream

        self.state.system_prompt = new_persona.system_prompt
        self.state.current_persona = new_persona
        print(
            f"{utils.SYSTEM_MSG}--> Switched to persona: '{new_persona.name}'{utils.RESET_COLOR}"
        )
        self.state.history.append(
            utils.construct_user_message(
                self.state.engine.name,
                f"[SYSTEM] Persona switched to '{new_persona.name}'.",
                [],
            )
        )

    def _get_history_for_helpers(self) -> str:
        return "\n".join(
            f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}"
            for msg in self.state.history
        )

    def _save_session_to_file(self, filename: str) -> bool:
        safe_name = utils.sanitize_filename(filename.rsplit(".", 1)[0]) + ".json"
        filepath = config.SESSIONS_DIRECTORY / safe_name

        state_dict = asdict(self.state)
        state_dict["engine_name"] = self.state.engine.name
        del state_dict["engine"]
        state_dict["attachments"] = {
            str(k): v for k, v in self.state.attachments.items()
        }
        state_dict["current_persona"] = (
            self.state.current_persona.filename if self.state.current_persona else None
        )

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
            print(
                f"{utils.SYSTEM_MSG}--> Session saved to: {filepath}{utils.RESET_COLOR}"
            )
            return True
        except (OSError, TypeError) as e:
            log.error("Failed to save session state: %s", e)
            print(f"{utils.SYSTEM_MSG}--> Error saving session: {e}{utils.RESET_COLOR}")
            return False

    def _rename_log_file(self, old_path: Path, new_name_base: str) -> None:
        new_path = old_path.with_name(f"{utils.sanitize_filename(new_name_base)}.jsonl")
        try:
            old_path.rename(new_path)
            print(
                f"{utils.SYSTEM_MSG}--> Session log renamed to: {new_path.name}{utils.RESET_COLOR}"
            )
        except OSError as e:
            log.error("Failed to rename session log: %s", e)

    def _rename_log_with_ai(self, log_filepath: Path) -> None:
        print(
            f"{utils.SYSTEM_MSG}--> Generating descriptive name for session log...{utils.RESET_COLOR}"
        )
        prompt_text = LOG_RENAMING_PROMPT.format(
            log_content=self._get_history_for_helpers()
        )
        suggested_name, _ = self._perform_helper_request(
            prompt_text, app_settings.settings["log_rename_max_tokens"]
        )
        if suggested_name:
            self._rename_log_file(log_filepath, suggested_name)

    def _save_debug_log(self, log_filename_base: str) -> None:
        debug_filepath = config.LOG_DIRECTORY / f"debug_{log_filename_base}.jsonl"
        print(f"Saving debug log to: {debug_filepath}")
        try:
            with open(debug_filepath, "w", encoding="utf-8") as f:
                for entry in self.state.session_raw_logs:
                    f.write(json.dumps(entry) + "\n")
        except OSError as e:
            log.error("Could not save debug log file: %s", e)


class SingleChatManager:
    """Manages the lifecycle and I/O for an interactive single-chat session."""

    def __init__(self, session_manager: SessionManager, session_name: str | None):
        self.session = session_manager
        self.session_name = session_name

    def run(self) -> None:
        log_filename_base = (
            self.session_name
            or f"chat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.session.state.engine.name}"
        )
        log_filepath = config.LOG_DIRECTORY / f"{log_filename_base}.jsonl"
        print(
            f"Starting interactive chat with {self.session.state.engine.name.capitalize()} ({self.session.state.model})."
        )
        print(
            f"Type '/help' for commands or '/exit' to end. Log file: {log_filepath.name}"
        )
        cli_history = InMemoryHistory(self.session.state.command_history)
        first_turn = not self.session.state.history
        try:
            while True:
                prompt_message = f"\n{utils.USER_PROMPT}You: {utils.RESET_COLOR}"
                user_input = prompt(ANSI(prompt_message), history=cli_history).strip()
                if not user_input:
                    sys.stdout.write("\x1b[1A\x1b[2K")
                    sys.stdout.flush()
                    continue
                if user_input.startswith("/"):
                    sys.stdout.write("\x1b[1A\x1b[2K")
                    sys.stdout.flush()
                    if self._handle_slash_command(user_input, cli_history):
                        break
                    continue
                self.session.take_turn(user_input, first_turn)
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
            f"{utils.SYSTEM_MSG}--> Unknown command: {command_str}. Type /help.{utils.RESET_COLOR}"
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


class MultiChatManager:
    """Manages the lifecycle and I/O for an interactive multi-chat session."""

    def __init__(
        self,
        initial_state: MultiChatSessionState,
        session_name: str | None,
        initial_prompt: str | None,
    ):
        self.state = initial_state
        self.session_name = session_name
        self.initial_prompt = initial_prompt
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

    def run(self) -> None:
        log_filename_base = (
            self.session_name
            or f"multichat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        log_filepath = config.LOG_DIRECTORY / f"{log_filename_base}.jsonl"
        print(
            f"Starting interactive multi-chat. Primary: {self.primary_engine_name.capitalize()}. Log: {log_filepath.name}"
        )
        print("Type /help for commands or /exit to end.")

        cli_history = InMemoryHistory(self.state.command_history)
        if self.initial_prompt:
            self._process_turn(self.initial_prompt, log_filepath, is_first_turn=True)

        try:
            while True:
                prompt_message = (
                    f"\n{utils.DIRECTOR_PROMPT}Director> {utils.RESET_COLOR}"
                )
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
                    self._process_turn(user_input, log_filepath)
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
            self._process_turn(
                user_input, log_filepath
            )  # Special case handled by process_turn
            return False
        handler = commands.MULTICHAT_COMMAND_MAP.get(command_str)
        if handler:
            return handler(args, self.state, cli_history) or False
        print(
            f"{utils.SYSTEM_MSG}--> Unknown command: {command_str}.{utils.RESET_COLOR}"
        )
        return False

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
            cleaned = utils.clean_ai_response_text(engine.name, text)
            result_queue.put({"engine_name": engine.name, "text": cleaned})
        except Exception as e:
            log.error("Secondary worker failed: %s", e)
            result_queue.put({"engine_name": engine.name, "text": f"Error: {e}"})

    def _process_turn(
        self, prompt_text: str, log_filepath: Path, is_first_turn: bool = False
    ) -> None:
        srl_list = self.state.session_raw_logs if self.state.debug_active else None
        if prompt_text.lower().lstrip().startswith("/ai"):
            parts = prompt_text.lstrip().split(" ", 2)
            if len(parts) < 2 or parts[1].lower() not in self.engine_aliases:
                print(
                    f"{utils.SYSTEM_MSG}--> Usage: /ai <gpt|gem> [prompt]{utils.RESET_COLOR}"
                )
                return
            target_engine_name = self.engine_aliases[parts[1].lower()]
            target_prompt = parts[2] if len(parts) > 2 else CONTINUATION_PROMPT
            user_msg_text = (
                f"Director to {target_engine_name.capitalize()}: {target_prompt}"
            )
            print(f"\n{utils.DIRECTOR_PROMPT}{user_msg_text}{utils.RESET_COLOR}")
            user_msg = utils.construct_user_message(
                target_engine_name,
                user_msg_text,
                self.state.initial_image_data if is_first_turn else [],
            )
            current_history = utils.translate_history(
                self.state.shared_history + [user_msg], target_engine_name
            )
            engine, model = (
                self.engines[target_engine_name],
                self.models[target_engine_name],
            )
            print(
                f"\n{utils.ASSISTANT_PROMPT}[{engine.name.capitalize()}]: {utils.RESET_COLOR}",
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
            cleaned = utils.clean_ai_response_text(engine.name, raw_response)
            asst_msg_text = f"[{engine.name.capitalize()}]: {cleaned}"
            asst_msg = utils.construct_assistant_message("openai", asst_msg_text)
            self.state.shared_history.extend(
                [utils.construct_user_message("openai", user_msg_text, []), asst_msg]
            )
            self._log_multichat_turn(log_filepath, self.state.shared_history[-2:])
        else:
            primary_engine = self.engines[self.primary_engine_name]
            secondary_engine = self.engines[
                "gemini" if self.primary_engine_name == "openai" else "openai"
            ]
            user_msg_text = f"Director to All: {prompt_text}"
            user_msg = utils.construct_user_message(
                "openai",
                user_msg_text,
                self.state.initial_image_data if is_first_turn else [],
            )
            result_queue = queue.Queue()
            history_primary = utils.translate_history(
                self.state.shared_history + [user_msg], primary_engine.name
            )
            history_secondary = utils.translate_history(
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
                f"\n{utils.ASSISTANT_PROMPT}[{primary_engine.name.capitalize()}]: {utils.RESET_COLOR}",
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
                f"{utils.ASSISTANT_PROMPT}[{secondary_result['engine_name'].capitalize()}]: {utils.RESET_COLOR}{secondary_result['text']}"
            )

            primary_cleaned = utils.clean_ai_response_text(
                primary_engine.name, primary_raw
            )
            primary_msg = utils.construct_assistant_message(
                "openai", f"[{primary_engine.name.capitalize()}]: {primary_cleaned}"
            )
            secondary_msg = utils.construct_assistant_message(
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
