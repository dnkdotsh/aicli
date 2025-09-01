# src/aicli/commands.py
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

"""
This module contains the implementation for all interactive slash commands
and the associated session management helpers (save, load, etc.).
"""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from . import api_client, config, utils
from . import personas as persona_manager
from . import settings as app_settings
from .engine import get_engine
from .logger import log
from .prompts import (
    DIRECT_MEMORY_INJECTION_PROMPT,
    LOG_RENAMING_PROMPT,
    MEMORY_INTEGRATION_PROMPT,
)

if TYPE_CHECKING:
    from .session_manager import MultiChatSessionState, SessionState


def _format_bytes(byte_count: int) -> str:
    """Converts a byte count to a human-readable string (KB, MB, etc.)."""
    if byte_count is None:
        return "0 B"
    power = 1024
    n = 0
    power_labels = {0: "B", 1: "KB", 2: "MB", 3: "GB", 4: "TB"}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}"


def _generate_session_name(engine, history: list) -> str | None:
    """Generates a descriptive name for the session using an AI model."""
    log_content = "\n".join(
        [
            f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}"
            for msg in history[:10]
        ]
    )  # Use first 5 turns
    prompt_text = LOG_RENAMING_PROMPT.format(log_content=log_content)

    helper_model_key = (
        "helper_model_openai" if engine.name == "openai" else "helper_model_gemini"
    )
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(engine.name, prompt_text, [])]

    suggested_name, _ = api_client.perform_chat_request(
        engine=engine,
        model=task_model,
        messages_or_contents=messages,
        system_prompt=None,
        max_tokens=app_settings.settings["log_rename_max_tokens"],
        stream=False,
    )

    if suggested_name and not suggested_name.startswith("API Error:"):
        return utils.sanitize_filename(suggested_name.strip())

    reason = suggested_name or "No response from helper model."
    log.warning(
        "Could not generate a descriptive name for the session log. Reason: %s", reason
    )
    return None


def _save_session_to_file(state: "SessionState", filename: str) -> bool:
    """Serializes the current SessionState to a JSON file."""
    if not filename.endswith(".json"):
        filename += ".json"

    safe_name = utils.sanitize_filename(filename.rsplit(".", 1)[0]) + ".json"
    filepath = config.SESSIONS_DIRECTORY / safe_name

    state_dict = asdict(state)
    state_dict["engine_name"] = state.engine.name
    del state_dict["engine"]
    # Convert non-serializable types to strings for saving
    state_dict["attachments"] = {str(k): v for k, v in state.attachments.items()}
    if state.current_persona:
        state_dict["current_persona"] = state.current_persona.filename
    else:
        state_dict["current_persona"] = None

    try:
        utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        print(
            f"{utils.SYSTEM_MSG}--> Session saved successfully to: {filepath}{utils.RESET_COLOR}"
        )
        return True
    except (OSError, TypeError) as e:
        log.error("Failed to save session state: %s", e)
        print(f"{utils.SYSTEM_MSG}--> Error saving session: {e}{utils.RESET_COLOR}")
        return False


def consolidate_session_into_memory(engine, model: str, history: list):
    """Summarizes the session and integrates it into persistent memory."""
    print(
        f"{utils.SYSTEM_MSG}--> Updating persistent memory with session content...{utils.RESET_COLOR}"
    )
    session_content = "\n".join(
        [
            f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}"
            for msg in history
        ]
    )
    existing_ltm = ""
    try:
        if config.PERSISTENT_MEMORY_FILE.exists():
            existing_ltm = config.PERSISTENT_MEMORY_FILE.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("Could not read persistent memory file: %s", e)

    prompt_text = MEMORY_INTEGRATION_PROMPT.format(
        existing_ltm=existing_ltm, session_content=session_content
    )
    helper_model_key = (
        "helper_model_openai" if engine.name == "openai" else "helper_model_gemini"
    )
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(engine.name, prompt_text, [])]

    updated_memory, _ = api_client.perform_chat_request(
        engine=engine,
        model=task_model,
        messages_or_contents=messages,
        system_prompt=None,
        max_tokens=None,
        stream=False,
    )

    if updated_memory and not updated_memory.startswith("API Error:"):
        try:
            config.PERSISTENT_MEMORY_FILE.write_text(
                updated_memory.strip(), encoding="utf-8"
            )
            print(
                f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}"
            )
        except OSError as e:
            log.error("Failed to write to persistent memory file: %s", e)
    else:
        reason = updated_memory or "No response from helper model."
        log.warning("Memory integration failed. Reason: %s", reason)


def _inject_fact_into_memory(engine, fact: str):
    """Injects a single fact into persistent memory."""
    print(
        f"{utils.SYSTEM_MSG}--> Injecting fact into persistent memory...{utils.RESET_COLOR}"
    )
    existing_ltm = ""
    try:
        if config.PERSISTENT_MEMORY_FILE.exists():
            existing_ltm = config.PERSISTENT_MEMORY_FILE.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("Could not read persistent memory file: %s", e)

    prompt_text = DIRECT_MEMORY_INJECTION_PROMPT.format(
        existing_ltm=existing_ltm, new_fact=fact
    )
    helper_model_key = (
        "helper_model_openai" if engine.name == "openai" else "helper_model_gemini"
    )
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(engine.name, prompt_text, [])]

    updated_memory, _ = api_client.perform_chat_request(
        engine=engine,
        model=task_model,
        messages_or_contents=messages,
        system_prompt=None,
        max_tokens=None,
        stream=False,
    )

    if updated_memory and not updated_memory.startswith("API Error:"):
        try:
            config.PERSISTENT_MEMORY_FILE.write_text(
                updated_memory.strip(), encoding="utf-8"
            )
            print(
                f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}"
            )
        except OSError as e:
            log.error("Failed to write to persistent memory file: %s", e)
    else:
        reason = updated_memory or "No response from helper model."
        log.warning("Memory integration failed. Reason: %s", reason)


def load_session_from_file(filepath: Path) -> "SessionState | None":
    """Loads a session from a file and reconstructs the SessionState."""
    from .session_manager import SessionState

    try:
        if not filepath.exists():
            raise FileNotFoundError("The specified session file does not exist.")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        if "attachments" in data:
            data["attachments"] = {Path(k): v for k, v in data["attachments"].items()}
        if "attached_text_files" in data:
            del data["attached_text_files"]
        if "initial_image_data" in data:
            data["attached_images"] = data.pop("initial_image_data")

        # Load persona object from its filename
        if "current_persona" in data and data["current_persona"]:
            data["current_persona"] = persona_manager.load_persona(
                data["current_persona"]
            )

        data.pop("force_quit", None)
        data.pop("custom_log_rename", None)

        engine_name = data.pop("engine_name")
        api_key = api_client.check_api_keys(engine_name)
        engine_instance = get_engine(engine_name, api_key)

        state = SessionState(engine=engine_instance, **data)

        print(
            f"{utils.SYSTEM_MSG}--> Session loaded successfully from: {filepath}{utils.RESET_COLOR}"
        )
        print(f"{utils.SYSTEM_MSG}--- Last few messages ---{utils.RESET_COLOR}")
        history_slice = state.history[-4:]
        for msg in history_slice:
            role = msg.get("role")
            content = utils.extract_text_from_message(msg)
            if role == "user":
                print(f"{utils.USER_PROMPT}You: {utils.RESET_COLOR}{content}")
            elif role in ["assistant", "model"]:
                print(
                    f"{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{content}"
                )
        print(f"{utils.SYSTEM_MSG}-------------------------{utils.RESET_COLOR}")

        return state
    except (
        OSError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        api_client.MissingApiKeyError,
    ) as e:
        log.error("Failed to load session from %s: %s", filepath, e)
        return None


def rename_session_log(engine, history: list, original_filepath: str):
    """Generates a descriptive name for the session log and renames the file."""
    print(
        f"{utils.SYSTEM_MSG}--> Generating descriptive name for session log...{utils.RESET_COLOR}"
    )
    suggested_name = _generate_session_name(engine, history)

    if suggested_name:
        new_filename = f"{suggested_name}.jsonl"
        new_filepath = os.path.join(config.LOG_DIRECTORY, new_filename)
        try:
            os.rename(original_filepath, new_filepath)
            print(
                f"{utils.SYSTEM_MSG}--> Session log renamed to: {new_filepath}{utils.RESET_COLOR}"
            )
        except OSError as e:
            log.error("Failed to rename session log: %s", e)
    else:
        log.warning("Could not generate a descriptive name for the session log.")


# --- Single-Chat Command Handler Functions ---


def handle_exit(
    args: list, state: "SessionState", cli_history: InMemoryHistory
) -> bool:
    if args:
        state.custom_log_rename = " ".join(args)
    return True


def handle_quit(
    args: list, state: "SessionState", cli_history: InMemoryHistory
) -> bool:
    state.force_quit = True
    return True


def handle_help(args: list, state: "SessionState", cli_history: InMemoryHistory):
    utils.display_help("chat")


def handle_stream(args: list, state: "SessionState", cli_history: InMemoryHistory):
    state.stream_active = not state.stream_active
    status = "ENABLED" if state.stream_active else "DISABLED"
    print(
        f"{utils.SYSTEM_MSG}--> Response streaming is now {status} for this session.{utils.RESET_COLOR}"
    )


def handle_debug(args: list, state: "SessionState", cli_history: InMemoryHistory):
    state.debug_active = not state.debug_active
    status = "ENABLED" if state.debug_active else "DISABLED"
    print(
        f"{utils.SYSTEM_MSG}--> Session-specific debug logging is now {status}.{utils.RESET_COLOR}"
    )


def handle_memory(args: list, state: "SessionState", cli_history: InMemoryHistory):
    try:
        with open(config.PERSISTENT_MEMORY_FILE, encoding="utf-8") as f:
            content = f.read()
        print(f"{utils.SYSTEM_MSG}--- Persistent Memory ---{utils.RESET_COLOR}")
        print(content)
        print(f"{utils.SYSTEM_MSG}-------------------------{utils.RESET_COLOR}")
    except FileNotFoundError:
        print(
            f"{utils.SYSTEM_MSG}--> Persistent memory is currently empty.{utils.RESET_COLOR}"
        )
    except OSError as e:
        log.error("Could not read persistent memory file: %s", e)
        print(
            f"{utils.SYSTEM_MSG}--> Error reading memory file: {e}{utils.RESET_COLOR}"
        )


def handle_remember(args: list, state: "SessionState", cli_history: InMemoryHistory):
    if not args:
        if not state.history:
            print(
                f"{utils.SYSTEM_MSG}--> Nothing to consolidate; conversation history is empty.{utils.RESET_COLOR}"
            )
            return
        consolidate_session_into_memory(state.engine, state.model, state.history)
    else:
        fact_to_remember = " ".join(args)
        _inject_fact_into_memory(state.engine, fact_to_remember)


def handle_max_tokens(args: list, state: "SessionState", cli_history: InMemoryHistory):
    if args and args[0].isdigit():
        state.max_tokens = int(args[0])
        print(
            f"{utils.SYSTEM_MSG}--> Max tokens for this session set to: {state.max_tokens}.{utils.RESET_COLOR}"
        )
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")


def handle_clear(args: list, state: "SessionState", cli_history: InMemoryHistory):
    confirm = prompt(
        "This will clear all conversation history. Type `proceed` to confirm: "
    )
    if confirm.lower() == "proceed":
        state.history.clear()
        print(
            f"{utils.SYSTEM_MSG}--> Conversation history has been cleared.{utils.RESET_COLOR}"
        )
    else:
        print(f"{utils.SYSTEM_MSG}--> Clear cancelled.{utils.RESET_COLOR}")


def handle_model(args: list, state: "SessionState", cli_history: InMemoryHistory):
    from . import handlers  # Local import to avoid circular dependency

    if args:
        state.model = args[0]
        print(
            f"{utils.SYSTEM_MSG}--> Model temporarily set to: {state.model}{utils.RESET_COLOR}"
        )
    else:
        state.model = handlers.select_model(state.engine, "chat")
        print(
            f"{utils.SYSTEM_MSG}--> Model changed to: {state.model}{utils.RESET_COLOR}"
        )


def handle_engine(args: list, state: "SessionState", cli_history: InMemoryHistory):
    new_engine_name = (
        args[0] if args else ("gemini" if state.engine.name == "openai" else "openai")
    )
    if new_engine_name not in ["openai", "gemini"]:
        print(
            f"{utils.SYSTEM_MSG}--> Unknown engine: {new_engine_name}. Use 'openai' or 'gemini'.{utils.RESET_COLOR}"
        )
        return
    try:
        new_api_key = api_client.check_api_keys(new_engine_name)
        state.history = utils.translate_history(state.history, new_engine_name)
        state.engine = get_engine(new_engine_name, new_api_key)
        model_key = (
            "default_openai_chat_model"
            if new_engine_name == "openai"
            else "default_gemini_model"
        )
        state.model = app_settings.settings[model_key]
        print(
            f"{utils.SYSTEM_MSG}--> Engine switched to {state.engine.name.capitalize()}. Model set to default: {state.model}. History translated.{utils.RESET_COLOR}"
        )
    except api_client.MissingApiKeyError:
        print(
            f"{utils.SYSTEM_MSG}--> Switch to {new_engine_name.capitalize()} failed: API key not found.{utils.RESET_COLOR}"
        )


def handle_history(args: list, state: "SessionState", cli_history: InMemoryHistory):
    print(json.dumps(state.history, indent=2))


def handle_state(args: list, state: "SessionState", cli_history: InMemoryHistory):
    print(f"{utils.SYSTEM_MSG}--- Session State ---{utils.RESET_COLOR}")
    if state.current_persona:
        print(
            f"  Active Persona: {state.current_persona.name} ({state.current_persona.filename})"
        )
    else:
        print("  Active Persona: None")
    print(f"  Engine: {state.engine.name}")
    print(f"  Model: {state.model}")
    print(f"  Max Tokens: {state.max_tokens or 'Default'}")
    print(f"  Streaming: {'On' if state.stream_active else 'Off'}")
    print(f"  Memory Enabled (on exit): {'On' if state.memory_enabled else 'Off'}")
    print(f"  Debug Logging: {'On' if state.debug_active else 'Off'}")
    print(f"  System Prompt: {'Active' if state.system_prompt else 'None'}")
    if state.attachments:
        total_size = sum(p.stat().st_size for p in state.attachments if p.exists())
        print(
            f"  Attached Text Files: {len(state.attachments)} ({_format_bytes(total_size)})"
        )
    if state.attached_images:
        print(f"  Attached Images: {len(state.attached_images)}")


def handle_set(args: list, state: "SessionState", cli_history: InMemoryHistory):
    if not args:
        print(f"{utils.SYSTEM_MSG}--- Current Settings ---{utils.RESET_COLOR}")
        for key, value in sorted(app_settings.settings.items()):
            print(f"  {key}: {value}")
        print(f"{utils.SYSTEM_MSG}----------------------{utils.RESET_COLOR}")
    elif len(args) == 2:
        app_settings.save_setting(args[0], args[1])
    else:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /set <key> <value> OR /set to list all settings.{utils.RESET_COLOR}"
        )


def handle_save(
    args: list, state: "SessionState", cli_history: InMemoryHistory
) -> bool:
    should_remember, should_stay = False, False
    filename_parts = [arg for arg in args if arg not in ("--remember", "--stay")]
    if "--remember" in args:
        should_remember = True
    if "--stay" in args:
        should_stay = True

    filename = " ".join(filename_parts) if filename_parts else None

    if not filename:
        print(
            f"{utils.SYSTEM_MSG}--> Generating descriptive name for session...{utils.RESET_COLOR}"
        )
        filename = _generate_session_name(state.engine, state.history)
        if not filename:
            print(
                f"{utils.SYSTEM_MSG}--> Could not auto-generate a name. Save cancelled.{utils.RESET_COLOR}"
            )
            return False

    state.command_history = cli_history.get_strings()

    if _save_session_to_file(state, filename):
        if should_stay:
            return False
        if not should_remember:
            state.exit_without_memory = True
        return True
    return False


def handle_load(args: list, state: "SessionState", cli_history: InMemoryHistory):
    if args:
        filename = " ".join(args)
        if not filename.endswith(".json"):
            filename += ".json"
        filepath = config.SESSIONS_DIRECTORY / filename
        new_state = load_session_from_file(filepath)
        if new_state:
            # Replace current state with loaded state
            for key, value in asdict(new_state).items():
                setattr(state, key, value)
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /load <filename>{utils.RESET_COLOR}")


def handle_refresh(args: list, state: "SessionState", cli_history: InMemoryHistory):
    search_term = " ".join(args)
    if not args:
        if not state.attachments:
            print(
                f"{utils.SYSTEM_MSG}--> No files attached to refresh.{utils.RESET_COLOR}"
            )
            return
        paths_to_refresh = list(state.attachments)
    else:
        paths_to_refresh = [p for p in state.attachments if search_term in p.name]
        if not paths_to_refresh:
            print(
                f"{utils.SYSTEM_MSG}--> No attached files found matching '{search_term}'.{utils.RESET_COLOR}"
            )
            return

    updated_files, files_to_remove = [], []
    for path in paths_to_refresh:
        try:
            if path.suffix.lower() == ".zip":
                continue
            with open(path, encoding="utf-8", errors="ignore") as f:
                state.attachments[path] = f.read()
            updated_files.append(path.name)
        except (OSError, FileNotFoundError) as e:
            log.warning(
                "Could not re-read '%s': %s. Removing from context.", path.name, e
            )
            print(
                f"{utils.SYSTEM_MSG}--> Warning: Could not re-read '{path.name}'. Removing from session.{utils.RESET_COLOR}"
            )
            files_to_remove.append(path)

    for path in files_to_remove:
        del state.attachments[path]

    if updated_files:
        print(
            f"{utils.SYSTEM_MSG}--> Refreshed {len(updated_files)} file(s): {', '.join(updated_files)}{utils.RESET_COLOR}"
        )
        notification_text = f"[SYSTEM] The content of the following file(s) has been updated: {', '.join(updated_files)}."
        state.history.append(
            utils.construct_user_message(state.engine.name, notification_text, [])
        )
        print(
            f"{utils.SYSTEM_MSG}--> Notified AI of the context update.{utils.RESET_COLOR}"
        )


def handle_files(args: list, state: "SessionState", cli_history: InMemoryHistory):
    if not state.attachments:
        print(
            f"{utils.SYSTEM_MSG}--> No text files are currently attached.{utils.RESET_COLOR}"
        )
        return

    file_list = sorted(
        [(p, p.stat().st_size if p.exists() else 0) for p in state.attachments],
        key=lambda item: item[1],
        reverse=True,
    )
    print(
        f"{utils.SYSTEM_MSG}--- Attached Text Files (Sorted by Size) ---{utils.RESET_COLOR}"
    )
    for path, size in file_list:
        print(f"  - {path.name} ({_format_bytes(size)})")
    print(
        f"{utils.SYSTEM_MSG}------------------------------------------{utils.RESET_COLOR}"
    )


def handle_attach(args: list, state: "SessionState", cli_history: InMemoryHistory):
    if not args:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /attach <path_to_file_or_dir>{utils.RESET_COLOR}"
        )
        return

    path_str = " ".join(args)
    path = Path(path_str).resolve()
    if not path.exists():
        print(
            f"{utils.SYSTEM_MSG}--> Error: Path not found: {path_str}{utils.RESET_COLOR}"
        )
        return
    if path in state.attachments:
        print(
            f"{utils.SYSTEM_MSG}--> Error: File '{path.name}' is already attached.{utils.RESET_COLOR}"
        )
        return

    if utils.is_supported_text_file(path):
        try:
            state.attachments[path] = path.read_text(encoding="utf-8", errors="ignore")
            print(
                f"{utils.SYSTEM_MSG}--> Attached file: {path.name} ({_format_bytes(path.stat().st_size)}){utils.RESET_COLOR}"
            )
            notification_text = (
                f"[SYSTEM] The user has attached a new file: {path.name}."
            )
            state.history.append(
                utils.construct_user_message(state.engine.name, notification_text, [])
            )
        except (OSError, UnicodeDecodeError) as e:
            log.warning("Failed to attach file '%s': %s", path.name, e)
            print(
                f"{utils.SYSTEM_MSG}--> Error reading file '{path.name}': {e}{utils.RESET_COLOR}"
            )
    elif utils.is_supported_archive_file(path):
        print(
            f"{utils.SYSTEM_MSG}--> Processing archive: {path.name}...{utils.RESET_COLOR}"
        )
        _, new_attachments, _ = utils.process_files(
            [str(path)], use_memory=False, exclusions=[]
        )
        if new_attachments:
            state.attachments.update(new_attachments)
            # Estimate size from the first (and only) value in the new attachments dict
            content_size = len(next(iter(new_attachments.values()), "").encode("utf-8"))
            print(
                f"{utils.SYSTEM_MSG}--> Attached content from archive: {path.name} ({_format_bytes(content_size)}){utils.RESET_COLOR}"
            )
            notification_text = f"[SYSTEM] The user has attached and processed the archive: {path.name}."
            state.history.append(
                utils.construct_user_message(state.engine.name, notification_text, [])
            )
        else:
            print(
                f"{utils.SYSTEM_MSG}--> No supported text files found within the archive: {path.name}{utils.RESET_COLOR}"
            )
    else:
        print(
            f"{utils.SYSTEM_MSG}--> Error: File type '{''.join(path.suffixes)}' is not supported for attachment.{utils.RESET_COLOR}"
        )


def handle_detach(args: list, state: "SessionState", cli_history: InMemoryHistory):
    if not args:
        print(f"{utils.SYSTEM_MSG}--> Usage: /detach <filename>{utils.RESET_COLOR}")
        return

    search_term = " ".join(args)
    path_to_remove = next((p for p in state.attachments if p.name == search_term), None)

    if path_to_remove:
        del state.attachments[path_to_remove]
        print(
            f"{utils.SYSTEM_MSG}--> Detached file: {path_to_remove.name}{utils.RESET_COLOR}"
        )
        notification_text = (
            f"[SYSTEM] The user has detached the file: {path_to_remove.name}."
        )
        state.history.append(
            utils.construct_user_message(state.engine.name, notification_text, [])
        )
    else:
        print(
            f"{utils.SYSTEM_MSG}--> No attached file found with the name '{search_term}'.{utils.RESET_COLOR}"
        )


def handle_personas(args: list, state: "SessionState", cli_history: InMemoryHistory):
    """Lists available personas."""
    personas = persona_manager.list_personas()
    if not personas:
        print(
            f"{utils.SYSTEM_MSG}--> No personas found in {config.PERSONAS_DIRECTORY}.{utils.RESET_COLOR}"
        )
        return

    print(f"{utils.SYSTEM_MSG}--- Available Personas ---{utils.RESET_COLOR}")
    for p in personas:
        print(f"  - {p.filename.replace('.json', '')}: {p.description}")
    print(f"{utils.SYSTEM_MSG}------------------------{utils.RESET_COLOR}")


def handle_persona(args: list, state: "SessionState", cli_history: InMemoryHistory):
    """Switches to or clears a persona."""
    if not args:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /persona <name> OR /persona clear{utils.RESET_COLOR}"
        )
        return

    command = args[0].lower()
    if command == "clear":
        if not state.current_persona:
            print(
                f"{utils.SYSTEM_MSG}--> No active persona to clear.{utils.RESET_COLOR}"
            )
            return

        state.system_prompt = state.initial_system_prompt
        state.current_persona = None
        print(
            f"{utils.SYSTEM_MSG}--> Persona cleared. System prompt reverted to its original state for this session.{utils.RESET_COLOR}"
        )
        notification_text = "[SYSTEM] The current persona has been cleared. Default instructions are now in effect."
        state.history.append(
            utils.construct_user_message(state.engine.name, notification_text, [])
        )
        return

    # Load and apply the new persona
    persona_name = " ".join(args)
    new_persona = persona_manager.load_persona(persona_name)
    if not new_persona:
        print(
            f"{utils.SYSTEM_MSG}--> Persona '{persona_name}' not found or is invalid.{utils.RESET_COLOR}"
        )
        return

    # Apply engine if specified and different
    if new_persona.engine and new_persona.engine != state.engine.name:
        try:
            new_api_key = api_client.check_api_keys(new_persona.engine)
            state.history = utils.translate_history(state.history, new_persona.engine)
            state.engine = get_engine(new_persona.engine, new_api_key)
            # Use persona model, or default for new engine
            default_model_key = (
                "default_openai_chat_model"
                if new_persona.engine == "openai"
                else "default_gemini_model"
            )
            state.model = new_persona.model or app_settings.settings[default_model_key]
            print(
                f"{utils.SYSTEM_MSG}--> Engine switched to {state.engine.name.capitalize()} as per persona '{new_persona.name}'.{utils.RESET_COLOR}"
            )
        except api_client.MissingApiKeyError:
            print(
                f"{utils.SYSTEM_MSG}--> Could not switch to persona engine '{new_persona.engine}': API key not found. Aborting.{utils.RESET_COLOR}"
            )
            return

    # Apply model if specified (and engine didn't just change it)
    elif new_persona.model:
        state.model = new_persona.model

    # Apply other optional settings
    if new_persona.max_tokens is not None:
        state.max_tokens = new_persona.max_tokens
    if new_persona.stream is not None:
        state.stream_active = new_persona.stream

    # Apply system prompt and set current persona
    state.system_prompt = new_persona.system_prompt
    state.current_persona = new_persona

    print(
        f"{utils.SYSTEM_MSG}--> Switched to persona: '{new_persona.name}'{utils.RESET_COLOR}"
    )
    notification_text = f"[SYSTEM] Persona changed to '{new_persona.name}'. New instructions are now in effect."
    state.history.append(
        utils.construct_user_message(state.engine.name, notification_text, [])
    )


# --- Multi-Chat Command Handler Functions ---


def _save_multichat_session_to_file(
    state: "MultiChatSessionState", filename: str
) -> bool:
    if not filename.endswith(".json"):
        filename += ".json"
    safe_name = utils.sanitize_filename(filename.rsplit(".", 1)[0]) + ".json"
    filepath = config.SESSIONS_DIRECTORY / safe_name

    state_dict = asdict(state)
    state_dict["session_type"] = "multichat"
    # Engine objects are not serializable and can be recreated on load
    del state_dict["openai_engine"]
    del state_dict["gemini_engine"]

    try:
        utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        print(
            f"{utils.SYSTEM_MSG}--> Multi-chat session saved successfully to: {filepath}{utils.RESET_COLOR}"
        )
        return True
    except (OSError, TypeError) as e:
        log.error("Failed to save multi-chat session state: %s", e)
        print(f"{utils.SYSTEM_MSG}--> Error saving session: {e}{utils.RESET_COLOR}")
        return False


def handle_multichat_exit(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
) -> bool:
    if args:
        state.custom_log_rename = " ".join(args)
    return True


def handle_multichat_quit(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
) -> bool:
    state.force_quit = True
    return True


def handle_multichat_help(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
):
    utils.display_help("multichat")


def handle_multichat_history(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
):
    print(json.dumps(state.shared_history, indent=2))


def handle_multichat_debug(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
):
    state.debug_active = not state.debug_active
    status = "ENABLED" if state.debug_active else "DISABLED"
    print(
        f"{utils.SYSTEM_MSG}--> Session-specific debug logging is now {status}.{utils.RESET_COLOR}"
    )


def handle_multichat_remember(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
):
    primary_engine_name = app_settings.settings["default_engine"]
    primary_engine = (
        state.openai_engine if primary_engine_name == "openai" else state.gemini_engine
    )
    primary_model_key = (
        "helper_model_openai"
        if primary_engine_name == "openai"
        else "helper_model_gemini"
    )
    primary_model = app_settings.settings[primary_model_key]

    if not args:
        if not state.shared_history:
            print(
                f"{utils.SYSTEM_MSG}--> Nothing to consolidate; conversation history is empty.{utils.RESET_COLOR}"
            )
            return
        consolidate_session_into_memory(
            primary_engine, primary_model, state.shared_history
        )
    else:
        fact_to_remember = " ".join(args)
        _inject_fact_into_memory(primary_engine, fact_to_remember)


def handle_multichat_max_tokens(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
):
    if args and args[0].isdigit():
        state.max_tokens = int(args[0])
        print(
            f"{utils.SYSTEM_MSG}--> Max tokens for this session set to: {state.max_tokens}.{utils.RESET_COLOR}"
        )
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")


def handle_multichat_clear(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
):
    confirm = prompt(
        "This will clear all conversation history. Type `proceed` to confirm: "
    )
    if confirm.lower() == "proceed":
        state.shared_history.clear()
        print(
            f"{utils.SYSTEM_MSG}--> Conversation history has been cleared.{utils.RESET_COLOR}"
        )
    else:
        print(f"{utils.SYSTEM_MSG}--> Clear cancelled.{utils.RESET_COLOR}")


def handle_multichat_model(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
):
    if len(args) != 2:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /model <gpt|gem> <model_name>{utils.RESET_COLOR}"
        )
        return

    engine_alias, model_name = args[0].lower(), args[1]
    engine_map = {"gpt": "openai", "gem": "gemini"}

    if engine_alias not in engine_map:
        print(
            f"{utils.SYSTEM_MSG}--> Invalid engine alias. Use 'gpt' for OpenAI or 'gem' for Gemini.{utils.RESET_COLOR}"
        )
        return

    target_engine = engine_map[engine_alias]
    if target_engine == "openai":
        state.openai_model = model_name
        print(
            f"{utils.SYSTEM_MSG}--> OpenAI model set to: {model_name}{utils.RESET_COLOR}"
        )
    else:  # gemini
        state.gemini_model = model_name
        print(
            f"{utils.SYSTEM_MSG}--> Gemini model set to: {model_name}{utils.RESET_COLOR}"
        )


def handle_multichat_state(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
):
    print(f"{utils.SYSTEM_MSG}--- Multi-Chat Session State ---{utils.RESET_COLOR}")
    print(f"  OpenAI Model: {state.openai_model}")
    print(f"  Gemini Model: {state.gemini_model}")
    print(f"  Max Tokens: {state.max_tokens or 'Default'}")
    print(f"  Debug Logging: {'On' if state.debug_active else 'Off'}")
    print(f"  System Prompts: {'Active' if state.system_prompts else 'None'}")
    if state.initial_image_data:
        print(f"  Attached Images: {len(state.initial_image_data)}")


def handle_multichat_save(
    args: list, state: "MultiChatSessionState", cli_history: InMemoryHistory
) -> bool:
    should_remember, should_stay = False, False
    filename_parts = [arg for arg in args if arg not in ("--remember", "--stay")]
    if "--remember" in args:
        should_remember = True
    if "--stay" in args:
        should_stay = True

    filename = " ".join(filename_parts)
    if not filename:
        print(
            f"{utils.SYSTEM_MSG}--> Please provide a name for the session file.{utils.RESET_COLOR}"
        )
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /save <filename> [--stay] [--remember]{utils.RESET_COLOR}"
        )
        return False  # Stay in session

    state.command_history = cli_history.get_strings()

    if _save_multichat_session_to_file(state, filename):
        if should_stay:
            return False
        if not should_remember:
            state.exit_without_memory = True
        else:
            handle_multichat_remember([], state, cli_history)
        return True
    return False


# --- Command Dispatcher Maps ---

COMMAND_MAP = {
    "/exit": handle_exit,
    "/quit": handle_quit,
    "/help": handle_help,
    "/stream": handle_stream,
    "/debug": handle_debug,
    "/memory": handle_memory,
    "/remember": handle_remember,
    "/max-tokens": handle_max_tokens,
    "/clear": handle_clear,
    "/model": handle_model,
    "/engine": handle_engine,
    "/history": handle_history,
    "/state": handle_state,
    "/set": handle_set,
    "/save": handle_save,
    "/load": handle_load,
    "/refresh": handle_refresh,
    "/files": handle_files,
    "/attach": handle_attach,
    "/detach": handle_detach,
    "/personas": handle_personas,
    "/persona": handle_persona,
}

MULTICHAT_COMMAND_MAP = {
    "/exit": handle_multichat_exit,
    "/quit": handle_multichat_quit,
    "/help": handle_multichat_help,
    "/history": handle_multichat_history,
    "/debug": handle_multichat_debug,
    "/memory": handle_memory,  # Reuses read-only single-chat handler
    "/remember": handle_multichat_remember,
    "/max-tokens": handle_multichat_max_tokens,
    "/clear": handle_multichat_clear,
    "/model": handle_multichat_model,
    "/state": handle_multichat_state,
    "/set": handle_set,  # Reuses global settings single-chat handler
    "/save": handle_multichat_save,
}
