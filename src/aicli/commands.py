# aicli/commands.py
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
This module contains the implementation for all interactive slash commands.
These functions contain the business logic for each command and directly
manipulate the state of a SessionManager instance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import prompt
from prompt_toolkit.application import get_app
from prompt_toolkit.history import InMemoryHistory

from . import api_client, config, prompts, theme_manager, workflows
from . import personas as persona_manager
from .engine import get_engine
from .logger import log
from .session_state import SessionState
from .settings import save_setting, settings
from .utils.config_loader import get_default_model_for_engine
from .utils.formatters import (
    RESET_COLOR,
    SYSTEM_MSG,
    format_bytes,
    sanitize_filename,
)
from .utils.message_builder import construct_user_message, translate_history
from .utils.ui_helpers import display_help, select_model

if TYPE_CHECKING:
    from .managers.multichat_manager import MultiChatSession
    from .managers.session_manager import SessionManager


# --- Single-Chat Command Handler Functions ---


def handle_exit(args: list[str], session: SessionManager) -> bool:
    if args:
        session.state.custom_log_rename = " ".join(args)
    return True


def handle_quit(args: list[str], session: SessionManager) -> bool:
    session.state.force_quit = True
    return True


def handle_help(args: list[str], session: SessionManager) -> None:
    display_help("chat")


def handle_stream(args: list[str], session: SessionManager) -> None:
    session.state.stream_active = not session.state.stream_active
    status = "ENABLED" if session.state.stream_active else "DISABLED"
    print(f"{SYSTEM_MSG}--> Response streaming is now {status}.{RESET_COLOR}")


def handle_debug(args: list[str], session: SessionManager) -> None:
    session.state.debug_active = not session.state.debug_active
    status = "ENABLED" if session.state.debug_active else "DISABLED"
    print(
        f"{SYSTEM_MSG}--> Session-specific debug logging is now {status}.{RESET_COLOR}"
    )


def handle_memory(args: list[str], session: SessionManager) -> None:
    try:
        content = config.PERSISTENT_MEMORY_FILE.read_text(encoding="utf-8")
        print(f"{SYSTEM_MSG}--- Persistent Memory ---{RESET_COLOR}\n{content}")
    except FileNotFoundError:
        print(f"{SYSTEM_MSG}--> Persistent memory is currently empty.{RESET_COLOR}")
    except OSError as e:
        print(f"{SYSTEM_MSG}--> Error reading memory file: {e}{RESET_COLOR}")


def handle_remember(args: list[str], session: SessionManager) -> None:
    if not args:
        if not session.state.history:
            print(
                f"{SYSTEM_MSG}--> Nothing to consolidate; history is empty.{RESET_COLOR}"
            )
            return
        workflows.consolidate_memory(session)
    else:
        workflows.inject_memory(session, " ".join(args))


def handle_max_tokens(args: list[str], session: SessionManager) -> None:
    if args and args[0].isdigit():
        session.state.max_tokens = int(args[0])
        print(f"{SYSTEM_MSG}--> Max tokens set to: {int(args[0])}.{RESET_COLOR}")
    else:
        print(f"{SYSTEM_MSG}--> Usage: /max-tokens <number>{RESET_COLOR}")


def handle_clear(args: list[str], session: SessionManager) -> None:
    confirm = prompt(
        "This will clear all conversation history. Type `proceed` to confirm: "
    )
    if confirm.lower() == "proceed":
        session.state.history.clear()
        session.state.total_prompt_tokens = 0
        session.state.total_completion_tokens = 0
        session.state.last_turn_tokens = {}
        print(
            f"{SYSTEM_MSG}--> Conversation history and token counts have been cleared.{RESET_COLOR}"
        )
    else:
        print(f"{SYSTEM_MSG}--> Clear cancelled.{RESET_COLOR}")


def handle_model(args: list[str], session: SessionManager) -> None:
    if args:
        session.state.model = args[0]
        print(f"{SYSTEM_MSG}--> Model set to: {args[0]}.{RESET_COLOR}")
    else:
        new_model = select_model(session.state.engine, "chat")
        session.state.model = new_model
        print(f"{SYSTEM_MSG}--> Model set to: {new_model}.{RESET_COLOR}")


def handle_engine(args: list[str], session: SessionManager) -> None:
    new_engine_name = args[0] if args else None
    if not new_engine_name:
        new_engine_name = (
            "gemini" if session.state.engine.name == "openai" else "openai"
        )
    if new_engine_name not in ["openai", "gemini"]:
        print(f"{SYSTEM_MSG}--> Unknown engine: {new_engine_name}.{RESET_COLOR}")
        return
    try:
        api_key = api_client.check_api_keys(new_engine_name)
        session.state.history = translate_history(
            session.state.history, new_engine_name
        )
        session.state.engine = get_engine(new_engine_name, api_key)
        session.state.model = get_default_model_for_engine(new_engine_name)
        print(
            f"{SYSTEM_MSG}--> Engine switched to {session.state.engine.name.capitalize()}. Model set to {session.state.model}.{RESET_COLOR}"
        )
    except api_client.MissingApiKeyError as e:
        print(f"{SYSTEM_MSG}--> Switch failed: {e}{RESET_COLOR}")


def handle_history(args: list[str], session: SessionManager) -> None:
    print(json.dumps(session.state.history, indent=2))


def handle_state(args: list[str], session: SessionManager) -> None:
    print(f"{SYSTEM_MSG}--- Session State ---{RESET_COLOR}")
    persona_name = (
        session.state.current_persona.name if session.state.current_persona else "None"
    )
    print(f"  Active Persona: {persona_name}")
    print(f"  Engine: {session.state.engine.name}, Model: {session.state.model}")
    print(f"  Max Tokens: {session.state.max_tokens or 'Default'}")
    print(f"  Streaming: {'On' if session.state.stream_active else 'Off'}")
    print(f"  Memory on Exit: {'On' if session.state.memory_enabled else 'Off'}")
    print(f"  Debug Logging: {'On' if session.state.debug_active else 'Off'}")
    print(
        f"  Total Session I/O: {session.state.total_prompt_tokens}p / {session.state.total_completion_tokens}c"
    )
    print(f"  System Prompt: {'Active' if session.state.system_prompt else 'None'}")
    if session.state.attachments:
        total_size = sum(
            p.stat().st_size for p in session.state.attachments if p.exists()
        )
        print(
            f"  Attached Text Files: {len(session.state.attachments)} ({format_bytes(total_size)})"
        )
    if session.state.attached_images:
        print(f"  Attached Images: {len(session.state.attached_images)}")
    if session.image_workflow.img_prompt_crafting:
        print("  Image Crafting: ACTIVE")
        if session.image_workflow.img_prompt:
            print(f"  Current Prompt: {session.image_workflow.img_prompt[:50]}...")
    if session.image_workflow.last_img_prompt:
        print(f"  Last Image Prompt: {session.image_workflow.last_img_prompt[:50]}...")


def handle_set(args: list[str], session: SessionManager) -> None:
    if len(args) == 2:
        key, value = args[0], args[1]
        success, message = save_setting(key, value)
        print(f"{SYSTEM_MSG}--> {message}{RESET_COLOR}")
        if success:
            session.state.ui_refresh_needed = True
            # Force a full redraw for settings that might affect the UI layout
            if key in ["active_theme", "toolbar_enabled"]:
                theme_manager.reload_theme()
                get_app().invalidate()
    else:
        print(f"{SYSTEM_MSG}--> Usage: /set <key> <value>.{RESET_COLOR}")


def handle_toolbar(args: list[str], session: SessionManager) -> None:
    """Handles all toolbar configuration commands."""
    if not args:
        print(f"{SYSTEM_MSG}--- Toolbar Settings ---{RESET_COLOR}")
        print(f"  Toolbar Enabled: {settings['toolbar_enabled']}")
        print(f"  Component Order: {settings['toolbar_priority_order']}")
        print(f"  Separator: '{settings['toolbar_separator']}'")
        print(f"  Show Session I/O: {settings['toolbar_show_total_io']}")
        print(f"  Show Live Tokens: {settings['toolbar_show_live_tokens']}")
        print(f"  Show Model: {settings['toolbar_show_model']}")
        print(f"  Show Persona: {settings['toolbar_show_persona']}")
        return

    command = args[0].lower()
    valid_components = {
        "io": "toolbar_show_total_io",
        "live": "toolbar_show_live_tokens",
        "model": "toolbar_show_model",
        "persona": "toolbar_show_persona",
    }

    if command in ["on", "off"]:
        success, message = save_setting("toolbar_enabled", command)
        print(f"{SYSTEM_MSG}--> {message}{RESET_COLOR}")
        if success:
            session.state.ui_refresh_needed = True
            get_app().invalidate()
    elif command == "toggle" and len(args) == 2:
        component_key = args[1].lower()
        if component_key in valid_components:
            setting_key = valid_components[component_key]
            current_value = settings[setting_key]
            success, message = save_setting(setting_key, str(not current_value))
            print(f"{SYSTEM_MSG}--> {message}{RESET_COLOR}")
            if success:
                session.state.ui_refresh_needed = True
                get_app().invalidate()
        else:
            print(
                f"{SYSTEM_MSG}--> Unknown component '{component_key}'. Valid components: {list(valid_components.keys())}{RESET_COLOR}"
            )
    else:
        print(
            f"{SYSTEM_MSG}--> Usage: /toolbar [on|off|toggle <component>]{RESET_COLOR}"
        )


def handle_theme(args: list[str], session: SessionManager) -> None:
    """Handles listing and applying themes."""
    if not args:
        print(f"{SYSTEM_MSG}--- Available Themes ---{RESET_COLOR}")
        current_theme_name = settings.get("active_theme", "default")
        for name, desc in theme_manager.list_themes().items():
            prefix = " >" if name == current_theme_name else "  "
            print(f"{prefix} {name}: {desc}")
        return

    theme_name = args[0].lower()
    if theme_name not in theme_manager.list_themes():
        print(f"{SYSTEM_MSG}--> Theme '{theme_name}' not found.{RESET_COLOR}")
        return

    success, message = save_setting("active_theme", theme_name)
    if success:
        theme_manager.reload_theme()
        session.state.ui_refresh_needed = True
        get_app().invalidate()
        print(f"{SYSTEM_MSG}--> {message}{RESET_COLOR}")
    else:
        print(f"{SYSTEM_MSG}--> Error setting theme: {message}{RESET_COLOR}")


def _save_session_to_file(session: SessionManager, filename: str) -> bool:
    safe_name = sanitize_filename(filename.rsplit(".", 1)[0]) + ".json"
    filepath = config.SESSIONS_DIRECTORY / safe_name

    state_dict = asdict(session.state)
    state_dict["engine_name"] = session.state.engine.name
    del state_dict["engine"]
    state_dict["attachments"] = {
        str(k): v for k, v in session.state.attachments.items()
    }
    state_dict["persona_attachments"] = [
        str(p) for p in session.state.persona_attachments
    ]
    state_dict["current_persona"] = (
        session.state.current_persona.filename
        if session.state.current_persona
        else None
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


def handle_save(
    args: list[str], session: SessionManager, cli_history: InMemoryHistory
) -> bool:
    should_remember, should_stay = "--remember" in args, "--stay" in args
    filename_parts = [arg for arg in args if arg not in ("--remember", "--stay")]
    filename = " ".join(filename_parts)
    if not filename:
        print(f"{SYSTEM_MSG}--> Generating descriptive name...{RESET_COLOR}")
        filename, _ = session._perform_helper_request(
            prompts.LOG_RENAMING_PROMPT.format(
                log_content=session._get_history_for_helpers()
            ),
            50,
        )
        if not filename:
            print(
                f"{SYSTEM_MSG}--> Could not auto-generate a name. Save cancelled.{RESET_COLOR}"
            )
            return False
        filename = sanitize_filename(filename)

    session.state.command_history = cli_history.get_strings()
    if _save_session_to_file(session, filename):
        if not should_remember:
            session.state.exit_without_memory = True
        return not should_stay
    return False


def handle_load(args: list[str], session: SessionManager) -> bool:
    filename = " ".join(args)
    if not filename:
        print(f"{SYSTEM_MSG}--> Usage: /load <filename>{RESET_COLOR}")
        return False
    if not filename.endswith(".json"):
        filename += ".json"
    filepath = config.SESSIONS_DIRECTORY / filename
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        if "engine_name" not in data or "model" not in data:
            log.warning("Loaded session file %s is missing required keys.", filepath)
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
            data["attachments"] = {Path(k): v for k, v in data["attachments"].items()}

        if "persona_attachments" in data:
            data["persona_attachments"] = {Path(p) for p in data["persona_attachments"]}

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

        session.state = SessionState(**filtered_data)
        # Re-initialize context manager from the new state
        session.context_manager.attachments = session.state.attachments
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
        log.error("Error in handle_load: %s", e)
        print(f"{SYSTEM_MSG}--> Error loading session: {e}{RESET_COLOR}")
        return False


def handle_refresh(args: list[str], session: SessionManager) -> None:
    updated_files = session.context_manager.refresh_files(
        " ".join(args) if args else None
    )
    session.state.attachments = session.context_manager.attachments

    if updated_files:
        filenames = ", ".join(f"'{f}'" for f in updated_files)
        system_message_text = f"[SYSTEM] The content of the following files has been refreshed: {filenames}."
        system_message = construct_user_message(
            session.state.engine.name, system_message_text, []
        )
        session.state.history.append(system_message)


def handle_files(args: list[str], session: SessionManager) -> None:
    session.context_manager.list_files()


def handle_attach(args: list[str], session: SessionManager) -> None:
    if not args:
        print(f"{SYSTEM_MSG}--> Usage: /attach <path_to_file_or_dir>{RESET_COLOR}")
        return
    path_str = " ".join(args)

    before_paths = set(session.state.attachments.keys())
    session.context_manager.attach_file(path_str)
    session.state.attachments = session.context_manager.attachments
    after_paths = set(session.state.attachments.keys())

    newly_attached_paths = after_paths - before_paths
    if newly_attached_paths:
        path_names = ", ".join(f"'{p.name}'" for p in newly_attached_paths)
        system_message_text = f"[SYSTEM] The following content has been attached to the context: {path_names}."
        system_message = construct_user_message(
            session.state.engine.name, system_message_text, []
        )
        session.state.history.append(system_message)


def handle_detach(args: list[str], session: SessionManager) -> None:
    filename = " ".join(args)
    if not filename:
        print(f"{SYSTEM_MSG}--> Usage: /detach <filename>{RESET_COLOR}")
        return

    # Find the full path to remove, as context_manager tracks full paths
    path_to_remove = next(
        (p for p in session.state.attachments if p.name == filename), None
    )
    if path_to_remove:
        session.context_manager.detach_file(filename)
        # Also remove from persona tracking if it was a persona file
        session.state.persona_attachments.discard(path_to_remove)
        session.state.attachments = session.context_manager.attachments

        system_message_text = (
            f"[SYSTEM] The file '{filename}' has been detached from the context."
        )
        system_message = construct_user_message(
            session.state.engine.name, system_message_text, []
        )
        session.state.history.append(system_message)
    else:
        print(f"{SYSTEM_MSG}--> No attached file named '{filename}'.{RESET_COLOR}")


def handle_personas(args: list[str], session: SessionManager) -> None:
    personas = persona_manager.list_personas()
    if not personas:
        print(f"{SYSTEM_MSG}--> No personas found.{RESET_COLOR}")
        return
    print(f"{SYSTEM_MSG}--- Available Personas ---{RESET_COLOR}")
    for p in personas:
        print(f"  - {p.filename.replace('.json', '')}: {p.description}")


def handle_persona(args: list[str], session: SessionManager) -> None:
    name = " ".join(args)
    if not name:
        print(f"{SYSTEM_MSG}--> Usage: /persona <name> OR /persona clear{RESET_COLOR}")
        return

    # Remove attachments from the old persona, if any
    for path in session.state.persona_attachments:
        session.state.attachments.pop(path, None)
    session.state.persona_attachments.clear()

    if name.lower() == "clear":
        if not session.state.current_persona:
            print(f"{SYSTEM_MSG}--> No active persona to clear.{RESET_COLOR}")
            return
        session.state.system_prompt = session.state.initial_system_prompt
        session.state.current_persona = None
        print(f"{SYSTEM_MSG}--> Persona cleared.{RESET_COLOR}")
        session.state.history.append(
            construct_user_message(
                session.state.engine.name, "[SYSTEM] Persona cleared.", []
            )
        )
        return

    new_persona = persona_manager.load_persona(name)
    if not new_persona:
        print(f"{SYSTEM_MSG}--> Persona '{name}' not found.{RESET_COLOR}")
        return

    if new_persona.engine and new_persona.engine != session.state.engine.name:
        print(
            f"{SYSTEM_MSG}--> Switching engine to {new_persona.engine} for persona '{new_persona.name}'...{RESET_COLOR}"
        )
        handle_engine([new_persona.engine], session)

    if new_persona.model:
        session.state.model = new_persona.model
    if new_persona.max_tokens is not None:
        session.state.max_tokens = new_persona.max_tokens
    if new_persona.stream is not None:
        session.state.stream_active = new_persona.stream

    # Add attachments from the new persona
    if new_persona.attachments:
        # Import dynamically to avoid circular dependency at module level
        from .managers.context_manager import ContextManager

        temp_context = ContextManager(
            files_arg=new_persona.attachments, memory_enabled=False, exclude_arg=[]
        )
        session.state.attachments.update(temp_context.attachments)
        session.state.persona_attachments.update(temp_context.attachments.keys())
        print(
            f"{SYSTEM_MSG}--> Attached {len(temp_context.attachments)} file(s) from persona.{RESET_COLOR}"
        )

    session.state.system_prompt = new_persona.system_prompt
    session.state.current_persona = new_persona
    print(f"{SYSTEM_MSG}--> Switched to persona: '{new_persona.name}'{RESET_COLOR}")
    session.state.history.append(
        construct_user_message(
            session.state.engine.name,
            f"[SYSTEM] Persona switched to '{new_persona.name}'.",
            [],
        )
    )


def handle_image(args: list[str], session: SessionManager) -> None:
    session.image_workflow.run(args)


# --- Multi-Chat Command Handler Functions ---


def _save_multichat_session_to_file(session: MultiChatSession, filename: str) -> bool:
    safe_name = sanitize_filename(filename.rsplit(".", 1)[0]) + ".json"
    filepath = config.SESSIONS_DIRECTORY / safe_name

    state_dict = asdict(session.state)
    state_dict["session_type"] = "multichat"
    # Engines are not serializable, so we remove them
    del state_dict["openai_engine"]
    del state_dict["gemini_engine"]

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        print(f"{SYSTEM_MSG}--> Multi-chat session saved to: {filepath}{RESET_COLOR}")
        return True
    except (OSError, TypeError) as e:
        log.error("Failed to save multi-chat session state: %s", e)
        print(f"{SYSTEM_MSG}--> Error saving multi-chat session: {e}{RESET_COLOR}")
        return False


def handle_multichat_exit(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> bool:
    if args:
        session.state.custom_log_rename = " ".join(args)
    return True


def handle_multichat_quit(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> bool:
    session.state.force_quit = True
    return True


def handle_multichat_help(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> None:
    display_help("multichat")


def handle_multichat_history(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> None:
    import json

    print(json.dumps(session.state.shared_history, indent=2))


def handle_multichat_debug(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> None:
    session.state.debug_active = not session.state.debug_active
    status = "ENABLED" if session.state.debug_active else "DISABLED"
    print(
        f"{SYSTEM_MSG}--> Session-specific debug logging is now {status}.{RESET_COLOR}"
    )


def handle_multichat_remember(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> None:
    print(
        f"{SYSTEM_MSG}--> /remember is not yet implemented for multi-chat.{RESET_COLOR}"
    )


def handle_multichat_max_tokens(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> None:
    if args and args[0].isdigit():
        session.state.max_tokens = int(args[0])
        print(
            f"{SYSTEM_MSG}--> Max tokens for this session set to: {session.state.max_tokens}.{RESET_COLOR}"
        )
    else:
        print(f"{SYSTEM_MSG}--> Usage: /max-tokens <number>{RESET_COLOR}")


def handle_multichat_clear(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> None:
    confirm = prompt(
        "This will clear all conversation history. Type `proceed` to confirm: "
    )
    if confirm.lower() == "proceed":
        session.state.shared_history.clear()
        print(f"{SYSTEM_MSG}--> Conversation history has been cleared.{RESET_COLOR}")
    else:
        print(f"{SYSTEM_MSG}--> Clear cancelled.{RESET_COLOR}")


def handle_multichat_model(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> None:
    if len(args) != 2:
        print(f"{SYSTEM_MSG}--> Usage: /model <gpt|gem> <model_name>{RESET_COLOR}")
        return

    engine_alias, model_name = args[0].lower(), args[1]
    engine_map = {"gpt": "openai", "gem": "gemini"}

    if engine_alias not in engine_map:
        print(f"{SYSTEM_MSG}--> Invalid engine alias. Use 'gpt' or 'gem'.{RESET_COLOR}")
        return

    target_engine = engine_map[engine_alias]
    if target_engine == "openai":
        session.state.openai_model = model_name
        print(f"{SYSTEM_MSG}--> OpenAI model set to: {model_name}{RESET_COLOR}")
    else:  # gemini
        session.state.gemini_model = model_name
        print(f"{SYSTEM_MSG}--> Gemini model set to: {model_name}{RESET_COLOR}")


def handle_multichat_state(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> None:
    print(f"{SYSTEM_MSG}--- Multi-Chat Session State ---{RESET_COLOR}")
    print(f"  OpenAI Model: {session.state.openai_model}")
    print(f"  Gemini Model: {session.state.gemini_model}")
    print(f"  Max Tokens: {session.state.max_tokens or 'Default'}")
    print(f"  Debug Logging: {'On' if session.state.debug_active else 'Off'}")
    print(f"  System Prompts: {'Active' if session.state.system_prompts else 'None'}")
    if session.state.initial_image_data:
        print(f"  Attached Images: {len(session.state.initial_image_data)}")


def handle_multichat_save(
    args: list[str], session: MultiChatSession, cli_history: InMemoryHistory
) -> bool:
    should_remember, should_stay = "--remember" in args, "--stay" in args
    filename_parts = [arg for arg in args if arg not in ("--remember", "--stay")]
    filename = " ".join(filename_parts)
    if not filename:
        print(
            f"{SYSTEM_MSG}--> Usage: /save <filename> [--stay] [--remember]{RESET_COLOR}"
        )
        return False

    session.state.command_history = cli_history.get_strings()
    if _save_multichat_session_to_file(session, filename):
        if not should_remember:
            session.state.exit_without_memory = True
        return not should_stay
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
    "/toolbar": handle_toolbar,
    "/theme": handle_theme,
    "/save": handle_save,
    "/load": handle_load,
    "/refresh": handle_refresh,
    "/files": handle_files,
    "/attach": handle_attach,
    "/detach": handle_detach,
    "/personas": handle_personas,
    "/persona": handle_persona,
    "/image": handle_image,
}

MULTICHAT_COMMAND_MAP = {
    "/exit": handle_multichat_exit,
    "/quit": handle_multichat_quit,
    "/help": handle_multichat_help,
    "/history": handle_multichat_history,
    "/debug": handle_multichat_debug,
    "/remember": handle_multichat_remember,
    "/max-tokens": handle_multichat_max_tokens,
    "/clear": handle_multichat_clear,
    "/model": handle_multichat_model,
    "/state": handle_multichat_state,
    "/save": handle_multichat_save,
}
