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
These commands are thin wrappers that delegate state manipulation to a
SessionManager instance.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from . import config, theme_manager, utils
from .logger import log
from .settings import save_setting, settings

if TYPE_CHECKING:
    from .session_manager import MultiChatSessionState, SessionManager


# --- Single-Chat Command Handler Functions ---


def handle_exit(args: list[str], session: SessionManager) -> bool:
    if args:
        session.set_custom_log_rename(" ".join(args))
    return True


def handle_quit(args: list[str], session: SessionManager) -> bool:
    session.set_force_quit()
    return True


def handle_help(args: list[str], session: SessionManager) -> None:
    utils.display_help("chat")


def handle_stream(args: list[str], session: SessionManager) -> None:
    session.toggle_stream()


def handle_debug(args: list[str], session: SessionManager) -> None:
    session.toggle_debug()


def handle_memory(args: list[str], session: SessionManager) -> None:
    session.view_memory()


def handle_remember(args: list[str], session: SessionManager) -> None:
    if not args:
        session.consolidate_memory()
    else:
        session.inject_memory(" ".join(args))


def handle_max_tokens(args: list[str], session: SessionManager) -> None:
    if args and args[0].isdigit():
        session.set_max_tokens(int(args[0]))
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")


def handle_clear(args: list[str], session: SessionManager) -> None:
    confirm = prompt(
        "This will clear all conversation history. Type `proceed` to confirm: "
    )
    if confirm.lower() == "proceed":
        session.clear_history()
    else:
        print(f"{utils.SYSTEM_MSG}--> Clear cancelled.{utils.RESET_COLOR}")


def handle_model(args: list[str], session: SessionManager) -> None:
    if args:
        session.set_model(args[0])
    else:
        new_model = utils.select_model(session.state.engine, "chat")
        session.set_model(new_model)


def handle_engine(args: list[str], session: SessionManager) -> None:
    new_engine_name = args[0] if args else None
    session.switch_engine(new_engine_name)


def handle_history(args: list[str], session: SessionManager) -> None:
    session.view_history()


def handle_state(args: list[str], session: SessionManager) -> None:
    session.view_state()


def handle_set(args: list[str], session: SessionManager) -> None:
    if len(args) == 2:
        key, value = args[0], args[1]
        success, message = save_setting(key, value)
        print(f"{utils.SYSTEM_MSG}--> {message}{utils.RESET_COLOR}")
        if success and key == "active_theme":
            theme_manager.reload_theme()
            session.state.ui_refresh_needed = True
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /set <key> <value>.{utils.RESET_COLOR}")


def handle_toolbar(args: list[str], session: SessionManager) -> None:
    """Handles all toolbar configuration commands."""
    if not args:
        print(f"{utils.SYSTEM_MSG}--- Toolbar Settings ---{utils.RESET_COLOR}")
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
        _, message = save_setting("toolbar_enabled", command)
        print(f"{utils.SYSTEM_MSG}--> {message}{utils.RESET_COLOR}")
    elif command == "toggle" and len(args) == 2:
        component_key = args[1].lower()
        if component_key in valid_components:
            setting_key = valid_components[component_key]
            current_value = settings[setting_key]
            _, message = save_setting(setting_key, str(not current_value))
            print(f"{utils.SYSTEM_MSG}--> {message}{utils.RESET_COLOR}")
        else:
            print(
                f"{utils.SYSTEM_MSG}--> Unknown component '{component_key}'. Valid components: {list(valid_components.keys())}{utils.RESET_COLOR}"
            )
    else:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /toolbar [on|off|toggle <component>]{utils.RESET_COLOR}"
        )


def handle_theme(args: list[str], session: SessionManager) -> None:
    """Handles listing and applying themes."""
    if not args:
        print(f"{utils.SYSTEM_MSG}--- Available Themes ---{utils.RESET_COLOR}")
        current_theme_name = settings.get("active_theme", "default")
        for name, desc in theme_manager.list_themes().items():
            prefix = " >" if name == current_theme_name else "  "
            print(f"{prefix} {name}: {desc}")
        return

    theme_name = args[0].lower()
    if theme_name not in theme_manager.list_themes():
        print(
            f"{utils.SYSTEM_MSG}--> Theme '{theme_name}' not found.{utils.RESET_COLOR}"
        )
        return

    success, message = save_setting("active_theme", theme_name)
    if success:
        theme_manager.reload_theme()
        session.state.ui_refresh_needed = True
        print(f"{utils.SYSTEM_MSG}--> {message}{utils.RESET_COLOR}")
    else:
        print(
            f"{utils.SYSTEM_MSG}--> Error setting theme: {message}{utils.RESET_COLOR}"
        )


def handle_save(
    args: list[str], session: SessionManager, cli_history: InMemoryHistory
) -> bool:
    return session.save(args, cli_history.get_strings())


def handle_load(args: list[str], session: SessionManager) -> None:
    if args:
        session.load(" ".join(args))
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /load <filename>{utils.RESET_COLOR}")


def handle_refresh(args: list[str], session: SessionManager) -> None:
    session.refresh_files(" ".join(args) if args else None)


def handle_files(args: list[str], session: SessionManager) -> None:
    session.list_files()


def handle_attach(args: list[str], session: SessionManager) -> None:
    if not args:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /attach <path_to_file_or_dir>{utils.RESET_COLOR}"
        )
        return
    session.attach_file(" ".join(args))


def handle_detach(args: list[str], session: SessionManager) -> None:
    if not args:
        print(f"{utils.SYSTEM_MSG}--> Usage: /detach <filename>{utils.RESET_COLOR}")
        return
    session.detach_file(" ".join(args))


def handle_personas(args: list[str], session: SessionManager) -> None:
    session.list_personas()


def handle_persona(args: list[str], session: SessionManager) -> None:
    if not args:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /persona <name> OR /persona clear{utils.RESET_COLOR}"
        )
        return
    session.switch_persona(" ".join(args))


def handle_image(args: list[str], session: SessionManager) -> None:
    """
    Handles all image generation workflows by delegating to the SessionManager.
    This function acts as a simple dispatcher. It passes the user's arguments
    directly to the session manager, which contains the complex state and
    workflow logic. This keeps the command map clean and adheres to our
    architectural principle of centralizing state management in the session.
    """
    session.handle_image_command(args)


# --- Multi-Chat Command Handler Functions ---


def _save_multichat_session_to_file(
    state: MultiChatSessionState, filename: str
) -> bool:
    safe_name = utils.sanitize_filename(filename.rsplit(".", 1)[0]) + ".json"
    filepath = config.SESSIONS_DIRECTORY / safe_name

    state_dict = asdict(state)
    state_dict["session_type"] = "multichat"
    # Engines are not serializable, so we remove them
    del state_dict["openai_engine"]
    del state_dict["gemini_engine"]

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        print(
            f"{utils.SYSTEM_MSG}--> Multi-chat session saved to: {filepath}{utils.RESET_COLOR}"
        )
        return True
    except (OSError, TypeError) as e:
        log.error("Failed to save multi-chat session state: %s", e)
        print(
            f"{utils.SYSTEM_MSG}--> Error saving multi-chat session: {e}{utils.RESET_COLOR}"
        )
        return False


def handle_multichat_exit(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> bool:
    if args:
        state.custom_log_rename = " ".join(args)
    return True


def handle_multichat_quit(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> bool:
    state.force_quit = True
    return True


def handle_multichat_help(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    utils.display_help("multichat")


def handle_multichat_history(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    import json

    print(json.dumps(state.shared_history, indent=2))


def handle_multichat_debug(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    state.debug_active = not state.debug_active
    status = "ENABLED" if state.debug_active else "DISABLED"
    print(
        f"{utils.SYSTEM_MSG}--> Session-specific debug logging is now {status}.{utils.RESET_COLOR}"
    )


def handle_multichat_remember(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    print(
        f"{utils.SYSTEM_MSG}--> /remember is not yet implemented for multi-chat.{utils.RESET_COLOR}"
    )


def handle_multichat_max_tokens(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    if args and args[0].isdigit():
        state.max_tokens = int(args[0])
        print(
            f"{utils.SYSTEM_MSG}--> Max tokens for this session set to: {state.max_tokens}.{utils.RESET_COLOR}"
        )
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")


def handle_multichat_clear(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
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
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    if len(args) != 2:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /model <gpt|gem> <model_name>{utils.RESET_COLOR}"
        )
        return

    engine_alias, model_name = args[0].lower(), args[1]
    engine_map = {"gpt": "openai", "gem": "gemini"}

    if engine_alias not in engine_map:
        print(
            f"{utils.SYSTEM_MSG}--> Invalid engine alias. Use 'gpt' or 'gem'.{utils.RESET_COLOR}"
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
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    print(f"{utils.SYSTEM_MSG}--- Multi-Chat Session State ---{utils.RESET_COLOR}")
    print(f"  OpenAI Model: {state.openai_model}")
    print(f"  Gemini Model: {state.gemini_model}")
    print(f"  Max Tokens: {state.max_tokens or 'Default'}")
    print(f"  Debug Logging: {'On' if state.debug_active else 'Off'}")
    print(f"  System Prompts: {'Active' if state.system_prompts else 'None'}")
    if state.initial_image_data:
        print(f"  Attached Images: {len(state.initial_image_data)}")


def handle_multichat_save(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> bool:
    should_remember, should_stay = "--remember" in args, "--stay" in args
    filename_parts = [arg for arg in args if arg not in ("--remember", "--stay")]
    filename = " ".join(filename_parts)
    if not filename:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /save <filename> [--stay] [--remember]{utils.RESET_COLOR}"
        )
        return False

    state.command_history = cli_history.get_strings()
    if _save_multichat_session_to_file(state, filename):
        if not should_remember:
            state.exit_without_memory = True
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
