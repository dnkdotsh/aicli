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


import os
import sys
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import ANSI

from . import config
from . import api_client
from . import utils
from . import settings as app_settings
from . import personas as persona_manager
from .engine import AIEngine, get_engine
from .logger import log
from .prompts import (HISTORY_SUMMARY_PROMPT, MEMORY_INTEGRATION_PROMPT,
                      LOG_RENAMING_PROMPT, DIRECT_MEMORY_INJECTION_PROMPT)

def _format_bytes(byte_count: int) -> str:
    """Converts a byte count to a human-readable string (KB, MB, etc.)."""
    if byte_count is None: return "0 B"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while byte_count >= power and n < len(power_labels) -1 :
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}"

@dataclass
class SessionState:
    """A dataclass to hold the state of an interactive chat session."""
    engine: AIEngine
    model: str
    system_prompt: str | None
    initial_system_prompt: str | None # The original system prompt from startup
    current_persona: persona_manager.Persona | None
    max_tokens: int | None
    memory_enabled: bool
    attachments: dict = field(default_factory=dict) # Maps Path object to file content string
    attached_images: list = field(default_factory=list)
    history: list = field(default_factory=list)
    command_history: list[str] = field(default_factory=list)
    debug_active: bool = False
    stream_active: bool = True
    session_raw_logs: list = field(default_factory=list)
    exit_without_memory: bool = False
    force_quit: bool = False
    custom_log_rename: str | None = None

def _assemble_full_system_prompt(state: SessionState) -> str | None:
    """Combines base system prompt and attachments into a single string for the API."""
    prompt_parts = []
    if state.system_prompt:
        prompt_parts.append(state.system_prompt)

    if state.attachments:
        attachment_texts = []
        for path, content in state.attachments.items():
            attachment_texts.append(f"--- FILE: {path.as_posix()} ---\n{content}")
        prompt_parts.append(f"--- ATTACHED FILES ---\n" + "\n\n".join(attachment_texts))

    return "\n\n".join(prompt_parts) if prompt_parts else None

def _generate_session_name(engine: AIEngine, history: list) -> str | None:
    """Generates a descriptive name for the session using an AI model."""
    log_content = "\n".join([f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}" for msg in history[:10]]) # Use first 5 turns
    prompt_text = LOG_RENAMING_PROMPT.format(log_content=log_content)

    helper_model_key = 'helper_model_openai' if engine.name == 'openai' else 'helper_model_gemini'
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(engine.name, prompt_text, [])]

    suggested_name, _ = api_client.perform_chat_request(
        engine=engine, model=task_model, messages_or_contents=messages,
        system_prompt=None, max_tokens=app_settings.settings['log_rename_max_tokens'], stream=False
    )

    if suggested_name and not suggested_name.startswith("API Error:"):
        return utils.sanitize_filename(suggested_name.strip())

    reason = suggested_name or "No response from helper model."
    log.warning("Could not generate a descriptive name for the session log. Reason: %s", reason)
    return None

def _save_session_to_file(state: SessionState, filename: str) -> bool:
    """Serializes the current SessionState to a JSON file."""
    if not filename.endswith('.json'):
        filename += '.json'

    safe_name = utils.sanitize_filename(filename.rsplit('.', 1)[0]) + '.json'
    filepath = config.SESSIONS_DIRECTORY / safe_name

    state_dict = asdict(state)
    state_dict['engine_name'] = state.engine.name
    del state_dict['engine']
    # Convert non-serializable types to strings for saving
    state_dict['attachments'] = {str(k): v for k, v in state.attachments.items()}
    if state.current_persona:
        state_dict['current_persona'] = state.current_persona.filename
    else:
        state_dict['current_persona'] = None


    try:
        utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2)
        print(f"{utils.SYSTEM_MSG}--> Session saved successfully to: {filepath}{utils.RESET_COLOR}")
        return True
    except (IOError, TypeError) as e:
        log.error("Failed to save session state: %s", e)
        print(f"{utils.SYSTEM_MSG}--> Error saving session: {e}{utils.RESET_COLOR}")
        return False

def load_session_from_file(filepath: Path) -> SessionState | None:
    """Loads a session from a file and reconstructs the SessionState."""
    try:
        if not filepath.exists():
            raise FileNotFoundError("The specified session file does not exist.")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'attachments' in data:
            data['attachments'] = {Path(k): v for k, v in data['attachments'].items()}
        if 'attached_text_files' in data:
            del data['attached_text_files']
        if 'initial_image_data' in data:
            data['attached_images'] = data.pop('initial_image_data')

        # Load persona object from its filename
        if 'current_persona' in data and data['current_persona']:
            data['current_persona'] = persona_manager.load_persona(data['current_persona'])

        data.pop('force_quit', None)
        data.pop('custom_log_rename', None)

        engine_name = data.pop('engine_name')
        api_key = api_client.check_api_keys(engine_name)
        engine_instance = get_engine(engine_name, api_key)

        state = SessionState(engine=engine_instance, **data)

        print(f"{utils.SYSTEM_MSG}--> Session loaded successfully from: {filepath}{utils.RESET_COLOR}")
        print(f"{utils.SYSTEM_MSG}--- Last few messages ---{utils.RESET_COLOR}")
        history_slice = state.history[-4:]
        for msg in history_slice:
            role = msg.get('role')
            content = utils.extract_text_from_message(msg)
            if role == 'user':
                print(f"{utils.USER_PROMPT}You: {utils.RESET_COLOR}{content}")
            elif role in ['assistant', 'model']:
                print(f"{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{content}")
        print(f"{utils.SYSTEM_MSG}-------------------------{utils.RESET_COLOR}")

        return state
    except (FileNotFoundError, IOError, json.JSONDecodeError, KeyError, TypeError, api_client.MissingApiKeyError) as e:
        log.error("Failed to load session from %s: %s", filepath, e)
        return None

def _condense_chat_history(state: SessionState):
    """Summarizes the oldest turns and replaces them with a summary message."""
    print(f"\n{utils.SYSTEM_MSG}--> Condensing conversation history to preserve memory...{utils.RESET_COLOR}")
    num_messages_to_trim = config.HISTORY_SUMMARY_TRIM_TURNS * 2
    turns_to_summarize = state.history[:num_messages_to_trim]
    remaining_history = state.history[num_messages_to_trim:]

    log_content = "\n".join([f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}" for msg in turns_to_summarize])
    summary_prompt = HISTORY_SUMMARY_PROMPT.format(log_content=log_content)

    helper_model_key = 'helper_model_openai' if state.engine.name == 'openai' else 'helper_model_gemini'
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(state.engine.name, summary_prompt, [])]

    summary_text, _ = api_client.perform_chat_request(
        engine=state.engine, model=task_model, messages_or_contents=messages,
        system_prompt=None, max_tokens=app_settings.settings['summary_max_tokens'], stream=False
    )
    if not summary_text or summary_text.startswith("API Error:"):
        reason = summary_text or "No response from helper model."
        log.warning("History summarization failed. Proceeding with full history. Reason: %s", reason)
        return

    summary_message = utils.construct_user_message(state.engine.name, f"[PREVIOUSLY DISCUSSED]:\n{summary_text.strip()}", [])
    state.history = [summary_message] + remaining_history
    print(f"{utils.SYSTEM_MSG}--> History condensed successfully.{utils.RESET_COLOR}")

# --- Command Handler Functions ---

def _handle_command_exit(args: list, state: SessionState, cli_history: InMemoryHistory) -> bool:
    if args:
        state.custom_log_rename = ' '.join(args)
    return True

def _handle_command_quit(args: list, state: SessionState, cli_history: InMemoryHistory) -> bool:
    state.force_quit = True
    return True

def _handle_command_help(args: list, state: SessionState, cli_history: InMemoryHistory):
    utils.display_help('chat')

def _handle_command_stream(args: list, state: SessionState, cli_history: InMemoryHistory):
    state.stream_active = not state.stream_active
    status = "ENABLED" if state.stream_active else "DISABLED"
    print(f"{utils.SYSTEM_MSG}--> Response streaming is now {status} for this session.{utils.RESET_COLOR}")

def _handle_command_debug(args: list, state: SessionState, cli_history: InMemoryHistory):
    state.debug_active = not state.debug_active
    status = "ENABLED" if state.debug_active else "DISABLED"
    print(f"{utils.SYSTEM_MSG}--> Session-specific debug logging is now {status}.{utils.RESET_COLOR}")

def _handle_command_memory(args: list, state: SessionState, cli_history: InMemoryHistory):
    try:
        with open(config.PERSISTENT_MEMORY_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"{utils.SYSTEM_MSG}--- Persistent Memory ---{utils.RESET_COLOR}")
        print(content)
        print(f"{utils.SYSTEM_MSG}-------------------------{utils.RESET_COLOR}")
    except FileNotFoundError:
        print(f"{utils.SYSTEM_MSG}--> Persistent memory is currently empty.{utils.RESET_COLOR}")
    except IOError as e:
        log.error("Could not read persistent memory file: %s", e)
        print(f"{utils.SYSTEM_MSG}--> Error reading memory file: {e}{utils.RESET_COLOR}")

def _handle_command_remember(args: list, state: SessionState, cli_history: InMemoryHistory):
    if not args:
        if not state.history:
            print(f"{utils.SYSTEM_MSG}--> Nothing to consolidate; conversation history is empty.{utils.RESET_COLOR}")
            return
        _consolidate_session_into_memory(state.engine, state.model, state.history)
    else:
        fact_to_remember = ' '.join(args)
        _inject_fact_into_memory(state.engine, fact_to_remember)

def _handle_command_max_tokens(args: list, state: SessionState, cli_history: InMemoryHistory):
    if args and args[0].isdigit():
        state.max_tokens = int(args[0])
        print(f"{utils.SYSTEM_MSG}--> Max tokens for this session set to: {state.max_tokens}.{utils.RESET_COLOR}")
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")

def _handle_command_clear(args: list, state: SessionState, cli_history: InMemoryHistory):
    confirm = prompt("This will clear all conversation history. Type `proceed` to confirm: ")
    if confirm.lower() == 'proceed':
        state.history.clear()
        print(f"{utils.SYSTEM_MSG}--> Conversation history has been cleared.{utils.RESET_COLOR}")
    else:
        print(f"{utils.SYSTEM_MSG}--> Clear cancelled.{utils.RESET_COLOR}")

def _handle_command_model(args: list, state: SessionState, cli_history: InMemoryHistory):
    from . import handlers  # Local import to avoid circular dependency
    if args:
        state.model = args[0]
        print(f"{utils.SYSTEM_MSG}--> Model temporarily set to: {state.model}{utils.RESET_COLOR}")
    else:
        state.model = handlers.select_model(state.engine, 'chat')
        print(f"{utils.SYSTEM_MSG}--> Model changed to: {state.model}{utils.RESET_COLOR}")

def _handle_command_engine(args: list, state: SessionState, cli_history: InMemoryHistory):
    new_engine_name = args[0] if args else ('gemini' if state.engine.name == 'openai' else 'openai')
    if new_engine_name not in ['openai', 'gemini']:
        print(f"{utils.SYSTEM_MSG}--> Unknown engine: {new_engine_name}. Use 'openai' or 'gemini'.{utils.RESET_COLOR}")
        return
    try:
        from .engine import get_engine
        new_api_key = api_client.check_api_keys(new_engine_name)
        state.history = utils.translate_history(state.history, new_engine_name)
        state.engine = get_engine(new_engine_name, new_api_key)
        model_key = 'default_openai_chat_model' if new_engine_name == 'openai' else 'default_gemini_model'
        state.model = app_settings.settings[model_key]
        print(f"{utils.SYSTEM_MSG}--> Engine switched to {state.engine.name.capitalize()}. Model set to default: {state.model}. History translated.{utils.RESET_COLOR}")
    except api_client.MissingApiKeyError:
         print(f"{utils.SYSTEM_MSG}--> Switch to {new_engine_name.capitalize()} failed: API key not found.{utils.RESET_COLOR}")

def _handle_command_history(args: list, state: SessionState, cli_history: InMemoryHistory):
    print(json.dumps(state.history, indent=2))

def _handle_command_state(args: list, state: SessionState, cli_history: InMemoryHistory):
    print(f"{utils.SYSTEM_MSG}--- Session State ---{utils.RESET_COLOR}")
    if state.current_persona:
        print(f"  Active Persona: {state.current_persona.name} ({state.current_persona.filename})")
    else:
        print(f"  Active Persona: None")
    print(f"  Engine: {state.engine.name}")
    print(f"  Model: {state.model}")
    print(f"  Max Tokens: {state.max_tokens or 'Default'}")
    print(f"  Streaming: {'On' if state.stream_active else 'Off'}")
    print(f"  Memory Enabled (on exit): {'On' if state.memory_enabled else 'Off'}")
    print(f"  Debug Logging: {'On' if state.debug_active else 'Off'}")
    print(f"  System Prompt: {'Active' if state.system_prompt else 'None'}")
    if state.attachments:
        total_size = sum(p.stat().st_size for p in state.attachments.keys() if p.exists())
        print(f"  Attached Text Files: {len(state.attachments)} ({_format_bytes(total_size)})")
    if state.attached_images:
         print(f"  Attached Images: {len(state.attached_images)}")

def _handle_command_set(args: list, state: SessionState, cli_history: InMemoryHistory):
    if not args:
        print(f"{utils.SYSTEM_MSG}--- Current Settings ---{utils.RESET_COLOR}")
        for key, value in sorted(app_settings.settings.items()):
            print(f"  {key}: {value}")
        print(f"{utils.SYSTEM_MSG}----------------------{utils.RESET_COLOR}")
    elif len(args) == 2:
        app_settings.save_setting(args[0], args[1])
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /set <key> <value> OR /set to list all settings.{utils.RESET_COLOR}")

def _handle_command_save(args: list, state: SessionState, cli_history: InMemoryHistory) -> bool:
    should_remember, should_stay = False, False
    filename_parts = [arg for arg in args if arg not in ('--remember', '--stay')]
    if '--remember' in args: should_remember = True
    if '--stay' in args: should_stay = True

    filename = ' '.join(filename_parts) if filename_parts else None

    if not filename:
        print(f"{utils.SYSTEM_MSG}--> Generating descriptive name for session...{utils.RESET_COLOR}")
        filename = _generate_session_name(state.engine, state.history)
        if not filename:
            print(f"{utils.SYSTEM_MSG}--> Could not auto-generate a name. Save cancelled.{utils.RESET_COLOR}")
            return False

    state.command_history = cli_history.get_strings()

    if _save_session_to_file(state, filename):
        if should_stay:
            return False
        if not should_remember:
            state.exit_without_memory = True
        return True
    return False

def _handle_command_load(args: list, state: SessionState, cli_history: InMemoryHistory):
    if args:
        filename = ' '.join(args)
        if not filename.endswith('.json'):
            filename += '.json'
        filepath = config.SESSIONS_DIRECTORY / filename
        new_state = load_session_from_file(filepath)
        if new_state:
            # Replace current state with loaded state
            for key, value in asdict(new_state).items():
                setattr(state, key, value)
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /load <filename>{utils.RESET_COLOR}")

def _handle_command_refresh(args: list, state: SessionState, cli_history: InMemoryHistory):
    search_term = ' '.join(args)
    if not args:
        if not state.attachments:
            print(f"{utils.SYSTEM_MSG}--> No files attached to refresh.{utils.RESET_COLOR}")
            return
        paths_to_refresh = list(state.attachments.keys())
    else:
        paths_to_refresh = [p for p in state.attachments.keys() if search_term in p.name]
        if not paths_to_refresh:
            print(f"{utils.SYSTEM_MSG}--> No attached files found matching '{search_term}'.{utils.RESET_COLOR}")
            return

    updated_files, files_to_remove = [], []
    for path in paths_to_refresh:
        try:
            if path.suffix.lower() == '.zip': continue
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                state.attachments[path] = f.read()
            updated_files.append(path.name)
        except (IOError, FileNotFoundError) as e:
            log.warning("Could not re-read '%s': %s. Removing from context.", path.name, e)
            print(f"{utils.SYSTEM_MSG}--> Warning: Could not re-read '{path.name}'. Removing from session.{utils.RESET_COLOR}")
            files_to_remove.append(path)

    for path in files_to_remove:
        del state.attachments[path]

    if updated_files:
        print(f"{utils.SYSTEM_MSG}--> Refreshed {len(updated_files)} file(s): {', '.join(updated_files)}{utils.RESET_COLOR}")
        notification_text = f"[SYSTEM] The content of the following file(s) has been updated: {', '.join(updated_files)}."
        state.history.append(utils.construct_user_message(state.engine.name, notification_text, []))
        print(f"{utils.SYSTEM_MSG}--> Notified AI of the context update.{utils.RESET_COLOR}")

def _handle_command_files(args: list, state: SessionState, cli_history: InMemoryHistory):
    if not state.attachments:
        print(f"{utils.SYSTEM_MSG}--> No text files are currently attached.{utils.RESET_COLOR}")
        return

    file_list = sorted([(p, p.stat().st_size if p.exists() else 0) for p in state.attachments.keys()], key=lambda item: item[1], reverse=True)
    print(f"{utils.SYSTEM_MSG}--- Attached Text Files (Sorted by Size) ---{utils.RESET_COLOR}")
    for path, size in file_list:
        print(f"  - {path.name} ({_format_bytes(size)})")
    print(f"{utils.SYSTEM_MSG}------------------------------------------{utils.RESET_COLOR}")

def _handle_command_attach(args: list, state: SessionState, cli_history: InMemoryHistory):
    if not args:
        print(f"{utils.SYSTEM_MSG}--> Usage: /attach <path_to_file_or_dir>{utils.RESET_COLOR}")
        return

    path_str = ' '.join(args)
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"{utils.SYSTEM_MSG}--> Error: Path not found: {path_str}{utils.RESET_COLOR}")
        return
    if path in state.attachments:
        print(f"{utils.SYSTEM_MSG}--> Error: File '{path.name}' is already attached.{utils.RESET_COLOR}")
        return

    if utils.is_supported_text_file(path):
        try:
            state.attachments[path] = path.read_text(encoding='utf-8', errors='ignore')
            print(f"{utils.SYSTEM_MSG}--> Attached file: {path.name} ({_format_bytes(path.stat().st_size)}){utils.RESET_COLOR}")
            notification_text = f"[SYSTEM] The user has attached a new file: {path.name}."
            state.history.append(utils.construct_user_message(state.engine.name, notification_text, []))
        except (IOError, UnicodeDecodeError) as e:
            log.warning("Failed to attach file '%s': %s", path.name, e)
            print(f"{utils.SYSTEM_MSG}--> Error reading file '{path.name}': {e}{utils.RESET_COLOR}")
    else:
        print(f"{utils.SYSTEM_MSG}--> Error: File type '{path.suffix}' is not supported for attachment.{utils.RESET_COLOR}")

def _handle_command_detach(args: list, state: SessionState, cli_history: InMemoryHistory):
    if not args:
        print(f"{utils.SYSTEM_MSG}--> Usage: /detach <filename>{utils.RESET_COLOR}")
        return

    search_term = ' '.join(args)
    path_to_remove = next((p for p in state.attachments.keys() if p.name == search_term), None)

    if path_to_remove:
        del state.attachments[path_to_remove]
        print(f"{utils.SYSTEM_MSG}--> Detached file: {path_to_remove.name}{utils.RESET_COLOR}")
        notification_text = f"[SYSTEM] The user has detached the file: {path_to_remove.name}."
        state.history.append(utils.construct_user_message(state.engine.name, notification_text, []))
    else:
        print(f"{utils.SYSTEM_MSG}--> No attached file found with the name '{search_term}'.{utils.RESET_COLOR}")

def _handle_command_personas(args: list, state: SessionState, cli_history: InMemoryHistory):
    """Lists available personas."""
    personas = persona_manager.list_personas()
    if not personas:
        print(f"{utils.SYSTEM_MSG}--> No personas found in {config.PERSONAS_DIRECTORY}.{utils.RESET_COLOR}")
        return

    print(f"{utils.SYSTEM_MSG}--- Available Personas ---{utils.RESET_COLOR}")
    for p in personas:
        print(f"  - {p.filename.replace('.json', '')}: {p.description}")
    print(f"{utils.SYSTEM_MSG}------------------------{utils.RESET_COLOR}")

def _handle_command_persona(args: list, state: SessionState, cli_history: InMemoryHistory):
    """Switches to or clears a persona."""
    if not args:
        print(f"{utils.SYSTEM_MSG}--> Usage: /persona <name> OR /persona clear{utils.RESET_COLOR}")
        return

    command = args[0].lower()
    if command == 'clear':
        if not state.current_persona:
            print(f"{utils.SYSTEM_MSG}--> No active persona to clear.{utils.RESET_COLOR}")
            return

        state.system_prompt = state.initial_system_prompt
        state.current_persona = None
        print(f"{utils.SYSTEM_MSG}--> Persona cleared. System prompt reverted to its original state for this session.{utils.RESET_COLOR}")
        notification_text = "[SYSTEM] The current persona has been cleared. Default instructions are now in effect."
        state.history.append(utils.construct_user_message(state.engine.name, notification_text, []))
        return

    # Load and apply the new persona
    persona_name = ' '.join(args)
    new_persona = persona_manager.load_persona(persona_name)
    if not new_persona:
        print(f"{utils.SYSTEM_MSG}--> Persona '{persona_name}' not found or is invalid.{utils.RESET_COLOR}")
        return

    # Apply engine if specified and different
    if new_persona.engine and new_persona.engine != state.engine.name:
        try:
            new_api_key = api_client.check_api_keys(new_persona.engine)
            state.history = utils.translate_history(state.history, new_persona.engine)
            state.engine = get_engine(new_persona.engine, new_api_key)
            # Use persona model, or default for new engine
            default_model_key = 'default_openai_chat_model' if new_persona.engine == 'openai' else 'default_gemini_model'
            state.model = new_persona.model or app_settings.settings[default_model_key]
            print(f"{utils.SYSTEM_MSG}--> Engine switched to {state.engine.name.capitalize()} as per persona '{new_persona.name}'.{utils.RESET_COLOR}")
        except api_client.MissingApiKeyError:
            print(f"{utils.SYSTEM_MSG}--> Could not switch to persona engine '{new_persona.engine}': API key not found. Aborting.{utils.RESET_COLOR}")
            return

    # Apply model if specified (and engine didn't just change it)
    elif new_persona.model:
        state.model = new_persona.model

    # Apply other optional settings
    if new_persona.max_tokens is not None: state.max_tokens = new_persona.max_tokens
    if new_persona.stream is not None: state.stream_active = new_persona.stream

    # Apply system prompt and set current persona
    state.system_prompt = new_persona.system_prompt
    state.current_persona = new_persona

    print(f"{utils.SYSTEM_MSG}--> Switched to persona: '{new_persona.name}'{utils.RESET_COLOR}")
    notification_text = f"[SYSTEM] Persona changed to '{new_persona.name}'. New instructions are now in effect."
    state.history.append(utils.construct_user_message(state.engine.name, notification_text, []))


# --- Command Dispatcher ---

COMMAND_MAP = {
    '/exit': _handle_command_exit,
    '/quit': _handle_command_quit,
    '/help': _handle_command_help,
    '/stream': _handle_command_stream,
    '/debug': _handle_command_debug,
    '/memory': _handle_command_memory,
    '/remember': _handle_command_remember,
    '/max-tokens': _handle_command_max_tokens,
    '/clear': _handle_command_clear,
    '/model': _handle_command_model,
    '/engine': _handle_command_engine,
    '/history': _handle_command_history,
    '/state': _handle_command_state,
    '/set': _handle_command_set,
    '/save': _handle_command_save,
    '/load': _handle_command_load,
    '/refresh': _handle_command_refresh,
    '/files': _handle_command_files,
    '/attach': _handle_command_attach,
    '/detach': _handle_command_detach,
    '/personas': _handle_command_personas,
    '/persona': _handle_command_persona,
}

def _handle_slash_command(user_input: str, state: SessionState, cli_history: InMemoryHistory) -> bool:
    """Handles in-app slash commands by dispatching to helper functions."""
    parts = user_input.strip().split()
    command_str = parts[0].lower()
    args = parts[1:]

    handler = COMMAND_MAP.get(command_str)

    if handler:
        # A handler returns True if the session should end.
        # Ensure we return a boolean value.
        return handler(args, state, cli_history) or False
    else:
        print(f"{utils.SYSTEM_MSG}--> Unknown command: {command_str}. Type /help for a list of commands.{utils.RESET_COLOR}")
        return False


def perform_interactive_chat(initial_state: SessionState, session_name: str):
    """Manages the main loop for an interactive chat session."""
    log_filename_base = session_name or f"chat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{initial_state.engine.name}"
    log_filename = os.path.join(config.LOG_DIRECTORY, f"{log_filename_base}.jsonl")

    utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)

    print(f"Starting interactive chat with {initial_state.engine.name.capitalize()} ({initial_state.model}).")
    print("Type '/help' for commands or '/exit' to end.")
    print(f"Session log will be saved to: {log_filename}")

    if initial_state.current_persona: print(f"{utils.SYSTEM_MSG}Active Persona: {initial_state.current_persona.name}.{utils.RESET_COLOR}")
    if initial_state.system_prompt and not initial_state.current_persona: print(f"{utils.SYSTEM_MSG}System prompt is active.{utils.RESET_COLOR}")
    if initial_state.attachments:
        total_size = sum(p.stat().st_size for p in initial_state.attachments.keys() if p.exists())
        print(f"{utils.SYSTEM_MSG}Attached {len(initial_state.attachments)} text file(s). Total size: {_format_bytes(total_size)}{utils.RESET_COLOR}")
    if initial_state.attached_images: print(f"{utils.SYSTEM_MSG}Attached {len(initial_state.attached_images)} image(s) to this session.{utils.RESET_COLOR}")
    if initial_state.memory_enabled: print(f"{utils.SYSTEM_MSG}Persistent memory is enabled for this session.{utils.RESET_COLOR}")

    cli_history = InMemoryHistory()
    for command in initial_state.command_history:
        cli_history.append_string(command)

    first_turn = not initial_state.history
    large_attachment_confirmed = False

    try:
        if first_turn and initial_state.attachments:
            total_size = sum(p.stat().st_size for p in initial_state.attachments.keys() if p.exists())
            if total_size > config.LARGE_ATTACHMENT_THRESHOLD_BYTES and not large_attachment_confirmed:
                warning_msg = (
                    f"\n{utils.SYSTEM_MSG}WARNING: Total size of attached files is {_format_bytes(total_size)}.\n"
                    f"This may result in high API token usage and costs.\n"
                    f"Type 'yes' to proceed with the session: {utils.RESET_COLOR}"
                )
                confirmation = prompt(ANSI(warning_msg))
                if confirmation.lower().strip() != 'yes':
                    print(f"{utils.SYSTEM_MSG}--> Session aborted by user.{utils.RESET_COLOR}")
                    return
                large_attachment_confirmed = True

        while True:
            prompt_message = f"\n{utils.USER_PROMPT}You: {utils.RESET_COLOR}"
            user_input = prompt(ANSI(prompt_message), history=cli_history)

            if not user_input.strip():
                sys.stdout.write('\x1b[1A'); sys.stdout.write('\x1b[2K'); sys.stdout.flush()
                continue

            if user_input.startswith('/'):
                sys.stdout.write('\x1b[1A'); sys.stdout.write('\x1b[2K'); sys.stdout.flush()
                if _handle_slash_command(user_input, initial_state, cli_history):
                    break
                continue

            user_msg = utils.construct_user_message(initial_state.engine.name, user_input, initial_state.attached_images if first_turn else [])
            messages_or_contents = list(initial_state.history) + [user_msg]
            srl_list = initial_state.session_raw_logs if initial_state.debug_active else None

            if initial_state.stream_active:
                print(f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}", end='', flush=True)

            full_system_prompt = _assemble_full_system_prompt(initial_state)
            response_text, token_dict = api_client.perform_chat_request(
                engine=initial_state.engine, model=initial_state.model,
                messages_or_contents=messages_or_contents, system_prompt=full_system_prompt,
                max_tokens=initial_state.max_tokens, stream=initial_state.stream_active, session_raw_logs=srl_list
            )
            if not initial_state.stream_active:
                print(f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{response_text}", end='')

            print(utils.format_token_string(token_dict))
            asst_msg = utils.construct_assistant_message(initial_state.engine.name, response_text)
            initial_state.history.extend([user_msg, asst_msg])

            if len(initial_state.history) >= config.HISTORY_SUMMARY_THRESHOLD_TURNS * 2:
                _condense_chat_history(initial_state)

            try:
                log_entry = {"timestamp": datetime.datetime.now().isoformat(), "model": initial_state.model, "prompt": user_msg, "response": asst_msg, "tokens": token_dict}
                with open(log_filename, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except IOError as e:
                log.warning("Could not write to session log file: %s", e)
            first_turn = False
    except (KeyboardInterrupt, EOFError):
        print("\nSession interrupted by user.")
    finally:
        print("\nSession ended.")
        if initial_state.force_quit or initial_state.exit_without_memory:
            return

        if os.path.exists(log_filename) and initial_state.history:
            if initial_state.memory_enabled:
                _consolidate_session_into_memory(initial_state.engine, initial_state.model, initial_state.history)
            else:
                print(f"{utils.SYSTEM_MSG}--> Persistent memory not enabled, skipping update.{utils.RESET_COLOR}")

            if initial_state.custom_log_rename:
                new_filepath = log_filename.replace(os.path.basename(log_filename), f"{utils.sanitize_filename(initial_state.custom_log_rename)}.jsonl")
                try:
                    os.rename(log_filename, new_filepath)
                    print(f"{utils.SYSTEM_MSG}--> Session log renamed to: {new_filepath}{utils.RESET_COLOR}")
                except OSError as e:
                    log.error("Failed to rename session log: %s", e)
            elif not session_name:
                rename_session_log(initial_state.engine, initial_state.history, log_filename)

        if initial_state.debug_active:
            debug_filename = f"debug_{os.path.splitext(log_filename_base)[0]}.jsonl"
            debug_filepath = config.LOG_DIRECTORY / debug_filename
            print(f"Saving debug log to: {debug_filepath}")
            try:
                with open(debug_filepath, 'w', encoding='utf-8') as f:
                    for entry in initial_state.session_raw_logs:
                        f.write(json.dumps(entry) + '\n')
            except IOError as e:
                log.error("Could not save debug log file: %s", e)

def _consolidate_session_into_memory(engine: AIEngine, model: str, history: list):
    """Summarizes the session and integrates it into persistent memory."""
    print(f"{utils.SYSTEM_MSG}--> Updating persistent memory with session content...{utils.RESET_COLOR}")
    session_content = "\n".join([f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}" for msg in history])
    existing_ltm = ""
    try:
        if config.PERSISTENT_MEMORY_FILE.exists():
            existing_ltm = config.PERSISTENT_MEMORY_FILE.read_text(encoding='utf-8')
    except IOError as e:
        log.warning("Could not read persistent memory file: %s", e)

    prompt_text = MEMORY_INTEGRATION_PROMPT.format(existing_ltm=existing_ltm, session_content=session_content)
    helper_model_key = 'helper_model_openai' if engine.name == 'openai' else 'helper_model_gemini'
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(engine.name, prompt_text, [])]

    updated_memory, _ = api_client.perform_chat_request(
        engine=engine, model=task_model, messages_or_contents=messages,
        system_prompt=None, max_tokens=None, stream=False
    )

    if updated_memory and not updated_memory.startswith("API Error:"):
        try:
            config.PERSISTENT_MEMORY_FILE.write_text(updated_memory.strip(), encoding='utf-8')
            print(f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}")
        except IOError as e:
            log.error("Failed to write to persistent memory file: %s", e)
    else:
        reason = updated_memory or "No response from helper model."
        log.warning("Memory integration failed. Reason: %s", reason)

def _inject_fact_into_memory(engine: AIEngine, fact: str):
    """Injects a single fact into persistent memory."""
    print(f"{utils.SYSTEM_MSG}--> Injecting fact into persistent memory...{utils.RESET_COLOR}")
    existing_ltm = ""
    try:
        if config.PERSISTENT_MEMORY_FILE.exists():
            existing_ltm = config.PERSISTENT_MEMORY_FILE.read_text(encoding='utf-8')
    except IOError as e:
        log.warning("Could not read persistent memory file: %s", e)

    prompt_text = DIRECT_MEMORY_INJECTION_PROMPT.format(existing_ltm=existing_ltm, new_fact=fact)
    helper_model_key = 'helper_model_openai' if engine.name == 'openai' else 'helper_model_gemini'
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(engine.name, prompt_text, [])]

    updated_memory, _ = api_client.perform_chat_request(
        engine=engine, model=task_model, messages_or_contents=messages,
        system_prompt=None, max_tokens=None, stream=False
    )

    if updated_memory and not updated_memory.startswith("API Error:"):
        try:
            config.PERSISTENT_MEMORY_FILE.write_text(updated_memory.strip(), encoding='utf-8')
            print(f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}")
        except IOError as e:
            log.error("Failed to write to persistent memory file: %s", e)
    else:
        reason = updated_memory or "No response from helper model."
        log.warning("Memory integration failed. Reason: %s", reason)

def rename_session_log(engine: AIEngine, history: list, original_filepath: str):
    """Generates a descriptive name for the session log and renames the file."""
    print(f"{utils.SYSTEM_MSG}--> Generating descriptive name for session log...{utils.RESET_COLOR}")
    suggested_name = _generate_session_name(engine, history)

    if suggested_name:
        new_filename = f"{suggested_name}.jsonl"
        new_filepath = os.path.join(config.LOG_DIRECTORY, new_filename)
        try:
            os.rename(original_filepath, new_filepath)
            print(f"{utils.SYSTEM_MSG}--> Session log renamed to: {new_filepath}{utils.RESET_COLOR}")
        except OSError as e:
            log.error("Failed to rename session log: %s", e)
    else:
        log.warning("Could not generate a descriptive name for the session log.")
