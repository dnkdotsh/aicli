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
from . import handlers
from . import settings as app_settings
from .engine import AIEngine, get_engine
from .logger import log
from .prompts import HISTORY_SUMMARY_PROMPT, MEMORY_INTEGRATION_PROMPT, LOG_RENAMING_PROMPT

@dataclass
class SessionState:
    """A dataclass to hold the state of an interactive chat session."""
    engine: AIEngine
    model: str
    system_prompt: str | None
    max_tokens: int | None
    memory_enabled: bool
    initial_image_data: list = field(default_factory=list)
    history: list = field(default_factory=list)
    debug_active: bool = False
    stream_active: bool = True
    session_raw_logs: list = field(default_factory=list)
    exit_without_memory: bool = False

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
    
    # Sanitize the filename part, but keep the extension
    safe_name = utils.sanitize_filename(filename.rsplit('.', 1)[0]) + '.json'
    filepath = config.SESSIONS_DIRECTORY / safe_name

    state_dict = asdict(state)
    state_dict['engine_name'] = state.engine.name
    del state_dict['engine']

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

def _handle_slash_command(user_input: str, state: SessionState) -> bool:
    """Handles in-app slash commands. Returns True if the session should end."""
    parts = user_input.strip().split()
    command = parts[0].lower()
    args = parts[1:]

    if command == '/exit':
        return True
    
    elif command == '/help':
        utils.display_help('chat')

    elif command == '/stream':
        state.stream_active = not state.stream_active
        status = "ENABLED" if state.stream_active else "DISABLED"
        print(f"{utils.SYSTEM_MSG}--> Response streaming is now {status} for this session.{utils.RESET_COLOR}")

    elif command == '/debug':
        state.debug_active = not state.debug_active
        status = "ENABLED" if state.debug_active else "DISABLED"
        print(f"{utils.SYSTEM_MSG}--> Session-specific debug logging is now {status}.{utils.RESET_COLOR}")

    elif command == '/memory':
        state.memory_enabled = not state.memory_enabled
        status = "ENABLED" if state.memory_enabled else "DISABLED"
        print(f"{utils.SYSTEM_MSG}--> Persistent memory is now {status} for saving at the end of this session.{utils.RESET_COLOR}")
        
    elif command == '/max-tokens':
        if args and args[0].isdigit():
            state.max_tokens = int(args[0])
            print(f"{utils.SYSTEM_MSG}--> Max tokens for this session set to: {state.max_tokens}.{utils.RESET_COLOR}")
        else:
            print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")

    elif command == '/clear':
        confirm = prompt("This will clear all conversation history. Type `proceed` to confirm: ")
        if confirm.lower() == 'proceed':
            state.history.clear()
            print(f"{utils.SYSTEM_MSG}--> Conversation history has been cleared.{utils.RESET_COLOR}")
        else:
            print(f"{utils.SYSTEM_MSG}--> Clear cancelled.{utils.RESET_COLOR}")

    elif command == '/model':
        if args:
            state.model = args[0]
            print(f"{utils.SYSTEM_MSG}--> Model temporarily set to: {state.model}{utils.RESET_COLOR}")
        else:
            state.model = handlers.select_model(state.engine, 'chat')
            print(f"{utils.SYSTEM_MSG}--> Model changed to: {state.model}{utils.RESET_COLOR}")

    elif command == '/engine':
        new_engine_name = args[0] if args else ('gemini' if state.engine.name == 'openai' else 'openai')
        if new_engine_name not in ['openai', 'gemini']:
            print(f"{utils.SYSTEM_MSG}--> Unknown engine: {new_engine_name}. Use 'openai' or 'gemini'.{utils.RESET_COLOR}")
            return False
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

    elif command == '/history':
        print(json.dumps(state.history, indent=2))

    elif command == '/state':
        state_dict = asdict(state)
        state_dict['engine'] = state.engine.name
        state_dict.pop('session_raw_logs', None)
        state_dict.pop('history', None)
        print(json.dumps(state_dict, indent=2))

    elif command == '/set':
        if len(args) == 2:
            app_settings.save_setting(args[0], args[1])
        else:
            print(f"{utils.SYSTEM_MSG}--> Usage: /set <setting_key> <value>{utils.RESET_COLOR}")

    elif command == '/save':
        should_remember, should_stay = False, False
        filename_parts = []
        for arg in args:
            if arg == '--remember': should_remember = True
            elif arg == '--stay': should_stay = True
            else: filename_parts.append(arg)
        
        filename = ' '.join(filename_parts) if filename_parts else None

        if not filename:
            print(f"{utils.SYSTEM_MSG}--> Generating descriptive name for session...{utils.RESET_COLOR}")
            filename = _generate_session_name(state.engine, state.history)
            if not filename:
                print(f"{utils.SYSTEM_MSG}--> Could not auto-generate a name. Save cancelled.{utils.RESET_COLOR}")
                return False

        if _save_session_to_file(state, filename):
            if should_stay:
                return False  # Do not exit
            if not should_remember:
                state.exit_without_memory = True  # Signal to skip memory update on exit
            return True  # Exit
        return False # Stay in session if save fails

    elif command == '/load':
        if args:
            filename = ' '.join(args)
            if not filename.endswith('.json'):
                filename += '.json'
            filepath = config.SESSIONS_DIRECTORY / filename
            new_state = load_session_from_file(filepath)
            if new_state:
                state.engine = new_state.engine
                state.model = new_state.model
                state.system_prompt = new_state.system_prompt
                state.max_tokens = new_state.max_tokens
                state.memory_enabled = new_state.memory_enabled
                state.initial_image_data = new_state.initial_image_data
                state.history = new_state.history
                state.debug_active = new_state.debug_active
                state.stream_active = new_state.stream_active
        else:
            print(f"{utils.SYSTEM_MSG}--> Usage: /load <filename>{utils.RESET_COLOR}")

    else:
        print(f"{utils.SYSTEM_MSG}--> Unknown command: {command}. Type /help for a list of commands.{utils.RESET_COLOR}")

    return False

def perform_interactive_chat(initial_state: SessionState, session_name: str):
    """Manages the main loop for an interactive chat session."""
    log_filename_base = session_name or f"chat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{initial_state.engine.name}"
    log_filename = os.path.join(config.LOG_DIRECTORY, f"{log_filename_base}.jsonl")

    utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)

    print(f"Starting interactive chat with {initial_state.engine.name.capitalize()} ({initial_state.model}).")
    print("Type '/help' for commands or '/exit' to end.")
    print(f"Session log will be saved to: {log_filename}")

    if initial_state.system_prompt: print("System prompt is active.")
    if initial_state.initial_image_data: print(f"Attached {len(initial_state.initial_image_data)} image(s) to this session.")
    if initial_state.memory_enabled: print("Persistent memory is enabled for this session.")

    cli_history = InMemoryHistory()
    first_turn = not initial_state.history
    try:
        while True:
            prompt_message = f"\n{utils.USER_PROMPT}You: {utils.RESET_COLOR}"
            user_input = prompt(ANSI(prompt_message), history=cli_history)

            if not user_input.strip():
                sys.stdout.write('\x1b[1A'); sys.stdout.write('\x1b[2K'); sys.stdout.flush()
                continue

            if user_input.startswith('/'):
                sys.stdout.write('\x1b[1A'); sys.stdout.write('\x1b[2K'); sys.stdout.flush()
                if _handle_slash_command(user_input, initial_state):
                    break
                continue

            user_msg = utils.construct_user_message(initial_state.engine.name, user_input, initial_state.initial_image_data if first_turn else [])
            messages_or_contents = list(initial_state.history)
            messages_or_contents.append(user_msg)
            srl_list = initial_state.session_raw_logs if initial_state.debug_active else None

            if initial_state.stream_active:
                print(f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}", end='', flush=True)

            response_text, token_dict = api_client.perform_chat_request(
                engine=initial_state.engine, model=initial_state.model,
                messages_or_contents=messages_or_contents, system_prompt=initial_state.system_prompt,
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
        if initial_state.exit_without_memory:
            return

        if os.path.exists(log_filename) and initial_state.history:
            if initial_state.memory_enabled:
                _update_persistent_memory(initial_state.engine, initial_state.model, initial_state.history)
            else:
                print(f"{utils.SYSTEM_MSG}--> Persistent memory not enabled, skipping update.{utils.RESET_COLOR}")

            if not session_name:
                rename_session_log(initial_state.engine, initial_state.history, log_filename)

        if initial_state.debug_active:
            debug_filename = f"debug_{os.path.splitext(log_filename_base)[0]}.jsonl"
            debug_filepath = os.path.join(config.LOG_DIRECTORY, debug_filename)
            print(f"Saving debug log to: {debug_filepath}")
            try:
                with open(debug_filepath, 'w', encoding='utf-8') as f:
                    for entry in initial_state.session_raw_logs:
                        f.write(json.dumps(entry) + '\n')
            except IOError as e:
                log.error("Could not save debug log file: %s", e)

def _update_persistent_memory(engine: AIEngine, model: str, history: list):
    """Summarizes the session and integrates it into persistent memory."""
    print(f"{utils.SYSTEM_MSG}--> Updating persistent memory...{utils.RESET_COLOR}")
    session_content = "\n".join([f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}" for msg in history])
    existing_ltm = ""
    if os.path.exists(config.PERSISTENT_MEMORY_FILE):
        with open(config.PERSISTENT_MEMORY_FILE, 'r', encoding='utf-8') as f:
            existing_ltm = f.read()

    prompt_text = MEMORY_INTEGRATION_PROMPT.format(existing_ltm=existing_ltm, session_content=session_content)
    helper_model_key = 'helper_model_openai' if engine.name == 'openai' else 'helper_model_gemini'
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(engine.name, prompt_text, [])]

    updated_memory, _ = api_client.perform_chat_request(
        engine=engine, model=task_model, messages_or_contents=messages,
        system_prompt=None, max_tokens=app_settings.settings['summary_max_tokens'], stream=False
    )

    if updated_memory and not updated_memory.startswith("API Error:"):
        try:
            with open(config.PERSISTENT_MEMORY_FILE, 'w', encoding='utf-8') as f:
                f.write(updated_memory.strip())
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
