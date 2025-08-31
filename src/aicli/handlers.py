# aicli/handlers.py
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
import base64
import datetime
import requests
import threading
import queue
import copy
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import ANSI
import re # Added for regex for _format_multichat_response

from . import config
from . import api_client
from . import utils
from .engine import AIEngine, get_engine
from .session_manager import perform_interactive_chat, SessionState
from .settings import settings
from .logger import log
from .prompts import (MULTICHAT_SYSTEM_PROMPT_GEMINI, MULTICHAT_SYSTEM_PROMPT_OPENAI,
                     CONTINUATION_PROMPT)

def _format_multichat_response(engine_name: str, raw_response: str) -> str:
    """
    Formats an AI's raw response for multi-chat, stripping any self-labels the AI
    might have added despite system prompts, and then prepending the client's
    controlled label.
    """
    client_prefix = f"[{engine_name.capitalize()}]: "
    # Regex to match leading AI-generated labels (e.g., "[Openai]:", "[Gemini]:", possibly with optional colon and whitespace)
    # This pattern accounts for variations in spacing and capitalization.
    ai_label_pattern = re.compile(r"^\[" + re.escape(engine_name.capitalize()) + r"\]:?\s*", re.IGNORECASE)
    cleaned_response = ai_label_pattern.sub("", raw_response.lstrip())
    return client_prefix + cleaned_response

def select_model(engine: AIEngine, task: str) -> str:
    """Allows the user to select a model or use the default."""
    default_model = ""
    if task == 'chat':
        default_model = settings['default_openai_chat_model'] if engine.name == 'openai' else settings['default_gemini_model']
    elif task == 'image':
        default_model = settings['default_openai_image_model']

    use_default = prompt(f"Use default model ({default_model})? (Y/n): ").lower().strip()
    if use_default in ('', 'y', 'yes'):
        return default_model

    print("Fetching available models...")
    models = engine.fetch_available_models(task)
    if not models:
        print(f"Using default: {default_model}")
        return default_model

    print("\nPlease select a model:")
    for i, model_name in enumerate(models):
        print(f"  {i+1}. {model_name}")

    try:
        choice = prompt(f"Enter number (or press Enter for default): ")
        if not choice: return default_model
        index = int(choice) - 1
        if 0 <= index < len(models):
            return models[index]
    except (ValueError, IndexError):
        pass

    print(f"Invalid selection. Using default: {default_model}")
    return default_model

def handle_chat(engine: AIEngine, model: str, system_prompt: str, initial_prompt: str, image_data: list, attachments: dict, session_name: str, max_tokens: int, stream: bool, memory_enabled: bool, debug_enabled: bool):
    """Handles both single-shot and interactive chat sessions."""
    if initial_prompt:
        # For single-shot chat, we must pre-assemble the full system prompt
        attachment_texts = []
        for path, content in attachments.items():
            attachment_texts.append(f"--- FILE: {path.as_posix()} ---\n{content}")
        attachments_str = "\n\n".join(attachment_texts)

        full_system_prompt = system_prompt
        if attachments_str:
            full_system_prompt = (full_system_prompt or "") + f"\n\n--- ATTACHED FILES ---\n{attachments_str}"

        # Handle single-shot chat
        messages_or_contents = [utils.construct_user_message(engine.name, initial_prompt, image_data)]
        if stream:
             print(f"{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}", end='', flush=True)
        response, token_dict = api_client.perform_chat_request(engine, model, messages_or_contents, full_system_prompt, max_tokens, stream)
        if not stream:
            print(f"{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{response}", end='')
        print(utils.format_token_string(token_dict))
    else:
        # Delegate to the session manager for interactive chat
        initial_state = SessionState(
            engine=engine,
            model=model,
            system_prompt=system_prompt,
            attached_images=image_data,
            attachments=attachments,
            stream_active=stream,
            memory_enabled=memory_enabled,
            debug_active=debug_enabled,
            max_tokens=max_tokens
        )
        perform_interactive_chat(initial_state, session_name)

def handle_load_session(filepath_str: str):
    """Loads and starts an interactive session from a file."""
    # Deferred imports to avoid circular dependencies
    from .session_manager import load_session_from_file
    from pathlib import Path
    from . import config

    raw_path = Path(filepath_str).expanduser()

    # If the path is not absolute, assume it is in the sessions directory
    if raw_path.is_absolute():
        filepath = raw_path
    else:
        filepath = config.SESSIONS_DIRECTORY / raw_path

    # Ensure the .json extension is present
    if filepath.suffix != '.json':
        filepath = filepath.with_suffix('.json')

    try:
        initial_state = load_session_from_file(filepath)
    except api_client.MissingApiKeyError as e:
        log.error(e)
        sys.exit(1)

    if not initial_state:
        # This branch handles other load failures (e.g., file not found, JSON error)
        # that are logged by load_session_from_file itself.
        sys.exit(1)

    # Use the loaded file's name as the base for the new session log
    session_name = filepath.stem
    perform_interactive_chat(initial_state, session_name)

def _secondary_worker(engine, model, history, system_prompt, max_tokens, result_queue):
    """A worker function to be run in a thread for the secondary model."""
    try:
        # Perform chat request, do NOT print stream here
        response_text, _ = api_client.perform_chat_request(
            engine, model, history, system_prompt, max_tokens, stream=True, print_stream=False
        )
        # Apply formatting before putting into queue
        formatted_response = _format_multichat_response(engine.name, response_text)
        result_queue.put(formatted_response)
    except Exception as e:
        log.error("Secondary model thread failed: %s", e)
        # Ensure that engine.name always returns a string, not a MagicMock object, in tests.
        result_queue.put(f"Error: Could not get response from {engine.name}.")

def handle_multichat_session(initial_prompt: str | None, system_prompt: str, image_data: list, session_name: str, max_tokens: int, debug_enabled: bool):
    """Manages an interactive session with both OpenAI and Gemini."""
    try:
        openai_key = api_client.check_api_keys('openai')
        gemini_key = api_client.check_api_keys('gemini')
    except api_client.MissingApiKeyError as e:
        log.error(e)
        sys.exit(1) # Exit if API key is missing

    openai_engine = get_engine('openai', openai_key)
    gemini_engine = get_engine('gemini', gemini_key)
    openai_model = settings['default_openai_chat_model']
    gemini_model = settings['default_gemini_model']

    primary_engine_name = settings['default_engine']

    engines = {'openai': openai_engine, 'gemini': gemini_engine}
    models = {'openai': openai_model, 'gemini': gemini_model}

    engine_aliases = {
        'gpt': 'openai', 'openai': 'openai',
        'gem': 'gemini', 'gemini': 'gemini', 'google': 'gemini'
    }

    primary_engine = engines[primary_engine_name]
    secondary_engine = engines['gemini' if primary_engine_name == 'openai' else 'openai']

    # Combine the user-provided system prompt with the mode-specific instructions.
    final_sys_prompts = {}
    if system_prompt:
        final_sys_prompts['openai'] = f"{system_prompt}\n\n---\n\n{MULTICHAT_SYSTEM_PROMPT_OPENAI}"
        final_sys_prompts['gemini'] = f"{system_prompt}\n\n---\n\n{MULTICHAT_SYSTEM_PROMPT_GEMINI}"
    else:
        final_sys_prompts['openai'] = MULTICHAT_SYSTEM_PROMPT_OPENAI
        final_sys_prompts['gemini'] = MULTICHAT_SYSTEM_PROMPT_GEMINI

    log_filename_base = session_name or f"multichat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_filename = os.path.join(config.LOG_DIRECTORY, f"{log_filename_base}.jsonl")

    print(f"Starting interactive multi-chat. Primary engine: {primary_engine.name.capitalize()}.")
    print(f"Session log will be saved to: {log_filename}")
    print("Type /help for commands or /exit to end.")

    shared_history = []
    turn_counter = 0
    cli_history = InMemoryHistory()

    def process_turn(prompt_text):
        nonlocal turn_counter
        nonlocal shared_history
        turn_counter += 1

        # Handle slash commands for targeted prompts
        if prompt_text.lower().strip().startswith('/ai'):
            parts = prompt_text.strip().split(' ', 2)

            if len(parts) < 2 or parts[1].lower() not in engine_aliases:
                print(f"{utils.SYSTEM_MSG}--> Usage: /ai <gpt|gem> [prompt]{utils.RESET_COLOR}")
                return

            target_alias = parts[1].lower()
            target_engine_name = engine_aliases[target_alias]
            target_prompt = parts[2] if len(parts) > 2 else CONTINUATION_PROMPT

            target_engine = engines[target_engine_name]
            target_model = models[target_engine_name]

            user_msg_text = f"Director to {target_engine.name.capitalize()}: {target_prompt}"
            user_msg = utils.construct_user_message(target_engine.name, user_msg_text, image_data if turn_counter == 1 else [])
            current_history = utils.translate_history(shared_history + [user_msg], target_engine.name)

            print(f"\n{utils.ASSISTANT_PROMPT}[{target_engine.name.capitalize()}]: {utils.RESET_COLOR}", end='', flush=True)
            # Perform chat request, streaming to stdout. The raw_response will capture the full stream.
            raw_response, _ = api_client.perform_chat_request(target_engine, target_model, current_history, final_sys_prompts[target_engine_name], max_tokens, stream=True)
            print() # Ensure a newline after streamed output

            formatted_response = _format_multichat_response(target_engine.name, raw_response)
            asst_msg = utils.construct_assistant_message('openai', formatted_response)
            shared_history.extend([utils.construct_user_message('openai', user_msg_text, []), asst_msg])

        # Handle regular messages for broadcast
        else:
            user_msg_text = f"Director to All: {prompt_text}"
            user_msg_for_history = utils.construct_user_message('openai', user_msg_text, [])

            result_queue = queue.Queue()

            history_for_primary = utils.translate_history(shared_history + [user_msg_for_history], primary_engine.name)
            history_for_secondary = utils.translate_history(shared_history + [user_msg_for_history], secondary_engine.name)

            secondary_thread = threading.Thread(
                target=_secondary_worker,
                args=(secondary_engine, models[secondary_engine.name], history_for_secondary, final_sys_prompts[secondary_engine.name], max_tokens, result_queue)
            )
            secondary_thread.start()

            print(f"\n{utils.ASSISTANT_PROMPT}[{primary_engine.name.capitalize()}]: {utils.RESET_COLOR}", end='', flush=True)
            # Stream the *raw* primary response, then format the full response later for history
            primary_response_streamed, _ = api_client.perform_chat_request(primary_engine, models[primary_engine.name], history_for_primary, final_sys_prompts[primary_engine.name], max_tokens, stream=True)
            print() # CRITICAL FIX: Ensure a newline after the streamed content

            secondary_thread.join()
            secondary_response_formatted = result_queue.get() # Already formatted by _secondary_worker

            print(f"{utils.ASSISTANT_PROMPT}{secondary_response_formatted}{utils.RESET_COLOR}")

            formatted_primary_response = _format_multichat_response(primary_engine.name, primary_response_streamed)

            primary_msg = utils.construct_assistant_message('openai', formatted_primary_response)
            secondary_msg = utils.construct_assistant_message('openai', secondary_response_formatted) # Already formatted

            first_msg, second_msg = (primary_msg, secondary_msg) if primary_engine_name == 'openai' else (secondary_msg, primary_msg)
            shared_history.extend([user_msg_for_history, first_msg, second_msg])

        try:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"turn": turn_counter, "history_slice": shared_history[-3:] if not prompt_text.lower().strip().startswith('/ai') else shared_history[-2:]}) + '\n')
        except IOError as e:
            log.warning("Could not write to session log file: %s", e)

    if initial_prompt:
        process_turn(initial_prompt)

    try:
        while True:
            prompt_message = f"\n{utils.DIRECTOR_PROMPT}Director> {utils.RESET_COLOR}"
            user_input = prompt(ANSI(prompt_message), history=cli_history)

            if not user_input.strip():
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')
                sys.stdout.flush()
                continue

            is_command = user_input.lstrip().startswith('/')
            if is_command:
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')
                sys.stdout.flush()

            if user_input.lower().strip() == '/exit':
                break
            if user_input.lower().strip() == '/help':
                utils.display_help('multichat')
                continue
            if user_input.lower().strip() == '/history':
                print(json.dumps(shared_history, indent=2))
                continue

            process_turn(user_input)
    except (KeyboardInterrupt, EOFError):
        print("\nSession interrupted.")
    finally:
        print("\nSession ended.")

def handle_image_generation(api_key: str, engine: AIEngine, prompt: str):
    """Handles OpenAI image generation."""
    model = select_model(engine, 'image')
    if not prompt:
        prompt = sys.stdin.read().strip() if not sys.stdin.isatty() else input("Enter a description for the image: ")

    if not prompt:
        print("Image generation cancelled: No prompt provided.", file=sys.stderr)
        return

    print(f"Generating image with {model} for prompt: '{prompt}'...")
    payload = {"model": model, "prompt": prompt, "n": 1, "size": "1024x1024"}
    if model.startswith('dall-e'):
        payload['response_format'] = 'b64_json'

    url, headers = "https://api.openai.com/v1/images/generations", {"Authorization": f"Bearer {api_key}"}
    response_data = api_client.make_api_request(url, headers, payload)

    if response_data and 'data' in response_data:
        image_url = response_data['data'][0].get('url')
        b64_data = response_data['data'][0].get('b64_json')
        image_bytes = None

        if b64_data:
            try:
                image_bytes = base64.b64decode(b64_data)
            except base64.binascii.Error as e:
                print(f"Error decoding base64 image data: {e}", file=sys.stderr)
                return
        elif image_url:
            try:
                print(f"Downloading image from: {image_url}")
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                image_bytes = image_response.content
            except requests.exceptions.RequestException as e:
                print(f"Error downloading image: {e}", file=sys.stderr)
                return

        if image_bytes:
            try:
                safe_prompt = utils.sanitize_filename(prompt[:50])
                base_filename = f"image_{safe_prompt}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(config.IMAGE_DIRECTORY, base_filename)
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                print(f"Image saved successfully as: {filepath}")
                log_entry = {"timestamp": datetime.datetime.now().isoformat(), "model": model, "prompt": prompt, "file": filepath}
                with open(config.IMAGE_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except IOError as e:
                print(f"Error saving image to file: {e}", file=sys.stderr)
    else:
            print("Error: API response did not contain image data in a recognized format.", file=sys.stderr)
