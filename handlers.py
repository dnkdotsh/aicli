# aicli/handlers.py

# Unified Command-Line AI Client
# Copyright (C) 2025 <name of author>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import json
import base64
import datetime
import requests

import config
import api_client
import utils
from engine import AIEngine, get_engine
from session_manager import perform_interactive_chat, SessionState
from settings import settings

def select_model(engine: AIEngine, task: str) -> str:
    """Allows the user to select a model or use the default."""
    default_model = ""
    if task == 'chat':
        default_model = settings['default_openai_chat_model'] if engine.name == 'openai' else settings['default_gemini_model']
    elif task == 'image':
        default_model = settings['default_openai_image_model']

    use_default = input(f"Use default model ({default_model})? (Y/n): ").lower().strip()
    if use_default in ('', 'y', 'yes'):
        return default_model

    print("Fetching available models...")
    models = engine.fetch_available_models(task)
    if not models:
        print(f"Using default: {default_model}", file=sys.stderr)
        return default_model

    print("\nPlease select a model:")
    for i, model_name in enumerate(models):
        print(f"  {i+1}. {model_name}")

    try:
        choice = input(f"Enter number (or press Enter for default): ")
        if not choice: return default_model
        index = int(choice) - 1
        if 0 <= index < len(models):
            return models[index]
    except (ValueError, IndexError):
        pass

    print(f"Invalid selection. Using default: {default_model}")
    return default_model

def handle_chat(engine: AIEngine, model: str, system_prompt: str, initial_prompt: str, image_data: list, session_name: str, max_tokens: int, stream: bool, memory_enabled: bool):
    """Handles both single-shot and interactive chat sessions."""
    if initial_prompt:
        # Handle single-shot chat
        messages_or_contents = [utils.construct_user_message(engine.name, initial_prompt, image_data)]
        if stream:
             print("Assistant: ", end='', flush=True)
        response, token_dict = api_client.perform_chat_request(engine, model, messages_or_contents, system_prompt, max_tokens, stream)
        if not stream:
            print(f"Assistant: {response}", end='')
        print(utils.format_token_string(token_dict))
    else:
        # Delegate to the session manager for interactive chat
        initial_state = SessionState(
            engine=engine,
            model=model,
            system_prompt=system_prompt,
            initial_image_data=image_data,
            stream_active=stream,
            memory_enabled=memory_enabled
        )
        perform_interactive_chat(initial_state, session_name, max_tokens)

def handle_both_engines(system_prompt: str, prompt: str, image_data: list, max_tokens: int, stream: bool):
    """Sends a single-shot prompt to both OpenAI and Gemini and prints the results."""
    # --- OpenAI ---
    print("--- OpenAI Response ---")
    openai_key = api_client.check_api_keys('openai')
    openai_engine = get_engine('openai', openai_key)
    openai_model = select_model(openai_engine, 'chat')
    openai_messages = [utils.construct_user_message('openai', prompt, image_data)]
    if stream: print("Assistant: ", end='', flush=True)
    openai_response, openai_tokens = api_client.perform_chat_request(openai_engine, openai_model, openai_messages, system_prompt, max_tokens, stream)
    if not stream: print(f"Assistant: {openai_response}", end='')
    print(utils.format_token_string(openai_tokens))
    print("\n" + "="*50 + "\n")

    # --- Gemini ---
    print("--- Gemini Response ---")
    gemini_key = api_client.check_api_keys('gemini')
    gemini_engine = get_engine('gemini', gemini_key)
    gemini_model = select_model(gemini_engine, 'chat')
    gemini_messages = [utils.construct_user_message('gemini', prompt, image_data)]
    if stream: print("Assistant: ", end='', flush=True)
    gemini_response, gemini_tokens = api_client.perform_chat_request(gemini_engine, gemini_model, gemini_messages, system_prompt, max_tokens, stream)
    if not stream: print(f"Assistant: {gemini_response}", end='')
    print(utils.format_token_string(gemini_tokens))
    print("\n" + "="*50 + "\n")

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
    else:
        print("Image generation failed.", file=sys.stderr)

