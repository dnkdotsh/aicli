# aicli/utils.py
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


import os
import sys
import base64
import mimetypes
import zipfile
import re
import json
import requests
from pathlib import Path

import config
from logger import log

# ANSI color codes for UI theming
USER_PROMPT = '\033[96m'      # Bright Cyan
ASSISTANT_PROMPT = '\033[93m' # Bright Yellow
SYSTEM_MSG = '\033[90m'      # Bright Black (Gray)
DIRECTOR_PROMPT = '\033[95m'  # Bright Magenta
RESET_COLOR = '\033[0m'      # Reset

def display_help(mode: str = 'chat'):
    """Displays a comprehensive help menu for interactive modes."""
    print(f"{SYSTEM_MSG}--- Help Menu ---")
    print("/help              Show this help menu.")
    print("/exit              End the current session.")
    print("/clear             Clear the conversation history for this session.")
    print("/stream            Toggle response streaming on/off.")
    print("/debug             Toggle session-specific debug logging on/off.")
    print("/memory            Toggle persistent memory on/off for session end.")
    print("/model [name]      Change the model (prompts for selection if no name is given).")
    print("/engine [name]     Switch the AI engine (e.g., to 'openai' or 'gemini').")
    print("/max-tokens [num]  Set the max tokens for responses in this session.")
    print("/history           Show the raw conversation history as JSON.")
    print("/state             Show the current session state.")
    print("/set <key> <val>   Change and save a default setting (e.g., /set default_engine openai).")
    if mode == 'multichat':
        print("\n--- Multi-Chat Only ---")
        print("/ai <eng> [prompt] Send a prompt to a specific engine (eng: gpt, gem, openai, etc.).")
        print("                   If no prompt is given, the AI will be asked to continue.")
    print(f"-------------------{RESET_COLOR}")


def format_token_string(token_dict: dict) -> str:
    """Formats the token dictionary into a consistent string for display."""
    if not token_dict or token_dict.get('total', 0) == 0:
        return ""
    p = token_dict.get('prompt', 0)
    c = token_dict.get('completion', 0)
    r = token_dict.get('reasoning', 0)
    t = token_dict.get('total', 0)
    return f" [{p}/{c}/{r}/{t}]"

def process_stream(engine_name: str, response: requests.Response, print_stream: bool = True) -> tuple[str, dict]:
    """Processes a streaming API response, printing deltas and returning the full text and token counts."""
    full_text = ""
    tokens = {}
    try:
        for line in response.iter_lines():
            if not line or not line.startswith(b'data: '):
                continue
            line_data = line[6:]
            if engine_name == 'openai' and line_data == b'[DONE]':
                break
            try:
                chunk = json.loads(line_data)
                delta = ""
                if engine_name == 'openai':
                    delta = chunk['choices'][0]['delta'].get('content', '')
                elif engine_name == 'gemini' and 'candidates' in chunk:
                    delta = chunk['candidates'][0]['content']['parts'][0]['text']
                    if chunk['candidates'][0].get('finishReason') == 'STOP':
                        p, c, r, t = parse_token_counts(engine_name, chunk)
                        tokens = {'prompt': p, 'completion': c, 'reasoning': r, 'total': t}
                if delta:
                    if print_stream:
                        print(delta, end='', flush=True)
                    full_text += delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    except KeyboardInterrupt:
        if print_stream:
            print(f"\n{SYSTEM_MSG}--> Stream cancelled by user.{RESET_COLOR}", file=sys.stderr)
    except requests.exceptions.ChunkedEncodingError:
        if print_stream:
            print(f"\n{SYSTEM_MSG}--> Warning: Stream connection interrupted.{RESET_COLOR}", file=sys.stderr)
    if print_stream:
        print() # Ensure there's a newline after streaming finishes or is cancelled.
    return full_text, tokens

def parse_token_counts(engine_name: str, response_data: dict) -> tuple[int, int, int, int]:
    """Parses token counts from a response, accommodating different structures."""
    prompt_tokens, completion_tokens, reasoning_tokens, total_tokens = 0, 0, 0, 0
    try:
        if engine_name == 'openai':
            usage = response_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
        elif engine_name == 'gemini':
            usage = response_data.get('usageMetadata', {})
            prompt_tokens = usage.get('promptTokenCount', 0)
            completion_tokens = usage.get('candidatesTokenCount', 0)
            total_tokens = usage.get('totalTokenCount', 0)
    except (KeyError, IndexError):
        log.warning("Could not parse token counts from API response.")
    return prompt_tokens, completion_tokens, reasoning_tokens, total_tokens

def extract_text_from_message(message: dict) -> str:
    """Extracts the text content from an OpenAI or Gemini message object."""
    content = ""
    if 'content' in message:
        if isinstance(message['content'], str):
            content = message['content']
        elif isinstance(message['content'], list):
            text_part = next((item['text'] for item in message['content'] if item.get('type') == 'text'), '')
            content = text_part
    elif 'parts' in message:
        text_part = next((part['text'] for part in message.get('parts', []) if 'text' in part), '')
        content = text_part
    return content

def construct_user_message(engine_name: str, text: str, image_data: list) -> dict:
    """Constructs a user message in the format required by the specified engine."""
    if engine_name == 'openai':
        content = [{"type": "text", "text": text}]
        if image_data:
            content.extend([{"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}} for img in image_data])
        return {"role": "user", "content": content}
    # gemini
    parts = [{"text": text}]
    if image_data:
        parts.extend([{"inline_data": {"mime_type": img['mime_type'], "data": img['data']}} for img in image_data])
    return {"role": "user", "parts": parts}

def construct_assistant_message(engine_name: str, text: str) -> dict:
    """Constructs an assistant message in the format required by the specified engine."""
    if engine_name == 'openai':
        return {"role": "assistant", "content": text}
    # gemini
    return {"role": "model", "parts": [{"text": text}]}

def ensure_dir_exists(directory: Path):
    """Creates a directory if it doesn't already exist."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.critical("Could not create directory '%s': %s", directory, e)
        sys.exit(1)

def sanitize_filename(name: str) -> str:
    """Takes a string and returns a safe version for a filename."""
    if not name: return ""
    name = name.replace(' ', '_').lower()
    name = re.sub(r'[^\w\._-]', '', name)
    name = re.sub(r'__+', '_', name)
    return name.strip('_.-')

def translate_history(history: list, target_engine: str) -> list:
    """Translates a conversation history between OpenAI and Gemini formats."""
    translated_history = []
    for message in history:
        role = message.get('role')
        text = extract_text_from_message(message)
        if target_engine == 'gemini':
            new_role = 'model' if role == 'assistant' else 'user'
            translated_history.append(construct_assistant_message(target_engine, text) if new_role == 'model' else construct_user_message(target_engine, text, []))
        elif target_engine == 'openai':
            new_role = 'assistant' if role == 'model' else 'user'
            translated_history.append(construct_assistant_message(target_engine, text) if new_role == 'assistant' else construct_user_message(target_engine, text, []))
    return translated_history

def _process_file_content(filepath: str, file_content: bytes, text_prompts: list, image_data: list):
    """Determines if content is text or image and processes it accordingly."""
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type and mime_type.startswith('image/'):
        encoded_image = base64.b64encode(file_content).decode('utf-8')
        image_data.append({"mime_type": mime_type, "data": encoded_image})
        log.info("Attached image: %s", filepath)
    else:
        try:
            text_content = file_content.decode('utf-8')
            filename = os.path.basename(filepath)
            text_prompts.append(f"--- START FILE: {filename} ---\n{text_content}\n--- END FILE: {filename} ---")
        except UnicodeDecodeError:
            log.warning("Could not decode file %s as UTF-8 text. Skipping.", filepath)

def _process_directory(path: str, exclude_abs: set, exclude_base: set, text_prompts: list, image_data: list):
    """Recursively processes files in a directory."""
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) not in exclude_abs and d not in exclude_base]
        for name in files:
            full_path = os.path.join(root, name)
            if os.path.abspath(full_path) in exclude_abs or name in exclude_base:
                continue
            try:
                with open(full_path, 'rb') as f:
                    _process_file_content(full_path, f.read(), text_prompts, image_data)
            except IOError as e:
                log.warning("Could not read file %s: %s", full_path, e)

def _process_zipfile(path: str, exclude_base: set, text_prompts: list, image_data: list):
    """Processes files within a zip archive."""
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                if filename.endswith('/') or os.path.basename(filename) in exclude_base:
                    continue
                with zip_ref.open(filename) as zf:
                    _process_file_content(filename, zf.read(), text_prompts, image_data)
    except (zipfile.BadZipFile, IOError) as e:
        log.warning("Could not process zip file %s: %s", path, e)

def process_files(file_paths: list, use_memory: bool, exclude_paths: list | None) -> tuple[str, list]:
    """Processes files, memories, and archives to build the system prompt and image data."""
    text_prompts, image_data = [], []
    exclude_abs = {os.path.abspath(p) for p in exclude_paths or []}
    exclude_base = {os.path.basename(p) for p in exclude_paths or []}
    if use_memory and os.path.exists(config.PERSISTENT_MEMORY_FILE):
        try:
            with open(config.PERSISTENT_MEMORY_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if content:
                text_prompts.append(f"--- MEMORY ---\n{content}\n---")
        except IOError as e:
            log.warning("Could not read %s: %s", config.PERSISTENT_MEMORY_FILE, e)
    for path in file_paths or []:
        abs_path = os.path.abspath(path)
        if abs_path in exclude_abs or os.path.basename(path) in exclude_base:
            continue
        if os.path.isfile(path):
            if path.endswith('.zip'):
                _process_zipfile(path, exclude_base, text_prompts, image_data)
            else:
                try:
                    with open(path, 'rb') as f:
                        _process_file_content(path, f.read(), text_prompts, image_data)
                except IOError as e:
                    log.warning("Could not read file %s: %s", path, e)
        elif os.path.isdir(path):
            _process_directory(path, exclude_abs, exclude_base, text_prompts, image_data)

    system_prompt = "\n\n".join(text_prompts)
    return system_prompt, image_data
