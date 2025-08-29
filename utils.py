# aicli/utils.py

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

# ANSI color codes for UI theming
USER_PROMPT = '\033[96m'      # Bright Cyan
ASSISTANT_PROMPT = '\033[93m' # Bright Yellow
SYSTEM_MSG = '\033[90m'      # Bright Black (Gray)
RESET_COLOR = '\033[0m'      # Reset

def format_token_string(token_dict: dict) -> str:
    """Formats the token dictionary into a consistent string for display."""
    if not token_dict or token_dict.get('total', 0) == 0:
        return ""
    p = token_dict.get('prompt', 0)
    c = token_dict.get('completion', 0)
    r = token_dict.get('reasoning', 0)
    t = token_dict.get('total', 0)
    return f" [{p}/{c}/{r}/{t}]"

def process_stream(engine_name: str, response: requests.Response) -> tuple[str, dict]:
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
                    print(delta, end='', flush=True)
                    full_text += delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    except KeyboardInterrupt:
        print(f"\n{SYSTEM_MSG}--> Stream cancelled by user.{RESET_COLOR}", file=sys.stderr)
    except requests.exceptions.ChunkedEncodingError:
        print(f"\n{SYSTEM_MSG}--> Warning: Stream connection interrupted.{RESET_COLOR}", file=sys.stderr)
    print() # Ensure there's a newline after streaming finishes or is cancelled.
    return full_text, tokens

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
        print(f"Error: Could not create directory '{directory}': {e}", file=sys.stderr)
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
        print(f"Attached image: {filepath}", file=sys.stderr)
    else:
        try:
            text_content = file_content.decode('utf-8')
            filename = os.path.basename(filepath)
            text_prompts.append(f"--- START FILE: {filename} ---\n{text_content}\n--- END FILE: {filename} ---")
        except UnicodeDecodeError:
            print(f"Warning: Could not decode file {filepath} as UTF-8 text. Skipping.", file=sys.stderr)

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
                print(f"Warning: Could not read file {full_path}: {e}", file=sys.stderr)

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
        print(f"Warning: Could not process zip file {path}: {e}", file=sys.stderr)

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
            print(f"Warning: Could not read {config.PERSISTENT_MEMORY_FILE}: {e}", file=sys.stderr)
    for path in file_paths or []:
        abs_path = os.path.abspath(path)
        if abs_path in exclude_abs or os.path.basename(abs_path) in exclude_base:
            print(f"Excluding: {path}", file=sys.stderr)
            continue
        if os.path.isdir(path):
            _process_directory(path, exclude_abs, exclude_base, text_prompts, image_data)
        elif os.path.isfile(path):
            if path.endswith('.zip'):
                _process_zipfile(path, exclude_base, text_prompts, image_data)
            else:
                try:
                    with open(path, 'rb') as f:
                        _process_file_content(path, f.read(), text_prompts, image_data)
                except IOError as e:
                    print(f"Warning: Could not read file {path}: {e}", file=sys.stderr)
    system_prompt = "\n\n".join(text_prompts) if text_prompts else None
    return system_prompt, image_data

def parse_token_counts(engine: str, response_data: dict | None) -> tuple[int, int, int, int]:
    """Extracts prompt, completion, reasoning, and total tokens from an API response."""
    if not response_data: return 0, 0, 0, 0
    prompt_tokens, completion_tokens, reasoning_tokens, total_tokens = 0, 0, 0, 0
    if engine == 'openai':
        usage = response_data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        total_output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        reasoning_tokens = usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
        completion_tokens = total_output_tokens - reasoning_tokens
    elif engine == 'gemini' and 'usageMetadata' in response_data:
        usage = response_data.get('usageMetadata', {})
        prompt_tokens = usage.get('promptTokenCount', 0)
        completion_tokens = usage.get('candidatesTokenCount', 0)
        reasoning_tokens = usage.get('thoughtsTokenCount', 0)
        total_tokens = usage.get('totalTokenCount', 0)
    return prompt_tokens, completion_tokens, reasoning_tokens, total_tokens
