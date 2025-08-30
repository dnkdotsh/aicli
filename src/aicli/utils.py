# aicli/utils.py
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
import re
import datetime
import zipfile
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Tuple

from . import config
from .logger import log

# ANSI color codes for prompts
USER_PROMPT = "\033[94m"  # Bright Blue
ASSISTANT_PROMPT = "\033[92m"  # Bright Green
SYSTEM_MSG = "\033[93m"  # Bright Yellow
DIRECTOR_PROMPT = "\033[95m" # Bright Magenta
RESET_COLOR = "\033[0m"

SUPPORTED_TEXT_EXTENSIONS = {
    '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
    '.csv', '.sh', '.bash', '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.php',
    '.rb', '.pl', '.sql', '.r', '.swift', '.kt', '.scala', '.ts', '.tsx', '.jsx', '.vue'
}
SUPPORTED_IMAGE_MIMETYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}

def read_system_prompt(prompt_or_path: str) -> str:
    """Reads a system prompt from a file path or returns the string directly."""
    if os.path.exists(prompt_or_path):
        with open(prompt_or_path, 'r', encoding='utf-8') as f:
            return f.read()
    return prompt_or_path

def ensure_dir_exists(directory_path: Path):
    """Creates a directory if it does not already exist."""
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Failed to create directory at %s: %s", directory_path, e)
        sys.exit(1)

def is_supported_text_file(filepath: Path) -> bool:
    """Check if a file is a supported text file based on its extension."""
    return filepath.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS

def is_supported_image_file(filepath: Path) -> bool:
    """Check if a file is a supported image file based on its MIME type."""
    mimetype, _ = mimetypes.guess_type(filepath)
    return mimetype in SUPPORTED_IMAGE_MIMETYPES

def process_files(paths: list | None, use_memory: bool, exclusions: list | None) -> tuple[str | None, str | None, list]:
    """
    Processes files, directories, and memory to build context.
    Returns memory content, attachment content, and image data separately.
    """
    if paths is None: paths = []
    if exclusions is None: exclusions = []

    memory_content_parts = []
    attachments_content_parts = []
    image_data_parts = []

    if use_memory and os.path.exists(config.PERSISTENT_MEMORY_FILE):
        try:
            with open(config.PERSISTENT_MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory_content_parts.append(f.read())
        except IOError as e:
            log.warning("Could not read persistent memory file: %s", e)

    exclusion_set = {Path(p).name for p in exclusions} | set(exclusions)

    def should_exclude(path: Path) -> bool:
        return path.name in exclusion_set or str(path) in exclusion_set

    def process_path(path_obj: Path):
        if should_exclude(path_obj): return
        if not path_obj.exists():
            log.warning("Path not found, skipping: %s", path_obj)
            return

        if path_obj.is_dir():
            for root, _, files in os.walk(path_obj):
                root_path = Path(root)
                if should_exclude(root_path): continue
                for file in files:
                    file_path = root_path / file
                    if not should_exclude(file_path): process_path(file_path)
        elif path_obj.is_file():
            if path_obj.suffix.lower() == '.zip':
                process_zip(path_obj)
            elif is_supported_text_file(path_obj):
                try:
                    with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        attachments_content_parts.append(f"--- FILE: {path_obj} ---\n{content}")
                except IOError as e:
                    log.warning("Could not read file %s: %s", path_obj, e)
            elif is_supported_image_file(path_obj):
                try:
                    with open(path_obj, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        mimetype, _ = mimetypes.guess_type(path_obj)
                        image_data_parts.append({"type": "image", "data": encoded_string, "mime_type": mimetype})
                except IOError as e:
                    log.warning("Could not read image file %s: %s", path_obj, e)

    def process_zip(zip_path: Path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                for filename in z.namelist():
                    if not filename.endswith('/'):
                        file_path = Path(filename)
                        if should_exclude(file_path): continue
                        if is_supported_text_file(file_path):
                            with z.open(filename) as f:
                                content = f.read().decode('utf-8', errors='ignore')
                                attachments_content_parts.append(f"--- FILE (from {zip_path.name}): {filename} ---\n{content}")
        except (zipfile.BadZipFile, IOError) as e:
            log.warning("Could not process zip file %s: %s", zip_path, e)

    for p in paths: process_path(Path(p))

    memory_str = "\n".join(memory_content_parts) if memory_content_parts else None
    attachments_str = "\n\n".join(attachments_content_parts) if attachments_content_parts else None

    return memory_str, attachments_str, image_data_parts

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '_', name)
    return name

def translate_history(history: list, target_engine: str) -> list:
    """Translates a conversation history to the target engine's format."""
    translated = []
    for msg in history:
        role = msg.get('role')
        if target_engine == 'gemini':
            if role == 'user':
                translated.append({'role': 'user', 'parts': msg.get('content')})
            elif role == 'assistant':
                translated.append({'role': 'model', 'parts': [{'text': extract_text_from_message(msg)}]})
        elif target_engine == 'openai':
            if role in ['user', 'model']:
                translated.append({'role': 'user' if role == 'user' else 'assistant', 'content': msg.get('parts')})
            else:
                translated.append(msg)
    return translated

def construct_user_message(engine_name: str, text: str, image_data: list) -> dict:
    """Constructs a user message in the format expected by the specified engine."""
    content = []
    if engine_name == 'openai':
        content.append({"type": "text", "text": text})
    else:  # Gemini
        content.append({"text": text})

    if image_data:
        for img in image_data:
            if engine_name == 'openai':
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}
                })
            else:  # Gemini
                content.append({
                    "inline_data": {"mime_type": img['mime_type'], "data": img['data']}
                })

    if engine_name == 'openai':
        return {"role": "user", "content": content}
    return {"role": "user", "parts": content}

def construct_assistant_message(engine_name: str, text: str) -> dict:
    """Constructs an assistant message in the format expected by the specified engine."""
    if engine_name == 'openai':
        return {"role": "assistant", "content": text}
    return {"role": "model", "parts": [{"text": text}]}

def extract_text_from_message(message: Dict[str, Any]) -> str:
    """Extracts the text part from a potentially complex message object."""
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                return part.get('text', '')
    return ''

def parse_token_counts(engine_name: str, response_data: dict) -> tuple[int, int, int, int]:
    """Parses token counts from a non-streaming API response."""
    p, c, r, t = 0, 0, 0, 0
    if not response_data:
        return 0, 0, 0, 0

    if engine_name == 'openai':
        if 'usage' in response_data:
            p = response_data['usage'].get('prompt_tokens', 0)
            c = response_data['usage'].get('completion_tokens', 0)
            t = response_data['usage'].get('total_tokens', 0)
    elif engine_name == 'gemini':
        usage = response_data.get('usageMetadata', {})
        p = usage.get('promptTokenCount', 0)
        c = usage.get('candidatesTokenCount', 0)
        t = usage.get('totalTokenCount', 0)

    return p, c, r, t

def process_stream(engine: str, response: Any, print_stream: bool = True) -> Tuple[str, dict]:
    """Processes a streaming API response."""
    full_response, p, c, r, t = "", 0, 0, 0, 0
    try:
        for chunk in response.iter_lines():
            if not chunk: continue
            decoded_chunk = chunk.decode('utf-8')
            if engine == 'openai':
                if decoded_chunk.startswith("data:"):
                    if "[DONE]" in decoded_chunk: break
                    try:
                        data = json.loads(decoded_chunk.split("data: ", 1)[1])
                        if 'choices' in data and data['choices'][0].get('delta', {}).get('content'):
                            text_chunk = data['choices'][0]['delta']['content']
                            if print_stream: sys.stdout.write(text_chunk); sys.stdout.flush()
                            full_response += text_chunk
                        if 'usage' in data and data['usage']:
                            p = data['usage'].get('prompt_tokens', 0)
                            c = data['usage'].get('completion_tokens', 0)
                            t = data['usage'].get('total_tokens', 0)
                    except json.JSONDecodeError: continue
            elif engine == 'gemini':
                try:
                    data = json.loads(decoded_chunk.split("data: ", 1)[1])
                    if 'candidates' in data:
                        text_chunk = data['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text', '')
                        if print_stream: sys.stdout.write(text_chunk); sys.stdout.flush()
                        full_response += text_chunk
                    if 'usageMetadata' in data:
                        p = data['usageMetadata'].get('promptTokenCount', 0)
                        c = data['usageMetadata'].get('candidatesTokenCount', 0)
                        r = data['usageMetadata'].get('cachedContentTokenCount', 0)
                        t = data['usageMetadata'].get('totalTokenCount', 0)
                except (json.JSONDecodeError, IndexError): continue
    except Exception as e:
        if print_stream: print(f"\n{SYSTEM_MSG}--> Stream interrupted: {e}{RESET_COLOR}")
        log.warning("Stream processing error: %s", e)

    tokens = {'prompt': p, 'completion': c, 'reasoning': r, 'total': t}
    return full_response, tokens

def format_token_string(token_dict: dict) -> str:
    """Formats the token dictionary into a consistent string for display."""
    p = token_dict.get('prompt', 0)
    c = token_dict.get('completion', 0)
    t = token_dict.get('total', 0)
    r_api = token_dict.get('reasoning', 0)
    
    if not any([p, c, t]):
        return ""

    # Prefer the API's reasoning count, otherwise calculate it.
    # Ensure it's not negative.
    r = r_api if r_api > 0 else max(0, t - (p + c))
    
    return f"\n{SYSTEM_MSG}[P:{p}/C:{c}/R:{r}/T:{t}]{RESET_COLOR}"

def display_help(context: str):
    """Displays help information for the given context (chat or multichat)."""
    if context == 'chat':
        help_text = """
Interactive Chat Commands:
  /exit             End the current session.
  /help             Display this help message.
  /stream           Toggle response streaming on/off.
  /debug            Toggle session-specific raw API logging.
  /memory           Toggle use of persistent memory for session conclusion.
  /clear            Clear the current conversation history.
  /history          Print the JSON of the current conversation history.
  /state            Print the current session state (engine, model, etc.).
  /engine [name]    Switch AI engine (openai/gemini). Translates history.
  /model [name]     Select a new model for the current engine.
  /set key value    Change a setting in settings.py (e.g., /set stream false).
  /max-tokens [num] Set max tokens for the session.
"""
    elif context == 'multichat':
        help_text = """
Multi-Chat Commands:
  /exit                     End the current session.
  /help                     Display this help message.
  /history                  Print the JSON of the shared conversation history.
  /ai <gpt|gem> [prompt]    Send a targeted prompt to only one AI.
                            If no prompt is given, the AI is asked to continue.
"""
    print(help_text)
