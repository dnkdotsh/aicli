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
import requests # Added for requests.exceptions.RequestException
import tarfile # Added for .tar.gz support
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
    '.rb', '.pl', '.sql', '.r', '.swift', '.kt', '.scala', '.ts', '.tsx', '.jsx', '.vue',
    '.jsonl', '.diff', '.log',
}
SUPPORTED_IMAGE_MIMETYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
SUPPORTED_EXTENSIONLESS_FILENAMES = {
    'dockerfile', 'makefile', 'vagrantfile', 'jenkinsfile', 'procfile', 'rakefile', '.gitignore',
    'license'
}

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
    """Check if a file is a supported text file based on its extension or name."""
    if filepath.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
        return True
    # If there's no extension, check against the list of known filenames.
    if not filepath.suffix:
        return filepath.name.lower() in SUPPORTED_EXTENSIONLESS_FILENAMES
    return False

def is_supported_image_file(filepath: Path) -> bool:
    """Check if a file is a supported image file based on its MIME type."""
    mimetype, _ = mimetypes.guess_type(filepath)
    return mimetype in SUPPORTED_IMAGE_MIMETYPES

def process_files(paths: list | None, use_memory: bool, exclusions: list | None) -> tuple[str | None, dict, list]:
    """
    Processes files, directories, and memory to build context.
    Returns memory content, a dictionary of attachment paths to content, and image data.
    """
    paths = paths or []
    exclusions = exclusions or []

    memory_content_parts = []
    attachments_dict = {}
    image_data_parts = []

    if use_memory and os.path.exists(config.PERSISTENT_MEMORY_FILE):
        try:
            with open(config.PERSISTENT_MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory_content_parts.append(f.read())
        except IOError as e:
            log.warning("Could not read persistent memory file: %s", e)

    exclusion_paths = {Path(p).resolve() for p in exclusions}

    def _process_text_file(filepath: Path):
        """Helper to read and store text file content in the dictionary."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                attachments_dict[filepath] = f.read()
        except IOError as e:
            log.warning("Could not read file %s: %s", filepath, e)

    def _process_image_file(filepath: Path):
        """Helper to read and encode image file data."""
        try:
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                mimetype, _ = mimetypes.guess_type(filepath)
                if mimetype in SUPPORTED_IMAGE_MIMETYPES:
                    image_data_parts.append({"type": "image", "data": encoded_string, "mime_type": mimetype})
        except IOError as e:
            log.warning("Could not read image file %s: %s", filepath, e)

    def _process_zip_file(zip_path: Path):
        """Helper to process text files within a zip archive."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                zip_content_parts = []
                for filename in z.namelist():
                    if filename.endswith('/'): continue
                    if Path(filename).name in {p.name for p in exclusion_paths}:
                        continue
                    if is_supported_text_file(Path(filename)):
                        with z.open(filename) as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            zip_content_parts.append(f"--- FILE (from {zip_path.name}): {filename} ---\n{content}")
                if zip_content_parts:
                    attachments_dict[zip_path] = "\n\n".join(zip_content_parts)
        except (zipfile.BadZipFile, IOError) as e:
            log.warning("Could not process zip file %s: %s", zip_path, e)

    def _process_tar_file(tar_path: Path):
        """Helper to process text files within a .tar.gz or .tgz archive."""
        try:
            # 'r:gz' mode transparently handles gzip decompression
            with tarfile.open(tar_path, "r:gz") as tar:
                tar_content_parts = []
                for member in tar.getmembers():
                    if not member.isfile(): continue
                    member_path = Path(member.name)
                    if member_path.name in {p.name for p in exclusion_paths}:
                        continue
                    if is_supported_text_file(member_path):
                        # extractfile returns a file-like object, read from it
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            content = file_obj.read().decode('utf-8', errors='ignore')
                            tar_content_parts.append(f"--- FILE (from {tar_path.name}): {member.name} ---\n{content}")
                if tar_content_parts:
                    attachments_dict[tar_path] = "\n\n".join(tar_content_parts)
        except (tarfile.TarError, IOError) as e:
            log.warning("Could not process tar file %s: %s", tar_path, e)

    for p_str in paths:
        path_obj = Path(p_str).resolve()

        if path_obj in exclusion_paths:
            continue
        if not path_obj.exists():
            log.warning("Path not found, skipping: %s", path_obj)
            continue

        if path_obj.is_file():
            file_name_lower = path_obj.name.lower()
            if file_name_lower.endswith('.zip'):
                _process_zip_file(path_obj)
            elif file_name_lower.endswith('.tar.gz') or file_name_lower.endswith('.tgz'):
                _process_tar_file(path_obj)
            elif is_supported_text_file(path_obj):
                _process_text_file(path_obj)
            elif is_supported_image_file(path_obj):
                _process_image_file(path_obj)
        elif path_obj.is_dir():
            for root, dirs, files in os.walk(path_obj, topdown=True):
                root_path = Path(root).resolve()
                dirs[:] = [d for d in dirs if (root_path / d).resolve() not in exclusion_paths]
                for name in files:
                    file_path = (root_path / name).resolve()
                    if file_path in exclusion_paths:
                        continue
                    if is_supported_text_file(file_path):
                        _process_text_file(file_path)
                    elif is_supported_image_file(file_path):
                        _process_image_file(file_path)

    memory_str = "\n".join(memory_content_parts) if memory_content_parts else None
    return memory_str, attachments_dict, image_data_parts

def sanitize_filename(name: str) -> str:
    r"""
    Sanitizes a string to be a valid filename.
    Allows unicode word characters as regex \w (default) includes them.
    """
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '_', name)
    return name or "unnamed_log"

def translate_history(history: list, target_engine: str) -> list:
    """Translates a conversation history to the target engine's format."""
    translated = []
    for msg in history:
        role = msg.get('role')
        if role not in ['user', 'assistant', 'model']:
            continue
        text_content = extract_text_from_message(msg)
        if role in ['user']:
            translated.append(construct_user_message(target_engine, text_content, []))
        elif role in ['assistant', 'model']:
            translated.append(construct_assistant_message(target_engine, text_content))
    return translated

def construct_user_message(engine_name: str, text: str, image_data: list) -> dict:
    """Constructs a user message in the format expected by the specified engine."""
    content = []
    if engine_name == 'openai':
        content.append({"type": "text", "text": text})
    else:
        content.append({"text": text})
    if image_data:
        for img in image_data:
            if engine_name == 'openai':
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}
                })
            else:
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
    """
    Extracts the text part from a potentially complex message object from
    either OpenAI or Gemini format.
    """
    content = message.get('content')
    if isinstance(content, str): return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                return part.get('text', '')
    parts = message.get('parts')
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and 'text' in part:
                return part.get('text', '')
    return message.get('text', '')

def parse_token_counts(engine_name: str, response_data: dict) -> tuple[int, int, int, int]:
    """Parses token counts from a non-streaming API response."""
    p, c, r, t = 0, 0, 0, 0
    if not response_data: return 0, 0, 0, 0
    if engine_name == 'openai':
        if 'usage' in response_data:
            p = response_data['usage'].get('prompt_tokens', 0)
            c = response_data['usage'].get('completion_tokens', 0)
            t = response_data['usage'].get('total_tokens', 0)
    elif engine_name == 'gemini':
        usage = response_data.get('usageMetadata', {})
        p = usage.get('promptTokenCount', 0)
        c = usage.get('candidatesTokenCount', 0)
        r = usage.get('cachedContentTokenCount', 0)
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
                        if 'choices' in data and data['choices'] and data['choices'][0].get('delta', {}).get('content'):
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
    except KeyboardInterrupt:
        if print_stream:
            # A newline is needed to move the cursor to the next line after the partial response.
            print(f"\n{SYSTEM_MSG}--> Stream interrupted by user.{RESET_COLOR}")
    except Exception as e:
        if print_stream: print(f"\n{SYSTEM_MSG}--> Stream interrupted by network/API error: {e}{RESET_COLOR}")
        log.warning("Stream processing error: %s", e)
    tokens = {'prompt': p, 'completion': c, 'reasoning': r, 'total': t}
    return full_response, tokens

def format_token_string(token_dict: dict) -> str:
    """Formats the token dictionary into a consistent string for display."""
    p = token_dict.get('prompt', 0); c = token_dict.get('completion', 0); t = token_dict.get('total', 0); r_api = token_dict.get('reasoning', 0)
    if not any([p, c, t]): return ""
    r = r_api if r_api > 0 else max(0, t - (p + c))
    return f"\n{SYSTEM_MSG}[P:{p}/C:{c}/R:{r}/T:{t}]{RESET_COLOR}"

def display_help(context: str):
    """Displays help information for the given context (chat or multichat)."""
    if context == 'chat':
        help_text = """
Interactive Chat Commands:
  /exit [name]      End the session. Optionally provide a name for the log file.
  /quit             Exit immediately without updating memory or renaming the log.
  /help             Display this help message.
  /stream           Toggle response streaming on/off.
  /debug            Toggle session-specific raw API logging.
  /memory           View the content of the persistent memory file.
  /remember [text]  If text is provided, inject it into persistent memory.
                    If no text, consolidates current chat into memory.
  /clear            Clear the current conversation history.
  /history          Print the JSON of the current conversation history.
  /state            Print the current session state (engine, model, etc.).
  /refresh [name]   Re-read attached files to update the context.
                    If no name is given, all files are refreshed.
                    Otherwise, refreshes all files containing 'name'.
  /files            List all currently attached text files, sorted by size.
  /attach <path>    Attach a file or directory to the session context.
  /detach <name>    Detach a file from the context by its filename.
  /save [name] [--stay] [--remember]
                    Save the session. Auto-generates a name if not provided.
                    --stay:      Do not exit after saving.
                    --remember:  Update persistent memory before exiting.
                    (Default is to save and exit without updating memory).
  /load <filename>  Load a session, replacing the current one.
  /engine [name]    Switch AI engine (openai/gemini). Translates history.
  /model [name]     Select a new model for the current engine.
  /persona <name>   Switch to a different persona. Use `/persona clear` to remove.
  /personas         List all available personas.
  /set [key] [val]  Change a setting (e.g., /set stream false).
                    Run without arguments to list all settings.
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
