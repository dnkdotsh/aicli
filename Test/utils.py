# utils.py

import os
import sys
import base64
import mimetypes
import zipfile
import re
from io import BytesIO

import config

# --- NEW: Helper function to extract text from different message formats ---
def extract_text_from_message(message: dict) -> str:
    """Extracts the text content from an OpenAI or Gemini message object."""
    content = ""
    # Handle OpenAI format
    if 'content' in message:
        if isinstance(message['content'], str):
            content = message['content']
        elif isinstance(message['content'], list):
            # For multi-modal, find the text part
            text_part = next((item['text'] for item in message['content'] if item.get('type') == 'text'), '')
            content = text_part
    # Handle Gemini format
    elif 'parts' in message:
        text_part = next((part['text'] for part in message.get('parts', []) if 'text' in part), '')
        content = text_part
    return content

def ensure_dir_exists(directory: str):
    """Creates a directory if it doesn't already exist."""
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create directory '{directory}': {e}", file=sys.stderr)
        sys.exit(1)

def ensure_log_dir_exists():
    """Creates the log directory if it doesn't already exist."""
    ensure_dir_exists(config.LOG_DIRECTORY)

def sanitize_filename(name: str) -> str:
    """
    Takes a string and returns a safe version for a filename.
    Replaces spaces with underscores and removes disallowed characters.
    """
    if not name:
        return ""
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\._-]', '', name)
    name = re.sub(r'__+', '_', name)
    name = re.sub(r'--+', '-', name)
    name = name.strip('_.-')
    return name.lower()

def translate_history(history: list, target_engine: str) -> list:
    """Translates a conversation history between OpenAI and Gemini formats."""
    translated_history = []
    for message in history:
        role = message.get('role')
        translated_message = {}

        if target_engine == 'gemini':
            # From OpenAI format: {'role': 'user'/'assistant', 'content': '...'}
            # To Gemini format: {'role': 'user'/'model', 'parts': [{...}]}
            translated_message['role'] = 'model' if role == 'assistant' else 'user'
            content = message.get('content', '')
            parts = []
            if isinstance(content, str):
                parts.append({'text': content})
            elif isinstance(content, list): # Handle multi-modal
                for item in content:
                    if item.get('type') == 'text':
                         parts.append({'text': item.get('text', '')})
                    elif item.get('type') == 'image_url':
                        url = item.get('image_url', {}).get('url', '')
                        if url.startswith('data:'):
                            header, b64_data = url.split(',', 1)
                            mime_type = header.replace('data:', '').split(';')[0]
                            parts.append({'inline_data': {'mime_type': mime_type, 'data': b64_data}})
            translated_message['parts'] = parts
        
        elif target_engine == 'openai':
            # From Gemini format: {'role': 'user'/'model', 'parts': [{...}]}
            # To OpenAI format: {'role': 'user'/'assistant', 'content': '...'}
            translated_message['role'] = 'assistant' if role == 'model' else 'user'
            parts = message.get('parts', [])
            content_list = []
            text_parts = []
            has_images = False
            for part in parts:
                if 'text' in part:
                    text_parts.append(part['text'])
                elif 'inline_data' in part:
                    has_images = True
                    mime = part['inline_data'].get('mime_type')
                    data = part['inline_data'].get('data')
                    content_list.append({
                        'type': 'image_url',
                        'image_url': {'url': f'data:{mime};base64,{data}'}
                    })
            
            full_text = '\n'.join(text_parts)
            if has_images:
                content_list.insert(0, {'type': 'text', 'text': full_text})
                translated_message['content'] = content_list
            else: # Text only
                translated_message['content'] = full_text

        translated_history.append(translated_message)
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
            text_prompts.append(f"This is an attached file called '{filename}':\n---\n{text_content}\n---")
        except UnicodeDecodeError:
            print(f"Warning: Could not decode file {filepath} as UTF-8 text. Skipping.", file=sys.stderr)

def process_files(file_paths: list, use_memory: bool, exclude_paths: list | None) -> tuple[str, list]:
    """
    Processes files, memories, and archives to build the system prompt and image data.
    """
    text_prompts = []
    image_data = [] 
    
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

    paths_to_process = list(file_paths or [])

    for path in paths_to_process:
        abs_path = os.path.abspath(path)
        if abs_path in exclude_abs or os.path.basename(abs_path) in exclude_base:
            print(f"Excluding: {path}", file=sys.stderr)
            continue

        if os.path.isdir(path):
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
        
        elif os.path.isfile(path):
            if path.endswith('.zip'):
                try:
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        for filename in zip_ref.namelist():
                            if filename.endswith('/'): continue 
                            if os.path.basename(filename) in exclude_base: continue
                            
                            with zip_ref.open(filename) as zf:
                                _process_file_content(filename, zf.read(), text_prompts, image_data)
                
                except (zipfile.BadZipFile, IOError) as e:
                    print(f"Warning: Could not process zip file {path}: {e}", file=sys.stderr)
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
    if not response_data:
        return 0, 0, 0, 0
        
    prompt_tokens, completion_tokens, reasoning_tokens, total_tokens = 0, 0, 0, 0
    
    if engine == 'openai':
        usage = response_data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        total_output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        completion_details = usage.get('completion_tokens_details', {})
        reasoning_tokens = completion_details.get('reasoning_tokens', 0)
        completion_tokens = total_output_tokens - reasoning_tokens

    elif engine == 'gemini' and 'usageMetadata' in response_data:
        usage = response_data.get('usageMetadata', {})
        prompt_tokens = usage.get('promptTokenCount', 0)
        completion_tokens = usage.get('candidatesTokenCount', 0)
        reasoning_tokens = usage.get('thoughtsTokenCount', 0)
        total_tokens = usage.get('totalTokenCount', 0)

    return prompt_tokens, completion_tokens, reasoning_tokens, total_tokens
