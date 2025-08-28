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
import datetime
import base64
import requests

import config
import api_client
import utils

def select_model(api_key: str, engine: str, task: str) -> str:
    """Allows the user to select a model or use the default."""
    default_model = ""
    if task == 'chat':
        default_model = config.DEFAULT_OPENAI_CHAT_MODEL if engine == 'openai' else config.DEFAULT_GEMINI_MODEL
    elif task == 'image':
        default_model = config.DEFAULT_OPENAI_IMAGE_MODEL

    use_default = input(f"Use default model ({default_model})? (Y/n): ").lower().strip()
    if use_default in ('', 'y', 'yes'):
        return default_model

    print("Fetching available models...")
    models = api_client.fetch_available_models(api_key, engine, task)
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

def _format_token_string(token_dict: dict) -> str:
    """Formats the token dictionary into a consistent string for display."""
    if not token_dict or token_dict.get('total', 0) == 0:
        return ""
    p = token_dict.get('prompt', 0)
    c = token_dict.get('completion', 0)
    r = token_dict.get('reasoning', 0)
    t = token_dict.get('total', 0)
    return f" [{p}/{c}/{r}/{t}]"

def _process_stream(engine: str, response: requests.Response) -> tuple[str, dict]:
    """Processes a streaming API response, printing deltas and returning the full text and token counts."""
    full_text = ""
    tokens = {}
    
    try:
        for line in response.iter_lines():
            if not line:
                continue
                
            if line.startswith(b'data: '):
                line_data = line[6:]
            else:
                continue

            if engine == 'openai' and line_data == b'[DONE]':
                break
            
            try:
                chunk = json.loads(line_data)
                delta = ""
                if engine == 'openai':
                    delta = chunk['choices'][0]['delta'].get('content', '')
                elif engine == 'gemini' and 'candidates' in chunk:
                    delta = chunk['candidates'][0]['content']['parts'][0]['text']
                    if chunk['candidates'][0].get('finishReason') == 'STOP':
                        p, c, r, t = utils.parse_token_counts(engine, chunk)
                        tokens = {'prompt': p, 'completion': c, 'reasoning': r, 'total': t}
    
                if delta:
                    print(delta, end='', flush=True)
                    full_text += delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    except requests.exceptions.ChunkedEncodingError:
        print("\nWarning: Stream connection interrupted.", file=sys.stderr)
    
    print() 
    return full_text, tokens

def _perform_chat_request(api_key: str, engine: str, model: str, messages_or_contents: list, system_prompt: str | None, max_tokens: int, stream: bool, session_raw_logs: list | None = None) -> tuple[str, dict]:
    """
    Executes a single chat request, handling both streaming and non-streaming,
    and returns the response text and a token dictionary.
    """
    payload = {}
    url = ""
    headers = {}

    if engine == 'openai':
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        all_messages = messages_or_contents.copy()
        if system_prompt:
            all_messages.insert(0, {"role": "system", "content": system_prompt})
        payload = {"model": model, "messages": all_messages, "stream": stream}
        
        if max_tokens:
            if 'o' in model or 'mini' in model:
                payload['max_completion_tokens'] = max_tokens
            else:
                payload['max_tokens'] = max_tokens
    
    elif engine == 'gemini':
        headers = {"Content-Type": "application/json"}
        if stream:
            url = f"https://generativelaunguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={api_key}&alt=sse"
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {"contents": messages_or_contents}
        if system_prompt:
            payload['system_instruction'] = {'parts': [{'text': system_prompt}]}
        if max_tokens:
            payload['generationConfig'] = {'maxOutputTokens': max_tokens}
            
    response_obj = api_client.make_api_request(url, headers, payload, stream, session_raw_logs)

    if not response_obj:
        return "", {}

    if stream:
        return _process_stream(engine, response_obj)
        
    response_data = response_obj
    assistant_response = ""
    if engine == 'openai' and 'choices' in response_data:
        assistant_response = utils.extract_text_from_message(response_data['choices'][0]['message'])
    elif engine == 'gemini' and 'candidates' in response_data:
        try:
            assistant_response = response_data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            finish_reason = response_data['candidates'][0].get('finishReason', 'UNKNOWN')
            print(f"Warning: Could not extract Gemini response part. Finish reason: {finish_reason}", file=sys.stderr)

    p, c, r, t = utils.parse_token_counts(engine, response_data)
    tokens = {'prompt': p, 'completion': c, 'reasoning': r, 'total': t}
    return assistant_response, tokens

def _handle_slash_command(user_input: str, state: dict) -> bool:
    """Handles in-app slash commands. Returns True if the session should end."""
    command = user_input.lower().strip()

    if command == '/exit':
        return True
    
    elif command == '/stream':
        state['stream_active'] = not state['stream_active']
        status = "ENABLED" if state['stream_active'] else "DISABLED"
        print(f"--> Response streaming is now {status}.")

    elif command == '/debug':
        state['debug_active'] = not state['debug_active']
        status = "ENABLED" if state['debug_active'] else "DISABLED"
        print(f"--> Session-specific debug logging is now {status}.")
    
    elif command == '/clear':
        confirm = input("This will clear all conversation history, including initial context. If you're sure, type `proceed`: ")
        if confirm.lower() == 'proceed':
            state['history'].clear()
            print("--> Conversation history has been cleared.")
        else:
            print("--> Clear cancelled.")

    elif command == '/forget':
        state['history'].clear()
        print("--> Conversation history forgotten. Initial context remains.")

    elif command == '/model':
        state['model'] = select_model(state['api_key'], state['engine'], 'chat')
        print(f"--> Model changed to: {state['model']}")

    elif command == '/engine':
        new_engine = 'gemini' if state['engine'] == 'openai' else 'openai'
        try:
            new_api_key = api_client.check_api_keys(new_engine)
            state['history'] = utils.translate_history(state['history'], new_engine)
            state['engine'] = new_engine
            state['api_key'] = new_api_key
            state['model'] = config.DEFAULT_OPENAI_CHAT_MODEL if new_engine == 'openai' else config.DEFAULT_GEMINI_MODEL
            print(f"--> Engine switched to {state['engine'].capitalize()}. Model set to default: {state['model']}. History translated.")
        except api_client.MissingApiKeyError as e:
             print(f"--> Switch to {new_engine.capitalize()} failed: API key not found.")
    
    
    else:
        print(f"--> Unknown command: {command}")

    return False

def _summarize_and_trim_history(session_state: dict) -> list:
    """
    Summarizes the oldest part of the chat history to conserve context space.
    """
    try:
        # Determine the number of messages to trim (1 turn = 2 messages)
        trim_count = config.HISTORY_SUMMARY_TRIM_TURNS * 2
        history_to_summarize = session_state['history'][:trim_count]
        remaining_history = session_state['history'][trim_count:]

        # Format the history into a single string for the prompt
        log_text = "\n".join([f"{msg['role']}: {utils.extract_text_from_message(msg)}" for msg in history_to_summarize])

        prompt = (
            "You are a summarization agent. Concisely summarize the key facts, topics, and user intent "
            "from the following conversation excerpt. This summary will serve as a memory for the rest of "
            "the conversation.\n\n"
            f"--- EXCERPT ---\n{log_text}\n--- END EXCERPT ---\n\n"
            "SUMMARY:"
        )

        engine = session_state['engine']
        task_model = config.DEFAULT_HELPER_MODEL_OPENAI if engine == 'openai' else config.DEFAULT_HELPER_MODEL_GEMINI
        
        messages_or_contents = []
        if engine == 'openai':
            messages_or_contents = [{"role": "user", "content": prompt}]
        else: # gemini
            messages_or_contents = [{"role": "user", "parts": [{"text": prompt}]}]
        
        summary_text, _ = _perform_chat_request(
            api_key=session_state['api_key'],
            engine=engine,
            model=task_model,
            messages_or_contents=messages_or_contents,
            system_prompt=None,
            max_tokens=config.SUMMARY_MAX_TOKENS,
            stream=False
        )

        if not summary_text:
            # If summarization fails, return the original history to avoid data loss
            print("Warning: History summarization failed. Conversation context may be lost.", file=sys.stderr)
            return session_state['history']

        # Inject the summary as the first message in the remaining history.
        # We format it as a user message for maximum compatibility between models.
        summary_block = f"--- Summary of Prior Conversation ---\n{summary_text}\n---"
        if engine == 'openai':
            summary_msg = {"role": "user", "content": summary_block}
        else: # gemini
            summary_msg = {"role": "user", "parts": [{"text": summary_block}]}
        
        return [summary_msg] + remaining_history

    except Exception as e:
        print(f"Error during history summarization: {e}", file=sys.stderr)
        # Fallback to returning the original history
        return session_state['history']

def _perform_interactive_chat(api_key: str, engine: str, model: str, system_prompt: str, initial_image_data: list, session_name: str, max_tokens: int, stream: bool):
    if session_name:
        log_filename_base = f"{session_name}.jsonl"
    else:
        log_filename_base = f"chat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{engine}.jsonl"
    log_filename = os.path.join(config.LOG_DIRECTORY, log_filename_base)

    print(f"Starting interactive chat with {engine.capitalize()} ({model}). Type '/exit' to end.")
    print(f"Available commands: /exit, /debug, /clear, /forget, /model, /engine, /stream")
    print(f"Session log will be saved to: {log_filename}")

    session_state = {
        'api_key': api_key,
        'engine': engine,
        'model': model,
        'system_prompt': system_prompt,
        'initial_image_data': initial_image_data,
        'history': [],
        'debug_active': False,
        'stream_active': stream,
        'session_raw_logs': []
    }
    
    if system_prompt: print("System prompt is active.")
    if initial_image_data: print(f"Attached {len(initial_image_data)} image(s) to this session.")

    first_turn = True
    try:
        while True:
            # Check history size and summarize if necessary BEFORE getting new input
            if len(session_state['history']) >= config.HISTORY_SUMMARY_THRESHOLD_TURNS * 2:
                print("\n--> Compressing conversation history to conserve context...", file=sys.stderr)
                session_state['history'] = _summarize_and_trim_history(session_state)
                print("--> Compression complete.", file=sys.stderr)

            user_input = input("\nYou: ")
            if not user_input.strip(): continue

            
            if user_input.startswith('/'):
                if _handle_slash_command(user_input, session_state):
                    break 
                continue

            messages_or_contents = list(session_state['history'])
            
            if session_state['engine'] == 'openai':
                user_content = [{"type": "text", "text": user_input}]
                if first_turn and session_state['initial_image_data']:
                    user_content.extend([{"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}} for img in session_state['initial_image_data']])
                user_msg = {"role": "user", "content": user_content}
            else: # gemini
                user_parts = [{"text": user_input}]
                if first_turn and session_state['initial_image_data']:
                    user_parts.extend([{"inline_data": {"mime_type": img['mime_type'], "data": img['data']}} for img in session_state['initial_image_data']])
                user_msg = {"role": "user", "parts": user_parts}
            
            
            messages_or_contents.append(user_msg)
            
            srl_list = session_state['session_raw_logs'] if session_state['debug_active'] else None
            
            if session_state['stream_active']:
                print(f"\nAssistant: ", end='', flush=True)
            
            
            response_text, token_dict = _perform_chat_request(
                api_key=session_state['api_key'],
                engine=session_state['engine'],
                model=session_state['model'],
                messages_or_contents=messages_or_contents,
                system_prompt=session_state['system_prompt'],
                max_tokens=max_tokens,
                stream=session_state['stream_active'],
                session_raw_logs=srl_list
            )
            
            if not session_state['stream_active']:
                print(f"\nAssistant: {response_text}", end='')

            
            token_str = _format_token_string(token_dict)
            print(token_str)
            
            if session_state['engine'] == 'openai':
                asst_msg = {"role": "assistant", "content": response_text}
            else:
                asst_msg = {"role": "model", "parts": [{"text": response_text}]}

            session_state['history'].extend([user_msg, asst_msg])
            
            try:
                log_entry = {"timestamp": datetime.datetime.now().isoformat(), "model": session_state['model'], "prompt": user_msg, "response": asst_msg, "tokens": token_dict}
                with open(log_filename, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except IOError as e:
                print(f"\nWarning: Could not write to session log file: {e}", file=sys.stderr)
            
            first_turn = False

    except (KeyboardInterrupt, EOFError):
        print("\nSession interrupted.")
    finally:
        print("\nSession ended.")
        if os.path.exists(log_filename):
            _update_persistent_memory(session_state['api_key'], session_state['engine'], session_state['model'], log_filename)
            if not session_name:
                rename_session_log(session_state['api_key'], session_state['engine'], log_filename)
        
        if session_state['debug_active']:
            debug_filename = f"debug_{os.path.splitext(log_filename_base)[0]}.jsonl"
            debug_filepath = os.path.join(config.LOG_DIRECTORY, debug_filename)
            print(f"Saving debug log to: {debug_filepath}")
            try:
                with open(debug_filepath, 'w', encoding='utf-8') as f:
                    for entry in session_state['session_raw_logs']:
                        f.write(json.dumps(entry) + '\n')
            except IOError as e:
                print(f"\nWarning: Could not write to debug log file: {e}", file=sys.stderr)

def _update_persistent_memory(api_key: str, engine: str, model: str, log_file: str):
    """
    Automatically summarizes the session and integrates it into the persistent memory.
    """
    try:
        print("Summarizing session for memory...")
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        summary_prompt = (
            "Provide a concise, third-person summary of the key topics, facts, and outcomes from the following chat log. "
            "Focus on information worth remembering for future sessions. Your summary must be under 1024 tokens.\n"
            f"CHAT LOG:\n---\n{log_content}\n---"
        )
        
        messages_or_contents = []
        if engine == 'openai':
            messages_or_contents = [{"role": "user", "content": summary_prompt}]
        else:
            messages_or_contents = [{"role": "user", "parts": [{"text": summary_prompt}]}]

        summary_text, _ = _perform_chat_request(api_key, engine, model, messages_or_contents, None, config.SUMMARY_MAX_TOKENS, stream=False)
        
        if not summary_text:
            print("Warning: Failed to generate session summary. Memory not updated.", file=sys.stderr)
            return

        print("Updating persistent memory...")
        existing_ltm = ""
        if os.path.exists(config.PERSISTENT_MEMORY_FILE):
            with open(config.PERSISTENT_MEMORY_FILE, 'r', encoding='utf-8') as f:
                existing_ltm = f.read()

        integration_prompt = (
            "You are a memory consolidation agent. Integrate the new session summary into the existing persistent memory. "
            "Combine related topics, update existing facts, and discard trivial data to keep the memory concise and relevant. "
            "The goal is a dense, factual summary of all interactions.\n\n"
            f"--- EXISTING PERSISTENT MEMORY ---\n{existing_ltm}\n\n"
            f"--- NEW SESSION SUMMARY TO INTEGRATE ---\n{summary_text}\n\n"
            "--- UPDATED PERSISTENT MEMORY ---"
        )
        
        task_model = config.DEFAULT_HELPER_MODEL_OPENAI if engine == 'openai' else config.DEFAULT_HELPER_MODEL_GEMINI
        
        if engine == 'openai':
            messages_or_contents = [{"role": "user", "content": integration_prompt}]
        else:
            messages_or_contents = [{"role": "user", "parts": [{"text": integration_prompt}]}]
            
        updated_ltm, _ = _perform_chat_request(api_key, engine, task_model, messages_or_contents, None, None, stream=False)

        if updated_ltm:
            with open(config.PERSISTENT_MEMORY_FILE, 'w', encoding='utf-8') as f:
                f.write(updated_ltm.strip())
            print("Persistent memory updated successfully.")
        else:
            print("Warning: Failed to update persistent memory.", file=sys.stderr)

    except Exception as e:
        print(f"Error during memory update: {e}", file=sys.stderr)

def rename_session_log(api_key: str, engine: str, log_file: str):
    """
    Generates a descriptive name for the chat log using an AI model and renames the file.
    """
    print("Generating smart name for session log...")
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = "".join(f.readlines()[:50]) 

        prompt = (
            "Based on the following chat log, generate a concise, descriptive, filename-safe title. "
            "Use snake_case. The title should be 3-5 words. Do not include any file extension like '.jsonl'. "
            "Example response: 'python_script_debugging_and_refactoring'\n\n"
            f"CHAT LOG:\n---\n{log_content}\n---"
        )
        
        task_model = config.DEFAULT_HELPER_MODEL_OPENAI if engine == 'openai' else config.DEFAULT_HELPER_MODEL_GEMINI
        
        if engine == 'openai':
            messages_or_contents = [{"role": "user", "content": prompt}]
        else:
            messages_or_contents = [{"role": "user", "parts": [{"text": prompt}]}]
            
        new_name, _ = _perform_chat_request(api_key, engine, task_model, messages_or_contents, None, 50, stream=False)
        sanitized_name = utils.sanitize_filename(new_name.strip())
        
        if sanitized_name:
            new_filename_base = f"{sanitized_name}.jsonl"
            new_filepath = os.path.join(config.LOG_DIRECTORY, new_filename_base)
            
            counter = 1
            while os.path.exists(new_filepath):
                new_filename_base = f"{sanitized_name}_{counter}.jsonl"
                new_filepath = os.path.join(config.LOG_DIRECTORY, new_filename_base)
                counter += 1

            os.rename(log_file, new_filepath)
            print(f"Session log saved as: {new_filepath}")
        else:
            print(f"Warning: Could not generate a valid name. Log remains as: {log_file}", file=sys.stderr)

    except (IOError, OSError, Exception) as e:
        print(f"Warning: Could not rename session log ({e}). Log remains as: {log_file}", file=sys.stderr)

def handle_chat(api_key: str, engine: str, model: str, system_prompt: str, initial_prompt: str, image_data: list, session_name: str, max_tokens: int, stream: bool):
    """Handles both single-shot and interactive chat sessions."""
    if initial_prompt:
        messages_or_contents = []
        if engine == 'openai':
            user_content = [{"type": "text", "text": initial_prompt}]
            if image_data:
                user_content.extend([{"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}} for img in image_data])
            messages_or_contents.append({"role": "user", "content": user_content})
        else: # gemini
            user_parts = [{"text": initial_prompt}]
            if image_data:
                user_parts.extend([{"inline_data": {"mime_type": img['mime_type'], "data": img['data']}} for img in image_data])
            messages_or_contents.append({"role": "user", "parts": user_parts})
        
        if stream:
             print("Assistant: ", end='', flush=True)

        response, token_dict = _perform_chat_request(api_key, engine, model, messages_or_contents, system_prompt, max_tokens, stream)
        
        if not stream:
            print(f"Assistant: {response}", end='')
        
        token_str = _format_token_string(token_dict)
        print(token_str)
    else:
        _perform_interactive_chat(api_key, engine, model, system_prompt, image_data, session_name, max_tokens, stream)

def handle_both_engines(openai_key: str, gemini_key: str, system_prompt: str, prompt: str, image_data: list, max_tokens: int, stream: bool):
    """Sends a single-shot prompt to both OpenAI and Gemini and prints the results."""
    print("--- OpenAI Response ---")
    
    if stream:
        print("Assistant: ", end='', flush=True)
    openai_model = select_model(openai_key, 'openai', 'chat')
    
    openai_user_content = [{"type": "text", "text": prompt}]
    if image_data:
        openai_user_content.extend([{"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}} for img in image_data])
    openai_messages = [{"role": "user", "content": openai_user_content}]
    
    openai_response, openai_tokens = _perform_chat_request(openai_key, 'openai', openai_model, openai_messages, system_prompt, max_tokens, stream)
    if not stream:
        print(f"Assistant: {openai_response}", end='')
    
    token_str = _format_token_string(openai_tokens)
    print(token_str)
    print("\n" + "="*50 + "\n")

    print("--- Gemini Response ---")
    if stream:
        print("Assistant: ", end='', flush=True)
    gemini_model = select_model(gemini_key, 'gemini', 'chat')
    
    gemini_user_parts = [{"text": prompt}]
    if image_data:
        gemini_user_parts.extend([{"inline_data": {"mime_type": img['mime_type'], "data": img['data']}} for img in image_data])
    gemini_contents = [{"role": "user", "parts": gemini_user_parts}]
        
    gemini_response, gemini_tokens = _perform_chat_request(gemini_key, 'gemini', gemini_model, gemini_contents, system_prompt, max_tokens, stream)
    if not stream:
        print(f"Assistant: {gemini_response}", end='')
    token_str = _format_token_string(gemini_tokens)
    print(token_str)
    print("\n" + "="*50 + "\n")

def handle_image_generation(api_key: str, prompt: str):
    """Handles OpenAI image generation."""
    model = select_model(api_key, 'openai', 'image')
    if not prompt:
        if sys.stdin.isatty():
             prompt = input("Enter a description for the image: ")
        else:
            prompt = sys.stdin.read().strip()

    if not prompt:
        print("Image generation cancelled: No prompt provided.", file=sys.stderr)
        return

    print(f"Generating image with {model} for prompt: '{prompt}'...")
    
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }
    
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
        
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": model,
                    "prompt": prompt,
                    "file": filepath
                }
                with open(config.IMAGE_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except IOError as e:
                print(f"Error saving image to file: {e}", file=sys.stderr)
        else:
            print("Error: API response did not contain image data in a recognized format (URL or b64_json).", file=sys.stderr)
    else:
        print("Image generation failed.", file=sys.stderr)
