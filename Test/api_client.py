# aicli/api_client.py

import os
import sys
import json
import requests
import datetime
import re
import copy
import config

class MissingApiKeyError(Exception):
    """Custom exception for missing API keys."""
    pass

def check_api_keys(engine: str):
    """Checks for the required API key in environment variables and returns it."""
    key_name = 'OPENAI_API_KEY' if engine == 'openai' else 'GEMINI_API_KEY'
    api_key = os.getenv(key_name)
    if not api_key:
        raise MissingApiKeyError(f"Error: Environment variable '{key_name}' is not set.")
    return api_key

def _redact_sensitive_info(log_entry: dict) -> dict:
    """Redacts sensitive information (API keys) from a log entry in place."""
    safe_log_entry = copy.deepcopy(log_entry)
    request = safe_log_entry.get("request", {})
    if "url" in request and "key=" in request["url"]:
        request["url"] = re.sub(r"key=([^&]+)", "key=[REDACTED]", request["url"])
    if "headers" in request and "Authorization" in request["headers"]:
        request["headers"]["Authorization"] = "Bearer [REDACTED]"
    return safe_log_entry

def make_api_request(url: str, headers: dict, payload: dict, stream: bool = False, session_raw_logs: list | None = None) -> dict | requests.Response:
    """
    Makes a POST request to the specified API endpoint and handles errors.
    Returns a dictionary for non-streaming responses or a requests.Response object for streaming.
    """
    response_data = None
    error_info = None
    
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "request": {"url": url, "headers": headers, "payload": payload},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            log_entry["response"] = {"status_code": response.status_code, "streaming": True}
            return response

        response_data = response.json()
        if 'error' in response_data:
            print(f"API Error: {response_data['error'].get('message', 'Unknown error')}", file=sys.stderr)
            error_info = response_data['error']
            return None
        return response_data
    except requests.exceptions.HTTPError as e:
        error_details = "No specific error message provided by the API."
        # --- FIX: Robust error message parsing ---
        # Rationale: The previous code assumed a fixed JSON structure for errors, causing a
        # TypeError when the Gemini API returned a different format. This new block
        # gracefully checks for different possible error structures before falling back to raw text.
        try:
            error_json = e.response.json()
            if isinstance(error_json, dict) and 'error' in error_json and 'message' in error_json.get('error', {}):
                error_details = error_json['error']['message']
            else:
                error_details = e.response.text
        except json.JSONDecodeError:
            error_details = e.response.text
        print(f"HTTP Request Error: {e}\nDETAILS: {error_details}", file=sys.stderr)
        error_info = {"code": e.response.status_code, "message": error_details}
        return None
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error: {e}", file=sys.stderr)
        error_info = {"message": str(e)}
        return None
    except json.JSONDecodeError:
        print("Error: Failed to decode API response.", file=sys.stderr)
        error_info = {"message": "Failed to decode API response."}
        return None
    finally:
        if not stream:
             log_entry["response"] = response_data or {"error": error_info}
        
        safe_log_entry = _redact_sensitive_info(log_entry)

        if session_raw_logs is not None:
            session_raw_logs.append(safe_log_entry)
        
        try:
            with open(config.RAW_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(safe_log_entry) + '\n')
        except IOError as e:
            print(f"\nWarning: Could not write to raw log file: {e}", file=sys.stderr)

def fetch_available_models(api_key: str, engine: str, task: str) -> list[str]:
    """Fetches a list of available models from the API."""
    try:
        if engine == 'openai':
            url, headers = "https://api.openai.com/v1/models", {"Authorization": f"Bearer {api_key}"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            model_list = response.json().get('data', [])
            model_ids = [m['id'] for m in model_list]

            if task == 'chat':
                chat_models = [mid for mid in model_ids if mid.startswith('gpt')]
                return sorted(chat_models)
            
            elif task == 'image':
                image_models = [mid for mid in model_ids if mid.startswith('dall-e') or 'image' in mid]
                return sorted(image_models)

        elif engine == 'gemini':
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            model_list = response.json().get('models', [])
            return sorted([m['name'].replace('models/', '') for m in model_list if 'generateContent' in m.get('supportedGenerationMethods', [])])
            
    except requests.exceptions.RequestException as e:
        print(f"\nWarning: Could not fetch model list ({e}).", file=sys.stderr)
        return []
    return []
