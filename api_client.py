# aicli/api_client.py

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
import requests
import datetime
import re
import copy
import config
import utils
from engine import AIEngine

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
    log_entry = { "timestamp": datetime.datetime.now().isoformat(), "request": {"url": url, "headers": headers, "payload": payload} }
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

def perform_chat_request(engine: AIEngine, model: str, messages_or_contents: list, system_prompt: str | None, max_tokens: int, stream: bool, session_raw_logs: list | None = None) -> tuple[str, dict]:
    """Executes a single chat request and returns the response text and token dictionary."""
    url = engine.get_chat_url(model, stream)
    payload = engine.build_chat_payload(messages_or_contents, system_prompt, max_tokens, stream, model)
    headers = {"Authorization": f"Bearer {engine.api_key}"} if engine.name == 'openai' else {"Content-Type": "application/json"}

    response_obj = make_api_request(url, headers, payload, stream, session_raw_logs)

    if not response_obj:
        return "", {}

    if stream:
        return utils.process_stream(engine.name, response_obj)

    response_data = response_obj
    assistant_response = engine.parse_chat_response(response_data)
    p, c, r, t = utils.parse_token_counts(engine.name, response_data)
    tokens = {'prompt': p, 'completion': c, 'reasoning': r, 'total': t}
    return assistant_response, tokens
