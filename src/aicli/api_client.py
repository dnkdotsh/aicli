# aicli/api_client.py
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


import copy
import datetime
import json
import os
import re
import sys

import requests

from . import config, utils
from .engine import AIEngine
from .logger import log
from .settings import settings


class MissingApiKeyError(Exception):
    """Custom exception for missing API keys."""

    pass


class ApiRequestError(Exception):
    """Custom exception for failed API requests."""

    pass


def check_api_keys(engine: str):
    """Checks for the required API key in environment variables and returns it."""
    key_name = "OPENAI_API_KEY" if engine == "openai" else "GEMINI_API_KEY"
    api_key = os.getenv(key_name)
    if not api_key:
        raise MissingApiKeyError(
            f"Error: Environment variable '{key_name}' is not set."
        )
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


def make_api_request(
    url: str,
    headers: dict,
    payload: dict,
    stream: bool = False,
    session_raw_logs: list | None = None,
) -> dict | requests.Response:
    """
    Makes a POST request to the specified API endpoint and handles errors.
    Returns a dictionary for non-streaming responses or a requests.Response object for streaming.
    Raises ApiRequestError on failure.
    """
    response_data = None
    error_details_for_log = None
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "request": {"url": url, "headers": headers, "payload": payload},
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=settings["api_timeout"],
        )
        response.raise_for_status()
        if stream:
            log_entry["response"] = {
                "status_code": response.status_code,
                "streaming": True,
            }
            return response
        response_data = response.json()
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            log.error("API Error: %s", error_msg)
            raise ApiRequestError(error_msg)
        return response_data
    except requests.exceptions.HTTPError as e:
        error_details = "No specific error message provided by the API."
        try:
            error_json = e.response.json()
            if (
                isinstance(error_json, dict)
                and "error" in error_json
                and "message" in error_json.get("error", {})
            ):
                error_details = error_json["error"]["message"]
            else:
                error_details = e.response.text
        except requests.exceptions.JSONDecodeError:
            error_details = e.response.text
        log.error("HTTP Request Error: %s\nDETAILS: %s", e, error_details)
        error_details_for_log = error_details
        raise ApiRequestError(error_details) from e
    except requests.exceptions.JSONDecodeError as e:
        log.error("Failed to decode API response.")
        error_details_for_log = "Failed to decode API response."
        raise ApiRequestError("Failed to decode API response.") from e
    except requests.exceptions.RequestException as e:
        log.error("Request Error: %s", e)
        error_details_for_log = str(e)
        raise ApiRequestError(str(e)) from e
    finally:
        if not stream:
            if error_details_for_log:
                log_entry["response"] = {"error": error_details_for_log}
            else:
                log_entry["response"] = response_data or {
                    "error": "Request failed, see logs for details."
                }
        safe_log_entry = _redact_sensitive_info(log_entry)
        if session_raw_logs is not None:
            session_raw_logs.append(safe_log_entry)
        try:
            with open(config.RAW_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_log_entry) + "\n")
        except OSError as e:
            log.warning("Could not write to raw log file: %s", e)


def _parse_token_counts(
    engine_name: str, response_data: dict
) -> tuple[int, int, int, int]:
    """Parses token counts from a non-streaming API response."""
    p, c, r, t = 0, 0, 0, 0
    if not response_data:
        return 0, 0, 0, 0
    if engine_name == "openai":
        if "usage" in response_data:
            p = response_data["usage"].get("prompt_tokens", 0)
            c = response_data["usage"].get("completion_tokens", 0)
            t = response_data["usage"].get("total_tokens", 0)
    elif engine_name == "gemini":
        usage = response_data.get("usageMetadata", {})
        p = usage.get("promptTokenCount", 0)
        c = usage.get("candidatesTokenCount", 0)
        r = usage.get("cachedContentTokenCount", 0)
        t = usage.get("totalTokenCount", 0)
    return p, c, r, t


def _process_stream(
    engine: str, response: requests.Response, print_stream: bool = True
) -> tuple[str, dict]:
    """Processes a streaming API response."""
    full_response, p, c, r, t = "", 0, 0, 0, 0
    try:
        for chunk in response.iter_lines():
            if not chunk:
                continue
            decoded_chunk = chunk.decode("utf-8")
            if engine == "openai":
                if decoded_chunk.startswith("data:"):
                    if "[DONE]" in decoded_chunk:
                        break
                    try:
                        data = json.loads(decoded_chunk.split("data: ", 1)[1])
                        if (
                            "choices" in data
                            and data["choices"]
                            and data["choices"][0].get("delta", {}).get("content")
                        ):
                            text_chunk = data["choices"][0]["delta"]["content"]
                            if print_stream:
                                sys.stdout.write(text_chunk)
                                sys.stdout.flush()
                            full_response += text_chunk
                        if "usage" in data and data["usage"]:
                            p = data["usage"].get("prompt_tokens", 0)
                            c = data["usage"].get("completion_tokens", 0)
                            t = data["usage"].get("total_tokens", 0)
                    except json.JSONDecodeError:
                        continue
            elif engine == "gemini":
                try:
                    data = json.loads(decoded_chunk.split("data: ", 1)[1])
                    if "candidates" in data:
                        text_chunk = (
                            data["candidates"][0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        if print_stream:
                            sys.stdout.write(text_chunk)
                            sys.stdout.flush()
                        full_response += text_chunk
                    if "usageMetadata" in data:
                        p = data["usageMetadata"].get("promptTokenCount", 0)
                        c = data["usageMetadata"].get("candidatesTokenCount", 0)
                        r = data["usageMetadata"].get("cachedContentTokenCount", 0)
                        t = data["usageMetadata"].get("totalTokenCount", 0)
                except (json.JSONDecodeError, IndexError):
                    continue
    except KeyboardInterrupt:
        if print_stream:
            # A newline is needed to move the cursor to the next line after the partial response.
            print(
                f"\n{utils.SYSTEM_MSG}--> Stream interrupted by user.{utils.RESET_COLOR}"
            )
    except Exception as e:
        if print_stream:
            print(
                f"\n{utils.SYSTEM_MSG}--> Stream interrupted by network/API error: {e}{utils.RESET_COLOR}"
            )
        log.warning("Stream processing error: %s", e)
    tokens = {"prompt": p, "completion": c, "reasoning": r, "total": t}
    return full_response, tokens


def perform_chat_request(
    engine: AIEngine,
    model: str,
    messages_or_contents: list,
    system_prompt: str | None,
    max_tokens: int,
    stream: bool,
    session_raw_logs: list | None = None,
    print_stream: bool = True,
) -> tuple[str, dict]:
    """Executes a single chat request and returns the response text and token dictionary."""
    url = engine.get_chat_url(model, stream)
    payload = engine.build_chat_payload(
        messages_or_contents, system_prompt, max_tokens, stream, model
    )
    headers = (
        {"Authorization": f"Bearer {engine.api_key}"}
        if engine.name == "openai"
        else {"Content-Type": "application/json"}
    )

    try:
        response_obj = make_api_request(url, headers, payload, stream, session_raw_logs)
    except ApiRequestError as e:
        # For non-streaming, return the error in the response tuple
        if not stream:
            return f"API Error: {e}", {}
        # For streaming, the error is already logged, so we just return empty
        return "", {}

    if stream:
        return _process_stream(engine.name, response_obj, print_stream=print_stream)

    response_data = response_obj
    assistant_response = engine.parse_chat_response(response_data)
    p, c, r, t = _parse_token_counts(engine.name, response_data)
    tokens = {"prompt": p, "completion": c, "reasoning": r, "total": t}
    return assistant_response, tokens
