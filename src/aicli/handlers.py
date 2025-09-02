# aicli/handlers.py
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


import base64
import datetime
import json
import os
import sys

import requests

from . import api_client, commands, config, utils
from . import personas as persona_manager
from .engine import AIEngine, get_engine
from .logger import log
from .prompts import MULTICHAT_SYSTEM_PROMPT_GEMINI, MULTICHAT_SYSTEM_PROMPT_OPENAI
from .session_manager import (
    MultiChatSessionState,
    SessionState,
    perform_interactive_chat,
    perform_multichat_session,
)
from .settings import settings


def handle_chat(
    engine: AIEngine,
    model: str,
    system_prompt: str,
    initial_prompt: str,
    image_data: list,
    attachments: dict,
    session_name: str,
    max_tokens: int,
    stream: bool,
    memory_enabled: bool,
    debug_enabled: bool,
    initial_system_prompt: str | None,
    persona: persona_manager.Persona | None,
):
    """Handles both single-shot and interactive chat sessions."""
    if initial_prompt:
        # For single-shot chat, we must pre-assemble the full system prompt
        attachment_texts = []
        for path, content in attachments.items():
            attachment_texts.append(f"--- FILE: {path.as_posix()} ---\n{content}")
        attachments_str = "\n\n".join(attachment_texts)

        full_system_prompt = system_prompt
        if attachments_str:
            full_system_prompt = (
                full_system_prompt or ""
            ) + f"\n\n--- ATTACHED FILES ---\n{attachments_str}"

        # Handle single-shot chat
        messages_or_contents = [
            utils.construct_user_message(engine.name, initial_prompt, image_data)
        ]
        if stream:
            print(
                f"{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}",
                end="",
                flush=True,
            )
        response, token_dict = api_client.perform_chat_request(
            engine, model, messages_or_contents, full_system_prompt, max_tokens, stream
        )
        if not stream:
            print(
                f"{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{response}",
                end="",
            )
        print(utils.format_token_string(token_dict))
    else:
        # Delegate to the session manager for interactive chat
        initial_state = SessionState(
            engine=engine,
            model=model,
            system_prompt=system_prompt,
            initial_system_prompt=initial_system_prompt,
            current_persona=persona,
            attached_images=image_data,
            attachments=attachments,
            stream_active=stream,
            memory_enabled=memory_enabled,
            debug_active=debug_enabled,
            max_tokens=max_tokens,
        )
        perform_interactive_chat(initial_state, session_name)


def handle_load_session(filepath_str: str):
    """Loads and starts an interactive session from a file."""
    from pathlib import Path

    raw_path = Path(filepath_str).expanduser()

    # If the path is not absolute, assume it is in the sessions directory
    if raw_path.is_absolute():
        filepath = raw_path
    else:
        filepath = config.SESSIONS_DIRECTORY / raw_path

    # Ensure the .json extension is present
    if filepath.suffix != ".json":
        filepath = filepath.with_suffix(".json")

    try:
        initial_state = commands.load_session_from_file(filepath)
    except api_client.MissingApiKeyError as e:
        log.error(e)
        sys.exit(1)

    if not initial_state:
        # This branch handles other load failures (e.g., file not found, JSON error)
        # that are logged by load_session_from_file itself.
        sys.exit(1)

    # Use the loaded file's name as the base for the new session log
    session_name = filepath.stem
    perform_interactive_chat(initial_state, session_name)


def handle_multichat_session(
    initial_prompt: str | None,
    system_prompt: str,
    image_data: list,
    session_name: str,
    max_tokens: int,
    debug_enabled: bool,
):
    """Sets up and delegates an interactive session with both OpenAI and Gemini."""
    try:
        openai_key = api_client.check_api_keys("openai")
        gemini_key = api_client.check_api_keys("gemini")
    except api_client.MissingApiKeyError as e:
        log.error(e)
        sys.exit(1)

    final_sys_prompts = {}
    if system_prompt:
        final_sys_prompts["openai"] = (
            f"{system_prompt}\n\n---\n\n{MULTICHAT_SYSTEM_PROMPT_OPENAI}"
        )
        final_sys_prompts["gemini"] = (
            f"{system_prompt}\n\n---\n\n{MULTICHAT_SYSTEM_PROMPT_GEMINI}"
        )
    else:
        final_sys_prompts["openai"] = MULTICHAT_SYSTEM_PROMPT_OPENAI
        final_sys_prompts["gemini"] = MULTICHAT_SYSTEM_PROMPT_GEMINI

    initial_state = MultiChatSessionState(
        openai_engine=get_engine("openai", openai_key),
        gemini_engine=get_engine("gemini", gemini_key),
        openai_model=settings["default_openai_chat_model"],
        gemini_model=settings["default_gemini_model"],
        max_tokens=max_tokens,
        system_prompts=final_sys_prompts,
        initial_image_data=image_data,
        debug_active=debug_enabled,
    )

    if initial_prompt:
        # For a non-interactive initial prompt, we can reuse the session loop logic
        # by simply pre-populating the first turn.
        perform_multichat_session(initial_state, session_name)
    else:
        # For interactive mode, just start the loop.
        perform_multichat_session(initial_state, session_name)


def handle_image_generation(api_key: str, engine: AIEngine, prompt: str):
    """Handles OpenAI image generation."""
    model = utils.select_model(engine, "image")
    if not prompt:
        prompt = (
            sys.stdin.read().strip()
            if not sys.stdin.isatty()
            else input("Enter a description for the image: ")
        )

    if not prompt:
        print("Image generation cancelled: No prompt provided.", file=sys.stderr)
        return

    print(f"Generating image with {model} for prompt: '{prompt}'...")
    payload = {"model": model, "prompt": prompt, "n": 1, "size": "1024x1024"}
    if model.startswith("dall-e"):
        payload["response_format"] = "b64_json"

    url, headers = (
        "https://api.openai.com/v1/images/generations",
        {"Authorization": f"Bearer {api_key}"},
    )
    response_data = api_client.make_api_request(url, headers, payload)

    if response_data and "data" in response_data:
        image_url = response_data["data"][0].get("url")
        b64_data = response_data["data"][0].get("b64_json")
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
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                print(f"Image saved successfully as: {filepath}")
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": model,
                    "prompt": prompt,
                    "file": filepath,
                }
                with open(config.IMAGE_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except OSError as e:
                print(f"Error saving image to file: {e}", file=sys.stderr)
    else:
        print(
            "Error: API response did not contain image data in a recognized format.",
            file=sys.stderr,
        )
