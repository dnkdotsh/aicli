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

import argparse
import base64
import datetime
import json
import sys
from pathlib import Path

from . import api_client, config, utils
from .engine import get_engine
from .prompts import MULTICHAT_SYSTEM_PROMPT_GEMINI, MULTICHAT_SYSTEM_PROMPT_OPENAI
from .session_manager import (
    MultiChatManager,
    MultiChatSessionState,
    SessionManager,
    SingleChatManager,
)
from .settings import settings


def handle_chat(initial_prompt: str | None, args: argparse.Namespace) -> None:
    """Handles both single-shot and interactive chat sessions."""
    config_params = utils.resolve_config_precedence(args)

    session = SessionManager(
        engine_name=config_params["engine_name"],
        model=config_params["model"],
        max_tokens=config_params["max_tokens"],
        stream_active=config_params["stream"],
        memory_enabled=config_params["memory_enabled"],
        debug_active=config_params["debug_enabled"],
        persona=config_params["persona"],
        system_prompt_arg=config_params["system_prompt_arg"],
        files_arg=config_params["files_arg"],
        exclude_arg=config_params["exclude_arg"],
    )

    if initial_prompt:
        # Single-shot mode
        session.handle_single_shot(initial_prompt)
    else:
        # Interactive mode
        chat_manager = SingleChatManager(session, config_params["session_name"])
        chat_manager.run()


def handle_load_session(filepath_str: str) -> None:
    """Loads and starts an interactive session from a file."""
    raw_path = Path(filepath_str).expanduser()
    filepath = (
        raw_path if raw_path.is_absolute() else config.SESSIONS_DIRECTORY / raw_path
    )
    if filepath.suffix != ".json":
        filepath = filepath.with_suffix(".json")

    # Create an empty session manager, then load state into it
    session = SessionManager(engine_name=settings["default_engine"])
    if not session.load(str(filepath)):
        sys.exit(1)

    # Use the loaded file's name as the base for the new session log
    session_name = filepath.stem
    chat_manager = SingleChatManager(session, session_name)
    chat_manager.run()


def handle_multichat_session(
    initial_prompt: str | None, args: argparse.Namespace
) -> None:
    """Sets up and delegates an interactive session with both OpenAI and Gemini."""
    openai_key = api_client.check_api_keys("openai")
    gemini_key = api_client.check_api_keys("gemini")

    _, attachments, image_data = utils.process_files(
        args.file, use_memory=False, exclusions=args.exclude
    )
    attachment_texts = [
        f"--- FILE: {path.as_posix()} ---\n{content}"
        for path, content in attachments.items()
    ]
    attachments_str = "\n\n".join(attachment_texts)

    base_system_prompt = ""
    if args.system_prompt:
        base_system_prompt = utils.read_system_prompt(args.system_prompt)
    if attachments_str:
        base_system_prompt += f"\n\n--- ATTACHED FILES ---\n{attachments_str}"

    final_sys_prompts: dict[str, str] = {}
    if base_system_prompt:
        final_sys_prompts["openai"] = (
            f"{base_system_prompt}\n\n---\n\n{MULTICHAT_SYSTEM_PROMPT_OPENAI}"
        )
        final_sys_prompts["gemini"] = (
            f"{base_system_prompt}\n\n---\n\n{MULTICHAT_SYSTEM_PROMPT_GEMINI}"
        )
    else:
        final_sys_prompts["openai"] = MULTICHAT_SYSTEM_PROMPT_OPENAI
        final_sys_prompts["gemini"] = MULTICHAT_SYSTEM_PROMPT_GEMINI

    initial_state = MultiChatSessionState(
        openai_engine=get_engine("openai", openai_key),
        gemini_engine=get_engine("gemini", gemini_key),
        openai_model=args.model or settings["default_openai_chat_model"],
        gemini_model=args.model or settings["default_gemini_model"],
        max_tokens=args.max_tokens or settings["default_max_tokens"],
        system_prompts=final_sys_prompts,
        initial_image_data=image_data,
        debug_active=args.debug,
    )

    multi_chat_manager = MultiChatManager(
        initial_state, args.session_name, initial_prompt
    )
    multi_chat_manager.run()


def handle_image_generation(prompt: str | None, args: argparse.Namespace) -> None:
    """Handles OpenAI image generation."""
    api_key = api_client.check_api_keys("openai")
    engine = get_engine("openai", api_key)

    model = args.model or utils.select_model(engine, "image")
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
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    url, headers = (
        "https://api.openai.com/v1/images/generations",
        {"Authorization": f"Bearer {api_key}"},
    )

    try:
        response_data = api_client.make_api_request(url, headers, payload)
    except api_client.ApiRequestError as e:
        print(f"Error: {e}", file=sys.stderr)
        return

    if (
        response_data
        and "data" in response_data
        and response_data["data"][0].get("b64_json")
    ):
        b64_data = response_data["data"][0]["b64_json"]
        try:
            image_bytes = base64.b64decode(b64_data)
            safe_prompt = utils.sanitize_filename(prompt[:50])
            base_filename = f"image_{safe_prompt}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = config.IMAGE_DIRECTORY / base_filename
            filepath.write_bytes(image_bytes)
            print(f"Image saved successfully as: {filepath}")
            # Log the successful generation
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": model,
                "prompt": prompt,
                "file": str(filepath),
            }
            with open(config.IMAGE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except (base64.binascii.Error, OSError) as e:
            print(f"Error saving image: {e}", file=sys.stderr)
    else:
        print("Error: API response did not contain image data.", file=sys.stderr)
