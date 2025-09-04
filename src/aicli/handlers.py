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
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from .session_manager import SessionManager


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


def _perform_image_generation(
    api_key: str,
    model: str,
    prompt: str,
    session_name: str | None = None,
    session_raw_logs: list | None = None,
) -> tuple[bool, str | None]:
    """
    Core image generation logic, serving as the single source of truth.

    This function encapsulates all API interaction for image generation. It is
    called by both standalone mode (`handle_image_generation`) and interactive
    session mode (`generate_image_from_session`). This centralization prevents
    code duplication and ensures consistent behavior.

    Args:
        api_key: The OpenAI API key for authentication.
        model: The specific model to use for generation (e.g., "dall-e-3").
        prompt: The text description of the image to generate.
        session_name: An optional session identifier for organized file naming.
        session_raw_logs: A list to append raw API logs to if debug is active.

    Returns:
        A tuple of (success: bool, filepath: str | None). The filepath is None
        if generation failed, allowing callers to handle failures gracefully.
    """
    print(
        f"Generating image with {model} for prompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'..."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response_data = api_client.make_api_request(
            url, headers, payload, session_raw_logs=session_raw_logs
        )
    except api_client.ApiRequestError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False, None

    if not (
        response_data
        and "data" in response_data
        and response_data["data"]
        and response_data["data"][0].get("b64_json")
    ):
        print("Error: API response did not contain image data.", file=sys.stderr)
        return False, None

    b64_data = response_data["data"][0]["b64_json"]
    try:
        image_bytes = base64.b64decode(b64_data)
        filepath = utils.save_image_and_get_path(prompt, image_bytes, session_name)
        print(f"Image saved successfully as: {filepath}")
        utils.log_image_generation(model, prompt, str(filepath), session_name)
        return True, str(filepath)
    except (base64.binascii.Error, OSError) as e:
        print(f"Error saving image: {e}", file=sys.stderr)
        return False, None


def generate_image_from_session(session: "SessionManager", prompt: str) -> bool:
    """
    Generates an image using session context and settings.

    This function acts as a bridge between the SessionManager's state and the
    core generation logic. It handles engine compatibility checks, model
    selection, and updating the conversation history post-generation.

    Args:
        session: The active SessionManager instance.
        prompt: The final, confirmed prompt for image generation.

    Returns:
        A boolean indicating whether the generation was successful.
    """
    if session.state.engine.name != "openai":
        print(
            f"{utils.SYSTEM_MSG}--> Switching to OpenAI for image generation...{utils.RESET_COLOR}"
        )
        original_engine = session.state.engine.name
        session.switch_engine("openai")
    else:
        original_engine = None

    current_model = session.state.model
    if current_model and "dall-e" in current_model.lower():
        model_to_use = current_model
    else:
        model_to_use = settings["default_openai_image_model"]
        print(
            f"{utils.SYSTEM_MSG}--> Using default image model: {model_to_use}{utils.RESET_COLOR}"
        )

    srl_list = session.state.session_raw_logs if session.state.debug_active else None
    session_name = getattr(session, "session_name", None)

    success, filepath = _perform_image_generation(
        session.state.engine.api_key,
        model_to_use,
        prompt,
        session_name,
        srl_list,
    )

    if success and filepath:
        user_msg = utils.construct_user_message(
            session.state.engine.name, f"Generate image: {prompt}", []
        )
        asst_msg_text = (
            f"I've generated an image and saved it to:\n{filepath}\n\n"
            f"Prompt: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"\n\n"
            "Note: Generated images are not kept in the conversation context to manage token usage."
        )
        asst_msg = utils.construct_assistant_message(
            session.state.engine.name, asst_msg_text
        )
        session.state.history.extend([user_msg, asst_msg])

        if original_engine and original_engine != "openai":
            print(
                f"\n{utils.SYSTEM_MSG}--> Switch back to {original_engine.capitalize()}? "
                f"(Type '/engine {original_engine}'){utils.RESET_COLOR}"
            )
        return True

    return False


def handle_image_generation(prompt: str | None, args: argparse.Namespace) -> None:
    """
    Handles standalone OpenAI image generation (via the -i flag).

    This function is a simple wrapper that processes command-line arguments
    and then calls the centralized `_perform_image_generation` function.
    """
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

    srl_list = [] if args.debug else None
    success, _ = _perform_image_generation(
        api_key, model, prompt, session_raw_logs=srl_list
    )

    if not success:
        sys.exit(1)
