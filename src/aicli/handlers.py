# aicli/handlers.py
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import sys
from pathlib import Path

from . import api_client, config, utils, workflows
from .chat_ui import MultiChatUI, SingleChatUI
from .engine import get_engine
from .prompts import MULTICHAT_SYSTEM_PROMPT_GEMINI, MULTICHAT_SYSTEM_PROMPT_OPENAI
from .session_manager import MultiChatSession, SessionManager
from .session_state import MultiChatSessionState
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

    # Check for large context size and warn the user.
    total_attachment_bytes = sum(
        len(content.encode("utf-8")) for content in session.state.attachments.values()
    )

    if total_attachment_bytes > config.LARGE_ATTACHMENT_THRESHOLD_BYTES:
        warning_msg = (
            f"{utils.SYSTEM_MSG}--> Warning: The total size of attached files "
            f"({utils.format_bytes(total_attachment_bytes)}) exceeds the recommended "
            f"threshold ({utils.format_bytes(config.LARGE_ATTACHMENT_THRESHOLD_BYTES)}).\n"
            f"    This may result in high API costs and slower responses.{utils.RESET_COLOR}"
        )
        print(warning_msg, file=sys.stderr)

        if not initial_prompt:  # Only ask for confirmation in interactive mode
            try:
                confirm = (
                    input("    Proceed with this context? (y/N): ").lower().strip()
                )
                if confirm not in ["y", "yes"]:
                    print("Operation cancelled by user.")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user.")
                sys.exit(0)

    if initial_prompt:
        # Single-shot mode
        session.handle_single_shot(initial_prompt)
    else:
        # Interactive mode
        chat_ui = SingleChatUI(session, config_params["session_name"])
        chat_ui.run()


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
    chat_ui = SingleChatUI(session, session_name)
    chat_ui.run()


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

    session = MultiChatSession(initial_state)
    multi_chat_ui = MultiChatUI(session, args.session_name, initial_prompt)
    multi_chat_ui.run()


def handle_image_generation(prompt: str | None, args: argparse.Namespace) -> None:
    """
    Handles standalone OpenAI image generation (via the -i flag).
    This function is a simple wrapper that processes command-line arguments
    and then calls the centralized workflow.
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
    success, _ = workflows._perform_image_generation(
        api_key, model, prompt, session_raw_logs=srl_list
    )

    if not success:
        sys.exit(1)
