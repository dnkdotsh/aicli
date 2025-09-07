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

from . import api_client, config, workflows
from .chat_ui import MultiChatUI, SingleChatUI
from .engine import get_engine
from .managers.context_manager import ContextManager
from .managers.multichat_manager import MultiChatSession
from .managers.session_manager import SessionManager
from .prompts import MULTICHAT_SYSTEM_PROMPT_GEMINI, MULTICHAT_SYSTEM_PROMPT_OPENAI
from .session_state import MultiChatSessionState, SessionState
from .settings import settings
from .utils.config_loader import resolve_config_precedence
from .utils.file_processor import read_system_prompt
from .utils.formatters import RESET_COLOR, SYSTEM_MSG, format_bytes
from .utils.ui_helpers import select_model


def handle_chat(initial_prompt: str | None, args: argparse.Namespace) -> None:
    """Handles both single-shot and interactive chat sessions."""
    # --- Phase 4: Centralized Session Setup ---
    # 1. Resolve all configuration from CLI, persona, and settings
    config_params = resolve_config_precedence(args)
    p = config_params  # Alias for brevity

    # 2. Prepare engine and context
    api_key = api_client.check_api_keys(p["engine_name"])
    engine_instance = get_engine(p["engine_name"], api_key)

    persona_attachment_path_strs = set(p["persona"].attachments if p["persona"] else [])
    all_files_to_process = list(p["files_arg"] or [])
    all_files_to_process.extend(persona_attachment_path_strs)

    context_manager = ContextManager(
        files_arg=all_files_to_process,
        memory_enabled=p["memory_enabled"],
        exclude_arg=p["exclude_arg"],
    )

    # 3. Assemble the system prompt
    initial_system_prompt = None
    if p["system_prompt_arg"]:
        initial_system_prompt = read_system_prompt(p["system_prompt_arg"])
    elif p["persona"] and p["persona"].system_prompt:
        initial_system_prompt = p["persona"].system_prompt

    system_prompt_parts = []
    if initial_system_prompt:
        system_prompt_parts.append(initial_system_prompt)
    if context_manager.memory_content:
        system_prompt_parts.append(
            f"--- PERSISTENT MEMORY ---\n{context_manager.memory_content}"
        )
    final_system_prompt = (
        "\n\n".join(system_prompt_parts) if system_prompt_parts else None
    )

    persona_attachments_set = {
        path
        for path in context_manager.attachments
        if str(path) in persona_attachment_path_strs
    }

    # 4. Create the complete SessionState object
    state = SessionState(
        engine=engine_instance,
        model=p["model"],
        system_prompt=final_system_prompt,
        initial_system_prompt=initial_system_prompt,
        current_persona=p["persona"],
        max_tokens=p["max_tokens"],
        memory_enabled=p["memory_enabled"],
        attachments=context_manager.attachments,
        persona_attachments=persona_attachments_set,
        attached_images=context_manager.image_data,
        debug_active=p["debug_enabled"],
        stream_active=p["stream"],
    )

    # 5. Finally, instantiate the SessionManager with the pre-built state
    session = SessionManager(state=state, context_manager=context_manager)

    # --- End of Centralized Setup ---

    # Check for large context size and warn the user.
    total_attachment_bytes = sum(
        len(content.encode("utf-8")) for content in session.state.attachments.values()
    )

    if total_attachment_bytes > config.LARGE_ATTACHMENT_THRESHOLD_BYTES:
        warning_msg = (
            f"{SYSTEM_MSG}--> Warning: The total size of attached files "
            f"({format_bytes(total_attachment_bytes)}) exceeds the recommended "
            f"threshold ({format_bytes(config.LARGE_ATTACHMENT_THRESHOLD_BYTES)}).\n"
            f"    This may result in high API costs and slower responses.{RESET_COLOR}"
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
        chat_ui = SingleChatUI(session, p["session_name"])
        chat_ui.run()


def handle_load_session(filepath_str: str) -> None:
    """Loads and starts an interactive session from a file."""
    from .commands import handle_load

    raw_path = Path(filepath_str).expanduser()
    filepath = (
        raw_path if raw_path.is_absolute() else config.SESSIONS_DIRECTORY / raw_path
    )
    if filepath.suffix != ".json":
        filepath = filepath.with_suffix(".json")

    # The new SessionManager constructor requires pre-built state.
    # We create a temporary, minimal manager to pass to handle_load.
    # handle_load will then replace its `state` object entirely.
    temp_engine = get_engine(
        settings["default_engine"],
        api_client.check_api_keys(settings["default_engine"]),
    )
    temp_state = SessionState(
        engine=temp_engine,
        model=settings["default_gemini_model"],
        system_prompt=None,
        initial_system_prompt=None,
        current_persona=None,
        max_tokens=None,
        memory_enabled=False,
    )
    temp_context = ContextManager(
        files_arg=None, memory_enabled=False, exclude_arg=None
    )
    session = SessionManager(state=temp_state, context_manager=temp_context)

    if not handle_load([str(filepath)], session):
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

    context = ContextManager(
        files_arg=args.file, memory_enabled=False, exclude_arg=args.exclude
    )
    attachments = context.attachments
    image_data = context.image_data

    attachment_texts = [
        f"--- FILE: {path.as_posix()} ---\n{content}"
        for path, content in attachments.items()
    ]
    attachments_str = "\n\n".join(attachment_texts)

    base_system_prompt = ""
    if args.system_prompt:
        base_system_prompt = read_system_prompt(args.system_prompt)
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

    model = args.model or select_model(engine, "image")

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
