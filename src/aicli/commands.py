# src/aicli/commands.py
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

"""
This module contains the implementation for all interactive slash commands.
These commands are thin wrappers that delegate state manipulation to a
SessionManager instance.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from . import config, utils
from .logger import log
from .settings import save_setting

if TYPE_CHECKING:
    from .session_manager import MultiChatSessionState, SessionManager


# --- Single-Chat Command Handler Functions ---


def handle_exit(args: list[str], session: SessionManager) -> bool:
    if args:
        session.set_custom_log_rename(" ".join(args))
    return True


def handle_quit(args: list[str], session: SessionManager) -> bool:
    session.set_force_quit()
    return True


def handle_help(args: list[str], session: SessionManager) -> None:
    utils.display_help("chat")


def handle_stream(args: list[str], session: SessionManager) -> None:
    session.toggle_stream()


def handle_debug(args: list[str], session: SessionManager) -> None:
    session.toggle_debug()


def handle_memory(args: list[str], session: SessionManager) -> None:
    session.view_memory()


def handle_remember(args: list[str], session: SessionManager) -> None:
    if not args:
        session.consolidate_memory()
    else:
        session.inject_memory(" ".join(args))


def handle_max_tokens(args: list[str], session: SessionManager) -> None:
    if args and args[0].isdigit():
        session.set_max_tokens(int(args[0]))
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")


def handle_clear(args: list[str], session: SessionManager) -> None:
    confirm = prompt(
        "This will clear all conversation history. Type `proceed` to confirm: "
    )
    if confirm.lower() == "proceed":
        session.clear_history()
    else:
        print(f"{utils.SYSTEM_MSG}--> Clear cancelled.{utils.RESET_COLOR}")


def handle_model(args: list[str], session: SessionManager) -> None:
    if args:
        session.set_model(args[0])
    else:
        new_model = utils.select_model(session.state.engine, "chat")
        session.set_model(new_model)


def handle_engine(args: list[str], session: SessionManager) -> None:
    new_engine_name = args[0] if args else None
    session.switch_engine(new_engine_name)


def handle_history(args: list[str], session: SessionManager) -> None:
    session.view_history()


def handle_state(args: list[str], session: SessionManager) -> None:
    session.view_state()


def handle_set(args: list[str], session: SessionManager) -> None:
    if len(args) == 2:
        _success, message = save_setting(args[0], args[1])
        print(f"{utils.SYSTEM_MSG}--> {message}{utils.RESET_COLOR}")
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /set <key> <value>.{utils.RESET_COLOR}")


def handle_save(
    args: list[str], session: SessionManager, cli_history: InMemoryHistory
) -> bool:
    return session.save(args, cli_history.get_strings())


def handle_load(args: list[str], session: SessionManager) -> None:
    if args:
        session.load(" ".join(args))
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /load <filename>{utils.RESET_COLOR}")


def handle_refresh(args: list[str], session: SessionManager) -> None:
    session.refresh_files(" ".join(args) if args else None)


def handle_files(args: list[str], session: SessionManager) -> None:
    session.list_files()


def handle_attach(args: list[str], session: SessionManager) -> None:
    if not args:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /attach <path_to_file_or_dir>{utils.RESET_COLOR}"
        )
        return
    session.attach_file(" ".join(args))


def handle_detach(args: list[str], session: SessionManager) -> None:
    if not args:
        print(f"{utils.SYSTEM_MSG}--> Usage: /detach <filename>{utils.RESET_COLOR}")
        return
    session.detach_file(" ".join(args))


def handle_personas(args: list[str], session: SessionManager) -> None:
    session.list_personas()


def handle_persona(args: list[str], session: SessionManager) -> None:
    if not args:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /persona <name> OR /persona clear{utils.RESET_COLOR}"
        )
        return
    session.switch_persona(" ".join(args))


def handle_image(args: list[str], session: SessionManager) -> None:
    """
    Manages image generation workflows with intelligent state-based routing.

    This command acts as a sophisticated dispatcher that examines the current
    session state and user intent to determine the appropriate image generation
    workflow. It's designed to feel intuitive - users don't need to remember
    multiple commands, just /image and optional arguments.

    Think of this command as having multiple "personalities" that activate based
    on context. Sometimes it's a creative assistant helping you craft the perfect
    prompt. Sometimes it's a quick executor that generates immediately. And
    sometimes it's a helpful guide offering to refine previous work.

    Supported workflows:
    1. Fresh start: /image (with no existing prompt)
    2. Initial prompt: /image a beautiful sunset over mountains
    3. Previous prompt refinement: /image (when last_img_prompt exists)
    4. Immediate generation: /image --send-prompt

    The command maintains a careful balance between being helpful and not being
    intrusive. It provides clear feedback about what's happening and what options
    are available, making the image generation process feel conversational rather
    than transactional.
    """

    # First, let's check for the special immediate generation flag
    # This flag takes priority over all other logic because it represents
    # an explicit user intent to generate NOW. It's like pressing the
    # "express checkout" button - skip all the browsing and just complete the purchase
    send_immediately = "--send-prompt" in args

    # Remove the flag from args so we can process the rest normally
    # This cleanup is important because the remaining args might contain the prompt text
    # We're essentially separating the "how" (the flag) from the "what" (the prompt)
    if send_immediately:
        args = [arg for arg in args if arg != "--send-prompt"]

    # Now we need to understand what prompt material we're working with
    # There are three possible sources, like three different drafts on your desk:
    # 1. New text the user just typed (args)
    # 2. A prompt we've been crafting together (current_prompt)
    # 3. Something we generated before (previous_prompt)
    prompt_from_args = " ".join(args) if args else None
    current_prompt = session.state.img_prompt
    previous_prompt = session.state.last_img_prompt

    # Check if we're already in crafting mode
    # This is a safety check - normally users wouldn't use /image while already crafting,
    # but we handle it gracefully. It's like someone asking to start a conversation
    # when you're already talking to them
    if session.state.img_prompt_crafting:
        print(
            f"{utils.SYSTEM_MSG}--> You're already in image crafting mode. "
            f"Continue refining your prompt or type 'yes' to generate.{utils.RESET_COLOR}"
        )
        return

    # Now let's handle the immediate generation workflow
    # This is for users who know exactly what they want and don't need our help
    # It's the "I've got this" path through the command
    if send_immediately:
        # Determine which prompt to use for immediate generation
        # We have a clear priority order here, like choosing which draft to submit:
        # First choice: new prompt from args (freshest input)
        # Second choice: current crafted prompt (what we've been working on)
        # Third choice: previous prompt (what worked before)
        prompt_to_generate = None

        if prompt_from_args:
            # User provided a fresh prompt with the command
            prompt_to_generate = prompt_from_args
            print(
                f"{utils.SYSTEM_MSG}--> Generating image immediately with: "
                f"{prompt_from_args[:50]}{'...' if len(prompt_from_args) > 50 else ''}"
                f"{utils.RESET_COLOR}"
            )
        elif current_prompt:
            # We have a prompt we've been crafting
            prompt_to_generate = current_prompt
            print(
                f"{utils.SYSTEM_MSG}--> Generating with current prompt: "
                f"{current_prompt[:50]}{'...' if len(current_prompt) > 50 else ''}"
                f"{utils.RESET_COLOR}"
            )
        elif previous_prompt:
            # Fall back to the last successful prompt
            prompt_to_generate = previous_prompt
            print(
                f"{utils.SYSTEM_MSG}--> Regenerating previous image: "
                f"{previous_prompt[:50]}{'...' if len(previous_prompt) > 50 else ''}"
                f"{utils.RESET_COLOR}"
            )
        else:
            # No prompt available anywhere - we need something to work with!
            print(
                f"{utils.SYSTEM_MSG}--> No prompt available for generation. "
                f"Use '/image <description>' to start crafting.{utils.RESET_COLOR}"
            )
            return

        # Perform the actual generation using the session's method
        # This delegates to the SessionManager which knows how to coordinate
        # with the handlers module for the actual API calls
        session.generate_image(prompt_to_generate)
        return

    # Now let's handle the interactive crafting workflows
    # These are for users who want help refining their prompts
    # Think of this section as the "let's work together" paths

    if prompt_from_args:
        # Workflow 2: User provided an initial prompt
        # Example: /image a majestic dragon perched on a mountain
        # They have an idea but might want our help making it better
        print(
            f"\n{utils.SYSTEM_MSG}--> Starting image prompt crafting with your idea..."
            f"{utils.RESET_COLOR}"
        )
        session.start_image_prompt_crafting(prompt_from_args)

    elif not current_prompt and not previous_prompt:
        # Workflow 1: Fresh start - no prompts anywhere
        # This is for users who want to brainstorm from scratch
        # It's like starting with a blank canvas and asking "what should we create?"
        print(
            f"\n{utils.SYSTEM_MSG}--> Starting fresh image prompt crafting..."
            f"{utils.RESET_COLOR}"
        )
        session.start_image_prompt_crafting()

    elif not current_prompt and previous_prompt:
        # Workflow 3: Offer to refine the previous prompt
        # This is helpful when users want to iterate on something they made before
        # It's like asking "should we try a variation of what we did last time?"
        print(f"\n{utils.SYSTEM_MSG}Previous image prompt found:{utils.RESET_COLOR}")
        print(f"  '{previous_prompt}'\n")

        # We present options in a friendly, conversational way
        # Notice how we offer shortcuts (single letters) for experienced users
        # but also full words for clarity
        choice = (
            input(
                "Would you like to:\n"
                "  1. Refine this prompt (enter 'r' or 'refine')\n"
                "  2. Regenerate as-is (enter 'g' or 'generate')\n"
                "  3. Start fresh (enter 'n' or 'new')\n"
                "Choice: "
            )
            .lower()
            .strip()
        )

        if choice in ["r", "refine", "1"]:
            # User wants to tweak the previous prompt
            print(
                f"\n{utils.SYSTEM_MSG}--> Loading previous prompt for refinement..."
                f"{utils.RESET_COLOR}"
            )
            # Start crafting with the previous prompt as the initial seed
            # This gives them a starting point to work from
            session.start_image_prompt_crafting(previous_prompt)

        elif choice in ["g", "generate", "2"]:
            # User wants the exact same prompt again
            # Maybe they want a different variation or the last one failed
            print(
                f"\n{utils.SYSTEM_MSG}--> Regenerating previous image..."
                f"{utils.RESET_COLOR}"
            )
            # Directly generate without entering crafting mode
            session.generate_image(previous_prompt)

        elif choice in ["n", "new", "3"]:
            # User wants to start over with something completely different
            print(
                f"\n{utils.SYSTEM_MSG}--> Starting fresh image prompt crafting..."
                f"{utils.RESET_COLOR}"
            )
            session.start_image_prompt_crafting()

        else:
            # Invalid input - cancel the operation
            # We don't want to guess what the user meant, so we safely abort
            print(
                f"{utils.SYSTEM_MSG}--> Invalid choice. Cancelled.{utils.RESET_COLOR}"
            )

    elif current_prompt:
        # Edge case: There's a current prompt but we're not in crafting mode
        # This might happen if crafting was interrupted somehow
        # We help the user recover gracefully
        print(f"\n{utils.SYSTEM_MSG}Current prompt found:{utils.RESET_COLOR}")
        print(f"  '{current_prompt}'\n")
        print(
            f"{utils.SYSTEM_MSG}--> Resuming image crafting mode. "
            f"Type 'yes' to generate or continue refining.{utils.RESET_COLOR}"
        )
        # Re-enable crafting mode so the user can continue where they left off
        session.state.img_prompt_crafting = True


# --- Multi-Chat Command Handler Functions ---


def _save_multichat_session_to_file(
    state: MultiChatSessionState, filename: str
) -> bool:
    safe_name = utils.sanitize_filename(filename.rsplit(".", 1)[0]) + ".json"
    filepath = config.SESSIONS_DIRECTORY / safe_name

    state_dict = asdict(state)
    state_dict["session_type"] = "multichat"
    # Engines are not serializable, so we remove them
    del state_dict["openai_engine"]
    del state_dict["gemini_engine"]

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        print(
            f"{utils.SYSTEM_MSG}--> Multi-chat session saved to: {filepath}{utils.RESET_COLOR}"
        )
        return True
    except (OSError, TypeError) as e:
        log.error("Failed to save multi-chat session state: %s", e)
        print(
            f"{utils.SYSTEM_MSG}--> Error saving multi-chat session: {e}{utils.RESET_COLOR}"
        )
        return False


def handle_multichat_exit(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> bool:
    if args:
        state.custom_log_rename = " ".join(args)
    return True


def handle_multichat_quit(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> bool:
    state.force_quit = True
    return True


def handle_multichat_help(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    utils.display_help("multichat")


def handle_multichat_history(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    import json

    print(json.dumps(state.shared_history, indent=2))


def handle_multichat_debug(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    state.debug_active = not state.debug_active
    status = "ENABLED" if state.debug_active else "DISABLED"
    print(
        f"{utils.SYSTEM_MSG}--> Session-specific debug logging is now {status}.{utils.RESET_COLOR}"
    )


def handle_multichat_remember(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    print(
        f"{utils.SYSTEM_MSG}--> /remember is not yet implemented for multi-chat.{utils.RESET_COLOR}"
    )


def handle_multichat_max_tokens(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    if args and args[0].isdigit():
        state.max_tokens = int(args[0])
        print(
            f"{utils.SYSTEM_MSG}--> Max tokens for this session set to: {state.max_tokens}.{utils.RESET_COLOR}"
        )
    else:
        print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")


def handle_multichat_clear(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    confirm = prompt(
        "This will clear all conversation history. Type `proceed` to confirm: "
    )
    if confirm.lower() == "proceed":
        state.shared_history.clear()
        print(
            f"{utils.SYSTEM_MSG}--> Conversation history has been cleared.{utils.RESET_COLOR}"
        )
    else:
        print(f"{utils.SYSTEM_MSG}--> Clear cancelled.{utils.RESET_COLOR}")


def handle_multichat_model(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    if len(args) != 2:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /model <gpt|gem> <model_name>{utils.RESET_COLOR}"
        )
        return

    engine_alias, model_name = args[0].lower(), args[1]
    engine_map = {"gpt": "openai", "gem": "gemini"}

    if engine_alias not in engine_map:
        print(
            f"{utils.SYSTEM_MSG}--> Invalid engine alias. Use 'gpt' or 'gem'.{utils.RESET_COLOR}"
        )
        return

    target_engine = engine_map[engine_alias]
    if target_engine == "openai":
        state.openai_model = model_name
        print(
            f"{utils.SYSTEM_MSG}--> OpenAI model set to: {model_name}{utils.RESET_COLOR}"
        )
    else:  # gemini
        state.gemini_model = model_name
        print(
            f"{utils.SYSTEM_MSG}--> Gemini model set to: {model_name}{utils.RESET_COLOR}"
        )


def handle_multichat_state(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> None:
    print(f"{utils.SYSTEM_MSG}--- Multi-Chat Session State ---{utils.RESET_COLOR}")
    print(f"  OpenAI Model: {state.openai_model}")
    print(f"  Gemini Model: {state.gemini_model}")
    print(f"  Max Tokens: {state.max_tokens or 'Default'}")
    print(f"  Debug Logging: {'On' if state.debug_active else 'Off'}")
    print(f"  System Prompts: {'Active' if state.system_prompts else 'None'}")
    if state.initial_image_data:
        print(f"  Attached Images: {len(state.initial_image_data)}")


def handle_multichat_save(
    args: list[str], state: MultiChatSessionState, cli_history: InMemoryHistory
) -> bool:
    should_remember, should_stay = "--remember" in args, "--stay" in args
    filename_parts = [arg for arg in args if arg not in ("--remember", "--stay")]
    filename = " ".join(filename_parts)
    if not filename:
        print(
            f"{utils.SYSTEM_MSG}--> Usage: /save <filename> [--stay] [--remember]{utils.RESET_COLOR}"
        )
        return False

    state.command_history = cli_history.get_strings()
    if _save_multichat_session_to_file(state, filename):
        if not should_remember:
            state.exit_without_memory = True
        return not should_stay
    return False


# --- Command Dispatcher Maps ---

COMMAND_MAP = {
    "/exit": handle_exit,
    "/quit": handle_quit,
    "/help": handle_help,
    "/stream": handle_stream,
    "/debug": handle_debug,
    "/memory": handle_memory,
    "/remember": handle_remember,
    "/max-tokens": handle_max_tokens,
    "/clear": handle_clear,
    "/model": handle_model,
    "/engine": handle_engine,
    "/history": handle_history,
    "/state": handle_state,
    "/set": handle_set,
    "/save": handle_save,
    "/load": handle_load,
    "/refresh": handle_refresh,
    "/files": handle_files,
    "/attach": handle_attach,
    "/detach": handle_detach,
    "/personas": handle_personas,
    "/persona": handle_persona,
    "/image": handle_image,  # Added image generation command
}

MULTICHAT_COMMAND_MAP = {
    "/exit": handle_multichat_exit,
    "/quit": handle_multichat_quit,
    "/help": handle_multichat_help,
    "/history": handle_multichat_history,
    "/debug": handle_multichat_debug,
    "/remember": handle_multichat_remember,
    "/max-tokens": handle_multichat_max_tokens,
    "/clear": handle_multichat_clear,
    "/model": handle_multichat_model,
    "/state": handle_multichat_state,
    "/save": handle_multichat_save,
}
