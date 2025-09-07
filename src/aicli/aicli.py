#!/usr/bin/env python3
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 David

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

# -*- coding: utf-8 -*-

"""
Unified Command-Line AI Client
Main entry point for the application.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import api_client, bootstrap, config, handlers, review
from . import personas as persona_manager
from .settings import settings
from .utils.formatters import RESET_COLOR, SYSTEM_MSG


class CustomHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Custom formatter for argparse help messages."""


def run_chat_command(args: argparse.Namespace) -> None:
    """Orchestrates the application flow for all chat-related commands."""
    prompt = args.prompt
    if args.both is not None and args.both != "":
        prompt = args.both

    if not sys.stdin.isatty() and not prompt and args.both is None and not args.load:
        prompt = sys.stdin.read().strip()

    is_single_shot = prompt is not None
    if is_single_shot and not prompt.strip():
        print(
            "Error: The provided prompt cannot be empty or contain only whitespace.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Argument Validation ---
    if args.image:
        # Pre-resolve engine for validation since persona can set it.
        effective_engine = args.engine
        if not effective_engine and args.persona:
            persona = persona_manager.load_persona(args.persona)
            if persona and persona.engine:
                effective_engine = persona.engine
        if not effective_engine:
            effective_engine = settings["default_engine"]

        if effective_engine != "openai":
            print(
                "Error: --image mode is only supported by the 'openai' engine.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.both is not None and args.prompt:
        print(
            'Error: Provide an initial prompt via --both "PROMPT" or --prompt "PROMPT", but not both.',
            file=sys.stderr,
        )
        sys.exit(1)
    if args.file:
        for path_str in args.file:
            if not Path(path_str).exists():
                print(
                    f"Error: The file or directory '{path_str}' does not exist.",
                    file=sys.stderr,
                )
                sys.exit(1)

    try:
        if args.load:
            handlers.handle_load_session(args.load)
        elif args.both is not None:
            handlers.handle_multichat_session(prompt, args)
        elif args.image:
            handlers.handle_image_generation(prompt, args)
        else:  # Default to single chat mode (interactive or single-shot)
            handlers.handle_chat(prompt, args)

    except api_client.MissingApiKeyError as e:
        print(
            f"{SYSTEM_MSG}Configuration Error:{RESET_COLOR}",
            file=sys.stderr,
        )
        print(f"  {e}", file=sys.stderr)
        print(
            "\nPlease create a .env file with your API keys at the following location:",
            file=sys.stderr,
        )
        print(f"  {config.DOTENV_FILE}", file=sys.stderr)
        print("\nExample .env content:", file=sys.stderr)
        print("  OPENAI_API_KEY=sk-...\n  GEMINI_API_KEY=AIza...", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Parses arguments and orchestrates the application flow."""
    bootstrap.ensure_project_structure()
    load_dotenv(dotenv_path=config.DOTENV_FILE)

    chat_parent_parser = argparse.ArgumentParser(add_help=False)
    core_group = chat_parent_parser.add_argument_group("Core Execution")
    mode_group = chat_parent_parser.add_argument_group("Operation Modes")
    context_group = chat_parent_parser.add_argument_group("Context & Input")
    session_group = chat_parent_parser.add_argument_group("Session Control")
    exclusive_mode_group = mode_group.add_mutually_exclusive_group()
    core_group.add_argument(
        "-e",
        "--engine",
        choices=["openai", "gemini"],
        default=None,
        help=f"Specify the AI provider. (default: {settings['default_engine']})",
    )
    core_group.add_argument(
        "-m",
        "--model",
        type=str,
        help="Specify the model to use, overriding the default.",
    )
    exclusive_mode_group.add_argument(
        "-c",
        "--chat",
        action="store_true",
        help="Activate chat mode for text generation.",
    )
    exclusive_mode_group.add_argument(
        "-i",
        "--image",
        action="store_true",
        help="Activate image generation mode (OpenAI only).",
    )
    exclusive_mode_group.add_argument(
        "-b",
        "--both",
        nargs="?",
        const="",
        type=str,
        metavar="PROMPT",
        help="Activate interactive multi-chat mode.\nOptionally provide an initial prompt.",
    )
    exclusive_mode_group.add_argument(
        "-l", "--load", type=str, metavar="FILEPATH", help="Load a saved chat session."
    )
    context_group.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Provide a prompt for single-shot chat or image mode.",
    )
    context_group.add_argument(
        "-P",
        "--persona",
        type=str,
        help="Load a session configuration from a persona file.",
    )
    context_group.add_argument(
        "--system-prompt",
        type=str,
        help="Specify a system prompt/instruction from a string or file path.",
    )
    context_group.add_argument(
        "-f",
        "--file",
        action="append",
        help="Attach content from files or directories (can be used multiple times).",
    )
    context_group.add_argument(
        "-x",
        "--exclude",
        action="append",
        help="Exclude a file or directory (can be used multiple times).",
    )
    context_group.add_argument(
        "--memory",
        "--mem",
        action="store_true",
        help="Toggle persistent memory for this session.",
    )
    session_group.add_argument(
        "-s",
        "--session-name",
        type=str,
        help="Provide a custom name for the chat log file.",
    )
    session_group.add_argument(
        "--stream",
        action="store_true",
        default=None,
        help="Enable streaming for chat responses.",
    )
    session_group.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming for chat responses.",
    )
    session_group.add_argument(
        "--max-tokens",
        type=int,
        help=f"Set the maximum number of tokens to generate. (default: {settings['default_max_tokens']})",
    )
    session_group.add_argument(
        "--debug",
        action="store_true",
        help="Start with session-specific debug logging enabled.",
    )

    parser = argparse.ArgumentParser(
        description="Unified Command-Line AI Client for OpenAI and Gemini.",
        formatter_class=CustomHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands: chat, review"
    )
    subparsers.add_parser(
        "chat",
        help="Start a chat session (default if no command is given).",
        parents=[chat_parent_parser],
        formatter_class=CustomHelpFormatter,
    )
    parser_review = subparsers.add_parser(
        "review", help="Review and manage logs and saved sessions."
    )
    parser_review.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Optional: Path to a specific file to replay directly.",
    )

    args_list = sys.argv[1:]
    is_review_command = len(args_list) > 0 and args_list[0] == "review"
    is_chat_command = len(args_list) > 0 and args_list[0] == "chat"

    if is_review_command:
        args = parser.parse_args(args_list)
        review.main(args)
    else:
        if is_chat_command:
            args_list = args_list[1:]

        # Handle `aicli` with no arguments for interactive mode.
        if not args_list and sys.stdin.isatty():
            # Create a default args namespace for interactive mode
            args = chat_parent_parser.parse_args([])
            args.prompt = None
            args.chat = True  # Ensure chat mode is active
            run_chat_command(args)
        else:
            args = chat_parent_parser.parse_args(args_list)
            run_chat_command(args)


if __name__ == "__main__":
    main()
