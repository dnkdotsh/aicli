#!/usr/bin/env python3
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

# -*- coding: utf-8 -*-

"""
Unified Command-Line AI Client
Main entry point for the application.
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from . import api_client
from . import handlers
from . import utils
from . import config
from . import review
from .settings import settings
from .engine import get_engine
from .logger import log

class CustomHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter for argparse help messages."""
    pass

def run_chat_command(args):
    """Orchestrates the application flow for all chat-related commands."""
    # --- Argument Validation ---
    # Perform all validation checks upfront before any processing.
    if args.image and args.engine != 'openai':
        print("Error: --image mode is only supported by the 'openai' engine.", file=sys.stderr)
        sys.exit(1)
    if args.both is not None and args.prompt:
        print("Error: Provide an initial prompt via --both \"PROMPT\" or --prompt \"PROMPT\", but not both.", file=sys.stderr)
        sys.exit(1)
    if args.file:
        for path in args.file:
            if not os.path.exists(path):
                print(f"Error: The file or directory '{path}' does not exist.", file=sys.stderr)
                sys.exit(1)

    prompt = args.prompt
    if args.both is not None and args.both != '':
        prompt = args.both

    # Handle piped input if no other prompt source is available
    if not sys.stdin.isatty() and not prompt and args.both is None and not args.load:
        prompt = sys.stdin.read().strip()

    is_single_shot = prompt is not None

    # Further prompt validation after potential stdin read
    if is_single_shot and not prompt.strip():
        print("Error: The provided prompt cannot be empty or contain only whitespace.", file=sys.stderr)
        sys.exit(1)

    # If no specific mode is chosen, but flags that imply chat are used, default to chat mode
    is_chat_implied = (args.prompt or args.file or args.memory or args.session_name or args.system_prompt)
    if not args.image and args.both is None and not args.load and is_chat_implied:
        args.chat = True
    elif not args.chat and not args.image and args.both is None and not args.load:
        args.chat = True

    # Get the default from settings, then toggle it if the flag is present.
    memory_enabled_for_session = settings['memory_enabled']
    if args.memory:
        memory_enabled_for_session = not memory_enabled_for_session

    # Override and disable memory for any single-shot, non-interactive prompt.
    if is_single_shot:
        memory_enabled_for_session = False

    # Process files and memory into their respective components
    memory_content, attachments, image_data = utils.process_files(
        args.file, memory_enabled_for_session, args.exclude
    )

    # Assemble the base system prompt (user-provided + memory)
    system_prompt_parts = []
    if args.system_prompt:
        try:
            system_prompt_parts.append(utils.read_system_prompt(args.system_prompt))
        except (IOError, FileNotFoundError) as e:
            print(f"Error reading system prompt file: {e}", file=sys.stderr)
            sys.exit(1)

    if memory_content:
        system_prompt_parts.append(f"--- PERSISTENT MEMORY ---\n{memory_content}")

    system_prompt = "\n\n".join(system_prompt_parts) if system_prompt_parts else None

    # Execute the appropriate handler based on the mode
    try:
        if args.load:
            handlers.handle_load_session(args.load)
        elif args.both is not None:
            # For multichat, we pre-assemble the file context into the prompt
            attachment_texts = []
            for path, content in attachments.items():
                attachment_texts.append(f"--- FILE: {path.as_posix()} ---\n{content}")
            attachments_str = "\n\n".join(attachment_texts)

            full_system_prompt = system_prompt
            if attachments_str:
                full_system_prompt = (full_system_prompt or "") + f"\n\n--- ATTACHED FILES ---\n{attachments_str}"

            handlers.handle_multichat_session(prompt, full_system_prompt, image_data, args.session_name, args.max_tokens, args.debug)
        elif args.chat:
            api_key = api_client.check_api_keys(args.engine)
            engine_instance = get_engine(args.engine, api_key)
            model_key = 'default_openai_chat_model' if args.engine == 'openai' else 'default_gemini_model'
            model = args.model or settings[model_key]
            handlers.handle_chat(engine_instance, model, system_prompt, prompt, image_data, attachments, args.session_name, args.max_tokens, args.stream, memory_enabled_for_session, args.debug)
        elif args.image:
            api_key = api_client.check_api_keys(args.engine)
            engine_instance = get_engine(args.engine, api_key)
            image_prompt = prompt or system_prompt
            handlers.handle_image_generation(api_key, engine_instance, image_prompt)
    except api_client.MissingApiKeyError as e:
        log.error(e)
        sys.exit(1)


def main():
    """Parses arguments and orchestrates the application flow."""
    load_dotenv()
    utils.ensure_dir_exists(config.CONFIG_DIR)
    utils.ensure_dir_exists(config.LOG_DIRECTORY)
    utils.ensure_dir_exists(config.IMAGE_DIRECTORY)
    utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)

    # Parent parser to define all common chat-related arguments
    chat_parent_parser = argparse.ArgumentParser(add_help=False)
    core_group = chat_parent_parser.add_argument_group('Core Execution')
    mode_group = chat_parent_parser.add_argument_group('Operation Modes')
    context_group = chat_parent_parser.add_argument_group('Context & Input')
    session_group = chat_parent_parser.add_argument_group('Session Control')
    exclusive_mode_group = mode_group.add_mutually_exclusive_group()
    core_group.add_argument('-e', '--engine', choices=['openai', 'gemini'], default=settings['default_engine'], help="Specify the AI provider.")
    core_group.add_argument('-m', '--model', type=str, help="Specify the model to use, overriding the default.")
    exclusive_mode_group.add_argument('-c', '--chat', action='store_true', help='Activate chat mode for text generation.')
    exclusive_mode_group.add_argument('-i', '--image', action='store_true', help='Activate image generation mode (OpenAI only).')
    exclusive_mode_group.add_argument('-b', '--both', nargs='?', const='', type=str, metavar='PROMPT', help='Activate interactive multi-chat mode.\nOptionally provide an initial prompt.')
    exclusive_mode_group.add_argument('-l', '--load', type=str, metavar='FILEPATH', help='Load a saved chat session.')
    context_group.add_argument('-p', '--prompt', type=str, help='Provide a prompt for single-shot chat or image mode.')
    context_group.add_argument('--system-prompt', type=str, help='Specify a system prompt/instruction from a string or file path.')
    context_group.add_argument('-f', '--file', action='append', help="Attach content from files or directories (can be used multiple times).")
    context_group.add_argument('-x', '--exclude', action='append', help="Exclude a file or directory (can be used multiple times).")
    context_group.add_argument('--memory', '--mem', action='store_true', help='Toggle persistent memory for this session.')
    session_group.add_argument('-s', '--session-name', type=str, help='Provide a custom name for the chat log file.')
    session_group.add_argument('--stream', action='store_true', default=None, help="Enable streaming for chat responses.")
    session_group.add_argument('--no-stream', action='store_false', dest='stream', help='Disable streaming for chat responses.')
    session_group.add_argument('--max-tokens', type=int, default=settings['default_max_tokens'], help="Set the maximum number of tokens to generate.")
    session_group.add_argument('--debug', action='store_true', help='Start with session-specific debug logging enabled.')
    chat_parent_parser.set_defaults(stream=settings['stream'])

    # Main parser
    parser = argparse.ArgumentParser(description="Unified Command-Line AI Client for OpenAI and Gemini.", formatter_class=CustomHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Available commands: chat, review')

    # Chat command parser (inherits from parent)
    subparsers.add_parser('chat', help='Start a chat session (default if no command is given).', parents=[chat_parent_parser], formatter_class=CustomHelpFormatter)

    # Review command parser
    parser_review = subparsers.add_parser('review', help='Review and manage logs and saved sessions.')
    parser_review.add_argument('file', nargs='?', type=Path, help="Optional: Path to a specific file to replay directly.")

    # --- Argument Parsing and Dispatch ---
    # Check if a subcommand is likely being used.
    # The first arg is the script name itself.
    args_list = sys.argv[1:]
    is_review_command = len(args_list) > 0 and args_list[0] == 'review'
    is_chat_command = len(args_list) > 0 and args_list[0] == 'chat'

    if is_review_command:
        args = parser.parse_args(args_list)
        review.main(args)
    else:
        # If 'chat' is explicitly typed, remove it so the parent parser doesn't see it as an unknown arg.
        if is_chat_command:
            args_list = args_list[1:]

        # Handle the default case (no subcommand) by just parsing against the parent.
        args = chat_parent_parser.parse_args(args_list)

        # This handles the no-arg `aicli` case for interactive chat.
        if not args_list and sys.stdin.isatty():
             # Mimic the original no-arg behavior
             print(f"Starting interactive chat with default engine ('{settings['default_engine']}')...")
             # PyEvolve Change: Load persistent memory and attachments for default interactive mode
             memory_content, attachments, _ = utils.process_files(None, settings['memory_enabled'], None)
             system_prompt_for_interactive = None
             if memory_content:
                 system_prompt_for_interactive = f"--- PERSISTENT MEMORY ---\n{memory_content}"

             handlers.handle_chat(
                 get_engine(settings['default_engine'], api_client.check_api_keys(settings['default_engine'])),
                 settings['default_gemini_model'],
                 system_prompt_for_interactive,  # Pass loaded system prompt
                 None,                           # No initial prompt for interactive mode
                 [],                             # No initial image data
                 attachments,                    # Pass loaded attachments (will be empty dict for no-arg mode)
                 None,
                 None,
                 True,
                 settings['memory_enabled'],
                 False
             )
        else:
             run_chat_command(args)


if __name__ == "__main__":
    main()
