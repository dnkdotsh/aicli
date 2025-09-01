#!/usr/bin/env python3
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

# This program is free software: you can redistribute it and/or modify
# it under a new-style BSD license. See the LICENSE file for details.
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
from . import personas as persona_manager
from .settings import settings
from .engine import get_engine
from .logger import log

class CustomHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter for argparse help messages."""
    pass

def run_chat_command(args):
    """Orchestrates the application flow for all chat-related commands."""
    prompt = args.prompt
    if args.both is not None and args.both != '':
        prompt = args.both

    if not sys.stdin.isatty() and not prompt and args.both is None and not args.load:
        prompt = sys.stdin.read().strip()

    is_single_shot = prompt is not None
    if is_single_shot and prompt is not None and not prompt.strip():
        print("Error: The provided prompt cannot be empty or contain only whitespace.", file=sys.stderr)
        sys.exit(1)

    # --- Persona Loading ---
    persona = None
    if args.persona:
        persona = persona_manager.load_persona(args.persona)
        if not persona:
            print(f"Error: Persona '{args.persona}' not found or is invalid.", file=sys.stderr)
            sys.exit(1)
    # If interactive and no other persona/prompt is set, load the default.
    elif not is_single_shot and not args.system_prompt:
        persona = persona_manager.load_persona(persona_manager.DEFAULT_PERSONA_NAME)

    # --- Configuration Precedence: CLI > Persona > Settings ---
    engine_to_use = settings['default_engine']
    max_tokens_to_use = settings['default_max_tokens']
    stream_to_use = settings['stream']

    if persona:
        if persona.engine: engine_to_use = persona.engine
        if persona.max_tokens is not None: max_tokens_to_use = persona.max_tokens
        if persona.stream is not None: stream_to_use = persona.stream

    engine_to_use = args.engine
    if args.max_tokens != settings['default_max_tokens']: max_tokens_to_use = args.max_tokens
    if args.stream is not None: stream_to_use = args.stream

    model_to_use = args.model
    if not model_to_use:
        if persona and persona.model:
            model_to_use = persona.model
        else:
            model_key = 'default_openai_chat_model' if engine_to_use == 'openai' else 'default_gemini_model'
            model_to_use = settings[model_key]

    # --- Argument Validation ---
    if args.image and engine_to_use != 'openai':
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

    is_chat_implied = (args.prompt or args.file or args.memory or args.session_name or args.system_prompt or args.persona)
    if not args.image and args.both is None and not args.load and is_chat_implied:
        args.chat = True
    elif not args.chat and not args.image and args.both is None and not args.load:
        args.chat = True

    memory_enabled_for_session = settings['memory_enabled']
    if args.memory: memory_enabled_for_session = not memory_enabled_for_session
    if is_single_shot: memory_enabled_for_session = False

    memory_content, attachments, image_data = utils.process_files(
        args.file, memory_enabled_for_session, args.exclude
    )

    initial_system_prompt = None
    if args.system_prompt:
        try:
            initial_system_prompt = utils.read_system_prompt(args.system_prompt)
        except (IOError, FileNotFoundError) as e:
            print(f"Error reading system prompt file: {e}", file=sys.stderr)
            sys.exit(1)
    elif persona and persona.system_prompt:
        initial_system_prompt = persona.system_prompt

    system_prompt_parts = []
    if initial_system_prompt: system_prompt_parts.append(initial_system_prompt)
    if memory_content: system_prompt_parts.append(f"--- PERSISTENT MEMORY ---\n{memory_content}")
    final_api_system_prompt = "\n\n".join(system_prompt_parts) if system_prompt_parts else None

    try:
        if args.load:
            handlers.handle_load_session(args.load)
        elif args.both is not None:
            attachment_texts = [f"--- FILE: {path.as_posix()} ---\n{content}" for path, content in attachments.items()]
            attachments_str = "\n\n".join(attachment_texts)
            full_system_prompt = final_api_system_prompt
            if attachments_str:
                full_system_prompt = (full_system_prompt or "") + f"\n\n--- ATTACHED FILES ---\n{attachments_str}"
            handlers.handle_multichat_session(prompt, full_system_prompt, image_data, args.session_name, max_tokens_to_use, args.debug)
        elif args.chat:
            api_key = api_client.check_api_keys(engine_to_use)
            engine_instance = get_engine(engine_to_use, api_key)
            handlers.handle_chat(engine_instance, model_to_use, final_api_system_prompt, prompt, image_data, attachments, args.session_name, max_tokens_to_use, stream_to_use, memory_enabled_for_session, args.debug, initial_system_prompt, persona)
        elif args.image:
            api_key = api_client.check_api_keys(engine_to_use)
            engine_instance = get_engine(engine_to_use, api_key)
            image_prompt = prompt or initial_system_prompt
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
    persona_manager.ensure_personas_directory_and_default()

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
    context_group.add_argument('-P', '--persona', type=str, help='Load a session configuration from a persona file.')
    context_group.add_argument('--system-prompt', type=str, help='Specify a system prompt/instruction from a string or file path.')
    context_group.add_argument('-f', '--file', action='append', help="Attach content from files or directories (can be used multiple times).")
    context_group.add_argument('-x', '--exclude', action='append', help="Exclude a file or directory (can be used multiple times).")
    context_group.add_argument('--memory', '--mem', action='store_true', help='Toggle persistent memory for this session.')
    session_group.add_argument('-s', '--session-name', type=str, help='Provide a custom name for the chat log file.')
    session_group.add_argument('--stream', action='store_true', default=None, help="Enable streaming for chat responses.")
    session_group.add_argument('--no-stream', action='store_false', dest='stream', help='Disable streaming for chat responses.')
    session_group.add_argument('--max-tokens', type=int, default=settings['default_max_tokens'], help="Set the maximum number of tokens to generate.")
    session_group.add_argument('--debug', action='store_true', help='Start with session-specific debug logging enabled.')

    parser = argparse.ArgumentParser(description="Unified Command-Line AI Client for OpenAI and Gemini.", formatter_class=CustomHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Available commands: chat, review')
    subparsers.add_parser('chat', help='Start a chat session (default if no command is given).', parents=[chat_parent_parser], formatter_class=CustomHelpFormatter)
    parser_review = subparsers.add_parser('review', help='Review and manage logs and saved sessions.')
    parser_review.add_argument('file', nargs='?', type=Path, help="Optional: Path to a specific file to replay directly.")

    args_list = sys.argv[1:]
    is_review_command = len(args_list) > 0 and args_list[0] == 'review'
    is_chat_command = len(args_list) > 0 and args_list[0] == 'chat'

    if is_review_command:
        args = parser.parse_args(args_list)
        review.main(args)
    else:
        if is_chat_command:
            args_list = args_list[1:]
        args = chat_parent_parser.parse_args(args_list)
        if not args_list and sys.stdin.isatty():
             # This is the no-argument `aicli` case. We now load the default persona.
             default_persona = persona_manager.load_persona(persona_manager.DEFAULT_PERSONA_NAME)
             engine_name = default_persona.engine if default_persona and default_persona.engine else settings['default_engine']
             model_name = default_persona.model if default_persona and default_persona.model else settings['default_gemini_model']
             initial_sys_prompt = default_persona.system_prompt if default_persona else None

             memory_content, attachments, _ = utils.process_files(None, settings['memory_enabled'], None)
             api_system_prompt = initial_sys_prompt
             if memory_content:
                 api_system_prompt = f"{initial_sys_prompt}\n\n--- PERSISTENT MEMORY ---\n{memory_content}" if initial_sys_prompt else f"--- PERSISTENT MEMORY ---\n{memory_content}"

             handlers.handle_chat(
                 get_engine(engine_name, api_client.check_api_keys(engine_name)),
                 model_name,
                 api_system_prompt,
                 None, [], attachments, None, None, True, settings['memory_enabled'], False,
                 initial_sys_prompt, default_persona
             )
        else:
             run_chat_command(args)

if __name__ == "__main__":
    main()
