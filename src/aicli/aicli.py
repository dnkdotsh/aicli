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
from dotenv import load_dotenv

from . import api_client
from . import handlers
from . import utils
from . import config
from .settings import settings
from .engine import get_engine
from .logger import log

class CustomHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter for argparse help messages."""
    pass

def main():
    """Parses arguments and orchestrates the application flow."""
    load_dotenv()
    # Ensure all required directories exist
    utils.ensure_dir_exists(config.CONFIG_DIR)
    utils.ensure_dir_exists(config.LOG_DIRECTORY)
    utils.ensure_dir_exists(config.IMAGE_DIRECTORY)
    utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)

    if len(sys.argv) == 1 and sys.stdin.isatty():
        print(f"Starting interactive chat with default engine ('{settings['default_engine']}')...")
        try:
            memory_content, attachments_content, _ = utils.process_files([], settings['memory_enabled'], [])
            
            system_prompt_parts = []
            if memory_content:
                system_prompt_parts.append(memory_content)
            if attachments_content:
                system_prompt_parts.append(attachments_content)
            system_prompt = "\n\n".join(system_prompt_parts) if system_prompt_parts else None

            api_key = api_client.check_api_keys(settings['default_engine'])
            engine_instance = get_engine(settings['default_engine'], api_key)
            model_key = 'default_openai_chat_model' if settings['default_engine'] == 'openai' else 'default_gemini_model'
            model = settings[model_key]
            handlers.handle_chat(engine_instance, model, system_prompt, None, [], None, settings.get('default_max_tokens'), settings.get('stream'), settings['memory_enabled'], False)
        except api_client.MissingApiKeyError as e:
            log.error(e)
            sys.exit(1)
        return

    parser = argparse.ArgumentParser(
        description="Unified Command-Line AI Client for OpenAI and Gemini.",
        formatter_class=CustomHelpFormatter
    )

    core_group = parser.add_argument_group('Core Execution')
    mode_group = parser.add_argument_group('Operation Modes')
    context_group = parser.add_argument_group('Context & Input')
    session_group = parser.add_argument_group('Session Control')

    exclusive_mode_group = mode_group.add_mutually_exclusive_group()

    core_group.add_argument('-e', '--engine', choices=['openai', 'gemini'], default=settings['default_engine'],
                        help="Specify the AI provider.")
    core_group.add_argument('-m', '--model', type=str,
                        help="Specify the model to use, overriding the default.")

    exclusive_mode_group.add_argument('-c', '--chat', action='store_true', help='Activate chat mode for text generation.')
    exclusive_mode_group.add_argument('-i', '--image', action='store_true', help='Activate image generation mode (OpenAI only).')
    exclusive_mode_group.add_argument('-b', '--both', nargs='?', const='', type=str, metavar='PROMPT',
                                  help='Activate interactive multi-chat mode with both OpenAI and Gemini.\nOptionally provide an initial prompt.')
    exclusive_mode_group.add_argument('-l', '--load', type=str, metavar='FILEPATH', help='Load a saved chat session from a file and start in interactive mode.')

    context_group.add_argument('-p', '--prompt', type=str, help='Provide a prompt for single-shot chat or image mode.')
    context_group.add_argument('--system-prompt', type=str, help='Specify a system prompt/instruction from a string or file path.')
    context_group.add_argument('-f', '--file', action='append', help="Attach content from files or directories (can be used multiple times).")
    context_group.add_argument('-x', '--exclude', action='append', help="Exclude a file or directory (can be used multiple times).")
    context_group.add_argument('--memory', '--mem', action='store_true', help='Toggle persistent memory for this session (enables if disabled, disables if enabled).')

    session_group.add_argument('-s', '--session-name', type=str, help='Provide a custom name for the chat log file (interactive chat only).')
    session_group.add_argument('--stream', action='store_true', default=None, help="Enable streaming for chat responses.")
    session_group.add_argument('--no-stream', action='store_false', dest='stream', help='Disable streaming for chat responses.')
    session_group.add_argument('--max-tokens', type=int, default=settings['default_max_tokens'], help="Set the maximum number of tokens to generate.")
    session_group.add_argument('--debug', action='store_true', help='Start with session-specific debug logging enabled.')

    parser.set_defaults(stream=settings['stream'])
    args = parser.parse_args()

    prompt = args.prompt
    if args.both is not None and args.both != '':
        prompt = args.both
    
    if not sys.stdin.isatty() and not prompt and args.both is None and not args.load:
        prompt = sys.stdin.read().strip()

    if prompt is not None and not prompt.strip():
        parser.error("The provided prompt cannot be empty or contain only whitespace.")

    if args.file:
        for path in args.file:
            if not os.path.exists(path):
                parser.error(f"The file or directory '{path}' does not exist.")

    if args.both is not None and args.prompt:
        parser.error("Provide an initial prompt via --both \"PROMPT\" or --prompt \"PROMPT\", but not both.")
    if args.image and args.engine != 'openai':
        parser.error("--image mode is only supported by the 'openai' engine.")

    if not args.image and args.both is None and not args.load and (args.prompt or args.file or args.memory or args.session_name or args.system_prompt):
        args.chat = True
    elif not args.chat and not args.image and args.both is None and not args.load:
        args.chat = True

    # Get the default from settings, then toggle it if the flag is present.
    memory_enabled_for_session = settings['memory_enabled']
    if args.memory:
        memory_enabled_for_session = not memory_enabled_for_session

    # Assemble the final system prompt from multiple sources
    system_prompt_parts = []
    if args.system_prompt:
        try:
            system_prompt_parts.append(utils.read_system_prompt(args.system_prompt))
        except FileNotFoundError:
            parser.error(f"The system prompt file '{args.system_prompt}' does not exist.")
        except IOError as e:
            parser.error(f"Error reading system prompt file: {e}")

    memory_content, attachments_content, image_data = utils.process_files(
        args.file, memory_enabled_for_session, args.exclude
    )

    if memory_content:
        system_prompt_parts.append(f"--- PERSISTENT MEMORY ---\n{memory_content}")

    if attachments_content:
        system_prompt_parts.append(f"--- ATTACHED FILES ---\n{attachments_content}")

    system_prompt = "\n\n".join(system_prompt_parts) if system_prompt_parts else None

    try:
        if args.load:
            handlers.handle_load_session(args.load)
        elif args.both is not None:
            handlers.handle_multichat_session(prompt, system_prompt, image_data, args.session_name, args.max_tokens, args.debug)
        elif args.chat:
            api_key = api_client.check_api_keys(args.engine)
            engine_instance = get_engine(args.engine, api_key)
            model_key = 'default_openai_chat_model' if args.engine == 'openai' else 'default_gemini_model'
            model = args.model or settings[model_key]
            handlers.handle_chat(engine_instance, model, system_prompt, prompt, image_data, args.session_name, args.max_tokens, args.stream, memory_enabled_for_session, args.debug)
        elif args.image:
            api_key = api_client.check_api_keys(args.engine)
            engine_instance = get_engine(args.engine, api_key)
            image_prompt = prompt or system_prompt
            handlers.handle_image_generation(api_key, engine_instance, image_prompt)

    except api_client.MissingApiKeyError as e:
        log.error(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
