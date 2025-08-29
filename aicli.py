#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Command-Line AI Client
Main entry point for the application.
"""

import argparse
import sys
import os

import api_client
import handlers
import utils
import config
from settings import settings
from engine import get_engine

class CustomHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter for argparse help messages."""
    pass

def main():
    """Parses arguments and orchestrates the application flow."""
    utils.ensure_dir_exists(config.LOG_DIRECTORY)
    utils.ensure_dir_exists(config.IMAGE_DIRECTORY)

    # If no arguments are provided and we're in an interactive session, start chat directly.
    if len(sys.argv) == 1 and sys.stdin.isatty():
        print(f"Starting interactive chat with default engine ('{settings['default_engine']}')...")
        try:
            system_prompt = None
            if settings['memory_enabled']:
                # Load memory content if the setting is enabled
                system_prompt, _ = utils.process_files([], settings['memory_enabled'], [])

            api_key = api_client.check_api_keys(settings['default_engine'])
            engine_instance = get_engine(settings['default_engine'], api_key)
            model = settings['default_openai_chat_model'] if settings['default_engine'] == 'openai' else settings['default_gemini_model']
            handlers.handle_chat(engine_instance, model, system_prompt, None, [], None, settings.get('default_max_tokens'), settings.get('stream'), settings['memory_enabled'], False)
        except api_client.MissingApiKeyError as e:
            print(e, file=sys.stderr)
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
    exclusive_mode_group.add_argument('-b', '--both', type=str, metavar='PROMPT', help='Send a prompt to both OpenAI and Gemini.')

    context_group.add_argument('-p', '--prompt', type=str, help='Provide a prompt for single-shot chat or image mode.')
    context_group.add_argument('-f', '--file', action='append', help="Attach content from files or directories (can be used multiple times).")
    context_group.add_argument('-x', '--exclude', action='append', help="Exclude a file or directory (can be used multiple times).")
    context_group.add_argument('--memory', action='store_true', help='Force persistent memory on for this session, overriding the default setting.')

    session_group.add_argument('-s', '--session-name', type=str, help='Provide a custom name for the chat log file (interactive chat only).')
    session_group.add_argument('--stream', action='store_true', default=None, help="Enable streaming for chat responses.")
    session_group.add_argument('--no-stream', action='store_false', dest='stream', help='Disable streaming for chat responses.')
    session_group.add_argument('--max-tokens', type=int, default=settings['default_max_tokens'], help="Set the maximum number of tokens to generate.")
    session_group.add_argument('--debug', action='store_true', help='Start with session-specific debug logging enabled.')
    
    parser.set_defaults(stream=settings['stream'])
    args = parser.parse_args()

    prompt = args.both or args.prompt
    if not sys.stdin.isatty() and not prompt:
        prompt = sys.stdin.read().strip()
    
    if prompt is not None and not prompt.strip():
        parser.error("The provided prompt cannot be empty or contain only whitespace.")

    if args.file:
        for path in args.file:
            if not os.path.exists(path):
                parser.error(f"The file or directory '{path}' does not exist.")

    if args.both and args.prompt:
        parser.error("Provide a prompt via --both \"PROMPT\" or --prompt \"PROMPT\", but not both.")
    if args.both and args.session_name:
        parser.error("--session-name cannot be used with --both mode.")
    if args.image and args.engine != 'openai':
        parser.error("--image mode is only supported by the 'openai' engine.")

    if not args.chat and not args.image and not args.both:
        args.chat = True

    # Determine if memory should be active for this session
    memory_enabled_for_session = settings['memory_enabled'] or args.memory
    system_prompt, image_data = utils.process_files(args.file, memory_enabled_for_session, args.exclude)

    try:
        if args.both:
            handlers.handle_both_engines(system_prompt, args.both, image_data, args.max_tokens, args.stream)
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
        print(e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
