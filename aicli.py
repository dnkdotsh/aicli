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

def main():
    """Parses arguments and orchestrates the application flow."""
    utils.ensure_dir_exists(config.LOG_DIRECTORY)
    utils.ensure_dir_exists(config.IMAGE_DIRECTORY)

    # --- TTY / Non-TTY Handling ---
    # If no arguments are provided and we're in an interactive session, start chat directly.
    if len(sys.argv) == 1 and sys.stdin.isatty():
        print(f"Starting interactive chat with default engine ('{settings['default_engine']}')...")
        try:
            api_key = api_client.check_api_keys(settings['default_engine'])
            engine_instance = get_engine(settings['default_engine'], api_key)
            model = settings['default_openai_chat_model'] if settings['default_engine'] == 'openai' else settings['default_gemini_model']
            handlers.handle_chat(engine_instance, model, None, None, [], None, settings.get('default_max_tokens'), settings.get('stream'), False)
        except api_client.MissingApiKeyError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        return

    parser = argparse.ArgumentParser(
        description="Unified Command-Line AI Client for OpenAI and Gemini.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-e', '--engine', choices=['openai', 'gemini'], default=settings['default_engine'],
                        help=f"Specify the AI provider (default: {settings['default_engine']}).")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('-c', '--chat', action='store_true', help='Activate chat mode for text generation.')
    mode_group.add_argument('-i', '--image', action='store_true', help='Activate image generation mode (OpenAI only).')
    mode_group.add_argument('-b', '--both', type=str, metavar='PROMPT', help='Send a prompt to both OpenAI and Gemini.')

    parser.add_argument('-f', '--file', action='append', help="Attach content from files, directories, or zip archives (can be used multiple times).")
    parser.add_argument('-x', '--exclude', action='append', help="Exclude a file or directory from processing (can be used multiple times).")
    parser.add_argument('-m', '--memory', action='store_true', help='Load the persistent memory file into context.')
    parser.add_argument('-s', '--session-name', type=str, help='Provide a custom name for the chat log file (interactive chat only).')
    parser.add_argument('-p', '--prompt', type=str, help='Provide a prompt for single-shot chat or image mode.')
    parser.add_argument('--stream', action='store_true', default=settings['stream'], help=f"Enable streaming for chat responses (default: {settings['stream']}).")
    parser.add_argument('--no-stream', action='store_false', dest='stream', help='Disable streaming for chat responses.')
    parser.add_argument('--max-tokens', type=int, default=settings['default_max_tokens'], help=f"Set the maximum number of tokens to generate (default: {settings['default_max_tokens']}).")

    args = parser.parse_args()

    # --- Argument Validation ---
    prompt = args.both or args.prompt

    # Handle piped input if no other prompt is provided
    if not sys.stdin.isatty() and not prompt:
        prompt = sys.stdin.read().strip()
        if not args.chat and not args.image and not args.both:
            args.chat = True # Default to chat mode for piped input

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
        parser.error("--image mode is only for OpenAI. Cannot be used with '--engine gemini'.")
        args.engine = 'openai' # Force engine for simplicity

    # --- Default Mode Handling ---
    if not args.chat and not args.image and not args.both:
        print("--> No mode specified, defaulting to chat mode.", file=sys.stderr)
        args.chat = True

    # --- Main Logic Dispatch ---
    system_prompt, image_data = utils.process_files(args.file, args.memory, args.exclude)

    try:
        if args.both:
            handlers.handle_both_engines(system_prompt, args.both, image_data, args.max_tokens, args.stream)
        elif args.chat:
            api_key = api_client.check_api_keys(args.engine)
            engine_instance = get_engine(args.engine, api_key)
            model_key = 'default_openai_chat_model' if args.engine == 'openai' else 'default_gemini_model'
            model = settings[model_key]
            # In single-shot mode, we don't ask the user to select a model.
            # In interactive mode (no prompt), handle_chat will trigger model selection if needed.
            handlers.handle_chat(engine_instance, model, system_prompt, prompt, image_data, args.session_name, args.max_tokens, args.stream, args.memory)
        elif args.image:
            api_key = api_client.check_api_keys(args.engine)
            engine_instance = get_engine(args.engine, api_key) # For model selection
            image_prompt = prompt or system_prompt
            handlers.handle_image_generation(api_key, engine_instance, image_prompt)

    except api_client.MissingApiKeyError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

