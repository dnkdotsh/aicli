#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Command-Line AI Client
Main entry point for the application.
"""

import argparse
import sys
import os

# Import modules from our project
# --- CHANGE: Import new custom exception ---
import api_client
import handlers
import utils
import config

def main():
    """Parses arguments and orchestrates the application flow."""
    # --- Application Setup ---
    utils.ensure_log_dir_exists()
    utils.ensure_dir_exists(config.IMAGE_DIRECTORY)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Unified Command-Line AI Client for OpenAI and Gemini.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('-e', '--engine', choices=['openai', 'gemini'], default=config.DEFAULT_ENGINE, help=f'Specify the AI provider (default: {config.DEFAULT_ENGINE}). Not used with --both.')

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('-c', '--chat', action='store_true', help='Activate chat mode for text generation.')
    mode_group.add_argument('-i', '--image', action='store_true', help='Activate image generation mode (OpenAI only).')
    mode_group.add_argument('-b', '--both', type=str, metavar='PROMPT', help='Send a prompt to both OpenAI and Gemini. Implies single-shot mode.')
    
    parser.add_argument('-f', '--file', action='append', help="Attach content from files, directories, or zip archives (can be used multiple times).")
    parser.add_argument('-x', '--exclude', action='append', help="Exclude a file or directory from processing (can be used multiple times).")
    parser.add_argument('-m', '--memory', action='store_true', help='Load the persistent memory file into context.')
    parser.add_argument('-s', '--session-name', type=str, help='Provide a custom name for the chat log file (interactive chat only).')
    parser.add_argument('-p', '--prompt', type=str, help='Provide a prompt for single-shot chat or image mode.')
    
    # --- NEW: --stream argument ---
    parser.add_argument('--stream', action='store_true', help='Enable streaming for chat responses.')
    
    parser.add_argument('--max-tokens', type=int, help=f'Set the maximum number of tokens to generate (default for model).')

    args = parser.parse_args()

    # --- Argument Validation and Default Mode Handling ---
    prompt = args.both or args.prompt
    
    if prompt is not None and not prompt.strip():
        parser.error("Error: The provided prompt cannot be empty or contain only whitespace.")

    if args.file:
        for path in args.file:
            if not os.path.exists(path):
                parser.error(f"Error: The file or directory '{path}' does not exist.")

    # --- FIX: Consolidated and corrected engine/mode validation ---
    # Rationale: This replaces multiple, brittle checks (including the faulty sys.argv check)
    # with a single, clear validation block that operates on the parsed arguments.
    if args.image:
        if args.engine != 'openai':
            parser.error("Error: --image mode is only for OpenAI. Cannot be used with '--engine gemini'.")
        # Force engine to openai if image is selected, for simplicity.
        args.engine = 'openai'

    if not args.chat and not args.image and not args.both:
        if len(sys.argv) == 1:
            # --- FIX: Updated interactive prompt to reflect Gemini default ---
            print("Choose an option:\n  1. Gemini Text (chat) [default]\n  2. OpenAI Text (chat)\n  3. OpenAI Image generation")
            choice = input("Enter choice [1]: ")
            if choice == '2': args.engine, args.chat = 'openai', True
            elif choice == '3': args.engine, args.image = 'openai', True
            else: args.engine, args.chat = config.DEFAULT_ENGINE, True # Default is Gemini
        else:
            print("--> No mode specified, defaulting to chat mode.", file=sys.stderr)
            args.chat = True

    if args.both and args.prompt:
        parser.error("Error: Provide a prompt via --both \"PROMPT\" or --prompt \"PROMPT\", but not both.")
    if args.both and ('-e' in sys.argv or '--engine' in sys.argv):
        parser.error("Error: --engine is not applicable with --both mode.")
    if args.both and args.session_name:
        parser.error("Error: --session-name cannot be used with --both mode.")
    
    # --- Main Logic Dispatch ---
    system_prompt, image_data = utils.process_files(args.file, args.memory, args.exclude)

    try:
        if args.both:
            openai_key = api_client.check_api_keys('openai')
            gemini_key = api_client.check_api_keys('gemini')
            handlers.handle_both_engines(openai_key, gemini_key, system_prompt, args.both, image_data, args.max_tokens, args.stream)
        
        elif args.chat:
            api_key = api_client.check_api_keys(args.engine)
            model = handlers.select_model(api_key, args.engine, 'chat')
            handlers.handle_chat(api_key, args.engine, model, system_prompt, prompt, image_data, args.session_name, args.max_tokens, args.stream)
        
        elif args.image:
            api_key = api_client.check_api_keys(args.engine)
            image_prompt = prompt or system_prompt
            handlers.handle_image_generation(api_key, image_prompt)

    # --- CHANGE: Centralized API key error handling ---
    # Rationale: Catches the new custom exception from api_client.py,
    # allowing for a clean exit without crashing the program.
    except api_client.MissingApiKeyError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
