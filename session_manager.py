# aicli/session_manager.py

import os
import sys
import json
import datetime
from dataclasses import dataclass, field, asdict

import config
import api_client
import utils
import handlers
import settings as app_settings
from engine import AIEngine
from logger import log

@dataclass
class SessionState:
    """A dataclass to hold the state of an interactive chat session."""
    engine: AIEngine
    model: str
    system_prompt: str | None
    max_tokens: int | None
    memory_enabled: bool
    initial_image_data: list = field(default_factory=list)
    history: list = field(default_factory=list)
    debug_active: bool = False
    stream_active: bool = True
    session_raw_logs: list = field(default_factory=list)

def _condense_chat_history(state: SessionState):
    """Summarizes the oldest turns and replaces them with a summary message."""
    print(f"\n{utils.SYSTEM_MSG}--> Condensing conversation history to preserve memory...{utils.RESET_COLOR}")
    num_messages_to_trim = config.HISTORY_SUMMARY_TRIM_TURNS * 2
    turns_to_summarize = state.history[:num_messages_to_trim]
    remaining_history = state.history[num_messages_to_trim:]

    log_content = "\n".join([f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}" for msg in turns_to_summarize])
    summary_prompt = (
        "Concisely summarize the key facts and takeaways from the following conversation excerpt in the third person. "
        "This summary will be used as context for the rest of the conversation.\n\n"
        f"--- EXCERPT ---\n{log_content}\n---"
    )

    helper_model_key = 'helper_model_openai' if state.engine.name == 'openai' else 'helper_model_gemini'
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(state.engine.name, summary_prompt, [])]

    summary_text, _ = api_client.perform_chat_request(
        engine=state.engine, model=task_model, messages_or_contents=messages,
        system_prompt=None, max_tokens=config.SUMMARY_MAX_TOKENS, stream=False
    )
    if not summary_text:
        log.warning("History summarization failed. Proceeding with full history.")
        return

    summary_message = utils.construct_user_message(state.engine.name, f"[PREVIOUSLY DISCUSSED]:\n{summary_text.strip()}", [])
    state.history = [summary_message] + remaining_history
    print(f"{utils.SYSTEM_MSG}--> History condensed successfully.{utils.RESET_COLOR}")

def _handle_slash_command(user_input: str, state: SessionState) -> bool:
    """Handles in-app slash commands. Returns True if the session should end."""
    parts = user_input.lower().strip().split()
    command = parts[0]
    args = parts[1:]

    if command == '/exit':
        return True

    elif command == '/stream':
        state.stream_active = not state.stream_active
        status = "ENABLED" if state.stream_active else "DISABLED"
        print(f"{utils.SYSTEM_MSG}--> Response streaming is now {status} for this session.{utils.RESET_COLOR}")

    elif command == '/debug':
        state.debug_active = not state.debug_active
        status = "ENABLED" if state.debug_active else "DISABLED"
        print(f"{utils.SYSTEM_MSG}--> Session-specific debug logging is now {status}.{utils.RESET_COLOR}")

    elif command == '/memory':
        state.memory_enabled = not state.memory_enabled
        status = "ENABLED" if state.memory_enabled else "DISABLED"
        print(f"{utils.SYSTEM_MSG}--> Persistent memory is now {status} for saving at the end of this session.{utils.RESET_COLOR}")
        
    elif command == '/max-tokens':
        if args and args[0].isdigit():
            state.max_tokens = int(args[0])
            print(f"{utils.SYSTEM_MSG}--> Max tokens for this session set to: {state.max_tokens}.{utils.RESET_COLOR}")
        else:
            print(f"{utils.SYSTEM_MSG}--> Usage: /max-tokens <number>{utils.RESET_COLOR}")

    elif command == '/clear':
        confirm = input("This will clear all conversation history. Type `proceed` to confirm: ")
        if confirm.lower() == 'proceed':
            state.history.clear()
            print(f"{utils.SYSTEM_MSG}--> Conversation history has been cleared.{utils.RESET_COLOR}")
        else:
            print(f"{utils.SYSTEM_MSG}--> Clear cancelled.{utils.RESET_COLOR}")

    elif command == '/model':
        if args:
            state.model = args[0]
            print(f"{utils.SYSTEM_MSG}--> Model temporarily set to: {state.model}{utils.RESET_COLOR}")
        else:
            state.model = handlers.select_model(state.engine, 'chat')
            print(f"{utils.SYSTEM_MSG}--> Model changed to: {state.model}{utils.RESET_COLOR}")

    elif command == '/engine':
        new_engine_name = args[0] if args else ('gemini' if state.engine.name == 'openai' else 'openai')
        if new_engine_name not in ['openai', 'gemini']:
            print(f"{utils.SYSTEM_MSG}--> Unknown engine: {new_engine_name}. Use 'openai' or 'gemini'.{utils.RESET_COLOR}")
            return False
        try:
            from engine import get_engine
            new_api_key = api_client.check_api_keys(new_engine_name)
            state.history = utils.translate_history(state.history, new_engine_name)
            state.engine = get_engine(new_engine_name, new_api_key)
            model_key = 'default_openai_chat_model' if new_engine_name == 'openai' else 'default_gemini_model'
            state.model = app_settings.settings[model_key]
            print(f"{utils.SYSTEM_MSG}--> Engine switched to {state.engine.name.capitalize()}. Model set to default: {state.model}. History translated.{utils.RESET_COLOR}")
        except api_client.MissingApiKeyError:
             print(f"{utils.SYSTEM_MSG}--> Switch to {new_engine_name.capitalize()} failed: API key not found.{utils.RESET_COLOR}")

    elif command == '/history':
        print(json.dumps(state.history, indent=2))

    elif command == '/state':
        state_dict = asdict(state)
        state_dict['engine'] = state.engine.name
        state_dict.pop('session_raw_logs', None)
        state_dict.pop('history', None)
        print(json.dumps(state_dict, indent=2))

    elif command == '/set':
        if len(args) == 2:
            app_settings.save_setting(args[0], args[1])
        else:
            print(f"{utils.SYSTEM_MSG}--> Usage: /set <setting_key> <value>{utils.RESET_COLOR}")

    else:
        print(f"{utils.SYSTEM_MSG}--> Unknown command: {command}. Available: /exit, /stream, /debug, /memory, /clear, /model, /engine, /max-tokens, /history, /state, /set{utils.RESET_COLOR}")

    return False

def perform_interactive_chat(initial_state: SessionState, session_name: str):
    """Manages the main loop for an interactive chat session."""
    log_filename_base = session_name or f"chat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{initial_state.engine.name}"
    log_filename = os.path.join(config.LOG_DIRECTORY, f"{log_filename_base}.jsonl")

    print(f"Starting interactive chat with {initial_state.engine.name.capitalize()} ({initial_state.model}).")
    print("Type '/exit' or press Ctrl+C to end. Use '/set <key> <value>' to change default settings.")
    print(f"Session log will be saved to: {log_filename}")

    if initial_state.system_prompt: print("System prompt is active.")
    if initial_state.initial_image_data: print(f"Attached {len(initial_state.initial_image_data)} image(s) to this session.")
    if initial_state.memory_enabled: print("Persistent memory is enabled for this session.")


    first_turn = True
    try:
        while True:
            user_input = input(f"\n{utils.USER_PROMPT}You: {utils.RESET_COLOR}")
            if not user_input.strip(): continue

            if user_input.startswith('/'):
                if _handle_slash_command(user_input, initial_state):
                    break
                continue

            user_msg = utils.construct_user_message(initial_state.engine.name, user_input, initial_state.initial_image_data if first_turn else [])
            messages_or_contents = list(initial_state.history)
            messages_or_contents.append(user_msg)
            srl_list = initial_state.session_raw_logs if initial_state.debug_active else None

            if initial_state.stream_active:
                print(f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}", end='', flush=True)

            response_text, token_dict = api_client.perform_chat_request(
                engine=initial_state.engine, model=initial_state.model,
                messages_or_contents=messages_or_contents, system_prompt=initial_state.system_prompt,
                max_tokens=initial_state.max_tokens, stream=initial_state.stream_active, session_raw_logs=srl_list
            )
            if not initial_state.stream_active:
                print(f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{response_text}", end='')

            print(utils.format_token_string(token_dict))
            asst_msg = utils.construct_assistant_message(initial_state.engine.name, response_text)
            initial_state.history.extend([user_msg, asst_msg])

            if len(initial_state.history) >= config.HISTORY_SUMMARY_THRESHOLD_TURNS * 2:
                _condense_chat_history(initial_state)

            try:
                log_entry = {"timestamp": datetime.datetime.now().isoformat(), "model": initial_state.model, "prompt": user_msg, "response": asst_msg, "tokens": token_dict}
                with open(log_filename, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except IOError as e:
                log.warning("Could not write to session log file: %s", e)
            first_turn = False
    except (KeyboardInterrupt, EOFError):
        print("\nSession interrupted by user.")
    finally:
        print("\nSession ended.")
        if os.path.exists(log_filename) and initial_state.history:
            if initial_state.memory_enabled:
                _update_persistent_memory(initial_state.engine, initial_state.model, initial_state.history)
            else:
                print(f"{utils.SYSTEM_MSG}--> Persistent memory not enabled, skipping update.{utils.RESET_COLOR}")

            if not session_name:
                rename_session_log(initial_state.engine, initial_state.history, log_filename)

        if initial_state.debug_active:
            debug_filename = f"debug_{os.path.splitext(log_filename_base)[0]}.jsonl"
            debug_filepath = os.path.join(config.LOG_DIRECTORY, debug_filename)
            print(f"Saving debug log to: {debug_filepath}")
            try:
                with open(debug_filepath, 'w', encoding='utf-8') as f:
                    for entry in initial_state.session_raw_logs:
                        f.write(json.dumps(entry) + '\n')
            except IOError as e:
                log.warning("Could not write to debug log file: %s", e)

def _update_persistent_memory(engine: AIEngine, model: str, history: list):
    """Integrates the session history into the persistent memory file in a single step."""
    try:
        print(f"{utils.SYSTEM_MSG}--> Integrating session into persistent memory...{utils.RESET_COLOR}")
        session_content = "\n".join([f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}" for msg in history])
        
        existing_ltm = ""
        if os.path.exists(config.PERSISTENT_MEMORY_FILE):
            with open(config.PERSISTENT_MEMORY_FILE, 'r', encoding='utf-8') as f:
                existing_ltm = f.read()

        integration_prompt = (
            "You are a memory consolidation agent. Integrate the key facts, topics, and outcomes from the new chat session "
            "into the existing persistent memory. Combine related topics, update existing facts, and discard trivial data to "
            "keep the memory concise and relevant. The goal is a dense, factual summary of all interactions.\n\n"
            f"--- EXISTING PERSISTENT MEMORY ---\n{existing_ltm}\n\n"
            f"--- NEW CHAT SESSION TO INTEGRATE ---\n{session_content}\n\n"
            "--- UPDATED PERSISTENT MEMORY ---"
        )
        
        helper_model_key = 'helper_model_openai' if engine.name == 'openai' else 'helper_model_gemini'
        task_model = app_settings.settings[helper_model_key]
        messages = [utils.construct_user_message(engine.name, integration_prompt, [])]
        
        updated_ltm, _ = api_client.perform_chat_request(engine, task_model, messages, None, config.SUMMARY_MAX_TOKENS, stream=False)
        
        if updated_ltm:
            with open(config.PERSISTENT_MEMORY_FILE, 'w', encoding='utf-8') as f:
                f.write(updated_ltm.strip())
            print(f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}")
        else:
            log.warning("Failed to update persistent memory.")
    except Exception as e:
        log.error("Error during memory update: %s", e)

def rename_session_log(engine: AIEngine, history: list, log_file: str):
    """Generates a descriptive name for the chat log from its history and renames the file."""
    print(f"{utils.SYSTEM_MSG}--> Generating smart name for session log...{utils.RESET_COLOR}")
    try:
        # Use the first 20 turns (40 messages) for context to avoid large prompts
        history_excerpt = history[:40]
        log_content = "\n".join([f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}" for msg in history_excerpt])

        prompt = (
            "Based on the following chat log, generate a concise, descriptive, filename-safe title. "
            "Use snake_case. The title should be 3-5 words. "
            "Do not include any file extension like '.jsonl'. "
            "Example response: 'python_script_debugging_and_refactoring'\n\n"
            f"CHAT LOG EXCERPT:\n---\n{log_content}\n---"
        )
        helper_model_key = 'helper_model_openai' if engine.name == 'openai' else 'helper_model_gemini'
        task_model = app_settings.settings[helper_model_key]
        messages = [utils.construct_user_message(engine.name, prompt, [])]
        
        new_name, _ = api_client.perform_chat_request(engine, task_model, messages, None, 1024, stream=False)
        sanitized_name = utils.sanitize_filename(new_name.strip())
        
        if sanitized_name:
            new_filename_base = f"{sanitized_name}.jsonl"
            new_filepath = os.path.join(config.LOG_DIRECTORY, new_filename_base)
            counter = 1
            while os.path.exists(new_filepath):
                new_filename_base = f"{sanitized_name}_{counter}.jsonl"
                new_filepath = os.path.join(config.LOG_DIRECTORY, new_filename_base)
                counter += 1
            os.rename(log_file, new_filepath)
            print(f"{utils.SYSTEM_MSG}--> Session log saved as: {new_filepath}{utils.RESET_COLOR}")
        else:
            log.warning("Could not generate a valid name. Log remains as: %s", log_file)
    except (IOError, OSError, Exception) as e:
        log.warning("Could not rename session log (%s). Log remains as: %s", e, log_file)
