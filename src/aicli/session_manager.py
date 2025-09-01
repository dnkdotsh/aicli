# aicli/session_manager.py
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


import os
import sys
import json
import datetime
from dataclasses import dataclass, field
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import ANSI

from . import config
from . import api_client
from . import utils
from . import settings as app_settings
from . import personas as persona_manager
from . import commands
from .engine import AIEngine
from .logger import log
from .prompts import HISTORY_SUMMARY_PROMPT

@dataclass
class SessionState:
    """A dataclass to hold the state of an interactive chat session."""
    engine: AIEngine
    model: str
    system_prompt: str | None
    initial_system_prompt: str | None # The original system prompt from startup
    current_persona: persona_manager.Persona | None
    max_tokens: int | None
    memory_enabled: bool
    attachments: dict = field(default_factory=dict) # Maps Path object to file content string
    attached_images: list = field(default_factory=list)
    history: list = field(default_factory=list)
    command_history: list[str] = field(default_factory=list)
    debug_active: bool = False
    stream_active: bool = True
    session_raw_logs: list = field(default_factory=list)
    exit_without_memory: bool = False
    force_quit: bool = False
    custom_log_rename: str | None = None

def _assemble_full_system_prompt(state: SessionState) -> str | None:
    """Combines base system prompt and attachments into a single string for the API."""
    prompt_parts = []
    if state.system_prompt:
        prompt_parts.append(state.system_prompt)

    if state.attachments:
        attachment_texts = []
        for path, content in state.attachments.items():
            attachment_texts.append(f"--- FILE: {path.as_posix()} ---\n{content}")
        prompt_parts.append(f"--- ATTACHED FILES ---\n" + "\n\n".join(attachment_texts))

    return "\n\n".join(prompt_parts) if prompt_parts else None

def _condense_chat_history(state: SessionState):
    """Summarizes the oldest turns and replaces them with a summary message."""
    print(f"\n{utils.SYSTEM_MSG}--> Condensing conversation history to preserve memory...{utils.RESET_COLOR}")
    num_messages_to_trim = config.HISTORY_SUMMARY_TRIM_TURNS * 2
    turns_to_summarize = state.history[:num_messages_to_trim]
    remaining_history = state.history[num_messages_to_trim:]

    log_content = "\n".join([f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}" for msg in turns_to_summarize])
    summary_prompt = HISTORY_SUMMARY_PROMPT.format(log_content=log_content)

    helper_model_key = 'helper_model_openai' if state.engine.name == 'openai' else 'helper_model_gemini'
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(state.engine.name, summary_prompt, [])]

    summary_text, _ = api_client.perform_chat_request(
        engine=state.engine, model=task_model, messages_or_contents=messages,
        system_prompt=None, max_tokens=app_settings.settings['summary_max_tokens'], stream=False
    )
    if not summary_text or summary_text.startswith("API Error:"):
        reason = summary_text or "No response from helper model."
        log.warning("History summarization failed. Proceeding with full history. Reason: %s", reason)
        return

    summary_message = utils.construct_user_message(state.engine.name, f"[PREVIOUSLY DISCUSSED]:\n{summary_text.strip()}", [])
    state.history = [summary_message] + remaining_history
    print(f"{utils.SYSTEM_MSG}--> History condensed successfully.{utils.RESET_COLOR}")

def _handle_slash_command(user_input: str, state: SessionState, cli_history: InMemoryHistory) -> bool:
    """Handles in-app slash commands by dispatching to the commands module."""
    parts = user_input.strip().split()
    command_str = parts[0].lower()
    args = parts[1:]

    handler = commands.COMMAND_MAP.get(command_str)

    if handler:
        # A handler returns True if the session should end.
        return handler(args, state, cli_history) or False
    else:
        print(f"{utils.SYSTEM_MSG}--> Unknown command: {command_str}. Type /help for a list of commands.{utils.RESET_COLOR}")
        return False

def perform_interactive_chat(initial_state: SessionState, session_name: str):
    """Manages the main loop for an interactive chat session."""
    log_filename_base = session_name or f"chat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{initial_state.engine.name}"
    log_filename = os.path.join(config.LOG_DIRECTORY, f"{log_filename_base}.jsonl")

    utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)

    print(f"Starting interactive chat with {initial_state.engine.name.capitalize()} ({initial_state.model}).")
    print("Type '/help' for commands or '/exit' to end.")
    print(f"Session log will be saved to: {log_filename}")

    if initial_state.current_persona: print(f"{utils.SYSTEM_MSG}Active Persona: {initial_state.current_persona.name}.{utils.RESET_COLOR}")
    if initial_state.system_prompt and not initial_state.current_persona: print(f"{utils.SYSTEM_MSG}System prompt is active.{utils.RESET_COLOR}")
    if initial_state.attachments:
        from .commands import _format_bytes
        total_size = sum(p.stat().st_size for p in initial_state.attachments.keys() if p.exists())
        print(f"{utils.SYSTEM_MSG}Attached {len(initial_state.attachments)} text file(s). Total size: {_format_bytes(total_size)}{utils.RESET_COLOR}")
    if initial_state.attached_images: print(f"{utils.SYSTEM_MSG}Attached {len(initial_state.attached_images)} image(s) to this session.{utils.RESET_COLOR}")
    if initial_state.memory_enabled: print(f"{utils.SYSTEM_MSG}Persistent memory is enabled for this session.{utils.RESET_COLOR}")

    cli_history = InMemoryHistory()
    for command in initial_state.command_history:
        cli_history.append_string(command)

    first_turn = not initial_state.history
    large_attachment_confirmed = False

    try:
        if first_turn and initial_state.attachments:
            from .commands import _format_bytes
            total_size = sum(p.stat().st_size for p in initial_state.attachments.keys() if p.exists())
            if total_size > config.LARGE_ATTACHMENT_THRESHOLD_BYTES and not large_attachment_confirmed:
                warning_msg = (
                    f"\n{utils.SYSTEM_MSG}WARNING: Total size of attached files is {_format_bytes(total_size)}.\n"
                    f"This may result in high API token usage and costs.\n"
                    f"Type 'yes' to proceed with the session: {utils.RESET_COLOR}"
                )
                confirmation = prompt(ANSI(warning_msg))
                if confirmation.lower().strip() != 'yes':
                    print(f"{utils.SYSTEM_MSG}--> Session aborted by user.{utils.RESET_COLOR}")
                    return
                large_attachment_confirmed = True

        while True:
            prompt_message = f"\n{utils.USER_PROMPT}You: {utils.RESET_COLOR}"
            user_input = prompt(ANSI(prompt_message), history=cli_history)

            if not user_input.strip():
                sys.stdout.write('\x1b[1A'); sys.stdout.write('\x1b[2K'); sys.stdout.flush()
                continue

            if user_input.startswith('/'):
                sys.stdout.write('\x1b[1A'); sys.stdout.write('\x1b[2K'); sys.stdout.flush()
                if _handle_slash_command(user_input, initial_state, cli_history):
                    break
                continue

            user_msg = utils.construct_user_message(initial_state.engine.name, user_input, initial_state.attached_images if first_turn else [])
            messages_or_contents = list(initial_state.history) + [user_msg]
            srl_list = initial_state.session_raw_logs if initial_state.debug_active else None

            if initial_state.stream_active:
                print(f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}", end='', flush=True)

            full_system_prompt = _assemble_full_system_prompt(initial_state)
            response_text, token_dict = api_client.perform_chat_request(
                engine=initial_state.engine, model=initial_state.model,
                messages_or_contents=messages_or_contents, system_prompt=full_system_prompt,
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
        if initial_state.force_quit or initial_state.exit_without_memory:
            return

        if os.path.exists(log_filename) and initial_state.history:
            if initial_state.memory_enabled:
                commands.consolidate_session_into_memory(initial_state.engine, initial_state.model, initial_state.history)
            else:
                print(f"{utils.SYSTEM_MSG}--> Persistent memory not enabled, skipping update.{utils.RESET_COLOR}")

            if initial_state.custom_log_rename:
                new_filepath = log_filename.replace(os.path.basename(log_filename), f"{utils.sanitize_filename(initial_state.custom_log_rename)}.jsonl")
                try:
                    os.rename(log_filename, new_filepath)
                    print(f"{utils.SYSTEM_MSG}--> Session log renamed to: {new_filepath}{utils.RESET_COLOR}")
                except OSError as e:
                    log.error("Failed to rename session log: %s", e)
            elif not session_name:
                commands.rename_session_log(initial_state.engine, initial_state.history, log_filename)

        if initial_state.debug_active:
            debug_filename = f"debug_{os.path.splitext(log_filename_base)[0]}.jsonl"
            debug_filepath = config.LOG_DIRECTORY / debug_filename
            print(f"Saving debug log to: {debug_filepath}")
            try:
                with open(debug_filepath, 'w', encoding='utf-8') as f:
                    for entry in initial_state.session_raw_logs:
                        f.write(json.dumps(entry) + '\n')
            except IOError as e:
                log.error("Could not save debug log file: %s", e)
