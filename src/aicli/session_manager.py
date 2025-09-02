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


import datetime
import json
import os
import queue
import sys
import threading
from dataclasses import dataclass, field

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory

from . import api_client, config, utils
from . import personas as persona_manager
from . import settings as app_settings
from .engine import AIEngine
from .logger import log
from .prompts import CONTINUATION_PROMPT, HISTORY_SUMMARY_PROMPT

# --- Single-Chat Session State and Logic ---


@dataclass
class SessionState:
    """A dataclass to hold the state of an interactive chat session."""

    engine: AIEngine
    model: str
    system_prompt: str | None
    initial_system_prompt: str | None  # The original system prompt from startup
    current_persona: persona_manager.Persona | None
    max_tokens: int | None
    memory_enabled: bool
    attachments: dict = field(
        default_factory=dict
    )  # Maps Path object to file content string
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
        prompt_parts.append("--- ATTACHED FILES ---\n" + "\n\n".join(attachment_texts))

    return "\n\n".join(prompt_parts) if prompt_parts else None


def _condense_chat_history(state: SessionState):
    """Summarizes the oldest turns and replaces them with a summary message."""
    print(
        f"\n{utils.SYSTEM_MSG}--> Condensing conversation history to preserve memory...{utils.RESET_COLOR}"
    )
    num_messages_to_trim = config.HISTORY_SUMMARY_TRIM_TURNS * 2
    turns_to_summarize = state.history[:num_messages_to_trim]
    remaining_history = state.history[num_messages_to_trim:]

    log_content = "\n".join(
        [
            f"{msg.get('role', 'unknown')}: {utils.extract_text_from_message(msg)}"
            for msg in turns_to_summarize
        ]
    )
    summary_prompt = HISTORY_SUMMARY_PROMPT.format(log_content=log_content)

    helper_model_key = (
        "helper_model_openai"
        if state.engine.name == "openai"
        else "helper_model_gemini"
    )
    task_model = app_settings.settings[helper_model_key]
    messages = [utils.construct_user_message(state.engine.name, summary_prompt, [])]

    summary_text, _ = api_client.perform_chat_request(
        engine=state.engine,
        model=task_model,
        messages_or_contents=messages,
        system_prompt=None,
        max_tokens=app_settings.settings["summary_max_tokens"],
        stream=False,
    )
    if not summary_text or summary_text.startswith("API Error:"):
        reason = summary_text or "No response from helper model."
        log.warning(
            "History summarization failed. Proceeding with full history. Reason: %s",
            reason,
        )
        return

    summary_message = utils.construct_user_message(
        state.engine.name, f"[PREVIOUSLY DISCUSSED]:\n{summary_text.strip()}", []
    )
    state.history = [summary_message] + remaining_history
    print(f"{utils.SYSTEM_MSG}--> History condensed successfully.{utils.RESET_COLOR}")


def _handle_slash_command(
    user_input: str, state: SessionState, cli_history: InMemoryHistory
) -> bool:
    """Handles in-app slash commands by dispatching to the commands module."""
    # Local import to break circular dependency: session_manager -> commands -> ... -> session_manager
    from . import commands

    parts = user_input.strip().split()
    command_str = parts[0].lower()
    args = parts[1:]

    handler = commands.COMMAND_MAP.get(command_str)

    if handler:
        # A handler returns True if the session should end.
        return handler(args, state, cli_history) or False

    print(
        f"{utils.SYSTEM_MSG}--> Unknown command: {command_str}. Type /help for a list of commands.{utils.RESET_COLOR}"
    )
    return False


def perform_interactive_chat(initial_state: SessionState, session_name: str):
    """Manages the main loop for an interactive chat session."""
    # Local import to break circular dependency for functions used in 'finally' block
    from . import commands

    log_filename_base = (
        session_name
        or f"chat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{initial_state.engine.name}"
    )
    log_filename = os.path.join(config.LOG_DIRECTORY, f"{log_filename_base}.jsonl")

    utils.ensure_dir_exists(config.SESSIONS_DIRECTORY)

    print(
        f"Starting interactive chat with {initial_state.engine.name.capitalize()} ({initial_state.model})."
    )
    print("Type '/help' for commands or '/exit' to end.")
    print(f"Session log will be saved to: {log_filename}")

    if initial_state.current_persona:
        print(
            f"{utils.SYSTEM_MSG}Active Persona: {initial_state.current_persona.name}.{utils.RESET_COLOR}"
        )
    if initial_state.system_prompt and not initial_state.current_persona:
        print(f"{utils.SYSTEM_MSG}System prompt is active.{utils.RESET_COLOR}")
    if initial_state.attachments:
        total_size = sum(
            p.stat().st_size for p in initial_state.attachments if p.exists()
        )
        print(
            f"{utils.SYSTEM_MSG}Attached {len(initial_state.attachments)} text file(s). Total size: {commands._format_bytes(total_size)}{utils.RESET_COLOR}"
        )
    if initial_state.attached_images:
        print(
            f"{utils.SYSTEM_MSG}Attached {len(initial_state.attached_images)} image(s) to this session.{utils.RESET_COLOR}"
        )
    if initial_state.memory_enabled:
        print(
            f"{utils.SYSTEM_MSG}Persistent memory is enabled for this session.{utils.RESET_COLOR}"
        )

    cli_history = InMemoryHistory()
    for command in initial_state.command_history:
        cli_history.append_string(command)

    first_turn = not initial_state.history
    large_attachment_confirmed = False

    should_exit = False
    try:
        if first_turn and initial_state.attachments:
            total_size = sum(
                p.stat().st_size for p in initial_state.attachments if p.exists()
            )
            if (
                total_size > config.LARGE_ATTACHMENT_THRESHOLD_BYTES
                and not large_attachment_confirmed
            ):
                warning_msg = (
                    f"\n{utils.SYSTEM_MSG}WARNING: Total size of attached files is {commands._format_bytes(total_size)}.\n"
                    f"This may result in high API token usage and costs.\n"
                    f"Type 'yes' to proceed with the session: {utils.RESET_COLOR}"
                )
                confirmation = prompt(ANSI(warning_msg))
                if confirmation.lower().strip() != "yes":
                    print(
                        f"{utils.SYSTEM_MSG}--> Session aborted by user.{utils.RESET_COLOR}"
                    )
                    should_exit = True
                large_attachment_confirmed = True

        while not should_exit:
            prompt_message = f"\n{utils.USER_PROMPT}You: {utils.RESET_COLOR}"
            user_input = prompt(ANSI(prompt_message), history=cli_history)

            if not user_input.strip():
                sys.stdout.write("\x1b[1A")
                sys.stdout.write("\x1b[2K")
                sys.stdout.flush()
                continue

            if user_input.startswith("/"):
                sys.stdout.write("\x1b[1A")
                sys.stdout.write("\x1b[2K")
                sys.stdout.flush()
                if _handle_slash_command(user_input, initial_state, cli_history):
                    break
                continue

            user_msg = utils.construct_user_message(
                initial_state.engine.name,
                user_input,
                initial_state.attached_images if first_turn else [],
            )
            messages_or_contents = list(initial_state.history) + [user_msg]
            srl_list = (
                initial_state.session_raw_logs if initial_state.debug_active else None
            )

            if initial_state.stream_active:
                print(
                    f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}",
                    end="",
                    flush=True,
                )

            full_system_prompt = _assemble_full_system_prompt(initial_state)
            response_text, token_dict = api_client.perform_chat_request(
                engine=initial_state.engine,
                model=initial_state.model,
                messages_or_contents=messages_or_contents,
                system_prompt=full_system_prompt,
                max_tokens=initial_state.max_tokens,
                stream=initial_state.stream_active,
                session_raw_logs=srl_list,
            )
            if not initial_state.stream_active:
                print(
                    f"\n{utils.ASSISTANT_PROMPT}Assistant: {utils.RESET_COLOR}{response_text}",
                    end="",
                )

            print(utils.format_token_string(token_dict))
            asst_msg = utils.construct_assistant_message(
                initial_state.engine.name, response_text
            )
            initial_state.history.extend([user_msg, asst_msg])

            if len(initial_state.history) >= config.HISTORY_SUMMARY_THRESHOLD_TURNS * 2:
                _condense_chat_history(initial_state)

            try:
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": initial_state.model,
                    "prompt": user_msg,
                    "response": asst_msg,
                    "tokens": token_dict,
                }
                with open(log_filename, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except OSError as e:
                log.warning("Could not write to session log file: %s", e)
            first_turn = False
    except (KeyboardInterrupt, EOFError):
        print("\nSession interrupted by user.")
    finally:
        print("\nSession ended.")
        if not initial_state.force_quit and not initial_state.exit_without_memory:
            if os.path.exists(log_filename) and initial_state.history:
                if initial_state.memory_enabled:
                    commands.consolidate_session_into_memory(
                        initial_state.engine, initial_state.model, initial_state.history
                    )
                else:
                    print(
                        f"{utils.SYSTEM_MSG}--> Persistent memory not enabled, skipping update.{utils.RESET_COLOR}"
                    )

                if initial_state.custom_log_rename:
                    new_filepath = log_filename.replace(
                        os.path.basename(log_filename),
                        f"{utils.sanitize_filename(initial_state.custom_log_rename)}.jsonl",
                    )
                    try:
                        os.rename(log_filename, new_filepath)
                        print(
                            f"{utils.SYSTEM_MSG}--> Session log renamed to: {new_filepath}{utils.RESET_COLOR}"
                        )
                    except OSError as e:
                        log.error("Failed to rename session log: %s", e)
                elif not session_name:
                    commands.rename_session_log(
                        initial_state.engine, initial_state.history, log_filename
                    )

            if initial_state.debug_active:
                debug_filename = f"debug_{os.path.splitext(log_filename_base)[0]}.jsonl"
                debug_filepath = config.LOG_DIRECTORY / debug_filename
                print(f"Saving debug log to: {debug_filepath}")
                try:
                    with open(debug_filepath, "w", encoding="utf-8") as f:
                        for entry in initial_state.session_raw_logs:
                            f.write(json.dumps(entry) + "\n")
                except OSError as e:
                    log.error("Could not save debug log file: %s", e)


# --- Multi-Chat Session State and Logic ---


@dataclass
class MultiChatSessionState:
    """A dataclass to hold the state of an interactive multi-chat session."""

    openai_engine: AIEngine
    gemini_engine: AIEngine
    openai_model: str
    gemini_model: str
    max_tokens: int
    system_prompts: dict = field(default_factory=dict)
    initial_image_data: list = field(default_factory=list)
    shared_history: list = field(default_factory=list)
    command_history: list[str] = field(default_factory=list)
    debug_active: bool = False
    session_raw_logs: list = field(default_factory=list)
    exit_without_memory: bool = False
    force_quit: bool = False
    custom_log_rename: str | None = None


def _secondary_worker(
    engine, model, history, system_prompt, max_tokens, result_queue, srl_list
):
    """A worker function to be run in a thread for the secondary model."""
    try:
        response_text, _ = api_client.perform_chat_request(
            engine,
            model,
            history,
            system_prompt,
            max_tokens,
            stream=True,
            print_stream=False,
            session_raw_logs=srl_list,
        )
        cleaned_response_text = utils.clean_ai_response_text(engine.name, response_text)
        result_queue.put(
            {"engine_name": engine.name, "cleaned_text": cleaned_response_text}
        )
    except Exception as e:
        log.error("Secondary model thread failed: %s", e)
        result_queue.put(
            {
                "engine_name": engine.name,
                "cleaned_text": f"Error: Could not get response from {engine.name}.",
            }
        )


def _handle_multichat_slash_command(
    user_input: str, state: MultiChatSessionState, cli_history: InMemoryHistory
) -> bool:
    """Handles in-app slash commands for multi-chat mode."""
    # Local import to break circular dependency
    from . import commands

    parts = user_input.strip().split()
    command_str = parts[0].lower()
    args = parts[1:]

    # The /ai command is a special case handled directly in the loop.
    if command_str == "/ai":
        return False

    handler = commands.MULTICHAT_COMMAND_MAP.get(command_str)
    if handler:
        return handler(args, state, cli_history) or False

    print(
        f"{utils.SYSTEM_MSG}--> Unknown command: {command_str}. Type /help for commands.{utils.RESET_COLOR}"
    )
    return False


def perform_multichat_session(initial_state: MultiChatSessionState, session_name: str):
    """Manages the main loop for an interactive multi-chat session."""
    log_filename_base = (
        session_name or f"multichat_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_filename = os.path.join(config.LOG_DIRECTORY, f"{log_filename_base}.jsonl")

    primary_engine_name = app_settings.settings["default_engine"]
    engines = {
        "openai": initial_state.openai_engine,
        "gemini": initial_state.gemini_engine,
    }
    models = {
        "openai": initial_state.openai_model,
        "gemini": initial_state.gemini_model,
    }
    engine_aliases = {
        "gpt": "openai",
        "openai": "openai",
        "gem": "gemini",
        "gemini": "gemini",
        "google": "gemini",
    }
    primary_engine = engines[primary_engine_name]
    secondary_engine = engines[
        "gemini" if primary_engine_name == "openai" else "openai"
    ]

    print(
        f"Starting interactive multi-chat. Primary engine: {primary_engine.name.capitalize()}."
    )
    print(f"Session log will be saved to: {log_filename}")
    print("Type /help for commands or /exit' to end.")

    turn_counter = 0
    cli_history = InMemoryHistory()
    for command in initial_state.command_history:
        cli_history.append_string(command)

    def process_turn(prompt_text):
        nonlocal turn_counter
        turn_counter += 1
        srl_list = (
            initial_state.session_raw_logs if initial_state.debug_active else None
        )

        parts = prompt_text.strip().split(" ", 2)
        if prompt_text.lower().strip().startswith("/ai") and (
            len(parts) < 2 or parts[1].lower() not in engine_aliases
        ):
            print(
                f"{utils.SYSTEM_MSG}--> Usage: /ai <gpt|gem> [prompt]{utils.RESET_COLOR}"
            )
            return

        if prompt_text.lower().strip().startswith("/ai"):
            target_engine_name = engine_aliases[parts[1].lower()]
            target_prompt = parts[2] if len(parts) > 2 else CONTINUATION_PROMPT
            target_engine = engines[target_engine_name]
            target_model = models[target_engine_name]

            print(
                f"\n{utils.DIRECTOR_PROMPT}Director to {target_engine.name.capitalize()}:{utils.RESET_COLOR} {target_prompt}"
            )

            user_msg_text = (
                f"Director to {target_engine.name.capitalize()}: {target_prompt}"
            )
            user_msg = utils.construct_user_message(
                target_engine.name,
                user_msg_text,
                initial_state.initial_image_data if turn_counter == 1 else [],
            )
            current_history = utils.translate_history(
                initial_state.shared_history + [user_msg], target_engine.name
            )

            print(
                f"\n{utils.ASSISTANT_PROMPT}[{target_engine.name.capitalize()}]: {utils.RESET_COLOR}",
                end="",
                flush=True,
            )
            raw_response, _ = api_client.perform_chat_request(
                target_engine,
                target_model,
                current_history,
                initial_state.system_prompts[target_engine_name],
                initial_state.max_tokens,
                stream=True,
                session_raw_logs=srl_list,
            )
            print()

            cleaned_response_text = utils.clean_ai_response_text(
                target_engine.name, raw_response
            )
            formatted_response_for_history = (
                f"[{target_engine.name.capitalize()}]: {cleaned_response_text}"
            )
            asst_msg = utils.construct_assistant_message(
                "openai", formatted_response_for_history
            )
            initial_state.shared_history.extend(
                [
                    utils.construct_user_message("openai", user_msg_text, []),
                    asst_msg,
                ]
            )
            # Log the two messages from this turn
            try:
                with open(log_filename, "a", encoding="utf-8") as f:
                    history_slice_to_log = initial_state.shared_history[-2:]
                    f.write(
                        json.dumps(
                            {
                                "turn": turn_counter,
                                "history_slice": history_slice_to_log,
                            }
                        )
                        + "\n"
                    )
            except OSError as e:
                log.warning("Could not write to session log file: %s", e)
        else:
            user_msg_text = f"Director to All: {prompt_text}"
            user_msg_for_history = utils.construct_user_message(
                "openai", user_msg_text, []
            )

            result_queue = queue.Queue()
            history_for_primary = utils.translate_history(
                initial_state.shared_history + [user_msg_for_history],
                primary_engine.name,
            )
            history_for_secondary = utils.translate_history(
                initial_state.shared_history + [user_msg_for_history],
                secondary_engine.name,
            )

            secondary_thread = threading.Thread(
                target=_secondary_worker,
                args=(
                    secondary_engine,
                    models[secondary_engine.name],
                    history_for_secondary,
                    initial_state.system_prompts[secondary_engine.name],
                    initial_state.max_tokens,
                    result_queue,
                    srl_list,
                ),
            )
            secondary_thread.start()

            print(
                f"\n{utils.ASSISTANT_PROMPT}[{primary_engine.name.capitalize()}]: {utils.RESET_COLOR}",
                end="",
                flush=True,
            )
            primary_response_streamed, _ = api_client.perform_chat_request(
                primary_engine,
                models[primary_engine.name],
                history_for_primary,
                initial_state.system_prompts[primary_engine.name],
                initial_state.max_tokens,
                stream=True,
                session_raw_logs=srl_list,
            )
            print("\n")

            secondary_thread.join()
            secondary_result = result_queue.get()
            secondary_engine_name = secondary_result["engine_name"]
            secondary_response_cleaned_text = secondary_result["cleaned_text"]
            print(
                f"{utils.ASSISTANT_PROMPT}[{secondary_engine_name.capitalize()}]: {utils.RESET_COLOR}{secondary_response_cleaned_text}"
            )

            formatted_primary = f"[{primary_engine.name.capitalize()}]: {utils.clean_ai_response_text(primary_engine.name, primary_response_streamed)}"
            formatted_secondary = f"[{secondary_engine_name.capitalize()}]: {secondary_response_cleaned_text}"
            primary_msg = utils.construct_assistant_message("openai", formatted_primary)
            secondary_msg = utils.construct_assistant_message(
                "openai", formatted_secondary
            )
            first_msg, second_msg = (
                (primary_msg, secondary_msg)
                if primary_engine_name == "openai"
                else (secondary_msg, primary_msg)
            )
            initial_state.shared_history.extend(
                [user_msg_for_history, first_msg, second_msg]
            )
            # Log the three messages from this turn
            try:
                with open(log_filename, "a", encoding="utf-8") as f:
                    history_slice_to_log = initial_state.shared_history[-3:]
                    f.write(
                        json.dumps(
                            {
                                "turn": turn_counter,
                                "history_slice": history_slice_to_log,
                            }
                        )
                        + "\n"
                    )
            except OSError as e:
                log.warning("Could not write to session log file: %s", e)

    should_exit = False
    try:
        while not should_exit:
            prompt_message = f"\n{utils.DIRECTOR_PROMPT}Director> {utils.RESET_COLOR}"
            user_input = prompt(ANSI(prompt_message), history=cli_history)

            if not user_input.strip():
                sys.stdout.write("\x1b[1A\x1b[2K")
                sys.stdout.flush()
                continue

            if user_input.lstrip().startswith("/"):
                # Clear the raw user input line for a cleaner interface
                sys.stdout.write("\x1b[1A\x1b[2K")
                sys.stdout.flush()

                is_ai_command = user_input.lstrip().lower().startswith("/ai")
                if is_ai_command:
                    process_turn(user_input)
                else:  # It's a meta-command like /help or /exit
                    if _handle_multichat_slash_command(
                        user_input, initial_state, cli_history
                    ):
                        break
            else:
                # This is a regular prompt, so we don't clear the line
                process_turn(user_input)
    except (KeyboardInterrupt, EOFError):
        print("\nSession interrupted.")
    finally:
        print("\nSession ended.")
        if (
            not initial_state.force_quit
            and not initial_state.exit_without_memory
            and os.path.exists(log_filename)
            and initial_state.shared_history
            and initial_state.custom_log_rename
        ):
            new_filepath = log_filename.replace(
                os.path.basename(log_filename),
                f"{utils.sanitize_filename(initial_state.custom_log_rename)}.jsonl",
            )
            try:
                os.rename(log_filename, new_filepath)
                print(
                    f"{utils.SYSTEM_MSG}--> Session log renamed to: {new_filepath}{utils.RESET_COLOR}"
                )
            except OSError as e:
                log.error("Failed to rename session log: %s", e)

        if (
            not initial_state.force_quit
            and not initial_state.exit_without_memory
            and initial_state.debug_active
        ):
            debug_filename = f"debug_{os.path.splitext(log_filename_base)[0]}.jsonl"
            debug_filepath = config.LOG_DIRECTORY / debug_filename
            print(f"Saving debug log to: {debug_filepath}")
            try:
                with open(debug_filepath, "w", encoding="utf-8") as f:
                    for entry in initial_state.session_raw_logs:
                        f.write(json.dumps(entry) + "\n")
            except OSError as e:
                log.error("Could not save debug log file: %s", e)
