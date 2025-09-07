# aicli/managers/multichat_manager.py
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

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

"""
This module contains the MultiChatSession, which manages the state and logic
for an interactive multi-chat session.
"""

import json
import queue
import threading
from pathlib import Path

from .. import api_client
from .. import settings as app_settings
from ..logger import log
from ..prompts import CONTINUATION_PROMPT
from ..session_state import MultiChatSessionState
from ..utils.formatters import (
    ASSISTANT_PROMPT,
    RESET_COLOR,
    SYSTEM_MSG,
    clean_ai_response_text,
)
from ..utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
    translate_history,
)


class MultiChatSession:
    """Manages the state and logic for an interactive multi-chat session."""

    def __init__(
        self,
        initial_state: MultiChatSessionState,
    ):
        self.state = initial_state
        self.primary_engine_name = app_settings.settings["default_engine"]
        self.engines = {
            "openai": self.state.openai_engine,
            "gemini": self.state.gemini_engine,
        }
        self.models = {
            "openai": self.state.openai_model,
            "gemini": self.state.gemini_model,
        }
        self.engine_aliases = {
            "gpt": "openai",
            "openai": "openai",
            "gem": "gemini",
            "gemini": "gemini",
        }

    def _secondary_worker(self, engine, model, history, system_prompt, result_queue):
        srl_list = self.state.session_raw_logs if self.state.debug_active else None
        try:
            text, _ = api_client.perform_chat_request(
                engine,
                model,
                history,
                system_prompt,
                self.state.max_tokens,
                stream=True,
                print_stream=False,
                session_raw_logs=srl_list,
            )
            cleaned = clean_ai_response_text(engine.name, text)
            result_queue.put({"engine_name": engine.name, "text": cleaned})
        except Exception as e:
            log.error("Secondary worker failed: %s", e)
            result_queue.put({"engine_name": engine.name, "text": f"Error: {e}"})

    def process_turn(
        self, prompt_text: str, log_filepath: Path, is_first_turn: bool = False
    ) -> None:
        srl_list = self.state.session_raw_logs if self.state.debug_active else None
        if prompt_text.lower().lstrip().startswith("/ai"):
            parts = prompt_text.lstrip().split(" ", 2)
            if len(parts) < 2 or parts[1].lower() not in self.engine_aliases:
                print(f"{SYSTEM_MSG}--> Usage: /ai <gpt|gem> [prompt]{RESET_COLOR}")
                return
            target_engine_name = self.engine_aliases[parts[1].lower()]
            target_prompt = parts[2] if len(parts) > 2 else CONTINUATION_PROMPT
            user_msg_text = (
                f"Director to {target_engine_name.capitalize()}: {target_prompt}"
            )
            print(f"\n{SYSTEM_MSG}{user_msg_text}{RESET_COLOR}")
            user_msg = construct_user_message(
                target_engine_name,
                user_msg_text,
                self.state.initial_image_data if is_first_turn else [],
            )
            current_history = translate_history(
                self.state.shared_history + [user_msg], target_engine_name
            )
            engine, model = (
                self.engines[target_engine_name],
                self.models[target_engine_name],
            )
            print(
                f"\n{ASSISTANT_PROMPT}[{engine.name.capitalize()}]: {RESET_COLOR}",
                end="",
                flush=True,
            )
            raw_response, _ = api_client.perform_chat_request(
                engine,
                model,
                current_history,
                self.state.system_prompts[target_engine_name],
                self.state.max_tokens,
                stream=True,
                session_raw_logs=srl_list,
            )
            print()
            cleaned = clean_ai_response_text(engine.name, raw_response)
            asst_msg_text = f"[{engine.name.capitalize()}]: {cleaned}"
            asst_msg = construct_assistant_message("openai", asst_msg_text)
            self.state.shared_history.extend(
                [construct_user_message("openai", user_msg_text, []), asst_msg]
            )
            self._log_multichat_turn(log_filepath, self.state.shared_history[-2:])
        else:
            primary_engine = self.engines[self.primary_engine_name]
            secondary_engine = self.engines[
                "gemini" if self.primary_engine_name == "openai" else "openai"
            ]
            user_msg_text = f"Director to All: {prompt_text}"
            user_msg = construct_user_message(
                "openai",
                user_msg_text,
                self.state.initial_image_data if is_first_turn else [],
            )
            result_queue = queue.Queue()
            history_primary = translate_history(
                self.state.shared_history + [user_msg], primary_engine.name
            )
            history_secondary = translate_history(
                self.state.shared_history + [user_msg], secondary_engine.name
            )

            thread = threading.Thread(
                target=self._secondary_worker,
                args=(
                    secondary_engine,
                    self.models[secondary_engine.name],
                    history_secondary,
                    self.state.system_prompts[secondary_engine.name],
                    result_queue,
                ),
            )
            thread.start()

            print(
                f"\n{ASSISTANT_PROMPT}[{primary_engine.name.capitalize()}]: {RESET_COLOR}",
                end="",
                flush=True,
            )
            primary_raw, _ = api_client.perform_chat_request(
                primary_engine,
                self.models[primary_engine.name],
                history_primary,
                self.state.system_prompts[primary_engine.name],
                self.state.max_tokens,
                stream=True,
                session_raw_logs=srl_list,
            )
            print("\n")
            thread.join()
            secondary_result = result_queue.get()
            print(
                f"{ASSISTANT_PROMPT}[{secondary_result['engine_name'].capitalize()}]: {RESET_COLOR}{secondary_result['text']}"
            )

            primary_cleaned = clean_ai_response_text(primary_engine.name, primary_raw)
            primary_msg = construct_assistant_message(
                "openai", f"[{primary_engine.name.capitalize()}]: {primary_cleaned}"
            )
            secondary_msg = construct_assistant_message(
                "openai",
                f"[{secondary_result['engine_name'].capitalize()}]: {secondary_result['text']}",
            )

            first, second = (
                (primary_msg, secondary_msg)
                if self.primary_engine_name == "openai"
                else (secondary_msg, primary_msg)
            )
            self.state.shared_history.extend([user_msg, first, second])
            self._log_multichat_turn(log_filepath, self.state.shared_history[-3:])

    def _log_multichat_turn(
        self, log_filepath: Path, history_slice: list[dict]
    ) -> None:
        try:
            with open(log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps({"history_slice": history_slice}) + "\n")
        except OSError as e:
            log.warning("Could not write to multi-chat session log file: %s", e)
