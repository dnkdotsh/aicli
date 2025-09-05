# aicli/workflows.py
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
This module encapsulates complex, multi-step workflows that often involve
their own internal state and multiple calls to helper AI models. This
isolates complex business logic from simple state management.
"""

from __future__ import annotations

import base64
import sys
from typing import TYPE_CHECKING

from . import api_client, prompts, utils
from . import settings as app_settings

if TYPE_CHECKING:
    from .session_manager import SessionManager


# --- AI Helper Workflows ---


def consolidate_memory(session: SessionManager) -> None:
    """Updates persistent memory by consolidating the current session's content."""
    print(
        f"{utils.SYSTEM_MSG}--> Updating persistent memory with session content...{utils.RESET_COLOR}"
    )
    prompt_text = prompts.MEMORY_INTEGRATION_PROMPT.format(
        existing_ltm=session._read_memory_file(),
        session_content=session._get_history_for_helpers(),
    )
    updated_memory, _ = session._perform_helper_request(prompt_text, None)
    if updated_memory:
        session._write_memory_file(updated_memory)
        print(
            f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}"
        )


def inject_memory(session: SessionManager, fact: str) -> None:
    """Injects a new fact directly into persistent memory."""
    print(
        f"{utils.SYSTEM_MSG}--> Injecting fact into persistent memory...{utils.RESET_COLOR}"
    )
    prompt_text = prompts.DIRECT_MEMORY_INJECTION_PROMPT.format(
        existing_ltm=session._read_memory_file(), new_fact=fact
    )
    updated_memory, _ = session._perform_helper_request(prompt_text, None)
    if updated_memory:
        session._write_memory_file(updated_memory)
        print(
            f"{utils.SYSTEM_MSG}--> Persistent memory updated successfully.{utils.RESET_COLOR}"
        )


def rename_log_with_ai(session: SessionManager, log_filepath) -> None:
    """Uses an AI helper to generate a descriptive name for a log file."""
    print(
        f"{utils.SYSTEM_MSG}--> Generating descriptive name for session log...{utils.RESET_COLOR}"
    )
    prompt_text = prompts.LOG_RENAMING_PROMPT.format(
        log_content=session._get_history_for_helpers()
    )
    suggested_name, _ = session._perform_helper_request(
        prompt_text, app_settings.settings["log_rename_max_tokens"]
    )
    if suggested_name:
        session._rename_log_file(log_filepath, suggested_name)


# --- Image Generation Workflow ---


def _perform_image_generation(
    api_key: str,
    model: str,
    prompt: str,
    session_name: str | None = None,
    session_raw_logs: list | None = None,
) -> tuple[bool, str | None]:
    """Core image generation logic, serving as the single source of truth."""
    print(
        f"Generating image with {model} for prompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'..."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    }
    if "dall-e" in model:
        payload["response_format"] = "b64_json"

    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response_data = api_client.make_api_request(
            url, headers, payload, session_raw_logs=session_raw_logs
        )
    except api_client.ApiRequestError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False, None

    b64_data = None
    if response_data and "data" in response_data and response_data["data"]:
        b64_data = response_data["data"][0].get("b64_json")

    if not b64_data:
        print("Error: API response did not contain image data.", file=sys.stderr)
        return False, None

    try:
        image_bytes = base64.b64decode(b64_data)
        filepath = utils.save_image_and_get_path(prompt, image_bytes, session_name)
        print(f"Image saved successfully as: {filepath}")
        utils.log_image_generation(model, prompt, str(filepath), session_name)
        return True, str(filepath)
    except (base64.binascii.Error, OSError) as e:
        print(f"Error saving image: {e}", file=sys.stderr)
        return False, None


class ImageGenerationWorkflow:
    """Manages the state and logic for the interactive image crafting process."""

    def __init__(self, session: SessionManager):
        self.session = session
        # Workflow-specific state
        self.img_prompt: str | None = None
        self.last_img_prompt: str | None = None
        self.img_prompt_crafting: bool = False
        self.pre_image_engine: str | None = None
        self.pre_image_model: str | None = None

    def run(self, args: list[str]) -> None:
        """Main entry point for the /image command."""
        if self.session.state.engine.name != "openai":
            print(
                f"{utils.SYSTEM_MSG}--> Image generation requires OpenAI. Temporarily switching engine...{utils.RESET_COLOR}"
            )
            self.pre_image_engine = self.session.state.engine.name
            self.pre_image_model = self.session.state.model
            self.session.switch_engine("openai")

        send_immediately = "--send-prompt" in args
        if send_immediately:
            args = [arg for arg in args if arg != "--send-prompt"]

        prompt_from_args = " ".join(args) if args else None

        if self.img_prompt_crafting:
            print(
                f"{utils.SYSTEM_MSG}--> You're already in image crafting mode. "
                f"Continue refining your prompt or type 'yes' to generate.{utils.RESET_COLOR}"
            )
            return

        if send_immediately:
            prompt_to_generate = (
                prompt_from_args or self.img_prompt or self.last_img_prompt
            )
            if prompt_to_generate:
                self._generate_image_from_session(prompt_to_generate)
            else:
                print(
                    f"{utils.SYSTEM_MSG}--> No prompt available. Use '/image <desc>' to start.{utils.RESET_COLOR}"
                )
            return

        if prompt_from_args:
            print(
                f"\n{utils.SYSTEM_MSG}--> Starting image prompt crafting...{utils.RESET_COLOR}"
            )
            self._start_prompt_crafting(prompt_from_args)
        elif self.last_img_prompt:
            print(
                f"\n{utils.SYSTEM_MSG}Previous prompt found: '{self.last_img_prompt}'{utils.RESET_COLOR}"
            )
            choice = (
                input("Refine, Regenerate, or start New? (r/g/n): ").lower().strip()
            )
            if choice in ["r", "refine"]:
                self._start_prompt_crafting(self.last_img_prompt)
            elif choice in ["g", "generate"]:
                self._generate_image_from_session(self.last_img_prompt)
            elif choice in ["n", "new"]:
                self._start_prompt_crafting()
            else:
                print(
                    f"{utils.SYSTEM_MSG}--> Invalid choice. Cancelled.{utils.RESET_COLOR}"
                )
        else:
            self._start_prompt_crafting()

    def _start_prompt_crafting(self, initial_prompt: str | None = None) -> None:
        """Transitions the session into image prompt crafting mode."""
        self.img_prompt_crafting = True
        token_limit = app_settings.settings["image_prompt_refinement_max_tokens"]

        if initial_prompt:
            refinement_request = prompts.IMAGE_PROMPT_INITIAL_REFINEMENT.format(
                initial_prompt=initial_prompt
            )
            refined_prompt, _ = self.session._perform_helper_request(
                refinement_request, token_limit
            )

            if refined_prompt and not refined_prompt.startswith("API Error:"):
                self.img_prompt = refined_prompt.strip()
                print(
                    f"\n{utils.ASSISTANT_PROMPT}Image Assistant:{utils.RESET_COLOR} Here's a refined version:\n\n"
                    f"{refined_prompt.strip()}\n\n"
                    "Type 'yes' to generate, or provide changes."
                )
            else:
                self.img_prompt = initial_prompt
                print(
                    f"\n{utils.ASSISTANT_PROMPT}Image Assistant:{utils.RESET_COLOR} Ready with prompt: {initial_prompt}\n"
                    "Type 'yes' to generate, or provide changes."
                )
        else:
            self.img_prompt = ""
            print(
                f"\n{utils.ASSISTANT_PROMPT}Image Assistant:{utils.RESET_COLOR} Describe the image you want to create."
            )

    def process_prompt_input(self, user_input: str) -> tuple[bool, str | None]:
        """Handles user input during image prompt crafting mode."""
        normalized_input = user_input.strip().lower()

        if normalized_input in ["yes", "y", "generate", "go"]:
            if self.img_prompt:
                return True, self.img_prompt
            else:
                print(
                    f"{utils.SYSTEM_MSG}--> No prompt to generate. Please describe an image.{utils.RESET_COLOR}"
                )
                return False, None

        if normalized_input in ["no", "n", "cancel", "stop"]:
            self.img_prompt_crafting = False
            self.img_prompt = None
            print(f"{utils.SYSTEM_MSG}--> Image crafting cancelled.{utils.RESET_COLOR}")
            self._revert_engine_after_crafting()
            return False, None

        token_limit = app_settings.settings["image_prompt_refinement_max_tokens"]
        refinement_request = prompts.IMAGE_PROMPT_SUBSEQUENT_REFINEMENT.format(
            current_prompt=self.img_prompt, user_input=user_input
        )
        refined_prompt, _ = self.session._perform_helper_request(
            refinement_request, token_limit
        )

        if refined_prompt and not refined_prompt.startswith("API Error:"):
            self.img_prompt = refined_prompt.strip()
            print(
                f"\n{utils.ASSISTANT_PROMPT}Image Assistant:{utils.RESET_COLOR} Updated prompt:\n\n{refined_prompt.strip()}\n\n"
                "Type 'yes' to generate, or refine further."
            )
        else:
            self.img_prompt = user_input
            print(
                f"\n{utils.ASSISTANT_PROMPT}Image Assistant:{utils.RESET_COLOR} Updated to: {user_input}\n"
                "Type 'yes' to generate, or refine further."
            )
        return False, None

    def _generate_image_from_session(self, prompt: str) -> bool:
        """Coordinates the image generation process from an active session."""
        print(
            f"\n{utils.SYSTEM_MSG}--> Initiating image generation...{utils.RESET_COLOR}"
        )

        self.last_img_prompt = prompt
        self.img_prompt = None
        self.img_prompt_crafting = False

        current_model = self.session.state.model
        if current_model and "dall-e" in current_model.lower():
            model_to_use = current_model
        else:
            model_to_use = app_settings.settings["default_openai_image_model"]
            print(
                f"{utils.SYSTEM_MSG}--> Using default image model: {model_to_use}{utils.RESET_COLOR}"
            )

        srl = (
            self.session.state.session_raw_logs
            if self.session.state.debug_active
            else None
        )
        session_name = getattr(self.session, "session_name", None)

        success, filepath = _perform_image_generation(
            self.session.state.engine.api_key,
            model_to_use,
            prompt,
            session_name,
            srl,
        )

        if success and filepath:
            user_msg = utils.construct_user_message(
                self.session.state.engine.name, f"Generate image: {prompt}", []
            )
            asst_msg_text = (
                f"I've generated an image and saved it to:\n{filepath}\n\n"
                f"Prompt: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"\n\n"
                "Note: Generated images are not kept in the conversation context to manage token usage."
            )
            asst_msg = utils.construct_assistant_message(
                self.session.state.engine.name, asst_msg_text
            )
            self.session.state.history.extend([user_msg, asst_msg])

        self._revert_engine_after_crafting()
        return success

    def _revert_engine_after_crafting(self) -> None:
        """Reverts to the original engine and model after an image workflow."""
        if self.pre_image_engine and self.pre_image_model:
            print(
                f"\n{utils.SYSTEM_MSG}--> Reverting to original session engine ({self.pre_image_engine})...{utils.RESET_COLOR}"
            )
            original_engine = self.pre_image_engine
            original_model = self.pre_image_model

            self.pre_image_engine = None
            self.pre_image_model = None

            self.session.switch_engine(original_engine)
            self.session.set_model(original_model)
