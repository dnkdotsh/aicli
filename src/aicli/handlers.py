# aicli/handlers.py
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

import argparse
import base64
import datetime
import json
import sys
from pathlib import Path

from . import api_client, config, utils
from .engine import get_engine
from .prompts import MULTICHAT_SYSTEM_PROMPT_GEMINI, MULTICHAT_SYSTEM_PROMPT_OPENAI
from .session_manager import (
    MultiChatManager,
    MultiChatSessionState,
    SessionManager,
    SingleChatManager,
)
from .settings import settings


def handle_chat(initial_prompt: str | None, args: argparse.Namespace) -> None:
    """Handles both single-shot and interactive chat sessions."""
    config_params = utils.resolve_config_precedence(args)

    session = SessionManager(
        engine_name=config_params["engine_name"],
        model=config_params["model"],
        max_tokens=config_params["max_tokens"],
        stream_active=config_params["stream"],
        memory_enabled=config_params["memory_enabled"],
        debug_active=config_params["debug_enabled"],
        persona=config_params["persona"],
        system_prompt_arg=config_params["system_prompt_arg"],
        files_arg=config_params["files_arg"],
        exclude_arg=config_params["exclude_arg"],
    )

    if initial_prompt:
        # Single-shot mode
        session.handle_single_shot(initial_prompt)
    else:
        # Interactive mode
        chat_manager = SingleChatManager(session, config_params["session_name"])
        chat_manager.run()


def handle_load_session(filepath_str: str) -> None:
    """Loads and starts an interactive session from a file."""
    raw_path = Path(filepath_str).expanduser()
    filepath = (
        raw_path if raw_path.is_absolute() else config.SESSIONS_DIRECTORY / raw_path
    )
    if filepath.suffix != ".json":
        filepath = filepath.with_suffix(".json")

    # Create an empty session manager, then load state into it
    session = SessionManager(engine_name=settings["default_engine"])
    if not session.load(str(filepath)):
        sys.exit(1)

    # Use the loaded file's name as the base for the new session log
    session_name = filepath.stem
    chat_manager = SingleChatManager(session, session_name)
    chat_manager.run()


def handle_multichat_session(
    initial_prompt: str | None, args: argparse.Namespace
) -> None:
    """Sets up and delegates an interactive session with both OpenAI and Gemini."""
    openai_key = api_client.check_api_keys("openai")
    gemini_key = api_client.check_api_keys("gemini")

    _, attachments, image_data = utils.process_files(
        args.file, use_memory=False, exclusions=args.exclude
    )
    attachment_texts = [
        f"--- FILE: {path.as_posix()} ---\n{content}"
        for path, content in attachments.items()
    ]
    attachments_str = "\n\n".join(attachment_texts)

    base_system_prompt = ""
    if args.system_prompt:
        base_system_prompt = utils.read_system_prompt(args.system_prompt)
    if attachments_str:
        base_system_prompt += f"\n\n--- ATTACHED FILES ---\n{attachments_str}"

    final_sys_prompts: dict[str, str] = {}
    if base_system_prompt:
        final_sys_prompts["openai"] = (
            f"{base_system_prompt}\n\n---\n\n{MULTICHAT_SYSTEM_PROMPT_OPENAI}"
        )
        final_sys_prompts["gemini"] = (
            f"{base_system_prompt}\n\n---\n\n{MULTICHAT_SYSTEM_PROMPT_GEMINI}"
        )
    else:
        final_sys_prompts["openai"] = MULTICHAT_SYSTEM_PROMPT_OPENAI
        final_sys_prompts["gemini"] = MULTICHAT_SYSTEM_PROMPT_GEMINI

    initial_state = MultiChatSessionState(
        openai_engine=get_engine("openai", openai_key),
        gemini_engine=get_engine("gemini", gemini_key),
        openai_model=args.model or settings["default_openai_chat_model"],
        gemini_model=args.model or settings["default_gemini_model"],
        max_tokens=args.max_tokens or settings["default_max_tokens"],
        system_prompts=final_sys_prompts,
        initial_image_data=image_data,
        debug_active=args.debug,
    )

    multi_chat_manager = MultiChatManager(
        initial_state, args.session_name, initial_prompt
    )
    multi_chat_manager.run()


def _perform_image_generation(
    api_key: str, engine, model: str, prompt: str, session_name: str | None = None
) -> tuple[bool, str | None]:
    """
    Core image generation logic that can be used by both standalone and session-based generation.

    This function encapsulates all the API interaction details - payload construction,
    request handling, response processing, and file saving. By extracting this logic,
    we ensure consistency between different generation paths and make the code easier
    to maintain.

    Think of this as the "engine" of image generation - it doesn't care where the
    request came from, it just knows how to turn a prompt into an image file.

    Args:
        api_key: The OpenAI API key for authentication
        engine: The AI engine instance (though currently only OpenAI is supported)
        model: The specific model to use for generation (e.g., "dall-e-3")
        prompt: The text description of the image to generate
        session_name: Optional session identifier for file naming

    Returns:
        A tuple of (success: bool, filepath: str | None) where filepath is None if
        generation failed. This allows callers to handle failures gracefully.

    The function follows a clear sequence:
    1. Build the API request payload
    2. Make the API call
    3. Process the response
    4. Save the image file
    5. Log the successful generation

    Each step includes proper error handling to ensure robustness.
    """
    print(
        f"Generating image with {model} for prompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'..."
    )

    # Build the API payload - this structure is consistent regardless of how we're called
    # The payload format is defined by OpenAI's API specification
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,  # Generate one image
        "size": "1024x1024",  # Standard size, could be made configurable
        "response_format": "b64_json",  # Get base64 encoded data for direct saving
    }

    # Set up the API endpoint and headers
    # This is specific to OpenAI's image generation API
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        # Make the API request using the existing pattern from the application
        # This maintains consistency with how the rest of the app makes API calls
        response_data = api_client.make_api_request(url, headers, payload)
    except api_client.ApiRequestError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False, None

    # Process the response and extract the image data
    # The response structure is defined by OpenAI's API
    if (
        response_data
        and "data" in response_data
        and response_data["data"]
        and response_data["data"][0].get("b64_json")
    ):
        b64_data = response_data["data"][0]["b64_json"]

        try:
            # Decode the base64 image data
            image_bytes = base64.b64decode(b64_data)

            # Generate a unique filename
            # If we have a session name, incorporate it for better organization
            # This helps users identify which conversation generated which image
            safe_prompt = utils.sanitize_filename(prompt[:50])
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if session_name:
                # Session-based generation: include session name in filename
                base_filename = (
                    f"session_{session_name}_img_{safe_prompt}_{timestamp}.png"
                )
            else:
                # Standalone generation: simpler filename
                base_filename = f"image_{safe_prompt}_{timestamp}.png"

            filepath = config.IMAGE_DIRECTORY / base_filename

            # Save the image file
            filepath.write_bytes(image_bytes)
            print(f"Image saved successfully as: {filepath}")

            # Log the successful generation for audit/history purposes
            # This creates a searchable record of all generated images
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": model,
                "prompt": prompt,
                "file": str(filepath),
                "session": session_name,  # Track which session generated this
            }
            with open(config.IMAGE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            return True, str(filepath)

        except (base64.binascii.Error, OSError) as e:
            print(f"Error saving image: {e}", file=sys.stderr)
            return False, None
    else:
        print("Error: API response did not contain image data.", file=sys.stderr)
        return False, None


def generate_image_from_session(session: SessionManager, prompt: str) -> bool:
    """
    Generates an image using session context and settings.

    This function bridges the gap between the SessionManager's state and the core
    generation logic. It respects the session's model preferences, handles the
    state transitions, and provides appropriate user feedback.

    Think of this as the "session-aware wrapper" around image generation - it knows
    about the conversation context and manages the state transitions appropriately.

    The function handles several important aspects:
    1. Engine compatibility (ensuring we're using OpenAI)
    2. Model selection (respecting session preferences)
    3. Session naming (for better file organization)
    4. Conversation history (recording what happened)
    5. Token management (explaining why images aren't kept in context)

    This separation of concerns keeps the core generation logic pure while
    allowing session-specific behavior to be handled at this layer.
    """
    # First, we need to ensure we're using OpenAI (currently the only supported engine for images)
    # This is a temporary limitation - future versions might support other image generation APIs
    if session.state.engine.name != "openai":
        print(
            f"{utils.SYSTEM_MSG}--> Switching to OpenAI for image generation...{utils.RESET_COLOR}"
        )
        # We need to temporarily switch to OpenAI for image generation
        # Store the current engine so we can offer to switch back later
        original_engine = session.state.engine.name
        session.switch_engine("openai")
    else:
        original_engine = None

    # Determine which model to use
    # Priority: session's current model (if it's an image model) > default image model
    current_model = session.state.model

    # Check if the current model is suitable for image generation
    # We're conservative here - only use the session model if we're sure it's an image model
    if current_model and (
        "dall-e" in current_model.lower() or "image" in current_model.lower()
    ):
        model_to_use = current_model
    else:
        # Fall back to the configured default image model
        model_to_use = settings["default_openai_image_model"]
        print(
            f"{utils.SYSTEM_MSG}--> Using image model: {model_to_use}{utils.RESET_COLOR}"
        )

    # Get the session name for file naming purposes
    # This helps users identify which conversation generated which image
    # Note: SingleChatManager sets this, but we check if it exists to be safe
    session_name = None
    if hasattr(session, "session_name"):
        session_name = session.session_name

    # Perform the actual generation using our extracted helper function
    # This is where we delegate to the core logic
    success, filepath = _perform_image_generation(
        session.state.engine.api_key,
        session.state.engine,
        model_to_use,
        prompt,
        session_name,
    )

    if success and filepath:
        # Add the generation event to conversation history
        # This creates a record of what happened in the conversation flow
        # We record both the user's request and the assistant's response
        user_msg = utils.construct_user_message(
            session.state.engine.name,
            f"Generate image: {prompt}",
            [],  # No image attachments for this message
        )

        # Craft a helpful assistant response that includes the filepath
        # and explains the token management strategy
        asst_msg = utils.construct_assistant_message(
            session.state.engine.name,
            f"I've successfully generated an image based on your prompt and saved it to:\n"
            f"{filepath}\n\n"
            f"The image was created with the prompt: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"\n\n"
            "Note: Generated images are automatically excluded from the conversation context "
            "to manage token usage. You can always reference them by describing what we created.",
        )

        session.state.history.extend([user_msg, asst_msg])

        # If we temporarily switched engines, offer to switch back
        # This maintains the user's preferred conversation context
        if original_engine and original_engine != "openai":
            print(
                f"\n{utils.SYSTEM_MSG}--> Would you like to switch back to {original_engine.capitalize()}? "
                f"(Type '/engine {original_engine}' if yes){utils.RESET_COLOR}"
            )

        return True

    return False


def handle_image_generation(prompt: str | None, args: argparse.Namespace) -> None:
    """
    Handles standalone OpenAI image generation (via -i command line flag).

    This function now serves as the "command-line interface" to image generation,
    handling argument processing and user interaction, while delegating the actual
    generation to our shared helper function.

    The refactoring maintains all existing functionality:
    - Model selection (from args or interactive)
    - Prompt collection (from args, stdin, or interactive)
    - API interaction (now via the helper)
    - Error handling and exit codes

    By using the extracted helper function, we avoid code duplication and ensure
    that both standalone and session-based generation use identical logic for
    the actual image creation process.
    """
    api_key = api_client.check_api_keys("openai")
    engine = get_engine("openai", api_key)

    # Handle model selection - either from args or interactive selection
    # This preserves the existing behavior where users can specify a model
    # or be prompted to select one
    model = args.model or utils.select_model(engine, "image")

    # Get the prompt - from args, stdin, or interactive input
    # This three-way fallback ensures the command works in various contexts:
    # 1. Direct command line: aicli -i -p "a red car"
    # 2. Piped input: echo "a red car" | aicli -i
    # 3. Interactive: aicli -i (then prompted)
    if not prompt:
        prompt = (
            sys.stdin.read().strip()
            if not sys.stdin.isatty()
            else input("Enter a description for the image: ")
        )

    if not prompt:
        print("Image generation cancelled: No prompt provided.", file=sys.stderr)
        return

    # Use our new helper function for the actual generation
    # Note that we pass None for session_name since this is standalone generation
    success, filepath = _perform_image_generation(
        api_key,
        engine,
        model,
        prompt,
        None,  # No session name for standalone generation
    )

    # The helper already provides user feedback, so we just need to handle the exit code
    # This maintains the existing behavior where standalone mode exits with an error code
    # on failure, which is important for scripting and automation
    if not success:
        sys.exit(1)  # Exit with error code for standalone mode
