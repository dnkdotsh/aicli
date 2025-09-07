# aicli/utils/ui_helpers.py
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
Contains helper functions that directly interact with the user via the
command line, such as displaying help text or prompting for choices.
"""

from typing import TYPE_CHECKING

from prompt_toolkit import prompt

from ..settings import settings
from .config_loader import get_default_model_for_engine

if TYPE_CHECKING:
    from ..engine import AIEngine


def select_model(engine: "AIEngine", task: str) -> str:
    """Allows the user to select a model or use the default."""
    default_model = ""
    if task == "chat":
        default_model = get_default_model_for_engine(engine.name)
    elif task == "image":
        default_model = settings["default_openai_image_model"]

    use_default = (
        prompt(f"Use default model ({default_model})? (Y/n): ").lower().strip()
    )
    if use_default in ("", "y", "yes"):
        return default_model

    print("Fetching available models...")
    models = engine.fetch_available_models(task)
    if not models:
        print(f"Using default: {default_model}")
        return default_model

    print("\nPlease select a model:")
    for i, model_name in enumerate(models):
        print(f"  {i + 1}. {model_name}")

    try:
        choice = prompt("Enter number (or press Enter for default): ")
        if not choice:
            return default_model
        index = int(choice) - 1
        if 0 <= index < len(models):
            return models[index]
    except (ValueError, IndexError):
        pass

    print(f"Invalid selection. Using default: {default_model}")
    return default_model


def display_help(context: str) -> None:
    """Displays help information for the given context (chat or multichat)."""
    if context == "chat":
        help_text = """
Interactive Chat Commands:
  /exit [name]      End the session. Optionally provide a name for the log file.
  /quit             Exit immediately without updating memory or renaming the log.
  /help             Display this help message.
  /stream           Toggle response streaming on/off.
  /debug            Toggle session-specific raw API logging.
  /memory           View the content of the persistent memory file.
  /remember [text]  If text is provided, inject it into persistent memory.
                    If no text, consolidates current chat into memory.
  /clear            Clear the current conversation history.
  /history          Print the JSON of the current conversation history.
  /state            Print the current session state (engine, model, etc.).
  /refresh [name]   Re-read attached files to update the context.
                    If no name is given, all files are refreshed.
                    Otherwise, refreshes all files containing 'name'.
  /files            List all currently attached text files, sorted by size.
  /attach <path>    Attach a file or directory to the session context.
  /detach <name>    Detach a file from the context by its filename.
  /save [name] [--stay] [--remember]
                    Save the session. Auto-generates a name if not provided.
                    --stay:      Do not exit after saving.
                    --remember:  Update persistent memory before exiting.
                    (Default is to save and exit without updating memory).
  /load <filename>  Load a session, replacing the current one.
  /engine [name]    Switch AI engine (openai/gemini). Translates history.
  /model [name]     Select a new model for the current engine.
  /persona <name>   Switch to a different persona. Use `/persona clear` to remove.
  /personas         List all available personas.
  /image [prompt]   Initiate the image generation workflow.
  /theme <name>     Switch the display theme. Run without a name to list themes.
  /toolbar [on|off|toggle <comp>]
                    Control the bottom toolbar. Components: io, live, model, persona.
  /set [key] [val]  Change a setting (e.g., /set stream false).
  /max-tokens [num] Set max tokens for the session.
"""
    elif context == "multichat":
        help_text = """
Multi-Chat Commands:
  /exit [name]              End the session. Optionally provide a name for the log file.
  /quit                     Exit immediately without saving.
  /help                     Display this help message.
  /history                  Print the JSON of the shared conversation history.
  /debug                    Toggle session-specific raw API logging.
  /remember                 Consolidates current chat into memory.
  /clear                    Clear the current conversation history.
  /state                    Print the current session state.
  /save <name> [--stay] [--remember]
                            Save the session. A name is required.
                            --stay:      Do not exit after saving.
                            --remember:  Update persistent memory before saving.
  /model <gpt|gem> <name>   Change the model for the specified engine.
  /max-tokens <num>         Set max output tokens for the session.
  /ai <gpt|gem> [prompt]    Send a targeted prompt to only one AI.
                            If no prompt, the AI is asked to continue.
"""
    else:
        help_text = "No help available for this context."
    print(help_text)
