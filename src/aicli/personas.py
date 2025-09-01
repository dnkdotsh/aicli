# src/aicli/personas.py
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

"""
Manages AI personas, including loading, listing, and creating defaults.
"""

import json
from dataclasses import dataclass, field

from . import config
from .logger import log

# Constants for the default persona
DEFAULT_PERSONA_NAME = "aicli_assistant"
DEFAULT_PERSONA_FILENAME = f"{DEFAULT_PERSONA_NAME}.json"

# This large string block will be embedded in the default persona's system prompt.
AICLI_DOCUMENTATION = """
## AICLI Tool Documentation

### Overview
AICLI is a command-line interface for interacting with AI models like OpenAI's GPT series and Google's Gemini. It supports interactive chat, single-shot questions, multi-model conversations, and image generation.

### Key Features
- **Personas**: Reusable configurations (.json files in `~/.config/aicli/personas/`) that define an AI's behavior, including its system prompt, model, and engine.
- **Context Management**: Attach local files and directories to the conversation using the `-f` flag or `/attach` command.
- **Persistent Memory**: A long-term memory file (`~/.local/share/aicli/persistent_memory.txt`) that the AI can learn from across sessions. Enabled by default.
- **Session Management**: Save and load entire conversations using `/save` and `/load` (or `aicli -l <file>`). Saved sessions are stored in `~/.local/share/aicli/sessions/`.
- **Multi-Chat**: Use the `-b` or `--both` flag to have OpenAI and Gemini respond to the same prompts simultaneously.

### Interactive Commands
- `/help`: Show this help message.
- `/exit [name]`: End the session, optionally naming the log file.
- `/quit`: Exit immediately without saving.
- `/persona <name>`: Switch to a different persona.
- `/personas`: List all available personas.
- `/attach <path>`: Add a file/directory to the context.
- `/detach <name>`: Remove a file from the context.
- `/files`: List currently attached files.
- `/refresh`: Re-read attached files.
- `/memory`: View the persistent memory.
- `/remember [text]`: Add text to memory, or consolidate the chat if no text is given.
- `/save [name]`: Save the current session to a file.
- `/load <name>`: Load a session from a file.
- `/engine <openai|gemini>`: Switch the AI engine.
- `/model <model_name>`: Change the current model.
- `/state`: Display the current session's configuration.
"""


@dataclass
class Persona:
    """Represents an AI persona configuration."""

    name: str
    filename: str
    description: str = ""
    system_prompt: str = ""
    engine: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    # This field is for internal use and not loaded from the JSON
    raw_content: dict = field(default_factory=dict, repr=False)


def _get_default_persona_content() -> dict:
    """Returns the content for the default persona file."""
    return {
        "name": "AICLI Assistant",
        "description": "A helpful assistant with expert knowledge of the AICLI tool itself.",
        "system_prompt": (
            "You are a versatile and helpful general-purpose AI assistant. "
            "In addition, you are an expert on the `aicli` command-line tool. "
            "When asked about the tool, you must use the provided documentation as your source of truth. "
            "Answer factually and concisely based on the text below.\n\n"
            f"--- AICLI DOCUMENTATION ---\n{AICLI_DOCUMENTATION}\n---"
        ),
        "engine": "gemini",
        "model": "gemini-1.5-flash-latest",
    }


def ensure_personas_directory_and_default():
    """
    Ensures the personas directory exists and creates the default persona
    if it does not.
    """
    try:
        config.PERSONAS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        default_persona_path = config.PERSONAS_DIRECTORY / DEFAULT_PERSONA_FILENAME
        if not default_persona_path.exists():
            with open(default_persona_path, "w", encoding="utf-8") as f:
                json.dump(_get_default_persona_content(), f, indent=2)
            log.info("Created default persona file at %s", default_persona_path)
    except OSError as e:
        log.error("Failed to create personas directory or default persona: %s", e)


def load_persona(name: str) -> Persona | None:
    """
    Loads a persona from a JSON file in the personas directory.
    The name can be with or without the .json extension.
    """
    if not name.endswith(".json"):
        name += ".json"

    persona_path = config.PERSONAS_DIRECTORY / name
    if not persona_path.exists():
        return None

    try:
        with open(persona_path, encoding="utf-8") as f:
            data = json.load(f)

        # Basic validation
        if "name" not in data or "system_prompt" not in data:
            log.warning("Persona file %s is missing 'name' or 'system_prompt'.", name)
            return None

        return Persona(
            filename=name,
            name=data.get("name"),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt"),
            engine=data.get("engine"),
            model=data.get("model"),
            max_tokens=data.get("max_tokens"),
            stream=data.get("stream"),
            raw_content=data,
        )
    except (OSError, json.JSONDecodeError) as e:
        log.error("Failed to load or parse persona file %s: %s", name, e)
        return None


def list_personas() -> list[Persona]:
    """Lists all valid personas found in the personas directory."""
    personas = []
    if not config.PERSONAS_DIRECTORY.exists():
        return []

    for file_path in config.PERSONAS_DIRECTORY.glob("*.json"):
        persona = load_persona(file_path.name)
        if persona:
            personas.append(persona)

    return sorted(personas, key=lambda p: p.name)
