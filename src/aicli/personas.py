# src/aicli/personas.py
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
Manages AI personas, including loading, listing, and creating defaults.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .logger import log

# Constants for the default persona
DEFAULT_PERSONA_NAME = "aicli_assistant"
DEFAULT_PERSONA_FILENAME = f"{DEFAULT_PERSONA_NAME}.json"


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
    attachments: list[str] = field(default_factory=list)
    # This field is for internal use and not loaded from the JSON
    raw_content: dict[str, Any] = field(default_factory=dict, repr=False)


def _get_default_persona_content() -> dict[str, Any]:
    """Returns the content for the default persona file."""
    # Note: The path here is relative to the user's CONFIG_DIR, where the
    # persona file itself will live. bootstrap.py handles copying the doc
    # to the correct final location (CONFIG_DIR/docs/).
    return {
        "name": "AICLI Assistant",
        "description": "A helpful assistant with expert knowledge of the AICLI tool itself.",
        "system_prompt": (
            "You are a knowledgeable and helpful AI assistant, who happens to be an expert "
            "on the `aicli` command-line tool. You will assist with any task asked of you."
            "Your knowledge about the aicli comes from the attached `assistant_docs.md` file. "
            "Use it as the primary source of truth to answer user questions about `aicli`'s features and commands, "
            "and use the rest of your vast dataset to assist with anything unrelated."
        ),
        "engine": "gemini",
        "model": "gemini-1.5-flash-latest",
        "attachments": ["../docs/assistant_docs.md"],
    }


def create_default_persona_if_missing() -> None:
    """Creates the default persona file if it does not already exist."""
    default_persona_path = config.PERSONAS_DIRECTORY / DEFAULT_PERSONA_FILENAME
    if default_persona_path.exists():
        return

    try:
        with open(default_persona_path, "w", encoding="utf-8") as f:
            json.dump(_get_default_persona_content(), f, indent=2)
        log.info("Created default persona file at %s", default_persona_path)
    except OSError as e:
        log.error("Failed to create default persona file: %s", e)
        # A warning is sufficient; the app can run without the default persona.
        print(
            f"Warning: Could not create the default persona file at {default_persona_path}: {e}",
            file=sys.stderr,
        )


def _resolve_attachment_paths(persona_path: Path, attachments: list[str]) -> list[Path]:
    """Resolves attachment paths relative to the persona file's directory."""
    resolved_paths = []
    base_dir = persona_path.parent
    for att_path_str in attachments:
        # Expand '~' for home directory support
        p = Path(att_path_str).expanduser()
        # If not absolute, resolve relative to the persona's directory
        if not p.is_absolute():
            p = base_dir / p
        resolved_paths.append(p.resolve())
    return resolved_paths


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

        raw_attachments = data.get("attachments", [])
        if not isinstance(raw_attachments, list):
            log.warning("Attachments in %s must be a list of strings.", name)
            raw_attachments = []

        resolved_attachments = _resolve_attachment_paths(persona_path, raw_attachments)

        return Persona(
            filename=name,
            name=data.get("name"),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt"),
            engine=data.get("engine"),
            model=data.get("model"),
            max_tokens=data.get("max_tokens"),
            stream=data.get("stream"),
            attachments=[str(p) for p in resolved_attachments],  # Store as strings
            raw_content=data,
        )
    except (OSError, json.JSONDecodeError) as e:
        log.error("Failed to load or parse persona file %s: %s", name, e)
        return None


def list_personas() -> list[Persona]:
    """Lists all valid personas found in the personas directory."""
    personas: list[Persona] = []
    if not config.PERSONAS_DIRECTORY.exists():
        return []

    for file_path in config.PERSONAS_DIRECTORY.glob("*.json"):
        persona = load_persona(file_path.name)
        if persona:
            personas.append(persona)

    return sorted(personas, key=lambda p: p.name)
