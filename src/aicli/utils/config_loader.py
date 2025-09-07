# aicli/utils/config_loader.py
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 David

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
Handles the complex logic of resolving the final application configuration
from command-line arguments, persona files, and default settings.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import config, personas
from ..settings import settings

if TYPE_CHECKING:
    import argparse


def get_default_model_for_engine(engine_name: str) -> str:
    """Returns the default chat model for a given engine from settings."""
    model_key = (
        f"default_{engine_name}_chat_model"
        if engine_name == "openai"
        else f"default_{engine_name}_model"
    )
    return settings.get(model_key, "")


def resolve_config_precedence(args: "argparse.Namespace") -> dict[str, Any]:
    """Determines the final configuration based on CLI args, persona, and settings."""
    is_single_shot = args.prompt is not None

    # --- Persona Loading ---
    persona = None
    if args.persona:
        persona = personas.load_persona(args.persona)
        if not persona:
            print(f"Warning: Persona '{args.persona}' not found.", file=sys.stderr)
    elif not is_single_shot and not args.system_prompt:
        persona = personas.load_persona(personas.DEFAULT_PERSONA_NAME)

    # --- Configuration Precedence: CLI > Persona > Settings ---

    # 1. Determine Engine
    if args.engine is not None:
        engine_to_use = args.engine
    elif persona and persona.engine:
        engine_to_use = persona.engine
    else:
        engine_to_use = settings["default_engine"]

    # 2. Determine Model (depends on final engine)
    if args.model is not None:
        model_to_use = args.model
    elif (
        persona
        and persona.model
        and (persona.engine is None or persona.engine == engine_to_use)
    ):
        model_to_use = persona.model
    else:
        model_to_use = get_default_model_for_engine(engine_to_use)

    # 3. Determine other parameters
    if args.max_tokens is not None:
        max_tokens_to_use = args.max_tokens
    elif persona and persona.max_tokens is not None:
        max_tokens_to_use = persona.max_tokens
    else:
        max_tokens_to_use = settings["default_max_tokens"]

    if args.stream is not None:
        stream_to_use = args.stream
    elif persona and persona.stream is not None:
        stream_to_use = persona.stream
    else:
        stream_to_use = settings["stream"]

    # 4. Determine memory status
    memory_enabled_for_session = settings["memory_enabled"]
    if hasattr(args, "memory") and args.memory:
        memory_enabled_for_session = not memory_enabled_for_session
    if is_single_shot:
        memory_enabled_for_session = False

    # 5. Combine file attachments from CLI and persona
    files_arg = args.file or []
    persona_attachments_arg = []
    if persona and persona.attachments:
        for p_str in persona.attachments:
            # Resolve path relative to CONFIG_DIR, but allow absolute/user-expanded paths
            path = Path(p_str)
            if path.is_absolute() or p_str.startswith("~"):
                resolved_path = path.expanduser().resolve()
            else:
                resolved_path = (config.CONFIG_DIR / path).resolve()
            persona_attachments_arg.append(str(resolved_path))

    return {
        "engine_name": engine_to_use,
        "model": model_to_use,
        "max_tokens": max_tokens_to_use,
        "stream": stream_to_use,
        "memory_enabled": memory_enabled_for_session,
        "debug_enabled": args.debug,
        "persona": persona,
        "session_name": args.session_name,
        "system_prompt_arg": args.system_prompt,
        "files_arg": files_arg,
        "persona_attachments_arg": persona_attachments_arg,
        "exclude_arg": args.exclude,
    }
