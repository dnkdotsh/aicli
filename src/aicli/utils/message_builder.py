# aicli/utils/message_builder.py
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
Handles the construction and parsing of API-specific message formats
for different AI providers (e.g., OpenAI vs. Gemini).
"""

from typing import Any


def translate_history(
    history: list[dict[str, Any]], target_engine: str
) -> list[dict[str, Any]]:
    """Translates a conversation history to the target engine's format."""
    translated = []
    for msg in history:
        role = msg.get("role")
        if role not in ["user", "assistant", "model"]:
            continue
        text_content = extract_text_from_message(msg)
        if role == "user":
            translated.append(construct_user_message(target_engine, text_content, []))
        elif role in ["assistant", "model"]:
            translated.append(construct_assistant_message(target_engine, text_content))
    return translated


def construct_user_message(
    engine_name: str, text: str, image_data: list[dict[str, Any]]
) -> dict[str, Any]:
    """Constructs a user message in the format expected by the specified engine."""
    content: list[dict[str, Any]] = []
    if engine_name == "openai":
        content.append({"type": "text", "text": text})
        for img in image_data:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['mime_type']};base64,{img['data']}"
                    },
                }
            )
        return {"role": "user", "content": content}
    else:  # Gemini
        content.append({"text": text})
        for img in image_data:
            content.append(
                {"inline_data": {"mime_type": img["mime_type"], "data": img["data"]}}
            )
        return {"role": "user", "parts": content}


def construct_assistant_message(engine_name: str, text: str) -> dict[str, Any]:
    """Constructs an assistant message in the format expected by the specified engine."""
    if engine_name == "openai":
        return {"role": "assistant", "content": text}
    return {"role": "model", "parts": [{"text": text}]}


def extract_text_from_message(message: dict[str, Any]) -> str:
    """Extracts the text part from a potentially complex message object."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return next(
            (
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ),
            "",
        )

    parts = message.get("parts")
    if isinstance(parts, list):
        return next(
            (p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p),
            "",
        )

    return message.get("text", "")
