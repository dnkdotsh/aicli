# aicli/utils.py
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


import base64
import mimetypes
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from prompt_toolkit import prompt

from . import config
from .engine import AIEngine
from .logger import log
from .settings import settings

# ANSI color codes for prompts
USER_PROMPT = "\033[94m"  # Bright Blue
ASSISTANT_PROMPT = "\033[92m"  # Bright Green
SYSTEM_MSG = "\033[93m"  # Bright Yellow
DIRECTOR_PROMPT = "\033[95m"  # Bright Magenta
RESET_COLOR = "\033[0m"

SUPPORTED_TEXT_EXTENSIONS: set[str] = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".html",
    ".css",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".csv",
    ".sh",
    ".bash",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".pl",
    ".sql",
    ".r",
    ".swift",
    ".kt",
    ".scala",
    ".ts",
    ".tsx",
    ".jsx",
    ".vue",
    ".jsonl",
    ".diff",
    ".log",
    ".toml",
}
SUPPORTED_IMAGE_MIMETYPES: set[str] = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}
SUPPORTED_ARCHIVE_EXTENSIONS: set[str] = {".zip", ".tar", ".gz", ".tgz"}
SUPPORTED_EXTENSIONLESS_FILENAMES: set[str] = {
    "dockerfile",
    "makefile",
    "vagrantfile",
    "jenkinsfile",
    "procfile",
    "rakefile",
    ".gitignore",
    "license",
}


def get_default_model_for_engine(engine_name: str) -> str:
    """Returns the default chat model for a given engine from settings."""
    model_key = f"default_{engine_name}_chat_model"
    return settings.get(model_key, "")


def select_model(engine: AIEngine, task: str) -> str:
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


def read_system_prompt(prompt_or_path: str) -> str:
    """Reads a system prompt from a file path or returns the string directly."""
    path = Path(prompt_or_path)
    if path.exists() and path.is_file():
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("Could not read system prompt file '%s': %s", prompt_or_path, e)
    return prompt_or_path


def ensure_dir_exists(directory_path: Path) -> None:
    """Creates a directory if it does not already exist."""
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Failed to create directory at %s: %s", directory_path, e)
        sys.exit(1)


def is_supported_text_file(filepath: Path) -> bool:
    """Check if a file is a supported text file based on its extension or name."""
    if filepath.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
        return True
    return (
        not filepath.suffix
        and filepath.name.lower() in SUPPORTED_EXTENSIONLESS_FILENAMES
    )


def is_supported_archive_file(filepath: Path) -> bool:
    """Check if a file is a supported archive file."""
    return any(
        filepath.name.lower().endswith(ext) for ext in SUPPORTED_ARCHIVE_EXTENSIONS
    )


def is_supported_image_file(filepath: Path) -> bool:
    """Check if a file is a supported image file based on its MIME type."""
    mimetype, _ = mimetypes.guess_type(filepath)
    return mimetype in SUPPORTED_IMAGE_MIMETYPES


def process_files(
    paths: list[str] | None, use_memory: bool, exclusions: list[str] | None
) -> tuple[str | None, dict[Path, str], list[dict[str, Any]]]:
    """
    Processes files, directories, and memory to build context.
    Returns memory content, a dictionary of attachment paths to content, and image data.
    """
    paths = paths or []
    exclusions = exclusions or []

    memory_content: str | None = None
    attachments_dict: dict[Path, str] = {}
    image_data_parts: list[dict[str, Any]] = []

    if use_memory and config.PERSISTENT_MEMORY_FILE.exists():
        try:
            memory_content = config.PERSISTENT_MEMORY_FILE.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("Could not read persistent memory file: %s", e)

    exclusion_paths = {Path(p).resolve() for p in exclusions}

    def process_text_file(filepath: Path):
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                attachments_dict[filepath] = f.read()
        except OSError as e:
            log.warning("Could not read file %s: %s", filepath, e)

    def process_image_file(filepath: Path):
        try:
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                mimetype, _ = mimetypes.guess_type(filepath)
                if mimetype in SUPPORTED_IMAGE_MIMETYPES:
                    image_data_parts.append(
                        {"type": "image", "data": encoded_string, "mime_type": mimetype}
                    )
        except OSError as e:
            log.warning("Could not read image file %s: %s", filepath, e)

    def process_zip_file(zip_path: Path):
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                zip_content_parts = []
                for filename in z.namelist():
                    if filename.endswith("/") or Path(filename).name in {
                        p.name for p in exclusion_paths
                    }:
                        continue
                    if is_supported_text_file(Path(filename)):
                        with z.open(filename) as f:
                            content = f.read().decode("utf-8", errors="ignore")
                            zip_content_parts.append(
                                f"--- FILE (from {zip_path.name}): {filename} ---\n{content}"
                            )
                if zip_content_parts:
                    attachments_dict[zip_path] = "\n\n".join(zip_content_parts)
        except (OSError, zipfile.BadZipFile) as e:
            log.warning("Could not process zip file %s: %s", zip_path, e)

    def process_tar_file(tar_path: Path):
        try:
            with tarfile.open(tar_path, "r:*") as t:
                tar_content_parts = []
                for member in t.getmembers():
                    if not member.isfile() or Path(member.name).name in {
                        p.name for p in exclusion_paths
                    }:
                        continue
                    if is_supported_text_file(Path(member.name)):
                        file_obj = t.extractfile(member)
                        if file_obj:
                            content = file_obj.read().decode("utf-8", errors="ignore")
                            tar_content_parts.append(
                                f"--- FILE (from {tar_path.name}): {member.name} ---\n{content}"
                            )
                if tar_content_parts:
                    attachments_dict[tar_path] = "\n\n".join(tar_content_parts)
        except (OSError, tarfile.TarError) as e:
            log.warning("Could not process tar file %s: %s", tar_path, e)

    for p_str in paths:
        path_obj = Path(p_str).resolve()
        if path_obj in exclusion_paths or not path_obj.exists():
            continue
        if path_obj.is_file():
            if is_supported_archive_file(path_obj):
                if path_obj.suffix.lower() == ".zip":
                    process_zip_file(path_obj)
                else:
                    process_tar_file(path_obj)
            elif is_supported_text_file(path_obj):
                process_text_file(path_obj)
            elif is_supported_image_file(path_obj):
                process_image_file(path_obj)
        elif path_obj.is_dir():
            for root, dirs, files in os.walk(path_obj, topdown=True):
                root_path = Path(root).resolve()
                dirs[:] = [
                    d for d in dirs if (root_path / d).resolve() not in exclusion_paths
                ]
                for name in files:
                    file_path = (root_path / name).resolve()
                    if file_path in exclusion_paths:
                        continue
                    if is_supported_text_file(file_path):
                        process_text_file(file_path)
                    elif is_supported_image_file(file_path):
                        process_image_file(file_path)

    return memory_content, attachments_dict, image_data_parts


def sanitize_filename(name: str) -> str:
    r"""Sanitizes a string to be a valid filename."""
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[-\s]+", "_", name)
    return name or "unnamed_log"


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


def format_token_string(token_dict: dict[str, int]) -> str:
    """Formats the token dictionary into a consistent string for display."""
    p, c, t = (
        token_dict.get("prompt", 0),
        token_dict.get("completion", 0),
        token_dict.get("total", 0),
    )
    r = token_dict.get("reasoning", 0) or max(0, t - (p + c))
    return (
        f"\n{SYSTEM_MSG}[P:{p}/C:{c}/R:{r}/T:{t}]{RESET_COLOR}"
        if any([p, c, t])
        else ""
    )


def format_bytes(byte_count: int) -> str:
    """Converts a byte count to a human-readable string (KB, MB, etc.)."""
    if byte_count is None:
        return "0 B"
    power, n = 1024, 0
    power_labels = {0: "B", 1: "KB", 2: "MB", 3: "GB"}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}"


def clean_ai_response_text(engine_name: str, raw_response: str) -> str:
    """Strips any self-labels the AI might have added."""
    pattern = re.compile(
        r"^\[" + re.escape(engine_name.capitalize()) + r"\]:?\s*", re.IGNORECASE
    )
    return pattern.sub("", raw_response.lstrip())


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
  /set [key] [val]  Change a setting (e.g., /set stream false).
                    Run without arguments to list all settings.
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
  /memory                   View the content of the persistent memory file.
  /remember [text]          If text, inject into memory. If no text, consolidate chat.
  /clear                    Clear the current conversation history.
  /state                    Print the current session state.
  /save <name> [--stay] [--remember]
                            Save the session. A name is required.
                            --stay:      Do not exit after saving.
                            --remember:  Update persistent memory before saving.
  /model <gpt|gem> <name>   Change the model for the specified engine.
  /set [key] [val]          Change a global setting.
  /max-tokens <num>         Set max output tokens for the session.
  /ai <gpt|gem> [prompt]    Send a targeted prompt to only one AI.
                            If no prompt, the AI is asked to continue.
"""
    else:
        help_text = "No help available for this context."
    print(help_text)
