# aicli/utils/file_processor.py
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
Handles processing of files and directories for context, including reading
text files, archives, and managing image data.
"""

import datetime
import json
import mimetypes
from pathlib import Path

from .. import config
from ..logger import log
from .formatters import sanitize_filename

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


def read_system_prompt(prompt_or_path: str) -> str:
    """Reads a system prompt from a file path or returns the string directly."""
    path = Path(prompt_or_path)
    if path.exists() and path.is_file():
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("Could not read system prompt file '%s': %s", prompt_or_path, e)
    return prompt_or_path


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


def save_image_and_get_path(
    prompt: str, image_bytes: bytes, session_name: str | None
) -> Path:
    """
    Saves image bytes to a uniquely named file and returns the path.

    Args:
        prompt: The image prompt, used for generating a descriptive filename.
        image_bytes: The raw bytes of the image to be saved.
        session_name: An optional session identifier for file organization.

    Returns:
        The Path object of the newly created image file.
    """
    safe_prompt = sanitize_filename(prompt[:50])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if session_name:
        base_filename = f"session_{session_name}_img_{safe_prompt}_{timestamp}.png"
    else:
        base_filename = f"image_{safe_prompt}_{timestamp}.png"
    filepath = config.IMAGE_DIRECTORY / base_filename
    filepath.write_bytes(image_bytes)
    return filepath


def log_image_generation(
    model: str, prompt: str, filepath: str, session_name: str | None
) -> None:
    """
    Writes a record of a successful image generation to the image log file.
    Args:
        model: The model used for generation.
        prompt: The full prompt used.
        filepath: The path where the final image was saved.
        session_name: The optional session identifier.
    """
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model,
            "prompt": prompt,
            "file": filepath,
            "session": session_name,
        }
        with open(config.IMAGE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except OSError as e:
        log.warning("Could not write to image log file: %s", e)
