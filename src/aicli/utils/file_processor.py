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

import base64
import datetime
import json
import mimetypes
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any

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

    exclusion_paths = {Path(p).expanduser().resolve() for p in exclusions}

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
        path_obj = Path(p_str).expanduser().resolve()
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
                root_path = Path(root).expanduser().resolve()
                dirs[:] = [
                    d
                    for d in dirs
                    if (root_path / d).expanduser().resolve() not in exclusion_paths
                ]
                for name in files:
                    file_path = (root_path / name).expanduser().resolve()
                    if file_path in exclusion_paths:
                        continue
                    if is_supported_text_file(file_path):
                        process_text_file(file_path)
                    elif is_supported_image_file(file_path):
                        process_image_file(file_path)

    return memory_content, attachments_dict, image_data_parts


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
