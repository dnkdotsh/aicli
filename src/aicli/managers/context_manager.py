# aicli/managers/context_manager.py
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
Manages the context for an AI session, including file attachments and
persistent memory.
"""

import base64
import mimetypes
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from .. import config
from ..logger import log
from ..utils.file_processor import (
    SUPPORTED_IMAGE_MIMETYPES,
    is_supported_archive_file,
    is_supported_image_file,
    is_supported_text_file,
)
from ..utils.formatters import RESET_COLOR, SYSTEM_MSG, format_bytes


class ContextManager:
    """Handles the aggregation and management of contextual data for a session."""

    def __init__(
        self,
        files_arg: list[str] | None,
        memory_enabled: bool,
        exclude_arg: list[str] | None,
    ):
        self.memory_content: str | None = None
        self.attachments: dict[Path, str] = {}
        self.image_data: list[dict[str, Any]] = []

        self._process_files(files_arg, memory_enabled, exclude_arg)

    def _read_memory_file(self) -> str:
        """Reads the persistent memory file, returning its content or an empty string."""
        if config.PERSISTENT_MEMORY_FILE.exists():
            try:
                return config.PERSISTENT_MEMORY_FILE.read_text(encoding="utf-8")
            except OSError as e:
                log.warning("Could not read persistent memory file: %s", e)
        return ""

    def _write_memory_file(self, content: str) -> None:
        """Writes content to the persistent memory file."""
        try:
            config.PERSISTENT_MEMORY_FILE.write_text(content.strip(), encoding="utf-8")
        except OSError as e:
            log.error("Failed to write to persistent memory file: %s", e)

    def _process_files(
        self,
        paths: list[str] | None,
        use_memory: bool,
        exclusions: list[str] | None,
    ) -> None:
        """
        Internal method to process all files, directories, and memory to build context.
        This populates the instance's state.
        """
        paths = paths or []
        exclusions = exclusions or []

        if use_memory:
            self.memory_content = self._read_memory_file()

        exclusion_paths = {Path(p).expanduser().resolve() for p in exclusions}

        for p_str in paths:
            path_obj = Path(p_str).expanduser().resolve()
            if path_obj in exclusion_paths or not path_obj.exists():
                continue
            if path_obj.is_file():
                if is_supported_archive_file(path_obj):
                    if path_obj.suffix.lower() == ".zip":
                        self._process_zip_file(path_obj, exclusion_paths)
                    else:
                        self._process_tar_file(path_obj, exclusion_paths)
                elif is_supported_text_file(path_obj):
                    self._process_text_file(path_obj)
                elif is_supported_image_file(path_obj):
                    self._process_image_file(path_obj)
            elif path_obj.is_dir():
                self._process_directory(path_obj, exclusion_paths)

    def _process_text_file(self, filepath: Path) -> None:
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                self.attachments[filepath] = f.read()
        except OSError as e:
            log.warning("Could not read file %s: %s", filepath, e)

    def _process_image_file(self, filepath: Path) -> None:
        try:
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                mimetype, _ = mimetypes.guess_type(filepath)
                if mimetype in SUPPORTED_IMAGE_MIMETYPES:
                    self.image_data.append(
                        {"type": "image", "data": encoded_string, "mime_type": mimetype}
                    )
        except OSError as e:
            log.warning("Could not read image file %s: %s", filepath, e)

    def _process_zip_file(self, zip_path: Path, exclusion_paths: set[Path]) -> None:
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
                    self.attachments[zip_path] = "\n\n".join(zip_content_parts)
        except (OSError, zipfile.BadZipFile) as e:
            log.warning("Could not process zip file %s: %s", zip_path, e)

    def _process_tar_file(self, tar_path: Path, exclusion_paths: set[Path]) -> None:
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
                    self.attachments[tar_path] = "\n\n".join(tar_content_parts)
        except (OSError, tarfile.TarError) as e:
            log.warning("Could not process tar file %s: %s", tar_path, e)

    def _process_directory(self, dir_path: Path, exclusion_paths: set[Path]) -> None:
        for root, dirs, files in os.walk(dir_path, topdown=True):
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
                    self._process_text_file(file_path)
                elif is_supported_image_file(file_path):
                    self._process_image_file(file_path)

    def refresh_files(self, search_term: str | None) -> list[str]:
        if not self.attachments:
            print(f"{SYSTEM_MSG}--> No files attached to refresh.{RESET_COLOR}")
            return []

        paths_to_refresh = [
            p for p in self.attachments if not search_term or search_term in p.name
        ]
        if not paths_to_refresh:
            print(f"{SYSTEM_MSG}--> No files matching '{search_term}'.{RESET_COLOR}")
            return []

        updated, removed = [], []
        for path in paths_to_refresh:
            try:
                self.attachments[path] = path.read_text(
                    encoding="utf-8", errors="ignore"
                )
                updated.append(path.name)
            except (OSError, FileNotFoundError):
                removed.append(path.name)
                del self.attachments[path]

        if updated:
            print(f"{SYSTEM_MSG}--> Refreshed: {', '.join(updated)}{RESET_COLOR}")
        if removed:
            print(
                f"{SYSTEM_MSG}--> Removed (not found): {', '.join(removed)}{RESET_COLOR}"
            )
        return updated

    def list_files(self) -> None:
        if not self.attachments:
            print(f"{SYSTEM_MSG}--> No text files are attached.{RESET_COLOR}")
            return

        print(f"{SYSTEM_MSG}--- Attached Files ---{RESET_COLOR}")

        paths = sorted(self.attachments.keys(), key=lambda p: str(p).lower())
        if not paths:
            return

        # Find the common base directory to make the tree cleaner
        try:
            # Use os.path.commonpath for robust handling of different path structures
            common_path_str = os.path.commonpath([str(p) for p in paths])
            common_base = Path(common_path_str)
            if not common_base.is_dir():
                common_base = common_base.parent
        except ValueError:
            common_base = Path("/")  # Fallback for paths on different drives (Windows)

        file_tree = {}
        for path in paths:
            try:
                relative_parts = path.relative_to(common_base).parts
            except ValueError:
                relative_parts = path.parts  # Show full path if not relative

            current_level = file_tree
            for part in relative_parts[:-1]:
                current_level = current_level.setdefault(part, {})
            current_level[relative_parts[-1]] = (
                path.stat().st_size if path.exists() else 0
            )

        def _print_tree(subtree: dict, prefix: str = ""):
            items = sorted(subtree.items())
            for i, (name, content) in enumerate(items):
                is_last = i == (len(items) - 1)
                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}{name}", end="")

                if isinstance(content, dict):
                    print()  # It's a directory, print a newline and recurse
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _print_tree(content, new_prefix)
                else:
                    # It's a file, print the size on the same line
                    print(f" ({format_bytes(content)})")

        print(f"{common_base}/")
        _print_tree(file_tree)

    def attach_file(self, path_str: str) -> None:
        path = Path(path_str).resolve()
        if not path.exists():
            print(f"{SYSTEM_MSG}--> Error: Path not found: {path_str}{RESET_COLOR}")
            return
        if path in self.attachments:
            print(
                f"{SYSTEM_MSG}--> Error: File '{path.name}' is already attached.{RESET_COLOR}"
            )
            return

        # Use a temporary instance to process just the new path
        temp_context = ContextManager(
            files_arg=[str(path)], memory_enabled=False, exclude_arg=[]
        )
        if temp_context.attachments:
            self.attachments.update(temp_context.attachments)
            print(f"{SYSTEM_MSG}--> Attached content from: {path.name}{RESET_COLOR}")
        else:
            print(
                f"{SYSTEM_MSG}--> No readable text content found at path: {path.name}{RESET_COLOR}"
            )

    def detach_file(self, filename: str) -> None:
        path_to_remove = next((p for p in self.attachments if p.name == filename), None)
        if path_to_remove:
            del self.attachments[path_to_remove]
            print(f"{SYSTEM_MSG}--> Detached file: {path_to_remove.name}{RESET_COLOR}")
        else:
            print(f"{SYSTEM_MSG}--> No attached file named '{filename}'.{RESET_COLOR}")
