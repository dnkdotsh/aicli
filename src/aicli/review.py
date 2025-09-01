# src/aicli/review.py
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
An interactive tool to review, manage, and re-enter aicli sessions and logs.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Platform-specific imports for single-character input
if platform.system() == "Windows":
    import msvcrt
else:
    import termios
    import tty

from . import config, utils


def get_single_char(prompt=""):
    """Gets a single character from standard input without requiring Enter."""
    if prompt:
        print(prompt, end="", flush=True)

    if platform.system() == "Windows":
        char = msvcrt.getch()
        return char.decode("utf-8")
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            char = sys.stdin.read(1)
            # Handle Ctrl+C manually
            if ord(char) == 3:
                raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return char


def present_numbered_menu(title: str, options: list) -> int | None:
    """Displays a multi-column, numbered menu and returns the user's choice index."""
    print(f"\n--- {title} ---")
    terminal_width = shutil.get_terminal_size().columns

    if len(options) > 8 and terminal_width >= 80:
        num_items = len(options)
        midpoint = (num_items + 1) // 2
        left_col_items, right_col_items = options[:midpoint], options[midpoint:]
        max_len_left = (
            max(len(f"  {i + 1}. {opt}") for i, opt in enumerate(left_col_items))
            if left_col_items
            else 0
        )
        col_width = max_len_left + 4

        for i in range(midpoint):
            left_item = f"  {i + 1}. {left_col_items[i]}"
            right_item = ""
            if i < len(right_col_items):
                right_idx = i + midpoint
                right_item = f"  {right_idx + 1}. {right_col_items[i]}"
            print(f"{left_item:<{col_width}}{right_item}")
    else:
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")

    try:
        choice_str = input("Select an option: ")
        if not choice_str.isdigit():
            return None
        choice_idx = int(choice_str) - 1
        if 0 <= choice_idx < len(options):
            return choice_idx
    except (ValueError, IndexError):
        return None
    return None


def present_action_menu(title: str, options: dict) -> str:
    """Displays a single-line, letter-based action menu and returns the chosen character."""
    print(f"\n--- {title} ---")
    menu_parts = []
    for action, char in options.items():
        pos = action.lower().find(char)
        display_action = (
            f"{action[:pos]}({action[pos].upper()}){action[pos + 1 :]}"
            if pos != -1
            else f"{action} ({char.upper()})"
        )
        menu_parts.append(display_action)

    print(" | ".join(menu_parts))
    choice = get_single_char()
    print()
    return choice.lower()


def get_turn_count(file_path: Path) -> int:
    """Quickly counts the number of conversation turns in a log or session file."""
    try:
        if file_path.suffix == ".jsonl":
            with open(file_path, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        elif file_path.suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                return len(data.get("history", [])) // 2
    except (OSError, json.JSONDecodeError):
        return 0
    return 0


def replay_file(file_path: Path):
    """Reads a log or session file and prints the conversation in a formatted, paged way."""
    print(
        f"\n{utils.SYSTEM_MSG}--- Start of replay for: {file_path.name} ---{utils.RESET_COLOR}\n"
    )

    turns = []
    try:
        if file_path.suffix == ".jsonl":
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        turns.append(json.loads(line))
        elif file_path.suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                history = data.get("history", [])
                for i in range(0, len(history), 2):
                    if i + 1 < len(history):
                        turns.append({"prompt": history[i], "response": history[i + 1]})
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        return

    total_turns = len(turns)
    if total_turns == 0:
        print(f"{utils.SYSTEM_MSG}No conversation history found.{utils.RESET_COLOR}")
        return

    for i, turn_data in enumerate(turns):
        try:
            if "prompt" in turn_data and "response" in turn_data:
                user_text = utils.extract_text_from_message(turn_data["prompt"])
                asst_text = utils.extract_text_from_message(turn_data["response"])
                print(f"{utils.USER_PROMPT}You:{utils.RESET_COLOR}\n{user_text}\n")
                print(
                    f"{utils.ASSISTANT_PROMPT}Assistant:{utils.RESET_COLOR}\n{asst_text}\n"
                )
            elif "history_slice" in turn_data:
                for message in turn_data.get("history_slice", []):
                    text = utils.extract_text_from_message(message)
                    role = "Director" if message.get("role") == "user" else "AI"
                    color = (
                        utils.DIRECTOR_PROMPT
                        if role == "Director"
                        else utils.ASSISTANT_PROMPT
                    )
                    print(f"{color}{role}:{utils.RESET_COLOR}\n{text}\n")

            if i < total_turns - 1:
                prompt_text = (
                    f"-- Turn {i + 1} of {total_turns} -- (Press any key, 'q' to quit)"
                )
                print(f"{utils.SYSTEM_MSG}{prompt_text}{utils.RESET_COLOR}", end="\r")
                choice = get_single_char()
                print(" " * (len(prompt_text) + 5), end="\r")
                if choice.lower() == "q":
                    break
        except KeyboardInterrupt:
            break

    print(f"\n{utils.SYSTEM_MSG}--- End of replay ---{utils.RESET_COLOR}")


def rename_file(file_path: Path):
    """Prompts for a new name and renames the given file."""
    try:
        new_base_name = input(f"Enter new name for '{file_path.stem}' (no extension): ")
        if not new_base_name.strip():
            print("Rename cancelled.")
            return None
        sanitized_name = utils.sanitize_filename(new_base_name)
        new_path = file_path.with_name(sanitized_name + file_path.suffix)
        if new_path.exists():
            print(
                f"Error: A file named '{new_path.name}' already exists.",
                file=sys.stderr,
            )
            return None
        file_path.rename(new_path)
        print(f"File renamed to '{new_path.name}'")
        return new_path
    except Exception as e:
        print(f"Error renaming file: {e}", file=sys.stderr)
    return None


def delete_file(file_path: Path):
    """Prompts for confirmation and deletes the given file."""
    print(
        f"\n{utils.SYSTEM_MSG}Permanently delete '{file_path.name}'? (y/N) {utils.RESET_COLOR}",
        end="",
        flush=True,
    )
    try:
        choice = get_single_char()
        print()
        if choice.lower() == "y":
            file_path.unlink()
            print("File deleted.")
            return True
    except Exception as e:
        print(f"Error deleting file: {e}", file=sys.stderr)

    print("Deletion cancelled.")
    return False


def reenter_session(file_path: Path):
    """Launches the main aicli application to load a session."""
    print(
        f"\n{utils.SYSTEM_MSG}--> Re-entering session '{file_path.name}'...{utils.RESET_COLOR}"
    )
    try:
        subprocess.run(["aicli", "--load", str(file_path)], check=True)
    except FileNotFoundError:
        print(
            "Error: 'aicli' command not found. Make sure it is installed and in your PATH.",
            file=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error launching aicli: {e}", file=sys.stderr)


def main(args):
    """Main application loop for review mode."""
    # Direct replay mode
    if args.file:
        replay_file(args.file.expanduser())
        return

    # Interactive mode
    while True:
        try:
            log_files = sorted(
                config.LOG_DIRECTORY.glob("*.jsonl"), key=os.path.getmtime, reverse=True
            )
            session_files = sorted(
                config.SESSIONS_DIRECTORY.glob("*.json"),
                key=os.path.getmtime,
                reverse=True,
            )
            all_files = session_files + log_files
            if not all_files:
                print("No logs or saved sessions found.")
                return

            options = [f"Session: {f.name}" for f in session_files] + [
                f"Log: {f.name}" for f in log_files
            ]
            options.append("Quit")

            choice_idx = present_numbered_menu("Select a file to review", options)

            if choice_idx is None or choice_idx == len(options) - 1:
                print("Exiting.")
                break

            selected_path = all_files[choice_idx]
            is_session = selected_path.suffix == ".json"

            while True:
                turn_count = get_turn_count(selected_path)
                menu_title = f"Actions for '{selected_path.name}' ({turn_count} turns)"

                action_map = {"Replay": "r", "Rename": "n", "Delete": "d", "Back": "b"}
                if is_session:
                    action_map.update({"Re-enter Session": "e"})

                choice_char = present_action_menu(menu_title, action_map)

                if choice_char == "b":
                    break
                elif choice_char == "r":
                    replay_file(selected_path)
                    post_map = {"Rename": "n", "Delete": "d", "Continue": "c"}
                    post_choice = present_action_menu("Post-Replay Actions", post_map)
                    if post_choice == "n":
                        new_path = rename_file(selected_path)
                        if new_path:
                            selected_path = new_path
                    elif post_choice == "d":
                        if delete_file(selected_path):
                            break
                elif choice_char == "e" and is_session:
                    reenter_session(selected_path)
                    return
                elif choice_char == "n":
                    new_path = rename_file(selected_path)
                    if new_path:
                        selected_path = new_path
                elif choice_char == "d":
                    if delete_file(selected_path):
                        break
                else:
                    print(f"{utils.SYSTEM_MSG}Unknown option.{utils.RESET_COLOR}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
