# tests/test_review.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aicli import review


# --- Fixtures ---
@pytest.fixture
def mock_get_single_char(mocker):
    """Mocks the get_single_char function."""
    return mocker.patch("aicli.review.get_single_char")


@pytest.fixture
def mock_input(mocker):
    """Mocks the built-in input function."""
    return mocker.patch("builtins.input")


@pytest.fixture
def mock_subprocess_run(mocker):
    """Mocks subprocess.run."""
    return mocker.patch("subprocess.run")


@pytest.fixture
def setup_file_system(fs):
    """Sets up a fake file system with log and session directories and files."""
    log_dir = Path(review.config.LOG_DIRECTORY)
    session_dir = Path(review.config.SESSIONS_DIRECTORY)
    fs.create_dir(log_dir)
    fs.create_dir(session_dir)

    # Create a sample .jsonl log file
    log_content = [
        {
            "prompt": {"role": "user", "content": "Hello"},
            "response": {"role": "assistant", "content": "Hi there"},
        },
        {
            "prompt": {"role": "user", "content": "How are you?"},
            "response": {"role": "assistant", "content": "I am fine"},
        },
    ]
    with open(log_dir / "test_log.jsonl", "w") as f:
        for entry in log_content:
            f.write(json.dumps(entry) + "\n")

    # Create a sample .json session file
    session_content = {
        "history": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]
    }
    with open(session_dir / "test_session.json", "w") as f:
        json.dump(session_content, f)

    return log_dir, session_dir


# --- Unit Tests ---


class TestHelperFunctions:
    """Tests for standalone helper functions."""

    def test_get_turn_count_jsonl(self, setup_file_system):
        log_dir, _ = setup_file_system
        count = review.get_turn_count(log_dir / "test_log.jsonl")
        assert count == 2

    def test_get_turn_count_json(self, setup_file_system):
        _, session_dir = setup_file_system
        count = review.get_turn_count(session_dir / "test_session.json")
        assert count == 2

    def test_get_turn_count_file_not_found(self):
        count = review.get_turn_count(Path("non_existent_file.json"))
        assert count == 0

    @patch("aicli.review.get_single_char", return_value="n")
    def test_delete_file_cancelled(self, mock_get_char, setup_file_system, capsys):
        _, session_dir = setup_file_system
        session_file = session_dir / "test_session.json"
        assert session_file.exists()
        deleted = review.delete_file(session_file)
        assert not deleted
        assert session_file.exists()
        assert "Deletion cancelled" in capsys.readouterr().out

    @patch("aicli.review.get_single_char", return_value="y")
    def test_delete_file_confirmed(self, mock_get_char, setup_file_system, capsys):
        _, session_dir = setup_file_system
        session_file = session_dir / "test_session.json"
        assert session_file.exists()
        deleted = review.delete_file(session_file)
        assert deleted
        assert not session_file.exists()
        assert "File deleted" in capsys.readouterr().out

    @patch("builtins.input", return_value="new_session_name")
    def test_rename_file_success(self, mock_input, setup_file_system):
        _, session_dir = setup_file_system
        session_file = session_dir / "test_session.json"
        new_path = review.rename_file(session_file)
        assert new_path is not None
        assert new_path.name == "new_session_name.json"
        assert not session_file.exists()
        assert new_path.exists()

    @patch("builtins.input", return_value="")
    def test_rename_file_cancelled(self, mock_input, setup_file_system):
        _, session_dir = setup_file_system
        session_file = session_dir / "test_session.json"
        new_path = review.rename_file(session_file)
        assert new_path is None
        assert session_file.exists()

    def test_reenter_session(self, mock_subprocess_run):
        file_path = Path("/fake/dir/my_session.json")
        review.reenter_session(file_path)
        mock_subprocess_run.assert_called_once_with(
            ["aicli", "--load", str(file_path)], check=True
        )

    def test_reenter_session_file_not_found_error(self, mock_subprocess_run, capsys):
        mock_subprocess_run.side_effect = FileNotFoundError
        review.reenter_session(Path("test.json"))
        captured = capsys.readouterr()
        assert "Error: 'aicli' command not found" in captured.err

    @patch("aicli.review.get_single_char", side_effect=["", "q"])
    def test_replay_file(self, mock_get_char, setup_file_system, capsys):
        _, session_dir = setup_file_system
        session_file = session_dir / "test_session.json"
        review.replay_file(session_file)
        captured = capsys.readouterr().out
        assert "--- Start of replay for: test_session.json ---" in captured
        assert "First question" in captured
        assert "First answer" in captured
        assert "Second question" in captured
        assert "Second answer" in captured
        assert "--- End of replay ---" in captured
        assert mock_get_char.call_count == 1  # FIX: Only called once between 2 turns


class TestMenuPresentation:
    """Tests for menu rendering and input handling."""

    @patch("shutil.get_terminal_size", return_value=MagicMock(columns=80))
    def test_present_numbered_menu_multi_column(
        self, mock_terminal_size, mock_input, capsys
    ):
        options = [f"item_{i}" for i in range(10)]
        mock_input.return_value = "3"
        choice = review.present_numbered_menu("Test Menu", options)
        assert choice == 2
        captured = capsys.readouterr().out
        # Check if items are printed on the same line (multi-column)
        assert "1. item_0" in captured
        assert "6. item_5" in captured.splitlines()[2]  # 1st item is on line 2

    def test_present_numbered_menu_single_column(self, mock_input, capsys):
        options = ["item_1", "item_2"]
        mock_input.return_value = "2"
        choice = review.present_numbered_menu("Test Menu", options)
        assert choice == 1
        captured = capsys.readouterr().out
        assert "1. item_1" in captured
        assert "2. item_2" in captured

    def test_present_numbered_menu_invalid_input(self, mock_input):
        mock_input.return_value = "abc"
        choice = review.present_numbered_menu("Test Menu", ["item"])
        assert choice is None

    def test_present_action_menu(self, mock_get_single_char, capsys):
        mock_get_single_char.return_value = "r"
        actions = {"Replay": "r", "Delete": "d"}
        choice = review.present_action_menu("Action Menu", actions)
        assert choice == "r"
        captured = capsys.readouterr().out
        assert "(R)eplay" in captured
        assert "(D)elete" in captured


class TestMainApplicationFlow:
    """Integration-style tests for the main() function loop."""

    def test_main_no_files_found(self, fs, capsys):
        # Setup empty directories
        fs.create_dir(review.config.LOG_DIRECTORY)
        fs.create_dir(review.config.SESSIONS_DIRECTORY)
        args = MagicMock(file=None)
        review.main(args)
        captured = capsys.readouterr().out
        assert "No logs or saved sessions found" in captured

    def test_main_direct_replay_mode(self, setup_file_system):
        _, session_dir = setup_file_system
        session_file = session_dir / "test_session.json"
        args = MagicMock(file=session_file)
        with patch("aicli.review.replay_file") as mock_replay:
            review.main(args)
            mock_replay.assert_called_once_with(session_file)

    @patch(
        "aicli.review.present_numbered_menu", side_effect=[0, 2]
    )  # Choose 1st file, then Quit
    @patch(
        "aicli.review.present_action_menu", side_effect=["r", "c", "b"]
    )  # Replay, Continue, Back
    @patch("aicli.review.replay_file")
    def test_main_interactive_replay_and_quit(
        self, mock_replay, mock_action_menu, mock_numbered_menu, setup_file_system
    ):
        args = MagicMock(file=None)
        review.main(args)

        assert mock_numbered_menu.call_count == 2
        assert mock_action_menu.call_count == 3
        mock_replay.assert_called_once()

    @patch(
        "aicli.review.present_numbered_menu", side_effect=[1, 1]
    )  # 2nd file (log), then quit
    @patch("aicli.review.present_action_menu", side_effect=["d"])  # delete
    @patch("aicli.review.get_single_char", return_value="y")  # confirm deletion
    def test_main_interactive_delete_and_quit(
        self, mock_get_char, mock_action_menu, mock_numbered_menu, setup_file_system
    ):
        log_dir, _ = setup_file_system
        log_file = log_dir / "test_log.jsonl"
        args = MagicMock(file=None)

        assert log_file.exists()
        review.main(args)
        assert not log_file.exists()
