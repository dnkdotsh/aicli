# tests/test_review.py
"""
Tests for the session review module in aicli/review.py.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, call

from aicli import review
from aicli import config
from aicli import utils

@pytest.fixture
def fake_log_file(fake_fs):
    """Creates a fake .jsonl log file with 3 turns."""
    log_path = config.LOG_DIRECTORY / "test_log.jsonl"
    content = [
        {"prompt": {"role": "user"}, "response": {"role": "assistant"}},
        {"prompt": {"role": "user"}, "response": {"role": "assistant"}},
        {"prompt": {"role": "user"}, "response": {"role": "assistant"}},
    ]
    log_path.write_text("\n".join(json.dumps(c) for c in content))
    return log_path

@pytest.fixture
def fake_session_file(fake_fs):
    """Creates a fake .json session file with 2 turns."""
    session_path = config.SESSIONS_DIRECTORY / "test_session.json"
    content = {
        "history": [
            {"role": "user"}, {"role": "assistant"}, # Turn 1
            {"role": "user"}, {"role": "assistant"}, # Turn 2
        ]
    }
    session_path.write_text(json.dumps(content))
    return session_path


class TestReviewHelpers:
    """Test suite for the helper functions in review.py."""

    def test_get_turn_count(self, fake_log_file, fake_session_file, fake_fs):
        """Tests that turn counting works for both log and session files."""
        assert review.get_turn_count(fake_log_file) == 3
        assert review.get_turn_count(fake_session_file) == 2

        empty_file = config.LOG_DIRECTORY / "empty.jsonl"
        empty_file.touch()
        assert review.get_turn_count(empty_file) == 0

        malformed_file = config.LOG_DIRECTORY / "bad.json"
        malformed_file.write_text("{ not json }")
        assert review.get_turn_count(malformed_file) == 0

    def test_replay_file_log(self, fake_log_file, capsys, mocker):
        """Tests replaying a .jsonl log file."""
        # Mock get_single_char to return 'q' to exit after the first turn
        mocker.patch('aicli.review.get_single_char', return_value='q')
        mocker.patch('aicli.utils.extract_text_from_message', side_effect=['User 1', 'Asst 1', 'User 2', 'Asst 2'])

        review.replay_file(fake_log_file)

        captured = capsys.readouterr()
        assert "--- Start of replay for: test_log.jsonl ---" in captured.out
        assert "You:" in captured.out
        assert "User 1" in captured.out
        assert "Assistant:" in captured.out
        assert "Asst 1" in captured.out
        # Should not contain the second turn because we quit
        assert "User 2" not in captured.out
        assert "--- End of replay ---" in captured.out

    def test_replay_file_session(self, fake_session_file, capsys, mocker):
        """Tests replaying a .json session file."""
        mocker.patch('aicli.review.get_single_char', return_value='any_key') # Press any key
        mocker.patch('aicli.utils.extract_text_from_message', side_effect=['User 1', 'Asst 1', 'User 2', 'Asst 2'])

        review.replay_file(fake_session_file)

        captured = capsys.readouterr()
        assert "--- Start of replay for: test_session.json ---" in captured.out
        assert "User 1" in captured.out
        assert "Asst 1" in captured.out
        # Should contain the second turn because we didn't quit
        assert "User 2" in captured.out
        assert "Asst 2" in captured.out
        assert "--- End of replay ---" in captured.out

    def test_rename_file_success(self, fake_log_file, mocker):
        """Tests successful file renaming."""
        mocker.patch('builtins.input', return_value="a new name")
        original_path = fake_log_file
        new_path = review.rename_file(original_path)

        assert not original_path.exists()
        assert new_path is not None
        assert new_path.name == "a_new_name.jsonl"
        assert new_path.exists()

    def test_rename_file_collision(self, fake_log_file, mocker, capsys):
        """Tests that renaming to an existing filename fails gracefully."""
        # Create the file that will cause the collision
        (config.LOG_DIRECTORY / "a_new_name.jsonl").touch()

        mocker.patch('builtins.input', return_value="a new name")
        original_path = fake_log_file
        new_path = review.rename_file(original_path)

        assert new_path is None # Should fail
        assert original_path.exists() # Original should still be there
        captured = capsys.readouterr()
        assert "Error: A file named 'a_new_name.jsonl' already exists." in captured.err

    def test_delete_file_confirmed(self, fake_log_file, mocker):
        """Tests that the file is deleted when user confirms with 'y'."""
        mocker.patch('aicli.review.get_single_char', return_value='y')
        assert fake_log_file.exists()
        was_deleted = review.delete_file(fake_log_file)
        assert was_deleted is True
        assert not fake_log_file.exists()

    def test_delete_file_cancelled(self, fake_log_file, mocker):
        """Tests that the file is NOT deleted when user cancels."""
        mocker.patch('aicli.review.get_single_char', return_value='n')
        assert fake_log_file.exists()
        was_deleted = review.delete_file(fake_log_file)
        assert was_deleted is False
        assert fake_log_file.exists()

    def test_reenter_session(self, fake_session_file, mocker):
        """Tests that reenter_session calls the correct subprocess command."""
        mock_subprocess_run = mocker.patch('subprocess.run')

        review.reenter_session(fake_session_file)

        mock_subprocess_run.assert_called_once_with(
            ['aicli', '--load', str(fake_session_file)],
            check=True
        )
