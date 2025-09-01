# tests/test_session_manager_commands.py
"""
Tests for the state-modifying slash commands in aicli/session_manager.py.
"""

import pytest
from pathlib import Path

from aicli.session_manager import _handle_slash_command, SessionState, CommandResult
from aicli import utils

@pytest.fixture
def session_state_with_history(mock_openai_engine):
    """Provides a SessionState with a pre-populated history for testing /del and /redo."""
    history = [
        utils.construct_user_message('openai', "First prompt", []),
        utils.construct_assistant_message('openai', "First answer"),
        utils.construct_user_message('openai', "Second prompt", []),
        utils.construct_assistant_message('openai', "Second answer"),
    ]
    state = SessionState(
        engine=mock_openai_engine,
        model="gpt-4o-mini",
        system_prompt=None,
        max_tokens=4096,
        memory_enabled=True,
    )
    state.history = history
    return state

class TestSessionManagerStatefulCommands:
    """Test suite for commands that modify session state."""

    def test_command_attach_file_success(self, mock_session_state, fake_fs):
        """Tests that /attach successfully adds a file to the session context."""
        test_file = Path("/test/app.py")
        fake_fs.create_file(test_file, contents="print('hello')")

        result = _handle_slash_command(f"/attach {test_file}", mock_session_state)

        assert not result.should_exit
        assert test_file.resolve() in mock_session_state.attachments
        assert mock_session_state.attachments[test_file.resolve()] == "print('hello')"
        # Verify AI notification message was added to history
        assert "user has attached" in utils.extract_text_from_message(mock_session_state.history[-1])
        assert "app.py" in utils.extract_text_from_message(mock_session_state.history[-1])

    def test_command_attach_file_not_found(self, mock_session_state, capsys):
        """Tests that /attach handles non-existent files gracefully."""
        initial_attachment_count = len(mock_session_state.attachments)
        _handle_slash_command("/attach /non/existent/file.txt", mock_session_state)
        captured = capsys.readouterr()

        assert "Error: Path not found" in captured.out
        assert len(mock_session_state.attachments) == initial_attachment_count
        assert len(mock_session_state.history) == 0

    def test_command_detach_file_success(self, mock_session_state):
        """Tests that /detach successfully removes a file from the context."""
        file_path = Path("/test/file.txt")
        mock_session_state.attachments = {file_path: "content"}

        result = _handle_slash_command("/detach file.txt", mock_session_state)

        assert not result.should_exit
        assert file_path not in mock_session_state.attachments
        assert "user has detached" in utils.extract_text_from_message(mock_session_state.history[-1])

    def test_command_detach_file_not_found(self, mock_session_state, capsys):
        """Tests that /detach handles attempts to remove non-attached files."""
        mock_session_state.attachments = {Path("/test/file.txt"): "content"}
        _handle_slash_command("/detach non_existent", mock_session_state)
        captured = capsys.readouterr()

        assert "No attached file found" in captured.out
        assert len(mock_session_state.attachments) == 1
        assert len(mock_session_state.history) == 0

    def test_command_del_default_one_turn(self, session_state_with_history):
        """Tests that /del removes the last turn by default."""
        initial_history_len = len(session_state_with_history.history)
        _handle_slash_command("/del", session_state_with_history)
        assert len(session_state_with_history.history) == initial_history_len - 2

    def test_command_del_multiple_turns(self, session_state_with_history):
        """Tests that /del 2 removes the last two turns."""
        _handle_slash_command("/del 2", session_state_with_history)
        assert len(session_state_with_history.history) == 0

    def test_command_del_too_many_turns(self, session_state_with_history, capsys):
        """Tests that /del handles requests to delete more turns than exist."""
        initial_history_len = len(session_state_with_history.history)
        _handle_slash_command("/del 5", session_state_with_history)
        captured = capsys.readouterr()

        assert "Error: Cannot delete 5 turn(s)" in captured.out
        assert len(session_state_with_history.history) == initial_history_len

    def test_command_redo_success(self, session_state_with_history):
        """Tests that /redo correctly pops history and returns the last prompt."""
        initial_history_len = len(session_state_with_history.history)
        result = _handle_slash_command("/redo", session_state_with_history)

        assert isinstance(result, CommandResult)
        assert not result.should_exit
        assert result.redo_prompt == "Second prompt"
        assert len(session_state_with_history.history) == initial_history_len - 2

    def test_command_redo_insufficient_history(self, mock_session_state, capsys):
        """Tests /redo with an empty history."""
        result = _handle_slash_command("/redo", mock_session_state)
        captured = capsys.readouterr()

        assert "Not enough history to redo" in captured.out
        assert result.redo_prompt is None
        assert len(mock_session_state.history) == 0
