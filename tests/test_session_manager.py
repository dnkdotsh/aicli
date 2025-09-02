# tests/test_session_manager.py
"""
Tests for the interactive session logic in aicli/session_manager.py.
"""

from unittest.mock import MagicMock, call

import pytest
from aicli.session_manager import (
    SessionState,
    _handle_slash_command,
    perform_interactive_chat,
)
from prompt_toolkit.history import InMemoryHistory


@pytest.fixture
def session_state(mock_openai_engine):
    """Provides a fresh, default SessionState for each test."""
    return SessionState(
        engine=mock_openai_engine,
        model="gpt-4o-mini",
        system_prompt=None,
        initial_system_prompt=None,
        current_persona=None,
        max_tokens=4096,
        memory_enabled=True,
        stream_active=True,
        debug_active=False,
    )


@pytest.fixture
def mock_cli_history(mocker):
    """Provides a mock InMemoryHistory object with pre-populated strings for testing saves."""
    history = mocker.MagicMock(spec=InMemoryHistory)
    history.get_strings.return_value = ["/help", "first prompt"]
    return history


class TestSessionManagerCommands:
    """Test suite for slash command handling."""

    def test_command_stream_toggle(self, session_state):
        """Tests that /stream toggles the stream_active flag."""
        assert session_state.stream_active is True
        _handle_slash_command("/stream", session_state, MagicMock())
        assert session_state.stream_active is False
        _handle_slash_command("/stream", session_state, MagicMock())
        assert session_state.stream_active is True

    def test_command_debug_toggle(self, session_state):
        """Tests that /debug toggles the debug_active flag."""
        assert session_state.debug_active is False
        _handle_slash_command("/debug", session_state, MagicMock())
        assert session_state.debug_active is True

    def test_command_max_tokens(self, session_state):
        """Tests that /max-tokens updates the session state."""
        _handle_slash_command("/max-tokens 8192", session_state, MagicMock())
        assert session_state.max_tokens == 8192

    def test_command_exit(self, session_state):
        """Tests that /exit signals the session should end."""
        should_exit = _handle_slash_command("/exit", session_state, MagicMock())
        assert should_exit is True

    def test_command_save_signals_exit_and_saves_history(
        self, session_state, fake_fs, mocker, mock_cli_history
    ):
        """Tests that /save returns True and saves the command history."""
        mocker.patch(
            "aicli.commands._generate_session_name", return_value="test_session"
        )
        should_exit = _handle_slash_command("/save", session_state, mock_cli_history)
        assert should_exit is True
        assert session_state.command_history == ["/help", "first prompt"]

    def test_command_save_stay_continues_session_and_saves_history(
        self, session_state, fake_fs, mocker, mock_cli_history
    ):
        """Tests that /save --stay returns False and saves the command history."""
        mocker.patch(
            "aicli.commands._generate_session_name", return_value="test_session"
        )
        should_exit = _handle_slash_command(
            "/save --stay", session_state, mock_cli_history
        )
        assert should_exit is False
        assert session_state.command_history == ["/help", "first prompt"]

    def test_command_unknown(self, session_state, capsys):
        """Tests that an unknown command prints an error and does not exit."""
        should_exit = _handle_slash_command(
            "/unknown_command", session_state, MagicMock()
        )
        assert should_exit is False
        captured = capsys.readouterr()
        assert "Unknown command: /unknown_command" in captured.out


class TestSessionLifecycle:
    """Tests for the overall session lifecycle management."""

    def test_perform_interactive_chat_loads_command_history(self, mocker):
        """
        Tests that when an interactive session starts, it populates the
        prompt_toolkit history from the session state.
        """
        # 1. Prepare Mocks and a mock initial_state
        mocker.patch(
            "aicli.session_manager.prompt", side_effect=EOFError
        )  # Exit loop immediately
        mock_history_instance = mocker.MagicMock(spec=InMemoryHistory)
        mocker.patch(
            "aicli.session_manager.InMemoryHistory", return_value=mock_history_instance
        )

        # Create a mock SessionState as if it were loaded from a file
        mock_engine = MagicMock()
        mock_engine.name = "openai"
        initial_state = SessionState(
            engine=mock_engine,
            model="gpt-4o-mini",
            system_prompt=None,
            initial_system_prompt=None,
            current_persona=None,
            max_tokens=None,
            memory_enabled=True,
            command_history=[
                "/help",
                "previous prompt",
            ],  # This is the key data for the test
        )

        # 2. Run the function that contains the logic to test
        perform_interactive_chat(
            initial_state=initial_state, session_name="test_session"
        )

        # 3. Assert that the InMemoryHistory was populated correctly
        calls = mock_history_instance.append_string.call_args_list
        assert len(calls) == 2
        assert calls[0] == call("/help")
        assert calls[1] == call("previous prompt")
