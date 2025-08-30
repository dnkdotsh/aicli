# tests/test_session_manager.py
"""
Tests for the interactive session logic in aicli/session_manager.py.
"""

import pytest

from aicli.session_manager import _handle_slash_command, SessionState

@pytest.fixture
def session_state(mock_openai_engine):
    """Provides a fresh, default SessionState for each test."""
    return SessionState(
        engine=mock_openai_engine,
        model="gpt-4o-mini",
        system_prompt=None,
        max_tokens=4096,
        memory_enabled=True,
        stream_active=True,
        debug_active=False
    )

class TestSessionManagerCommands:
    """Test suite for slash command handling."""

    def test_command_stream_toggle(self, session_state):
        """Tests that /stream toggles the stream_active flag."""
        assert session_state.stream_active is True
        _handle_slash_command("/stream", session_state)
        assert session_state.stream_active is False
        _handle_slash_command("/stream", session_state)
        assert session_state.stream_active is True

    def test_command_debug_toggle(self, session_state):
        """Tests that /debug toggles the debug_active flag."""
        assert session_state.debug_active is False
        _handle_slash_command("/debug", session_state)
        assert session_state.debug_active is True

    def test_command_memory_toggle(self, session_state):
        """Tests that /memory toggles the memory_enabled flag."""
        assert session_state.memory_enabled is True
        _handle_slash_command("/memory", session_state)
        assert session_state.memory_enabled is False

    def test_command_max_tokens(self, session_state):
        """Tests that /max-tokens updates the session state."""
        _handle_slash_command("/max-tokens 8192", session_state)
        assert session_state.max_tokens == 8192

    def test_command_exit(self, session_state):
        """Tests that /exit signals the session should end."""
        should_exit = _handle_slash_command("/exit", session_state)
        assert should_exit is True

    def test_command_save_signals_exit(self, session_state, fake_fs, mocker):
        """Tests that /save returns True to signal exiting the loop."""
        mocker.patch('aicli.session_manager._generate_session_name', return_value='test_session')
        should_exit = _handle_slash_command("/save", session_state)
        assert should_exit is True

    def test_command_save_stay_continues_session(self, session_state, fake_fs, mocker):
        """Tests that /save --stay returns False to continue the session."""
        mocker.patch('aicli.session_manager._generate_session_name', return_value='test_session')
        should_exit = _handle_slash_command("/save --stay", session_state)
        assert should_exit is False
