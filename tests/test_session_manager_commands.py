# tests/test_session_manager_commands.py
"""
Tests for the state-modifying and file-interacting slash commands
in aicli/commands.py.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from aicli.session_manager import SessionState, MultiChatSessionState
from aicli import utils
from aicli import config
from aicli import commands
from aicli.engine import OpenAIEngine, GeminiEngine
from aicli.prompts import CONTINUATION_PROMPT

@pytest.fixture
def mock_session_state(mock_openai_engine):
    """Provides a fresh, default SessionState for each test."""
    return SessionState(
        engine=mock_openai_engine,
        model="gpt-4o-mini",
        system_prompt=None,
        initial_system_prompt=None,
        current_persona=None,
        max_tokens=4096,
        memory_enabled=True,
    )

@pytest.fixture
def session_state_with_history(mock_openai_engine):
    """Provides a SessionState with a pre-populated history for testing."""
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
        initial_system_prompt=None,
        current_persona=None,
        max_tokens=4096,
        memory_enabled=True,
    )
    state.history = history
    return state

class TestSessionManagerFileCommands:
    """Test suite for commands that interact with files and memory."""

    def test_command_attach_file_success(self, mock_session_state, fake_fs):
        """Tests that /attach successfully adds a file to the session context."""
        test_file = Path("/test/app.py")
        fake_fs.create_file(test_file, contents="print('hello')")

        should_exit = commands.handle_attach([str(test_file)], mock_session_state, MagicMock())

        assert not should_exit
        assert test_file.resolve() in mock_session_state.attachments
        assert mock_session_state.attachments[test_file.resolve()] == "print('hello')"
        # Verify AI notification message was added to history
        assert "user has attached" in utils.extract_text_from_message(mock_session_state.history[-1])
        assert "app.py" in utils.extract_text_from_message(mock_session_state.history[-1])

    def test_command_attach_extensionless_file_success(self, mock_session_state, fake_fs):
        """Tests that /attach successfully adds a supported extensionless file."""
        dockerfile = Path("/test/Dockerfile")
        fake_fs.create_file(dockerfile, contents="FROM python:3.10")

        should_exit = commands.handle_attach([str(dockerfile)], mock_session_state, MagicMock())

        assert not should_exit
        assert dockerfile.resolve() in mock_session_state.attachments
        assert mock_session_state.attachments[dockerfile.resolve()] == "FROM python:3.10"
        assert "user has attached" in utils.extract_text_from_message(mock_session_state.history[-1])
        assert "Dockerfile" in utils.extract_text_from_message(mock_session_state.history[-1])

    def test_command_attach_file_not_found(self, mock_session_state, capsys):
        """Tests that /attach handles non-existent files gracefully."""
        initial_attachment_count = len(mock_session_state.attachments)
        commands.handle_attach(['/non/existent/file.txt'], mock_session_state, MagicMock())
        captured = capsys.readouterr()

        assert "Error: Path not found" in captured.out
        assert len(mock_session_state.attachments) == initial_attachment_count
        assert len(mock_session_state.history) == 0

    def test_command_detach_file_success(self, mock_session_state):
        """Tests that /detach successfully removes a file from the context."""
        file_path = Path("/test/file.txt").resolve()
        mock_session_state.attachments = {file_path: "content"}

        should_exit = commands.handle_detach(['file.txt'], mock_session_state, MagicMock())

        assert not should_exit
        assert file_path not in mock_session_state.attachments
        assert "user has detached" in utils.extract_text_from_message(mock_session_state.history[-1])

    def test_command_detach_file_not_found(self, mock_session_state, capsys):
        """Tests that /detach handles attempts to remove non-attached files."""
        mock_session_state.attachments = {Path("/test/file.txt"): "content"}
        commands.handle_detach(['non_existent'], mock_session_state, MagicMock())
        captured = capsys.readouterr()

        assert "No attached file found" in captured.out
        assert len(mock_session_state.attachments) == 1
        assert len(mock_session_state.history) == 0

    def test_command_remember_with_text(self, mock_session_state, fake_fs, mocker):
        """Tests that /remember <text> calls the API and updates the memory file."""
        fake_fs.create_file(config.PERSISTENT_MEMORY_FILE, contents="Initial memory.")
        mock_api = mocker.patch('aicli.api_client.perform_chat_request', return_value=("Initial memory. New fact.", {}))

        commands.handle_remember(['New', 'fact.'], mock_session_state, MagicMock())

        mock_api.assert_called_once()
        # Verify the prompt sent to the AI contains the right instruction
        call_kwargs = mock_api.call_args.kwargs
        messages = call_kwargs['messages_or_contents']
        api_prompt_text = utils.extract_text_from_message(messages[0])
        assert "--- NEW FACT TO INTEGRATE ---" in api_prompt_text
        assert "New fact." in api_prompt_text

        # Verify the memory file was updated
        with open(config.PERSISTENT_MEMORY_FILE, 'r') as f:
            assert f.read() == "Initial memory. New fact."

    def test_command_remember_no_text_consolidates_history(self, session_state_with_history, fake_fs, mocker):
        """Tests that /remember with no text consolidates session history into memory."""
        mock_api = mocker.patch('aicli.api_client.perform_chat_request', return_value=("Consolidated session.", {}))

        commands.handle_remember([], session_state_with_history, MagicMock())

        mock_api.assert_called_once()
        call_kwargs = mock_api.call_args.kwargs
        messages = call_kwargs['messages_or_contents']
        api_prompt_text = utils.extract_text_from_message(messages[0])
        assert "--- NEW CHAT SESSION TO INTEGRATE ---" in api_prompt_text
        assert "First prompt" in api_prompt_text
        assert "Second answer" in api_prompt_text

        with open(config.PERSISTENT_MEMORY_FILE, 'r') as f:
            assert f.read() == "Consolidated session."

    def test_command_quit_sets_flag(self, mock_session_state):
        """Tests that /quit sets the force_quit flag and signals exit."""
        should_exit = commands.handle_quit([], mock_session_state, MagicMock())
        assert should_exit is True
        assert mock_session_state.force_quit is True

    def test_command_exit_with_name_sets_rename(self, mock_session_state):
        """Tests that /exit [name] sets the custom_log_rename property."""
        should_exit = commands.handle_exit(['my', 'custom', 'log', 'name'], mock_session_state, MagicMock())
        assert should_exit is True
        assert mock_session_state.custom_log_rename == "my custom log name"

@pytest.fixture
def mock_multichat_state(mock_openai_engine, mock_gemini_engine):
    """Provides a fresh MultiChatSessionState for each test."""
    return MultiChatSessionState(
        openai_engine=mock_openai_engine,
        gemini_engine=mock_gemini_engine,
        openai_model="gpt-4o-mini",
        gemini_model="gemini-1.5-flash",
        max_tokens=1024,
        system_prompts={'openai': 'p1', 'gemini': 'p2'}
    )

class TestMultiChatCommands:
    """Test suite for multi-chat specific commands."""

    def test_multichat_exit(self, mock_multichat_state):
        """Tests that /exit returns True."""
        should_exit = commands.handle_multichat_exit([], mock_multichat_state, MagicMock())
        assert should_exit is True

    def test_multichat_help(self, mocker):
        """Tests that /help calls the correct display function."""
        mock_display_help = mocker.patch('aicli.utils.display_help')
        commands.handle_multichat_help([], MagicMock(), MagicMock())
        mock_display_help.assert_called_once_with('multichat')

    def test_multichat_history(self, mock_multichat_state, capsys):
        """Tests that /history prints the shared history."""
        mock_multichat_state.shared_history = [{"role": "user", "content": "test"}]
        commands.handle_multichat_history([], mock_multichat_state, MagicMock())
        captured = capsys.readouterr()
        assert '"role": "user"' in captured.out
        assert '"content": "test"' in captured.out
