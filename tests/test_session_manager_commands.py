# tests/test_session_manager_commands.py
"""
Tests for the state-modifying and file-interacting slash commands
in aicli/commands.py.
"""

import json
from pathlib import Path

import pytest
from aicli import commands, config, utils
from aicli import settings as app_settings
from aicli.session_manager import MultiChatSessionState, SessionManager
from prompt_toolkit.history import InMemoryHistory


@pytest.fixture
def mock_session_manager(mocker, mock_openai_engine) -> SessionManager:
    """Provides a fresh, default SessionManager for each test."""
    mocker.patch("aicli.api_client.check_api_keys", return_value="fake_key")
    return SessionManager(engine_name="openai")


@pytest.fixture
def session_manager_with_history(mocker, mock_openai_engine) -> SessionManager:
    """Provides a SessionManager with a pre-populated history for testing."""
    mocker.patch("aicli.api_client.check_api_keys", return_value="fake_key")
    manager = SessionManager(engine_name="openai")
    history = [
        utils.construct_user_message("openai", "First prompt", []),
        utils.construct_assistant_message("openai", "First answer"),
        utils.construct_user_message("openai", "Second prompt", []),
        utils.construct_assistant_message("openai", "Second answer"),
    ]
    manager.state.history = history
    return manager


@pytest.fixture
def mock_cli_history(mocker) -> InMemoryHistory:
    """Provides a mock InMemoryHistory object with pre-populated strings for testing saves."""
    history = mocker.MagicMock(spec=InMemoryHistory)
    history.get_strings.return_value = ["/help", "first prompt"]
    return history


class TestSessionManagerFileCommands:
    """Test suite for commands that interact with files and memory."""

    def test_command_attach_file_success(self, mock_session_manager, fake_fs):
        """Tests that /attach successfully adds a file to the session context."""
        test_file = Path("/test/app.py")
        fake_fs.create_file(test_file, contents="print('hello')")

        commands.handle_attach([str(test_file)], mock_session_manager)

        assert test_file.resolve() in mock_session_manager.state.attachments
        assert (
            mock_session_manager.state.attachments[test_file.resolve()]
            == "print('hello')"
        )
        # Verify no system message was added to history
        assert len(mock_session_manager.state.history) == 0

    def test_command_attach_extensionless_file_success(
        self, mock_session_manager, fake_fs
    ):
        """Tests that /attach successfully adds a supported extensionless file."""
        dockerfile = Path("/test/Dockerfile")
        fake_fs.create_file(dockerfile, contents="FROM python:3.10")

        commands.handle_attach([str(dockerfile)], mock_session_manager)

        assert dockerfile.resolve() in mock_session_manager.state.attachments
        assert (
            mock_session_manager.state.attachments[dockerfile.resolve()]
            == "FROM python:3.10"
        )
        assert len(mock_session_manager.state.history) == 0

    def test_command_attach_file_not_found(self, mock_session_manager, capsys):
        """Tests that /attach handles non-existent files gracefully."""
        initial_attachment_count = len(mock_session_manager.state.attachments)
        commands.handle_attach(["/non/existent/file.txt"], mock_session_manager)
        captured = capsys.readouterr()

        assert "Error: Path not found" in captured.out
        assert len(mock_session_manager.state.attachments) == initial_attachment_count
        assert len(mock_session_manager.state.history) == 0

    def test_command_detach_file_success(self, mock_session_manager):
        """Tests that /detach successfully removes a file from the context."""
        file_path = Path("/test/file.txt").resolve()
        mock_session_manager.state.attachments = {file_path: "content"}

        commands.handle_detach(["file.txt"], mock_session_manager)

        assert file_path not in mock_session_manager.state.attachments
        assert len(mock_session_manager.state.history) == 0

    def test_command_detach_file_not_found(self, mock_session_manager, capsys):
        """Tests that /detach handles attempts to remove non-attached files."""
        mock_session_manager.state.attachments = {Path("/test/file.txt"): "content"}
        commands.handle_detach(["non_existent"], mock_session_manager)
        captured = capsys.readouterr()

        assert "No attached file named 'non_existent'" in captured.out
        assert len(mock_session_manager.state.attachments) == 1
        assert len(mock_session_manager.state.history) == 0

    def test_command_remember_with_text(self, mock_session_manager, fake_fs, mocker):
        """Tests that /remember <text> calls the API and updates the memory file."""
        fake_fs.create_file(config.PERSISTENT_MEMORY_FILE, contents="Initial memory.")
        mock_helper = mocker.patch.object(
            mock_session_manager,
            "_perform_helper_request",
            return_value=("Initial memory. New fact.", {}),
        )

        commands.handle_remember(["New", "fact."], mock_session_manager)

        mock_helper.assert_called_once()
        # Verify the prompt sent to the AI contains the right instruction
        prompt_text = mock_helper.call_args.args[0]
        assert "--- NEW FACT TO INTEGRATE ---" in prompt_text
        assert "New fact." in prompt_text

        # Verify the memory file was updated
        assert config.PERSISTENT_MEMORY_FILE.read_text() == "Initial memory. New fact."

    def test_command_remember_no_text_consolidates_history(
        self, session_manager_with_history, fake_fs, mocker
    ):
        """Tests that /remember with no text consolidates session history into memory."""
        mock_helper = mocker.patch.object(
            session_manager_with_history,
            "_perform_helper_request",
            return_value=("Consolidated session.", {}),
        )

        commands.handle_remember([], session_manager_with_history)

        mock_helper.assert_called_once()
        prompt_text = mock_helper.call_args.args[0]
        assert "--- NEW CHAT SESSION TO INTEGRATE ---" in prompt_text
        assert "First prompt" in prompt_text
        assert "Second answer" in prompt_text

        assert config.PERSISTENT_MEMORY_FILE.read_text() == "Consolidated session."

    def test_command_quit_sets_flag(self, mock_session_manager):
        """Tests that /quit sets the force_quit flag and signals exit."""
        should_exit = commands.handle_quit([], mock_session_manager)
        assert should_exit is True
        assert mock_session_manager.state.force_quit is True

    def test_command_exit_with_name_sets_rename(self, mock_session_manager):
        """Tests that /exit [name] sets the custom_log_rename property."""
        should_exit = commands.handle_exit(
            ["my", "custom", "log", "name"], mock_session_manager
        )
        assert should_exit is True
        assert mock_session_manager.state.custom_log_rename == "my custom log name"

    def test_command_files(self, mock_session_manager, fake_fs, capsys):
        """Tests that /files correctly lists attached files."""
        file1 = Path("/test/small.txt")
        file2 = Path("/test/large.py")
        fake_fs.create_file(file1, contents="a")
        fake_fs.create_file(file2, contents="a" * 1024)
        mock_session_manager.state.attachments = {
            file1.resolve(): "a",
            file2.resolve(): "a" * 1024,
        }

        commands.handle_files([], mock_session_manager)
        captured = capsys.readouterr()

        assert "--- Attached Files ---" in captured.out
        assert captured.out.find("large.py (1.00 KB)") < captured.out.find(
            "small.txt (1.00 B)"
        )

    def test_command_refresh_success(self, mock_session_manager, fake_fs, capsys):
        """Tests that /refresh updates file content in the session state."""
        test_file = Path("/test/app.py")
        fake_fs.create_file(test_file, contents="initial content")
        mock_session_manager.state.attachments = {
            test_file.resolve(): "initial content"
        }

        test_file.write_text("updated content")

        commands.handle_refresh([], mock_session_manager)

        assert (
            mock_session_manager.state.attachments[test_file.resolve()]
            == "updated content"
        )
        assert "Refreshed: app.py" in capsys.readouterr().out

    def test_command_refresh_file_deleted(self, mock_session_manager, fake_fs, capsys):
        """Tests that /refresh removes a file from context if it's been deleted."""
        test_file = Path("/test/app.py").resolve()
        mock_session_manager.state.attachments = {test_file: "content"}

        commands.handle_refresh([], mock_session_manager)

        assert test_file not in mock_session_manager.state.attachments
        assert "Removed (not found): app.py" in capsys.readouterr().out

    def test_command_save_and_load_session(
        self, session_manager_with_history, fake_fs, mock_cli_history
    ):
        """An integration test for saving and then loading a session."""
        filename_to_save = "my_test_session"
        commands.handle_save(
            [filename_to_save], session_manager_with_history, mock_cli_history
        )

        saved_file_path = config.SESSIONS_DIRECTORY / f"{filename_to_save}.json"
        assert saved_file_path.exists()

        # Create a new manager and load the state into it
        new_manager = SessionManager(
            engine_name="gemini"
        )  # Start with a different engine
        assert new_manager.load(filename_to_save) is True

        assert new_manager.state.model == session_manager_with_history.state.model
        assert new_manager.state.engine.name == "openai"  # Engine should be loaded
        assert len(new_manager.state.history) == 4
        assert (
            utils.extract_text_from_message(new_manager.state.history[0])
            == "First prompt"
        )
        assert new_manager.state.command_history == ["/help", "first prompt"]


class TestSessionStateCommands:
    """Test suite for commands that modify the session's state directly."""

    def test_command_clear_confirmed(self, session_manager_with_history, mocker):
        """Tests that /clear empties the history after user confirmation."""
        mocker.patch("aicli.commands.prompt", return_value="proceed")

        assert len(session_manager_with_history.state.history) > 0
        commands.handle_clear([], session_manager_with_history)
        assert len(session_manager_with_history.state.history) == 0

    def test_command_clear_cancelled(self, session_manager_with_history, mocker):
        """Tests that /clear does nothing if the user cancels."""
        mocker.patch("aicli.commands.prompt", return_value="cancel")

        assert len(session_manager_with_history.state.history) > 0
        commands.handle_clear([], session_manager_with_history)
        assert len(session_manager_with_history.state.history) > 0

    def test_command_engine_switch(self, mock_session_manager, mocker):
        """Tests switching the AI engine and model."""
        mocker.patch("aicli.api_client.check_api_keys", return_value="gemini_key")
        mocker.patch(
            "aicli.utils.translate_history", return_value=[{"translated": True}]
        )

        assert mock_session_manager.state.engine.name == "openai"
        commands.handle_engine(["gemini"], mock_session_manager)
        assert mock_session_manager.state.engine.name == "gemini"
        assert (
            mock_session_manager.state.model
            == app_settings.settings["default_gemini_model"]
        )
        assert mock_session_manager.state.history == [{"translated": True}]

    def test_command_persona_switch_and_clear(self, mock_session_manager, fake_fs):
        """Tests switching to a new persona and then clearing it."""
        persona_content = {
            "name": "Code Reviewer",
            "system_prompt": "You are a code reviewer.",
            "engine": "openai",
            "model": "gpt-4-turbo",
        }
        fake_fs.create_file(
            config.PERSONAS_DIRECTORY / "reviewer.json",
            contents=json.dumps(persona_content),
        )
        mock_session_manager.state.initial_system_prompt = "initial prompt"

        commands.handle_persona(["reviewer"], mock_session_manager)

        assert mock_session_manager.state.current_persona is not None
        assert mock_session_manager.state.current_persona.name == "Code Reviewer"
        assert mock_session_manager.state.system_prompt == "You are a code reviewer."
        assert mock_session_manager.state.model == "gpt-4-turbo"
        assert "[SYSTEM] Persona switched" in utils.extract_text_from_message(
            mock_session_manager.state.history[-1]
        )

        commands.handle_persona(["clear"], mock_session_manager)

        assert mock_session_manager.state.current_persona is None
        assert mock_session_manager.state.system_prompt == "initial prompt"
        assert "[SYSTEM] Persona cleared" in utils.extract_text_from_message(
            mock_session_manager.state.history[-1]
        )


@pytest.fixture
def mock_multichat_state(mock_openai_engine, mock_gemini_engine):
    """Provides a fresh MultiChatSessionState for each test."""
    return MultiChatSessionState(
        openai_engine=mock_openai_engine,
        gemini_engine=mock_gemini_engine,
        openai_model="gpt-4o-mini",
        gemini_model="gemini-1.5-flash",
        max_tokens=1024,
        system_prompts={"openai": "p1", "gemini": "p2"},
    )


class TestMultiChatCommands:
    """Test suite for multi-chat specific commands."""

    def test_multichat_exit(self, mock_multichat_state):
        """Tests that /exit returns True."""
        should_exit = commands.handle_multichat_exit([], mock_multichat_state, None)
        assert should_exit is True

    def test_multichat_help(self, mocker):
        """Tests that /help calls the correct display function."""
        mock_display_help = mocker.patch("aicli.utils.display_help")
        commands.handle_multichat_help(None, None, None)
        mock_display_help.assert_called_once_with("multichat")

    def test_multichat_history(self, mock_multichat_state, capsys):
        """Tests that /history prints the shared history."""
        mock_multichat_state.shared_history = [{"role": "user", "content": "test"}]
        commands.handle_multichat_history([], mock_multichat_state, None)
        captured = capsys.readouterr()
        assert '"role": "user"' in captured.out
        assert '"content": "test"' in captured.out

    def test_multichat_model_command(self, mock_multichat_state):
        """Tests that /model correctly updates the targeted model in multichat state."""
        commands.handle_multichat_model(
            ["gpt", "new-gpt-model"], mock_multichat_state, None
        )
        assert mock_multichat_state.openai_model == "new-gpt-model"
        assert mock_multichat_state.gemini_model == "gemini-1.5-flash"

        commands.handle_multichat_model(
            ["gem", "new-gem-model"], mock_multichat_state, None
        )
        assert mock_multichat_state.openai_model == "new-gpt-model"
        assert mock_multichat_state.gemini_model == "new-gem-model"

    def test_multichat_save_command_success(
        self, mock_multichat_state, fake_fs, mock_cli_history
    ):
        """Tests that the /save command for multichat sessions works correctly."""
        should_exit = commands.handle_multichat_save(
            ["my-multi-session"], mock_multichat_state, mock_cli_history
        )

        assert should_exit is True
        saved_path = config.SESSIONS_DIRECTORY / "my_multi_session.json"
        assert saved_path.exists()

        with open(saved_path) as f:
            data = json.load(f)

        assert data["session_type"] == "multichat"
        assert data["openai_model"] == "gpt-4o-mini"
        assert "openai_engine" not in data  # Engines should be removed
        assert data["command_history"] == ["/help", "first prompt"]
