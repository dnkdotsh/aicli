# tests/test_session_manager.py
"""
Tests for the core logic and state management of the SessionManager class.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aicli import config, utils
from aicli.session_manager import SessionManager


@pytest.fixture
def mock_session_manager(mocker) -> SessionManager:
    """Provides a fresh, default SessionManager for each test."""
    mocker.patch("aicli.api_client.check_api_keys", return_value="fake_key")
    mocker.patch("aicli.engine.get_engine")
    return SessionManager(engine_name="openai")


class TestSessionManagerInitialization:
    """Tests for the SessionManager's constructor and initial state setup."""

    def test_init_with_files_and_memory(self, mocker, fake_fs):
        """Tests that the constructor correctly processes files and persistent memory."""
        mocker.patch("aicli.api_client.check_api_keys", return_value="fake_key")
        mocker.patch("aicli.engine.get_engine")

        fake_fs.create_file(config.PERSISTENT_MEMORY_FILE, contents="ltm data")
        fake_fs.create_file("/test/app.py", contents="app content")

        manager = SessionManager(
            engine_name="openai",
            files_arg=["/test/app.py"],
            memory_enabled=True,
        )

        assert "ltm data" in manager.state.system_prompt
        assert Path("/test/app.py").resolve() in manager.state.attachments
        assert (
            manager.state.attachments[Path("/test/app.py").resolve()] == "app content"
        )

    def test_init_system_prompt_precedence(self, mocker, fake_fs):
        """Tests that a direct system_prompt argument overrides a persona's prompt."""
        mocker.patch("aicli.api_client.check_api_keys", return_value="fake_key")
        mocker.patch("aicli.engine.get_engine")

        mock_persona = MagicMock()
        mock_persona.system_prompt = "persona prompt"

        manager = SessionManager(
            engine_name="openai",
            persona=mock_persona,
            system_prompt_arg="direct argument prompt",
        )

        assert manager.state.system_prompt == "direct argument prompt"
        assert manager.state.initial_system_prompt == "direct argument prompt"


class TestSessionManagerLogic:
    """Tests for the core methods and logic of SessionManager."""

    def test_toggle_stream(self, mock_session_manager):
        """Tests that toggle_stream correctly modifies the session state."""
        initial_state = mock_session_manager.state.stream_active
        mock_session_manager.toggle_stream()
        assert mock_session_manager.state.stream_active is not initial_state
        mock_session_manager.toggle_stream()
        assert mock_session_manager.state.stream_active is initial_state

    def test_toggle_debug(self, mock_session_manager):
        """Tests that toggle_debug correctly modifies the session state."""
        assert mock_session_manager.state.debug_active is False
        mock_session_manager.toggle_debug()
        assert mock_session_manager.state.debug_active is True

    def test_set_model(self, mock_session_manager):
        """Tests that set_model updates the model in the session state."""
        mock_session_manager.set_model("new-model-name")
        assert mock_session_manager.state.model == "new-model-name"

    def test_condense_chat_history_is_triggered(self, mocker):
        """
        Tests that taking a turn beyond the history threshold triggers condensation.
        """
        mocker.patch("aicli.api_client.check_api_keys", return_value="fake_key")
        mocker.patch("aicli.engine.get_engine")
        # Mock the helper request to control the summary and prevent network calls
        mock_helper_request = mocker.patch(
            "aicli.session_manager.SessionManager._perform_helper_request",
            return_value=("Condensed summary.", {}),
        )
        # Mock the main API call for taking a turn
        mocker.patch(
            "aicli.api_client.perform_chat_request", return_value=("answer", {})
        )

        manager = SessionManager(engine_name="openai")

        # Fill the history up to the threshold
        # A turn is 2 messages (user + assistant)
        for _ in range(config.HISTORY_SUMMARY_THRESHOLD_TURNS):
            manager.state.history.append(
                utils.construct_user_message("openai", "q", [])
            )
            manager.state.history.append(
                utils.construct_assistant_message("openai", "a")
            )

        assert mock_helper_request.call_count == 0

        # This turn should push it over the edge and trigger condensation
        manager.take_turn("trigger prompt", first_turn=False)

        mock_helper_request.assert_called_once()
        summary_text = utils.extract_text_from_message(manager.state.history[0])
        assert "Condensed summary." in summary_text


class TestSessionManagerCleanup:
    """Tests for the session cleanup logic."""

    @pytest.fixture
    def setup_mocks(self, mocker):
        mocker.patch("aicli.api_client.check_api_keys", return_value="fake_key")
        mocker.patch("aicli.engine.get_engine")
        mocker.patch(
            "aicli.session_manager.SessionManager._perform_helper_request",
            return_value=("summary", {}),
        )

    def test_cleanup_consolidates_memory(self, setup_mocks, fake_fs, mocker):
        """Tests that cleanup consolidates memory if enabled and not quitting."""
        log_file = config.LOG_DIRECTORY / "test.jsonl"
        log_file.touch()

        manager = SessionManager(engine_name="openai", memory_enabled=True)
        manager.state.history = [utils.construct_user_message("openai", "test", [])]
        mock_consolidate = mocker.spy(manager, "_consolidate_memory_from_session")

        manager.cleanup(session_name=None, log_filepath=log_file)

        mock_consolidate.assert_called_once()

    def test_cleanup_skips_memory_if_disabled(self, setup_mocks, fake_fs, mocker):
        """Tests that cleanup is skipped if memory is disabled."""
        log_file = config.LOG_DIRECTORY / "test.jsonl"
        log_file.touch()

        manager = SessionManager(engine_name="openai", memory_enabled=False)
        manager.state.history = [utils.construct_user_message("openai", "test", [])]
        mock_consolidate = mocker.spy(manager, "_consolidate_memory_from_session")

        manager.cleanup(session_name=None, log_filepath=log_file)

        mock_consolidate.assert_not_called()

    def test_cleanup_skips_on_force_quit(self, setup_mocks, fake_fs, mocker):
        """Tests that cleanup is skipped entirely on a force quit."""
        log_file = config.LOG_DIRECTORY / "test.jsonl"
        log_file.touch()

        manager = SessionManager(engine_name="openai", memory_enabled=True)
        manager.state.history = [utils.construct_user_message("openai", "test", [])]
        manager.set_force_quit()  # Set the force_quit flag
        mock_consolidate = mocker.spy(manager, "_consolidate_memory_from_session")
        mock_rename = mocker.spy(manager, "_rename_log_with_ai")

        manager.cleanup(session_name=None, log_filepath=log_file)

        mock_consolidate.assert_not_called()
        mock_rename.assert_not_called()

    def test_cleanup_renames_with_ai_if_no_session_name(
        self, setup_mocks, fake_fs, mocker
    ):
        """Tests that the AI is asked to rename the log if no name is provided."""
        log_file = config.LOG_DIRECTORY / "test.jsonl"
        log_file.touch()

        manager = SessionManager(engine_name="openai")
        manager.state.history = [utils.construct_user_message("openai", "test", [])]
        mock_rename_ai = mocker.spy(manager, "_rename_log_with_ai")

        manager.cleanup(session_name=None, log_filepath=log_file)

        mock_rename_ai.assert_called_once()

    def test_cleanup_uses_custom_rename(self, setup_mocks, fake_fs, mocker):
        """Tests that a custom rename from /exit is used."""
        log_file = config.LOG_DIRECTORY / "test.jsonl"
        log_file.touch()

        manager = SessionManager(engine_name="openai")
        manager.state.history = [utils.construct_user_message("openai", "test", [])]
        manager.set_custom_log_rename("my custom name")
        mock_rename_ai = mocker.spy(manager, "_rename_log_with_ai")
        mock_rename_file = mocker.spy(manager, "_rename_log_file")

        manager.cleanup(session_name=None, log_filepath=log_file)

        mock_rename_ai.assert_not_called()
        mock_rename_file.assert_called_once_with(log_file, "my custom name")
