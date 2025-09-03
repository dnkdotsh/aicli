# tests/test_handlers.py
"""
Tests for the main application logic handlers in aicli/handlers.py.
"""

import argparse
import base64

import pytest
from aicli import api_client, config, handlers


# Helper to create a standard mock args namespace
def create_mock_args(**kwargs):
    defaults = {
        "prompt": None,
        "both": None,
        "load": None,
        "image": False,
        "engine": "gemini",
        "model": None,
        "persona": None,
        "system_prompt": None,
        "file": None,
        "exclude": None,
        "memory": False,
        "session_name": None,
        "stream": None,
        "max_tokens": None,
        "debug": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture
def mock_session_manager(mocker):
    return mocker.patch("aicli.handlers.SessionManager")


@pytest.fixture
def mock_single_chat_manager(mocker):
    return mocker.patch("aicli.handlers.SingleChatManager")


@pytest.fixture
def mock_multi_chat_manager(mocker):
    return mocker.patch("aicli.handlers.MultiChatManager")


class TestHandleChat:
    """Tests for the main handle_chat function."""

    def test_single_shot_mode(self, mock_session_manager, mock_single_chat_manager):
        """Tests that a single-shot request correctly instantiates and calls SessionManager."""
        args = create_mock_args(engine="openai", model="gpt-4o")
        initial_prompt = "Test prompt."

        handlers.handle_chat(initial_prompt, args)

        mock_session_manager.assert_called_once()
        # Check that the SessionManager was instantiated with the correct args
        _, sm_kwargs = mock_session_manager.call_args
        assert sm_kwargs["engine_name"] == "openai"
        assert sm_kwargs["model"] == "gpt-4o"

        # Check that handle_single_shot was called and the interactive manager was not
        mock_session_manager.return_value.handle_single_shot.assert_called_once_with(
            initial_prompt
        )
        mock_single_chat_manager.assert_not_called()

    def test_interactive_mode(self, mock_session_manager, mock_single_chat_manager):
        """Tests that an interactive session correctly instantiates both managers."""
        args = create_mock_args(
            session_name="my_interactive_session", stream=False, debug=True
        )

        handlers.handle_chat(None, args)  # No initial prompt means interactive

        mock_session_manager.assert_called_once()
        _, sm_kwargs = mock_session_manager.call_args
        assert not sm_kwargs["stream_active"]
        assert sm_kwargs["debug_active"]

        mock_single_chat_manager.assert_called_once_with(
            mock_session_manager.return_value, "my_interactive_session"
        )
        mock_single_chat_manager.return_value.run.assert_called_once()


class TestHandleLoadSession:
    """Tests for the handle_load_session function."""

    def test_load_session_success(self, mock_session_manager, mock_single_chat_manager):
        """Tests successfully loading a session and starting the chat manager."""
        mock_session_manager.return_value.load.return_value = True
        filepath_str = "my_session.json"

        handlers.handle_load_session(filepath_str)

        # Check that SessionManager was instantiated and `load` was called
        mock_session_manager.assert_called_once()
        expected_path = str(config.SESSIONS_DIRECTORY / "my_session.json")
        mock_session_manager.return_value.load.assert_called_once_with(expected_path)

        # Check that the chat manager was started with the correct session name
        mock_single_chat_manager.assert_called_once_with(
            mock_session_manager.return_value, "my_session"
        )
        mock_single_chat_manager.return_value.run.assert_called_once()

    def test_load_session_failure_exits(self, mock_session_manager):
        """Tests that a failure in loading the session file causes a system exit."""
        mock_session_manager.return_value.load.return_value = False
        with pytest.raises(SystemExit) as excinfo:
            handlers.handle_load_session("non_existent.json")
        assert excinfo.value.code == 1


class TestHandleMultiChat:
    """Tests for the handle_multichat_session function."""

    @pytest.fixture(autouse=True)
    def setup_multichat_mocks(self, mocker):
        mocker.patch(
            "aicli.handlers.api_client.check_api_keys", return_value="fake_key"
        )
        mocker.patch("aicli.handlers.get_engine")

    def test_multichat_setup_and_delegation(self, mock_multi_chat_manager):
        """
        Tests that handle_multichat_session correctly initializes the state
        and delegates to the MultiChatManager.
        """
        args = create_mock_args(
            system_prompt="Global prompt",
            session_name="test_session",
            max_tokens=500,
        )
        initial_prompt = "Hello everyone"

        handlers.handle_multichat_session(initial_prompt, args)

        mock_multi_chat_manager.assert_called_once()
        mcm_args, _ = mock_multi_chat_manager.call_args
        state = mcm_args[0]  # The state object is the first positional argument
        session_name = mcm_args[1]
        prompt = mcm_args[2]

        assert session_name == "test_session"
        assert prompt == initial_prompt
        assert state.max_tokens == 500
        assert "Global prompt" in state.system_prompts["openai"]
        assert "You are OpenAI only" in state.system_prompts["openai"]
        assert "Global prompt" in state.system_prompts["gemini"]
        assert "You are Gemini only" in state.system_prompts["gemini"]

        mock_multi_chat_manager.return_value.run.assert_called_once()


class TestHandleImageGeneration:
    """Tests for the handle_image_generation function."""

    @pytest.fixture(autouse=True)
    def setup_image_mocks(self, mocker):
        mocker.patch("aicli.handlers.api_client.check_api_keys")
        mocker.patch("aicli.handlers.get_engine")
        mocker.patch("aicli.handlers.utils.select_model", return_value="dall-e-3")

    def test_image_generation_success(self, mocker, fake_fs):
        """Tests the successful image generation and saving logic."""
        mock_make_request = mocker.patch("aicli.api_client.make_api_request")
        fake_image_bytes = b"fake_png_data"
        b64_data = base64.b64encode(fake_image_bytes).decode("utf-8")
        mock_make_request.return_value = {"data": [{"b64_json": b64_data}]}

        args = create_mock_args()
        prompt = "A test image"

        handlers.handle_image_generation(prompt, args)

        saved_files = list(config.IMAGE_DIRECTORY.glob("*.png"))
        assert len(saved_files) == 1
        with open(saved_files[0], "rb") as f:
            assert f.read() == fake_image_bytes

    def test_image_generation_api_error(self, mocker, capsys):
        """Tests graceful failure when the API returns an error."""
        mocker.patch(
            "aicli.api_client.make_api_request",
            side_effect=api_client.ApiRequestError("API failed"),
        )
        args = create_mock_args()
        prompt = "A test image"
        handlers.handle_image_generation(prompt, args)
        captured = capsys.readouterr()
        assert "Error: API failed" in captured.err
