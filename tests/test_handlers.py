# tests/test_handlers.py
"""
Tests for the main application logic handlers in aicli/handlers.py.
"""

import pytest
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import io
import queue
import os
import logging

from aicli import handlers
from aicli import config
from aicli import api_client
from aicli.engine import OpenAIEngine, GeminiEngine
from aicli.session_manager import SessionState, MultiChatSessionState

def test_handle_chat_single_shot(mocker, capsys):
    """
    Tests that a single-shot chat request is handled correctly.
    """
    # Mock the API client to prevent network calls
    mock_perform_chat = mocker.patch(
        'aicli.api_client.perform_chat_request',
        return_value=("Test response.", {'prompt': 1, 'completion': 2, 'total': 3, 'reasoning': 0})
    )
    mock_engine = OpenAIEngine("fake_key")

    handlers.handle_chat(
        engine=mock_engine,
        model="gpt-4o-mini",
        system_prompt="Be brief.",
        initial_prompt="Test prompt.",
        image_data=[],
        attachments={},
        session_name=None,
        max_tokens=100,
        stream=False,
        memory_enabled=False,
        debug_enabled=False,
        initial_system_prompt=None,
        persona=None
    )

    # Verify the API client was called correctly
    mock_perform_chat.assert_called_once()
    call_args, _ = mock_perform_chat.call_args
    assert call_args[0] == mock_engine
    assert call_args[1] == "gpt-4o-mini"
    assert call_args[3] == "Be brief."

    # Verify the output was printed to the console
    captured = capsys.readouterr()
    assert "Assistant:" in captured.out
    assert "Test response." in captured.out
    assert "[P:1/C:2/R:0/T:3]" in captured.out

def test_handle_chat_interactive_mode_setup(mocker, mock_openai_engine):
    """
    Tests that handle_chat correctly sets up and delegates to perform_interactive_chat
    when no initial_prompt is given (interactive mode).
    """
    mock_perform_interactive_chat = mocker.patch('aicli.handlers.perform_interactive_chat')

    test_attachments = {Path('/a/file.txt'): 'content'}

    handlers.handle_chat(
        engine=mock_openai_engine,
        model="gpt-4o-mini",
        system_prompt="Interactive system prompt.",
        initial_prompt=None, # This triggers interactive mode
        image_data=['img1'],
        attachments=test_attachments,
        session_name=None,
        max_tokens=200,
        stream=True,
        memory_enabled=True,
        debug_enabled=True,
        initial_system_prompt="Interactive system prompt.",
        persona=None
    )

    mock_perform_interactive_chat.assert_called_once()
    call_args, _ = mock_perform_interactive_chat.call_args
    session_state: SessionState = call_args[0]
    session_name_arg = call_args[1]

    assert isinstance(session_state, SessionState)
    assert session_state.engine == mock_openai_engine
    assert session_state.model == "gpt-4o-mini"
    assert session_state.system_prompt == "Interactive system prompt."
    assert session_state.attached_images == ['img1']
    assert session_state.attachments == test_attachments
    assert session_state.stream_active is True
    assert session_state.memory_enabled is True
    assert session_state.debug_active is True
    assert session_state.max_tokens == 200
    assert session_name_arg is None


def test_handle_chat_single_shot_with_attachments_and_system_prompt(mocker):
    """
    Tests that for single-shot requests, file attachments and system prompt
    are correctly combined into the full_system_prompt.
    """
    mock_perform_chat = mocker.patch('aicli.api_client.perform_chat_request', return_value=("Response", {}))
    mock_engine = OpenAIEngine("fake_key")

    test_file_path = Path('/test/file.md')
    test_attachments = {test_file_path: 'File content here.'}

    handlers.handle_chat(
        engine=mock_engine,
        model="gpt-4o-mini",
        system_prompt="User's system instruction.",
        initial_prompt="My question.",
        image_data=[],
        attachments=test_attachments,
        session_name=None,
        max_tokens=None,
        stream=False,
        memory_enabled=False,
        debug_enabled=False,
        initial_system_prompt=None,
        persona=None
    )

    mock_perform_chat.assert_called_once()
    call_args, _ = mock_perform_chat.call_args

    # Check the full_system_prompt argument
    expected_full_system_prompt = (
        "User's system instruction."
        "\n\n--- ATTACHED FILES ---\n"
        f"--- FILE: {test_file_path.as_posix()} ---\nFile content here."
    )
    assert call_args[3] == expected_full_system_prompt

def test_handle_load_session_success(mocker):
    """Tests successfully loading a session and resuming interactive chat."""
    mock_load_session = mocker.patch('aicli.commands.load_session_from_file')
    mock_perform_chat = mocker.patch('aicli.handlers.perform_interactive_chat')

    fake_session_state = MagicMock(spec=SessionState)
    fake_session_state.history = []
    fake_session_state.command_history = []
    fake_session_state.engine = MagicMock(name='mock_engine_obj')
    fake_session_state.engine.name = 'openai'
    fake_session_state.model = 'gpt-4o-mini'
    fake_session_state.system_prompt = None
    fake_session_state.attachments = {}
    fake_session_state.attached_images = []
    fake_session_state.memory_enabled = True
    mock_load_session.return_value = fake_session_state

    test_filepath = Path("/fake/path/my_session.json")
    handlers.handle_load_session(str(test_filepath))

    mock_load_session.assert_called_once_with(test_filepath)
    mock_perform_chat.assert_called_once_with(fake_session_state, "my_session")

def test_handle_load_session_file_not_found_exits(mocker):
    """Tests that loading a non-existent session file causes a system exit."""
    mocker.patch('aicli.commands.load_session_from_file', return_value=None)

    with pytest.raises(SystemExit) as excinfo:
        handlers.handle_load_session("non_existent.json")
    assert excinfo.value.code == 1

def test_handle_load_session_missing_api_key_exits(mocker):
    """Tests that a missing API key on session load correctly exits the program."""
    mocker.patch(
        'aicli.commands.load_session_from_file',
        side_effect=api_client.MissingApiKeyError("Missing Fake Key")
    )

    with pytest.raises(SystemExit) as excinfo:
        handlers.handle_load_session("my_session.json")

    assert excinfo.value.code == 1

def test_handle_image_generation(mocker, fake_fs):
    """
    Tests the image generation and saving logic.
    """
    mocker.patch('aicli.handlers.select_model', return_value='dall-e-3')
    mock_make_request = mocker.patch('aicli.api_client.make_api_request')

    fake_image_bytes = b'fake_png_data'
    b64_data = base64.b64encode(fake_image_bytes).decode('utf-8')
    mock_make_request.return_value = {
        "data": [{"b64_json": b64_data}]
    }

    mock_engine = OpenAIEngine("fake_key")
    handlers.handle_image_generation(
        api_key="fake_key",
        engine=mock_engine,
        prompt="A test image"
    )

    saved_files = list(Path(config.IMAGE_DIRECTORY).glob('*.png'))
    assert len(saved_files) == 1
    with open(saved_files[0], 'rb') as f:
        assert f.read() == fake_image_bytes

class TestMultiChatHandler:
    @pytest.fixture(autouse=True)
    def setup_multichat_mocks(self, mocker):
        mocker.patch('aicli.api_client.check_api_keys', return_value="fake_key")
        self.mock_openai_engine = MagicMock(spec=OpenAIEngine)
        self.mock_openai_engine.name = 'openai'
        self.mock_gemini_engine = MagicMock(spec=GeminiEngine)
        self.mock_gemini_engine.name = 'gemini'
        # Patch get_engine where it is USED (in the handlers module), not where it is defined.
        mocker.patch('aicli.handlers.get_engine', side_effect=lambda name, key: self.mock_openai_engine if name == 'openai' else self.mock_gemini_engine)
        self.mock_perform_multichat = mocker.patch('aicli.handlers.perform_multichat_session')

    def test_handle_multichat_setup_and_delegation(self):
        """
        Tests that handle_multichat_session correctly initializes the state
        and delegates to the session manager.
        """
        handlers.handle_multichat_session(
            initial_prompt="Hello",
            system_prompt="Global prompt",
            image_data=["image1"],
            session_name="test_session",
            max_tokens=500,
            debug_enabled=False
        )

        self.mock_perform_multichat.assert_called_once()
        call_args, _ = self.mock_perform_multichat.call_args
        state: MultiChatSessionState = call_args[0]
        session_name_arg = call_args[1]

        assert isinstance(state, MultiChatSessionState)
        assert state.openai_engine == self.mock_openai_engine
        assert state.gemini_engine == self.mock_gemini_engine
        assert state.max_tokens == 500
        assert "Global prompt" in state.system_prompts['openai']
        assert "You are OpenAI only" in state.system_prompts['openai']
        assert "Global prompt" in state.system_prompts['gemini']
        assert "You are Gemini only" in state.system_prompts['gemini']
        assert state.initial_image_data == ["image1"]
        assert session_name_arg == "test_session"

    def test_handle_multichat_missing_api_key_exits(self, mocker, caplog):
        """
        Tests that a missing API key during setup causes a clean exit.
        """
        mocker.patch('aicli.api_client.check_api_keys', side_effect=api_client.MissingApiKeyError("Missing Key"))
        with pytest.raises(SystemExit) as excinfo, caplog.at_level(logging.ERROR):
            handlers.handle_multichat_session(None, None, [], None, 100, False)

        assert excinfo.value.code == 1
        assert "Missing Key" in caplog.text
        self.mock_perform_multichat.assert_not_called()

class TestSelectModel:
    @pytest.fixture(autouse=True)
    def setup_select_model_mocks(self, mocker, mock_prompt_toolkit):
        self.mock_engine = MagicMock(spec=OpenAIEngine)
        self.mock_engine.name = 'openai'
        self.mock_engine.fetch_available_models.return_value = ["model-a", "model-b", "model-c"]
        mocker.patch.dict(handlers.settings, {
            'default_openai_chat_model': 'gpt-4o-mini',
            'default_openai_image_model': 'dall-e-3',
            'default_gemini_model': 'gemini-2.5-flash'
        })
        mocker.patch('sys.stdout', new_callable=io.StringIO)
        self.mock_prompt_toolkit = mock_prompt_toolkit

    def test_select_model_default_selection_chat(self, mocker, mock_prompt_toolkit):
        """Tests that select_model returns the default chat model if user presses Enter."""
        mock_prompt_toolkit['input_queue'].append('y')
        model = handlers.select_model(self.mock_engine, 'chat')
        assert model == 'gpt-4o-mini'
        self.mock_engine.fetch_available_models.assert_not_called()

    def test_select_model_default_selection_image(self, mocker, mock_prompt_toolkit):
        """Tests that select_model returns the default image model if user presses Enter."""
        mock_prompt_toolkit['input_queue'].append('') # Enter is default 'y'
        model = handlers.select_model(self.mock_engine, 'image')
        assert model == 'dall-e-3'
        self.mock_engine.fetch_available_models.assert_not_called()

    def test_select_model_user_selection(self, mocker, mock_prompt_toolkit):
        """Tests that a valid numeric user choice for a model is returned."""
        mock_prompt_toolkit['input_queue'].extend(['n', '2'])
        model = handlers.select_model(self.mock_engine, 'chat')
        assert model == 'model-b'

    def test_select_model_invalid_selection_falls_back_to_default(self, mocker, capsys, mock_prompt_toolkit):
        """Tests that invalid input for model selection falls back to the default."""
        mock_prompt_toolkit['input_queue'].extend(['n', '99', 'invalid', ''])
        model = handlers.select_model(self.mock_engine, 'chat')
        assert model == 'gpt-4o-mini'
        output = capsys.readouterr().out
        assert "Invalid selection. Using default: gpt-4o-mini" in output

    def test_select_model_no_available_models_falls_back_to_default(self, mocker, capsys, mock_prompt_toolkit):
        """Tests behavior when fetch_available_models returns an empty list."""
        self.mock_engine.fetch_available_models.return_value = []
        mock_prompt_toolkit['input_queue'].extend(['n', ''])

        model = handlers.select_model(self.mock_engine, 'chat')
        assert model == 'gpt-4o-mini'
        output = capsys.readouterr().out
        assert "Fetching available models..." in output
        assert "Using default: gpt-4o-mini" in output
