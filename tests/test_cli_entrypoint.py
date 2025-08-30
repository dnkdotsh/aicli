# tests/test_cli_entrypoint.py
"""
Tests for the command-line argument parsing and dispatching in aicli/aicli.py.
"""

import pytest
import sys
import io

# Import the main module we want to test
from aicli import aicli

@pytest.fixture(autouse=True)
def mock_handlers(mocker):
    """
    A fixture to automatically mock all handler functions.
    This isolates the tests to only the argument parsing and dispatching logic
    in aicli.py, preventing the actual handler code from running.
    """
    mocker.patch('aicli.handlers.handle_chat')
    mocker.patch('aicli.handlers.handle_image_generation')
    mocker.patch('aicli.handlers.handle_multichat_session')
    mocker.patch('aicli.handlers.handle_load_session')
    mocker.patch('aicli.api_client.check_api_keys', return_value="fake_key")
    mocker.patch('aicli.utils.ensure_dir_exists')
    mocker.patch('aicli.engine.get_engine')


class TestCLIEntrypoint:
    """Test suite for the main CLI entrypoint."""

    def test_no_args_interactive_chat(self, monkeypatch):
        """Tests that running `aicli` with no args starts an interactive chat."""
        monkeypatch.setattr(sys, 'argv', ['aicli'])
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        aicli.main()
        aicli.handlers.handle_chat.assert_called_once()
        # Check that it's an interactive call (initial_prompt is None)
        call_args = aicli.handlers.handle_chat.call_args[0]
        assert call_args[3] is None

    def test_prompt_arg_calls_handle_chat(self, monkeypatch):
        """Tests that `aicli -p "prompt"` calls the chat handler."""
        monkeypatch.setattr(sys, 'argv', ['aicli', '-p', 'hello world'])
        aicli.main()
        aicli.handlers.handle_chat.assert_called_once()
        # Check that the prompt is passed correctly
        call_args = aicli.handlers.handle_chat.call_args
        assert call_args.kwargs['prompt'] == 'hello world'

    def test_piped_input_calls_handle_chat(self, monkeypatch):
        """Tests that piped input is correctly passed as a prompt."""
        monkeypatch.setattr(sys, 'argv', ['aicli'])
        # Simulate piped input
        monkeypatch.setattr('sys.stdin', io.StringIO('piped content'))
        aicli.main()
        aicli.handlers.handle_chat.assert_called_once()
        # Check that the piped content is used as the prompt
        call_args = aicli.handlers.handle_chat.call_args
        assert call_args.kwargs['prompt'] == 'piped content'

    def test_image_mode_calls_handle_image(self, monkeypatch):
        """Tests that `aicli --image` calls the image handler."""
        monkeypatch.setattr(sys, 'argv', ['aicli', '--image', '-p', 'a test image'])
        aicli.main()
        aicli.handlers.handle_image_generation.assert_called_once()
        aicli.handlers.handle_chat.assert_not_called()

    def test_both_mode_calls_multichat(self, monkeypatch):
        """Tests that `aicli --both` calls the multichat handler."""
        monkeypatch.setattr(sys, 'argv', ['aicli', '--both', 'a test prompt'])
        aicli.main()
        aicli.handlers.handle_multichat_session.assert_called_once()
        aicli.handlers.handle_chat.assert_not_called()

    def test_load_mode_calls_load_session(self, monkeypatch):
        """Tests that `aicli --load` calls the load session handler."""
        monkeypatch.setattr(sys, 'argv', ['aicli', '--load', 'my_session.json'])
        aicli.main()
        aicli.handlers.handle_load_session.assert_called_once_with('my_session.json')
        aicli.handlers.handle_chat.assert_not_called()

    def test_invalid_args_exits(self, monkeypatch):
        """Tests that an invalid argument causes a system exit."""
        monkeypatch.setattr(sys, 'argv', ['aicli', '--non-existent-arg'])
        with pytest.raises(SystemExit):
            aicli.main()

    def test_image_with_gemini_exits(self, monkeypatch):
        """Tests that using --image with the gemini engine correctly exits."""
        monkeypatch.setattr(sys, 'argv', ['aicli', '--image', '--engine', 'gemini'])
        with pytest.raises(SystemExit):
            aicli.main()
