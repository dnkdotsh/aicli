# tests/test_handlers.py
"""
Tests for the main application logic handlers in aicli/handlers.py.
"""

import pytest
import base64
from pathlib import Path

from aicli import handlers
from aicli import config
from aicli.engine import OpenAIEngine

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
        session_name=None,
        max_tokens=100,
        stream=False,
        memory_enabled=False,
        debug_enabled=False
    )

    # Verify the API client was called correctly
    mock_perform_chat.assert_called_once()
    call_args = mock_perform_chat.call_args[0]
    assert call_args[0] == mock_engine
    assert call_args[1] == "gpt-4o-mini"
    assert call_args[3] == "Be brief."

    # Verify the output was printed to the console
    captured = capsys.readouterr()
    assert "Assistant:" in captured.out
    assert "Test response." in captured.out
    assert "[P:1/C:2/R:0/T:3]" in captured.out

def test_handle_image_generation(mocker, fake_fs):
    """
    Tests the image generation and saving logic.
    """
    # Mock the model selection and API request
    mocker.patch('aicli.handlers.select_model', return_value='dall-e-3')
    mock_make_request = mocker.patch(
        'aicli.api_client.make_api_request'
    )

    # Create fake base64 image data
    fake_image_bytes = b'fake_png_data'
    b64_data = base64.b64encode(fake_image_bytes).decode('utf-8')
    mock_make_request.return_value = {
        "data": [{"b64_json": b64_data}]
    }

    # Set up the fake filesystem for saving the image
    fake_fs.create_dir(config.IMAGE_DIRECTORY)

    mock_engine = OpenAIEngine("fake_key")
    handlers.handle_image_generation(
        api_key="fake_key",
        engine=mock_engine,
        prompt="A test image"
    )

    # Verify the image file was created in the fake filesystem
    saved_files = list(Path(config.IMAGE_DIRECTORY).glob('*.png'))
    assert len(saved_files) == 1
    with open(saved_files[0], 'rb') as f:
        assert f.read() == fake_image_bytes
