# tests/conftest.py
"""
This module contains shared fixtures for the pytest suite.
Fixtures defined here are automatically available to all test functions.
"""

import logging

import pytest
from aicli import config  # Import config to access LOG_DIRECTORY
from aicli.engine import GeminiEngine, OpenAIEngine
from aicli.managers.session_manager import SessionState
from pyfakefs.fake_filesystem_unittest import Patcher


# Configure caplog to capture the 'aicli' logger
# This hook ensures the logger is configured before tests run
def pytest_configure(config):
    logging.getLogger("aicli").propagate = True


@pytest.fixture
def mock_openai_engine():
    """Provides a mock OpenAIEngine instance."""
    return OpenAIEngine(api_key="fake_openai_key")


@pytest.fixture
def mock_gemini_engine():
    """Provides a mock GeminiEngine instance."""
    return GeminiEngine(api_key="fake_gemini_key")


@pytest.fixture
def mock_session_state(mock_openai_engine):
    """Provides a basic SessionState instance for testing."""
    return SessionState(
        engine=mock_openai_engine,
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        initial_system_prompt="You are a helpful assistant.",
        current_persona=None,
        max_tokens=1024,
        memory_enabled=True,
        debug_active=False,
        stream_active=True,
    )


@pytest.fixture
def fake_fs():
    """
    Initializes a fake filesystem using pyfakefs for tests that
    require filesystem interactions (e.g., reading/writing settings,
    sessions, logs).
    Ensures core application directories exist in the fake filesystem.
    """
    patcher = Patcher()
    patcher.setUp()
    try:
        # Create all necessary application directories within the fake filesystem.
        config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        config.IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        config.SESSIONS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        config.PERSONAS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        yield patcher.fs
    finally:
        patcher.tearDown()


@pytest.fixture
def mock_requests_post(mocker):
    """
    A fixture that mocks `requests.post` to prevent actual network calls.
    Returns the mock object for customization within tests.
    """
    return mocker.patch("requests.post")


@pytest.fixture
def mock_openai_chat_response():
    """A fixture providing a standard, non-streaming OpenAI API chat response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_gemini_chat_response():
    """A fixture providing a standard, non-streaming Gemini API chat response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a test response."}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 25,
            "totalTokenCount": 40,
            "cachedContentTokenCount": 5,
        },
    }


@pytest.fixture
def mock_prompt_toolkit(mocker):
    """
    Mocks prompt_toolkit.prompt and underlying sys.stdin interactions
    to avoid EOFError in interactive tests.
    It provides an input_queue for tests to push expected inputs into.
    """
    input_queue = []

    # Mock sys.stdin.isatty to make prompt_toolkit think it's an interactive terminal.
    mocker.patch("sys.stdin.isatty", return_value=True)

    # Patch prompt_toolkit.prompt where it is USED (in the handlers module).
    def _mocked_prompt_side_effect(message="", **kwargs):
        if not input_queue:
            # When the queue is empty, raise EOFError to simulate the user
            # pressing Ctrl+D, allowing loops to terminate gracefully.
            raise EOFError
        return input_queue.pop(0)

    # Patch prompt in all modules where it might be called during tests
    mocker.patch("aicli.handlers.prompt", side_effect=_mocked_prompt_side_effect)
    mocker.patch("aicli.session_manager.prompt", side_effect=_mocked_prompt_side_effect)

    return {
        "input_queue": input_queue  # Tests will populate this list
    }
