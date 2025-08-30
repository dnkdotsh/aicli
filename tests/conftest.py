# tests/conftest.py
"""
This module contains shared fixtures for the pytest suite.
Fixtures defined here are automatically available to all test functions.
"""

import pytest
from unittest.mock import MagicMock
from pyfakefs.fake_filesystem_unittest import Patcher

from aicli.engine import OpenAIEngine, GeminiEngine, AIEngine
from aicli.session_manager import SessionState

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
        max_tokens=1024,
        memory_enabled=True,
        debug_active=False,
        stream_active=True
    )

@pytest.fixture
def fake_fs():
    """
    Initializes a fake filesystem using pyfakefs for tests that
    require filesystem interactions (e.g., reading/writing settings,
    sessions, logs).
    """
    with Patcher() as patcher:
        yield patcher.fs

@pytest.fixture
def mock_requests_post(mocker):
    """
    A fixture that mocks `requests.post` to prevent actual network calls.
    Returns the mock object for customization within tests.
    """
    return mocker.patch('requests.post')

@pytest.fixture
def mock_openai_chat_response():
    """A fixture providing a standard, non-streaming OpenAI API chat response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response.",
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def mock_gemini_chat_response():
    """A fixture providing a standard, non-streaming Gemini API chat response."""
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "This is a test response."
                }],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 25,
            "totalTokenCount": 40
        }
    }
