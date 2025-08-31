# tests/test_engine.py
"""
Unit tests for the AIEngine classes in aicli/engine.py.
These tests validate the logic for building API-specific payloads and parsing responses.
"""

import pytest
import requests
from unittest.mock import MagicMock
import logging # Import logging to specify levels for record_tuples

from aicli.engine import OpenAIEngine, GeminiEngine, get_engine
from aicli import settings # For settings['api_timeout']

class TestEngines:
    """Test suite for AI engine implementations."""

    def test_get_engine_factory(self, mock_openai_engine, mock_gemini_engine):
        """Tests the factory function for creating engine instances."""
        assert isinstance(get_engine('openai', 'fake_key'), OpenAIEngine)
        assert isinstance(get_engine('gemini', 'fake_key'), GeminiEngine)
        with pytest.raises(ValueError):
            get_engine('unknown_engine', 'fake_key')

    def test_openai_build_chat_payload(self, mock_openai_engine):
        """Tests that OpenAI chat payloads are constructed correctly for older models."""
        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "Be brief."
        payload = mock_openai_engine.build_chat_payload(messages, system_prompt, 100, True, "gpt-3.5-turbo") # Older model name

        assert payload['model'] == "gpt-3.5-turbo"
        assert payload['stream'] is True
        assert payload['max_tokens'] == 100 # Old field
        assert 'max_completion_tokens' not in payload
        assert len(payload['messages']) == 2
        assert payload['messages'][0] == {"role": "system", "content": "Be brief."}

    def test_openai_build_chat_payload_max_completion_tokens(self, mock_openai_engine):
        """Tests that max_completion_tokens is used for newer OpenAI models."""
        messages = [{"role": "user", "content": "Hello"}]
        payload = mock_openai_engine.build_chat_payload(messages, None, 150, False, "gpt-4o-mini") # Newer model name

        assert payload['model'] == "gpt-4o-mini"
        assert payload['stream'] is False
        assert 'max_tokens' not in payload
        assert payload['max_completion_tokens'] == 150 # New field
        assert len(payload['messages']) == 1 # No system prompt

    def test_gemini_build_chat_payload(self, mock_gemini_engine):
        """Tests that Gemini chat payloads are constructed correctly."""
        messages = [{"role": "user", "parts": [{"text": "Hello"}]}]
        system_prompt = "Be brief."
        payload = mock_gemini_engine.build_chat_payload(messages, system_prompt, 200, False, "gemini-1.5-flash")

        assert payload['system_instruction']['parts'][0]['text'] == "Be brief."
        assert payload['generationConfig']['maxOutputTokens'] == 200
        assert payload['contents'] == messages

    def test_openai_parse_chat_response(self, mock_openai_engine, mock_openai_chat_response):
        """Tests parsing of a standard OpenAI chat response."""
        response_text = mock_openai_engine.parse_chat_response(mock_openai_chat_response)
        assert response_text == "This is a test response."

    def test_openai_parse_chat_response_empty(self, mock_openai_engine):
        """Tests parsing of an empty OpenAI chat response."""
        assert mock_openai_engine.parse_chat_response({}) == ""
        assert mock_openai_engine.parse_chat_response({"choices": []}) == ""
        assert mock_openai_engine.parse_chat_response({"choices": [{"message": {}}]}) == ""


    def test_gemini_parse_chat_response_empty_or_malformed(self, mock_gemini_engine, caplog):
        """Tests parsing of empty or malformed Gemini chat responses."""
        with caplog.at_level(logging.WARNING): # Capture logs at WARNING level
            assert mock_gemini_engine.parse_chat_response({}) == ""
            # --- PyEvolve Change: Corrected log message string for assertion ---
            assert ("aicli", logging.WARNING, "Could not extract Gemini response part or finish reason. Full response: {}") in caplog.record_tuples
            # --- End PyEvolve Change ---

        with caplog.at_level(logging.WARNING):
            assert mock_gemini_engine.parse_chat_response({"candidates": []}) == ""
            assert ("aicli", logging.WARNING, "Could not parse Gemini response due to unexpected structure. Full response: {'candidates': []}") in caplog.record_tuples

        with caplog.at_level(logging.WARNING):
            assert mock_gemini_engine.parse_chat_response({"candidates": [{"content": {}}]}) == ""
            # --- PyEvolve Change: Corrected log message string for assertion ---
            assert ("aicli", logging.WARNING, "Could not extract Gemini response part or finish reason. Full response: {'candidates': [{'content': {}}]}") in caplog.record_tuples
            # --- End PyEvolve Change ---

        with caplog.at_level(logging.WARNING):
            assert mock_gemini_engine.parse_chat_response({"candidates": [{"content": {"parts": []}}]}) == ""
            # --- PyEvolve Change: Corrected log message string for assertion ---
            assert ("aicli", logging.WARNING, "Could not extract Gemini response part or finish reason. Full response: {'candidates': [{'content': {'parts': []}}]}") in caplog.record_tuples
            # --- End PyEvolve Change ---

        with caplog.at_level(logging.WARNING):
            assert mock_gemini_engine.parse_chat_response({"candidates": [{"content": {"parts": [{"not_text": "data"}]}}]}) == ""
            # --- PyEvolve Change: Corrected log message string for assertion ---
            assert ("aicli", logging.WARNING, "Could not extract Gemini response part or finish reason. Full response: {'candidates': [{'content': {'parts': [{'not_text': 'data'}]}}]}") in caplog.record_tuples
            # --- End PyEvolve Change ---


    def test_openai_fetch_available_models_request_error(self, mock_openai_engine, mocker, caplog):
        """Tests error handling when fetching OpenAI models."""
        mocker.patch('requests.get', side_effect=requests.exceptions.RequestException("Network Error"))
        with caplog.at_level(logging.WARNING):
            models = mock_openai_engine.fetch_available_models('chat')
            assert models == []
            assert ("aicli", logging.WARNING, "Could not fetch OpenAI model list (Network Error).") in caplog.record_tuples

    def test_gemini_fetch_available_models_request_error(self, mock_gemini_engine, mocker, caplog):
        """Tests error handling when fetching Gemini models."""
        mocker.patch('requests.get', side_effect=requests.exceptions.RequestException("Network Error"))
        with caplog.at_level(logging.WARNING):
            models = mock_gemini_engine.fetch_available_models('chat')
            assert models == []
            assert ("aicli", logging.WARNING, "Could not fetch Gemini model list (Network Error).") in caplog.record_tuples

    def test_openai_fetch_available_models_success(self, mock_openai_engine, mocker):
        """Tests successful fetching and filtering of OpenAI models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o-mini"}, {"id": "dall-e-3"}, {"id": "whisper-1"}, {"id": "gpt-3.5-turbo"}
            ]
        }
        mocker.patch('requests.get', return_value=mock_response)

        chat_models = mock_openai_engine.fetch_available_models('chat')
        assert sorted(chat_models) == ["gpt-3.5-turbo", "gpt-4o-mini"]

        image_models = mock_openai_engine.fetch_available_models('image')
        assert sorted(image_models) == ["dall-e-3"]

    def test_gemini_fetch_available_models_success(self, mock_gemini_engine, mocker):
        """Tests successful fetching and filtering of Gemini models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "models/gemini-1.5-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
                {"name": "models/gemini-pro", "supportedGenerationMethods": ["generateContent"]},
            ]
        }
        mocker.patch('requests.get', return_value=mock_response)

        chat_models = mock_gemini_engine.fetch_available_models('chat')
        assert sorted(chat_models) == ["gemini-1.5-flash", "gemini-pro"] # "models/" prefix should be removed
