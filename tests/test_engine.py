# tests/test_engine.py
"""
Unit tests for the AIEngine classes in aicli/engine.py.
These tests validate the logic for building API-specific payloads and parsing responses.
"""

import pytest

from aicli.engine import OpenAIEngine, GeminiEngine, get_engine

class TestEngines:
    """Test suite for AI engine implementations."""

    def test_get_engine_factory(self, mock_openai_engine, mock_gemini_engine):
        """Tests the factory function for creating engine instances."""
        assert isinstance(get_engine('openai', 'fake_key'), OpenAIEngine)
        assert isinstance(get_engine('gemini', 'fake_key'), GeminiEngine)
        with pytest.raises(ValueError):
            get_engine('unknown_engine', 'fake_key')

    def test_openai_build_chat_payload(self, mock_openai_engine):
        """Tests that OpenAI chat payloads are constructed correctly."""
        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "Be brief."
        payload = mock_openai_engine.build_chat_payload(messages, system_prompt, 100, True, "gpt-4")
        
        assert payload['model'] == "gpt-4"
        assert payload['stream'] is True
        assert payload['max_tokens'] == 100
        assert len(payload['messages']) == 2
        assert payload['messages'][0] == {"role": "system", "content": "Be brief."}

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

    def test_gemini_parse_chat_response(self, mock_gemini_engine, mock_gemini_chat_response):
        """Tests parsing of a standard Gemini chat response."""
        response_text = mock_gemini_engine.parse_chat_response(mock_gemini_chat_response)
        assert response_text == "This is a test response."
