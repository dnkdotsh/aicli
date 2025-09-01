# tests/test_api_client.py
"""
Integration tests for the API client in aicli/api_client.py.
These tests use mocking to simulate network requests and responses.
"""

import re  # Added for regex assertion
from unittest.mock import MagicMock

import pytest
import requests
from aicli import (
    api_client,
    config,  # For raw_log_file path
)
from aicli.engine import OpenAIEngine  # For perform_chat_request test


class TestApiClient:
    """Test suite for the API client."""

    def test_check_api_keys_success(self, monkeypatch):
        """Tests that API keys are correctly retrieved from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "openai_key_exists")
        assert api_client.check_api_keys("openai") == "openai_key_exists"

    def test_check_api_keys_missing(self, monkeypatch):
        """Tests that a MissingApiKeyError is raised when a key is not set."""
        # Ensure the environment variable is not set for this test
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(api_client.MissingApiKeyError):
            api_client.check_api_keys("gemini")

    def test_make_api_request_success(
        self, mock_requests_post, mock_openai_chat_response, fake_fs
    ):
        """Tests a successful, non-streaming API request."""
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response.json = lambda: mock_openai_chat_response
        mock_requests_post.return_value = mock_response

        response = api_client.make_api_request("http://fake.url", {}, {})
        assert response == mock_openai_chat_response

        # Verify log file was written (and ensure it exists)
        assert config.RAW_LOG_FILE.exists()

    def test_make_api_request_http_error(
        self, mock_requests_post, mocker, capsys, fake_fs
    ):
        """Tests that an ApiRequestError is raised on an HTTP error."""
        # Use a MagicMock for full control over the fake response
        mock_response = mocker.MagicMock(spec=requests.Response)
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.text = "The requested resource was not found."
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "doc", 0
        )  # Simulate non-JSON error body

        # Configure the mock to raise an HTTPError when raise_for_status is called
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error

        mock_requests_post.return_value = mock_response

        with pytest.raises(api_client.ApiRequestError) as excinfo:
            api_client.make_api_request("http://fake.url", {}, {})

        assert "The requested resource was not found." in str(excinfo.value)
        assert config.RAW_LOG_FILE.exists()  # Ensure it still logs errors

    def test_make_api_request_json_decode_error(
        self, mock_requests_post, mocker, capsys, fake_fs
    ):
        """Tests handling of a malformed JSON response from the API."""
        mock_response = mocker.MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "doc", 0
        )
        mock_response.text = "This is not JSON"
        mock_requests_post.return_value = mock_response

        with pytest.raises(api_client.ApiRequestError) as excinfo:
            api_client.make_api_request("http://fake.url", {}, {})

        # Changed assertion to use regex for robustness against slight variations in requests' error messages
        assert re.search(
            r"Failed to decode API response|Expecting value", str(excinfo.value)
        )
        assert config.RAW_LOG_FILE.exists()

    def test_make_api_request_timeout(
        self, mock_requests_post, mocker, capsys, fake_fs
    ):
        """Simulate a requests.exceptions.Timeout and verify ApiRequestError is raised."""
        mock_requests_post.side_effect = requests.exceptions.Timeout(
            "Request timed out"
        )

        with pytest.raises(api_client.ApiRequestError) as excinfo:
            api_client.make_api_request("http://fake.url", {}, {})

        assert "Request timed out" in str(excinfo.value)
        assert config.RAW_LOG_FILE.exists()

    def test_redact_sensitive_info_auth_header(self):
        """Verify API keys are correctly redacted from Authorization header."""
        log_entry = {
            "request": {
                "url": "https://api.openai.com/v1/chat/completions",
                "headers": {
                    "Authorization": "Bearer sk-12345ABCDEF",
                    "Content-Type": "application/json",
                },
                "payload": {"prompt": "test"},
            }
        }
        redacted_entry = api_client._redact_sensitive_info(log_entry)
        assert (
            redacted_entry["request"]["headers"]["Authorization"] == "Bearer [REDACTED]"
        )
        assert (
            log_entry["request"]["headers"]["Authorization"] == "Bearer sk-12345ABCDEF"
        )  # Original should be unchanged

    def test_redact_sensitive_info_query_param(self):
        """Verify API keys are correctly redacted from query parameters."""
        log_entry = {
            "request": {
                "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key=AIzaSyABCD",
                "headers": {"Content-Type": "application/json"},
                "payload": {"prompt": "test"},
            }
        }
        redacted_entry = api_client._redact_sensitive_info(log_entry)
        assert (
            redacted_entry["request"]["url"]
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key=[REDACTED]"
        )
        assert (
            log_entry["request"]["url"]
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key=AIzaSyABCD"
        )  # Original unchanged

    def test_perform_chat_request_streaming_error(
        self, mock_requests_post, mocker, capsys
    ):
        """Tests that streaming API errors result in an empty response string."""
        mock_engine = MagicMock(spec=OpenAIEngine, api_key="test-key", name="openai")
        mock_engine.get_chat_url.return_value = "http://fake.url"
        mock_engine.build_chat_payload.return_value = {}

        mock_response = mocker.MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.iter_lines.side_effect = requests.exceptions.RequestException(
            "Stream connection reset"
        )
        mock_requests_post.return_value = mock_response

        response_text, tokens = api_client.perform_chat_request(
            engine=mock_engine,
            model="gpt-4o-mini",
            messages_or_contents=[],
            system_prompt=None,
            max_tokens=100,
            stream=True,
        )

        assert response_text == ""  # Expect empty string for streaming error
        assert tokens == {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}
        captured = capsys.readouterr()
        assert (
            "--> Stream interrupted by network/API error: Stream connection reset"
            in captured.out
        )
