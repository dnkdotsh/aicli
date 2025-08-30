# tests/test_api_client.py
"""
Integration tests for the API client in aicli/api_client.py.
These tests use mocking to simulate network requests and responses.
"""

import pytest
import requests
import os

from aicli import api_client

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

    def test_make_api_request_success(self, mock_requests_post, mock_openai_chat_response):
        """Tests a successful, non-streaming API request."""
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response.json = lambda: mock_openai_chat_response
        mock_requests_post.return_value = mock_response

        response = api_client.make_api_request("http://fake.url", {}, {})
        assert response == mock_openai_chat_response

    def test_make_api_request_http_error(self, mock_requests_post, mocker):
        """Tests that an ApiRequestError is raised on an HTTP error."""
        # Use a MagicMock for full control over the fake response
        mock_response = mocker.MagicMock(spec=requests.Response)
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.text = "The requested resource was not found."
        
        # Configure the mock to raise an HTTPError when raise_for_status is called
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        
        # Configure the mock for the json() call inside the except block
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "doc", 0)

        mock_requests_post.return_value = mock_response

        with pytest.raises(api_client.ApiRequestError) as excinfo:
            api_client.make_api_request("http://fake.url", {}, {})
        
        assert "The requested resource was not found." in str(excinfo.value)
