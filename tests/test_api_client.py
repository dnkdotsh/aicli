# tests/test_api_client.py

from unittest.mock import MagicMock

import pytest
import requests
from aicli.api_client import (
    ApiRequestError,
    MissingApiKeyError,
    _redact_sensitive_info,
    check_api_keys,
    make_api_request,
)


@pytest.fixture
def mock_settings(mocker):
    """Fixture to mock the settings object."""
    return mocker.patch("aicli.api_client.settings", {"api_timeout": 10})


def test_check_api_keys_openai_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    assert check_api_keys("openai") == "test_key"


def test_check_api_keys_gemini_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    assert check_api_keys("gemini") == "test_key"


def test_check_api_keys_missing_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError) as excinfo:
        check_api_keys("openai")
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_check_api_keys_missing_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError) as excinfo:
        check_api_keys("gemini")
    assert "GEMINI_API_KEY" in str(excinfo.value)


def test_redact_sensitive_info():
    log_entry = {
        "request": {
            "url": "https://api.gemini.com/v1/models?key=supersecretkey",
            "headers": {"Authorization": "Bearer anothersecret"},
            "payload": {"data": "test"},
        }
    }
    redacted = _redact_sensitive_info(log_entry)
    assert "key=[REDACTED]" in redacted["request"]["url"]
    assert redacted["request"]["headers"]["Authorization"] == "Bearer [REDACTED]"
    # Ensure original is not modified
    assert "supersecretkey" in log_entry["request"]["url"]


def test_make_api_request_success(mocker, mock_settings):
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.return_value = {"success": True}
    mock_post.return_value = mock_response

    response = make_api_request("http://test.com", {}, {})
    assert response == {"success": True}
    mock_post.assert_called_once()
    mock_response.raise_for_status.assert_called_once()


def test_make_api_request_streaming_success(mocker, mock_settings):
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200  # Add status_code to the mock
    mock_post.return_value = mock_response

    response = make_api_request("http://test.com", {}, {}, stream=True)
    assert response is mock_response
    mock_post.assert_called_once_with(
        "http://test.com", headers={}, json={}, stream=True, timeout=10
    )
    mock_response.raise_for_status.assert_called_once()


def test_make_api_request_api_error_in_payload(mocker, mock_settings):
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": {"message": "API Error Occurred"}}
    mock_post.return_value = mock_response

    with pytest.raises(ApiRequestError, match="API Error Occurred"):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_http_error_with_json(mocker, mock_settings):
    """Test that an HTTP error with a valid JSON body is handled correctly."""
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

    # Create an HTTPError instance and attach the mock response to it
    http_error = requests.exceptions.HTTPError()
    http_error.response = mock_response
    mock_response.raise_for_status.side_effect = http_error
    mock_post.return_value = mock_response

    with pytest.raises(ApiRequestError, match="Invalid API key"):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_http_error_with_non_json(mocker, mock_settings):
    """Test that an HTTP error with a non-JSON body is handled."""
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
        "msg", "doc", 0
    )
    mock_response.text = "Internal Server Error"

    # Create an HTTPError instance and attach the mock response to it
    http_error = requests.exceptions.HTTPError()
    http_error.response = mock_response
    mock_response.raise_for_status.side_effect = http_error
    mock_post.return_value = mock_response

    with pytest.raises(ApiRequestError, match="Internal Server Error"):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_connection_error(mocker, mock_settings):
    """Test that a generic RequestException is caught and wrapped."""
    mocker.patch(
        "requests.post",
        side_effect=requests.exceptions.RequestException("Connection failed"),
    )
    with pytest.raises(ApiRequestError, match="Connection failed"):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_success_with_bad_json(mocker, mock_settings):
    """Test handling of a 200 OK response with an invalid JSON body."""
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
        "msg", "doc", 0
    )
    mock_post.return_value = mock_response

    with pytest.raises(ApiRequestError, match="Failed to decode API response."):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_logs_on_failure(mocker, mock_settings, fs):
    """Test that the request is logged even when an exception occurs."""
    mocker.patch(
        "requests.post",
        side_effect=requests.exceptions.RequestException("Connection failed"),
    )
    # Mock the log file path
    log_file_path = "/fake/home/.local/share/aicli/logs/raw_api_calls.log"
    fs.create_file(log_file_path, contents="")

    mocker.patch("aicli.api_client.config.RAW_LOG_FILE", log_file_path)

    session_logs = []
    with pytest.raises(ApiRequestError):
        make_api_request(
            "http://test.com", {}, {"payload": "data"}, session_raw_logs=session_logs
        )

    # Check that the in-memory session log was appended to
    assert len(session_logs) == 1
    assert session_logs[0]["request"]["payload"] == {"payload": "data"}
    assert "Connection failed" in session_logs[0]["response"]["error"]

    # Check that the file log was written to
    with open(log_file_path) as f:
        log_content = f.read()
        assert '"payload": "data"' in log_content
        assert "Connection failed" in log_content
