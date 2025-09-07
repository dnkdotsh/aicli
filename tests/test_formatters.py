# tests/utils/test_formatters.py
"""
Tests for the utility functions in aicli/utils/formatters.py.
"""

import pytest
from aicli.utils.formatters import (
    clean_ai_response_text,
    estimate_token_count,
    format_bytes,
    format_token_string,
    sanitize_filename,
)


class TestFormatters:
    """Test suite for formatting utilities."""

    @pytest.mark.parametrize(
        "input_name, expected_output",
        [
            ("My Test Log", "My_Test_Log"),
            ("  leading and trailing spaces  ", "leading_and_trailing_spaces"),
            ("file-with-hyphens", "file_with_hyphens"),
            ("file/with\\slashes", "filewithslashes"),
            ("file!@#$%^&*()_+=.log", "file_log"),
            ("", "unnamed_log"),
            ("  ", "unnamed_log"),
        ],
    )
    def test_sanitize_filename(self, input_name, expected_output):
        """Tests that filenames are correctly sanitized."""
        assert sanitize_filename(input_name) == expected_output

    @pytest.mark.parametrize(
        "token_dict, expected_output",
        [
            (
                {"prompt": 10, "completion": 20, "total": 35, "reasoning": 5},
                "[P:10/C:20/R:5/T:35]",
            ),
            (
                {"prompt": 100, "completion": 150, "total": 250},
                "[P:100/C:150/R:0/T:250]",
            ),
            ({}, ""),
            ({"prompt": 0, "completion": 0}, ""),
        ],
    )
    def test_format_token_string(self, token_dict, expected_output):
        """Tests the formatting of token dictionaries into strings."""
        assert format_token_string(token_dict) == expected_output

    @pytest.mark.parametrize(
        "text, expected_tokens",
        [
            ("Hello world", 2),
            ("def func(x):\n  return x * 2", 7),
            ("", 0),
            (
                "This is a longer sentence to test the estimation logic.",
                10,
            ),
        ],
    )
    def test_estimate_token_count(self, text, expected_tokens):
        """Tests the token count estimation logic."""
        assert estimate_token_count(text) == expected_tokens

    @pytest.mark.parametrize(
        "byte_count, expected_string",
        [
            (500, "500.00 B"),
            (1024, "1.00 KB"),
            (1536, "1.50 KB"),
            (1048576, "1.00 MB"),
            (0, "0.00 B"),
        ],
    )
    def test_format_bytes(self, byte_count, expected_string):
        """Tests the conversion of bytes to human-readable strings."""
        assert format_bytes(byte_count) == expected_string

    @pytest.mark.parametrize(
        "engine_name, raw_response, expected_clean",
        [
            ("openai", "[OpenAI]: Hello", "Hello"),
            ("gemini", "[Gemini] :  Hello again", "Hello again"),
            ("openai", "Just a regular response.", "Just a regular response."),
            ("gemini", " [Gemini]: Response", "Response"),
        ],
    )
    def test_clean_ai_response_text(self, engine_name, raw_response, expected_clean):
        """Tests the removal of AI self-labels from responses."""
        assert clean_ai_response_text(engine_name, raw_response) == expected_clean
