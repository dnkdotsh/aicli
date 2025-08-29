# tests/test_utils.py

import pytest
from aicli import utils

def test_sanitize_filename_standard():
    """Tests basic sanitization with spaces and mixed case."""
    input_name = "My Test Document"
    expected_output = "my_test_document"
    assert utils.sanitize_filename(input_name) == expected_output

def test_sanitize_filename_edge_cases():
    """Tests sanitization with special characters, repeated underscores, and leading/trailing spaces."""
    input_name = "  !!a/b\\c__d..e-f-  "
    expected_output = "abc_d..e-f"
    assert utils.sanitize_filename(input_name) == expected_output

def test_sanitize_filename_empty():
    """Tests that an empty input results in an empty output."""
    assert utils.sanitize_filename("") == ""

def test_sanitize_filename_no_changes():
    """Tests that a clean filename remains unchanged."""
    input_name = "clean_filename.txt"
    assert utils.sanitize_filename(input_name) == input_name
