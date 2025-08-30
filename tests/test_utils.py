# tests/test_utils.py
"""
Unit tests for the utility functions in aicli/utils.py.
These tests focus on data manipulation, file processing, and message formatting.
"""

import pytest
from pathlib import Path

from aicli import utils
from aicli import config

class TestUtils:
    """Test suite for utility functions."""

    def test_sanitize_filename(self):
        """Tests that filenames are correctly sanitized."""
        assert utils.sanitize_filename("  a test file!? ") == "a_test_file"
        assert utils.sanitize_filename("another-test_with_numbers_123") == "another_test_with_numbers_123"
        assert utils.sanitize_filename("") == "unnamed_log"
        assert utils.sanitize_filename("...///") == "unnamed_log"

    def test_construct_user_message_openai(self):
        """Tests user message construction for the OpenAI engine."""
        msg = utils.construct_user_message('openai', 'test prompt', [])
        assert msg == {"role": "user", "content": [{"type": "text", "text": "test prompt"}]}

    def test_construct_user_message_gemini_with_image(self):
        """Tests user message construction for the Gemini engine with image data."""
        image_data = [{"mime_type": "image/png", "data": "base64_string"}]
        msg = utils.construct_user_message('gemini', 'test prompt', image_data)
        assert msg['role'] == 'user'
        assert len(msg['parts']) == 2
        assert msg['parts'][0] == {"text": "test prompt"}
        assert msg['parts'][1] == {"inline_data": {"mime_type": "image/png", "data": "base64_string"}}

    def test_translate_history(self):
        """Tests translation of conversation history between engine formats."""
        openai_history = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi there"}
        ]
        gemini_history = utils.translate_history(openai_history, 'gemini')
        assert gemini_history[0] == {"role": "user", "parts": [{"text": "Hello"}]}
        assert gemini_history[1] == {"role": "model", "parts": [{"text": "Hi there"}]}

    def test_process_files_with_fake_fs(self, fake_fs):
        """Tests file and directory processing using a fake filesystem."""
        # Setup fake directories and files
        fake_fs.create_dir(config.DATA_DIR)
        fake_fs.create_file(config.PERSISTENT_MEMORY_FILE, contents="persistent memory data")

        project_dir = Path("/fake_project")
        fake_fs.create_dir(project_dir / "src")
        fake_fs.create_file(project_dir / "src/main.py", contents="print('hello')")
        fake_fs.create_file(project_dir / "README.md", contents="# Read Me")
        fake_fs.create_file(project_dir / "ignore.log", contents="log data") # Should be ignored

        # Test processing a directory
        mem, attachments, images, processed_files = utils.process_files(
            paths=[str(project_dir)],
            use_memory=True,
            exclusions=["ignore.log"]
        )

        assert "persistent memory data" in mem
        assert "--- FILE: /fake_project/src/main.py ---\nprint('hello')" in attachments
        assert "--- FILE: /fake_project/README.md ---\n# Read Me" in attachments
        assert "ignore.log" not in attachments
        assert images == []
        assert len(processed_files) == 2
