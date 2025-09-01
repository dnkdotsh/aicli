# tests/test_utils.py
"""
Unit tests for the utility functions in aicli/utils.py.
These tests focus on data manipulation, file processing, and message formatting.
"""

import pytest
import os
import zipfile
import base64
import io
import re # Added for regex assertion
import requests # Make sure requests is imported for exceptions
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        assert utils.sanitize_filename("fïlëñåmë_wïth_ünîcödë") == "fïlëñåmë_wïth_ünîcödë" # Keep Unicode as per current implementation's regex

    @pytest.mark.parametrize("filename, expected", [
        # Standard extensions
        ("file.txt", True),
        ("script.py", True),
        # Supported extensionless files (case-insensitive)
        ("Dockerfile", True),
        ("makefile", True),
        ("Jenkinsfile", True),
        # Unsupported extensionless files
        ("unknownfile", False),
        ("mybinary", False),
        # Unsupported extensions
        ("archive.zip", False),
        ("image.png", False),
    ])
    def test_is_supported_text_file(self, filename, expected):
        """Tests the is_supported_text_file logic for various filenames."""
        path = Path(f"/test/{filename}")
        assert utils.is_supported_text_file(path) == expected

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

        gemini_history_back_to_openai = utils.translate_history(gemini_history, 'openai')
        assert gemini_history_back_to_openai[0] == {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        assert gemini_history_back_to_openai[1] == {"role": "assistant", "content": "Hi there"}

    def test_process_files_with_fake_fs(self, fake_fs):
        """Tests file and directory processing using a fake filesystem."""
        # Setup fake directories and files
        # Removed: fake_fs.create_dir(config.DATA_DIR) - already created by conftest fixture
        # Ensure log directory is created for persistent memory if needed
        config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        fake_fs.create_file(config.PERSISTENT_MEMORY_FILE, contents="persistent memory data")

        project_dir = Path("/fake_project")
        fake_fs.create_dir(project_dir / "src")
        fake_fs.create_file(project_dir / "src/main.py", contents="print('hello')")
        fake_fs.create_file(project_dir / "README.md", contents="# Read Me")
        fake_fs.create_file(project_dir / "ignore.log", contents="log data") # Should be ignored

        # Test processing a directory
        mem, attachments, images = utils.process_files(
            paths=[str(project_dir)],
            use_memory=True,
            exclusions=["ignore.log"]
        )

        assert mem == "persistent memory data"

        main_py_path = (project_dir / "src/main.py").resolve()
        readme_md_path = (project_dir / "README.md").resolve()
        ignore_log_path = (project_dir / "ignore.log").resolve()

        assert main_py_path in attachments
        assert attachments[main_py_path] == "print('hello')"

        assert readme_md_path in attachments
        assert attachments[readme_md_path] == "# Read Me"

        assert ignore_log_path not in attachments # Check that the excluded file is not in attachments
        assert images == []
        assert len(attachments) == 2 # Only two files should have been processed into attachments

    def test_process_files_single_text_file(self, fake_fs):
        """Tests processing a single text file."""
        file_path = Path("/my_script.py")
        file_path.write_text("def func(): pass")

        mem, attachments, images = utils.process_files([str(file_path)], use_memory=False, exclusions=[])

        assert mem is None
        assert file_path.resolve() in attachments
        assert attachments[file_path.resolve()] == "def func(): pass"
        assert not images

    def test_process_files_directory_recursive(self, fake_fs):
        """Tests processing a directory with nested files and exclusions."""
        root_dir = Path("/my_repo")
        (root_dir / "src").mkdir(parents=True)
        (root_dir / "docs").mkdir()
        (root_dir / "ignore_dir").mkdir()

        (root_dir / "src/app.py").write_text("import os")
        (root_dir / "docs/README.md").write_text("Docs")
        (root_dir / "ignore_dir/secret.txt").write_text("sensitive info") # Excluded dir
        (root_dir / "config.json").write_text("{}")
        (root_dir / "image.jpg").write_bytes(b'fake_jpg_data') # Should be collected as image

        mem, attachments, images = utils.process_files(
            paths=[str(root_dir)],
            use_memory=False,
            exclusions=[str(root_dir / "ignore_dir")]
        )

        assert len(attachments) == 3 # app.py, README.md, config.json
        assert (root_dir / "src/app.py").resolve() in attachments
        assert (root_dir / "docs/README.md").resolve() in attachments
        assert (root_dir / "config.json").resolve() in attachments
        assert (root_dir / "ignore_dir/secret.txt").resolve() not in attachments
        assert len(images) == 1
        assert images[0]['mime_type'] == 'image/jpeg'


    def test_process_files_with_images(self, fake_fs):
        """Tests processing image files."""
        img_path = Path("/my_image.png")
        img_path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")) # A tiny transparent PNG

        mem, attachments, images = utils.process_files([str(img_path)], use_memory=False, exclusions=[])

        assert mem is None
        assert not attachments
        assert len(images) == 1
        assert images[0]['type'] == 'image'
        assert images[0]['mime_type'] == 'image/png'
        assert images[0]['data'] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    def test_process_files_with_zip_archive(self, fake_fs):
        """Tests processing text files within a zip archive."""
        zip_path = Path("/my_archive.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("code/script.py", "print('zip code')")
            zf.writestr("docs/notes.md", "Zip notes.")
            zf.writestr("images/pic.png", b'binary_data') # Not text, should be ignored by zip processing
            zf.writestr("excluded/temp.txt", "should be excluded") # excluded by exclusion list

        mem, attachments, images = utils.process_files(
            paths=[str(zip_path)],
            use_memory=False,
            exclusions=['temp.txt'] # Exclude by name
        )

        assert mem is None
        assert zip_path in attachments # The entire formatted content of the zip is stored under the zip's path
        assert "--- FILE (from my_archive.zip): code/script.py ---\nprint('zip code')" in attachments[zip_path]
        assert "--- FILE (from my_archive.zip): docs/notes.md ---\nZip notes." in attachments[zip_path]
        assert "images/pic.png" not in attachments[zip_path] # Binary should be ignored
        assert "excluded/temp.txt" not in attachments[zip_path] # Excluded by name
        assert not images # Images in zip files are not supported this way for now

    def test_process_files_no_memory_enabled(self, fake_fs):
        """Tests that persistent memory content is not loaded when use_memory is False."""
        # Removed: fake_fs.create_dir(config.DATA_DIR) - already created by conftest fixture
        # Ensure log directory is created for persistent memory if needed
        config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        fake_fs.create_file(config.PERSISTENT_MEMORY_FILE, contents="persistent memory data")

        mem, attachments, images = utils.process_files(paths=[], use_memory=False, exclusions=[])

        assert mem is None # Should be None because use_memory is False

    def test_extract_text_from_message_openai_str(self):
        """Extracts text from a simple OpenAI string content."""
        msg = {"role": "assistant", "content": "Hello there."}
        assert utils.extract_text_from_message(msg) == "Hello there."

    def test_extract_text_from_message_openai_list(self):
        """Extracts text from an OpenAI list-of-dict content."""
        msg = {"role": "user", "content": [{"type": "text", "text": "What's up?"}, {"type": "image_url", "url": "data:image/png;base64,..."}]}
        assert utils.extract_text_from_message(msg) == "What's up?"

    def test_extract_text_from_message_gemini(self):
        """Extracts text from a Gemini-style message with 'parts'."""
        msg = {"role": "model", "parts": [{"text": "How can I help?"}]}
        assert utils.extract_text_from_message(msg) == "How can I help?"

        msg_with_image = {"role": "user", "parts": [{"text": "Describe this."}, {"inline_data": {"mime_type": "image/jpeg", "data": "abc"}}]}
        assert utils.extract_text_from_message(msg_with_image) == "Describe this."

    def test_extract_text_from_message_empty_or_malformed(self):
        """Tests extraction from empty or malformed messages."""
        assert utils.extract_text_from_message({"role": "user"}) == ""
        assert utils.extract_text_from_message({"role": "user", "content": []}) == ""
        assert utils.extract_text_from_message({"role": "user", "content": [{"type": "image_url"}]}) == ""
        assert utils.extract_text_from_message({}) == ""

    def test_process_stream_openai_success(self, mocker, capsys):
        """Tests process_stream for OpenAI with valid streaming chunks."""
        mock_response = MagicMock()
        # Simulate OpenAI streaming chunks with usage data
        stream_chunks = [
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}],"usage":{}}',
            b'data: {"id":"chatcmpl-2","choices":[{"delta":{"content":" world."}}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}',
            b'data: {"id":"chatcmpl-3","choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
            b'data: [DONE]'
        ]
        mock_response.iter_lines.return_value = stream_chunks

        full_response, tokens = utils.process_stream('openai', mock_response)

        assert full_response == "Hello world."
        assert tokens['prompt'] == 10
        assert tokens['completion'] == 5
        assert tokens['total'] == 15
        assert capsys.readouterr().out == "Hello world."

    def test_process_stream_gemini_success(self, mocker, capsys):
        """Tests process_stream for Gemini with valid streaming chunks."""
        mock_response = MagicMock()
        # Simulate Gemini streaming chunks with usage data
        stream_chunks = [
            b'data: {"candidates":[{"content":{"parts":[{"text":"Gemini"}]}}],"usageMetadata":{}}',
            b'data: {"candidates":[{"content":{"parts":[{"text":" rocks!"}]}}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":1,"totalTokenCount":13}}',
            b'data: {"candidates":[{"content":{"parts":[{"text":""}]}}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":7,"cachedContentTokenCount":0,"totalTokenCount":19}}',
            b'' # Empty chunk
        ]
        mock_response.iter_lines.return_value = stream_chunks

        full_response, tokens = utils.process_stream('gemini', mock_response)

        assert full_response == "Gemini rocks!"
        assert tokens['prompt'] == 12
        assert tokens['completion'] == 7
        assert tokens['total'] == 19
        assert capsys.readouterr().out == "Gemini rocks!"

    def test_process_stream_interruption(self, mocker, capsys):
        """Tests that stream interruption (e.g., API error during stream) is handled gracefully."""
        mock_response = MagicMock()
        # Simulate a stream that raises an exception mid-way
        def raise_error_iter():
            yield b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Partial"}}]}'
            raise requests.exceptions.RequestException("Network error")
        mock_response.iter_lines.return_value = raise_error_iter()

        full_response, tokens = utils.process_stream('openai', mock_response)

        assert full_response == "Partial"
        assert tokens == {'prompt': 0, 'completion': 0, 'reasoning': 0, 'total': 0} # Token counts might be incomplete
        captured = capsys.readouterr()
        assert "Partial" in captured.out
        assert "Stream interrupted: Network error" in captured.out

    def test_parse_token_counts_openai(self):
        """Tests parsing token counts from a non-streaming OpenAI response."""
        openai_response = {
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150
            }
        }
        p, c, r, t = utils.parse_token_counts('openai', openai_response)
        assert (p, c, r, t) == (50, 100, 0, 150) # reasoning 'r' is not explicitly given by OpenAI's usage for chat

        openai_response_no_usage = {}
        p, c, r, t = utils.parse_token_counts('openai', openai_response_no_usage)
        assert (p, c, r, t) == (0, 0, 0, 0)

    def test_parse_token_counts_gemini(self):
        """Tests parsing token counts from a non-streaming Gemini response."""
        gemini_response = {
            "usageMetadata": {
                "promptTokenCount": 60,
                "candidatesTokenCount": 120,
                "totalTokenCount": 180,
                "cachedContentTokenCount": 5 # This is now correctly extracted
            }
        }
        p, c, r, t = utils.parse_token_counts('gemini', gemini_response)
        assert (p, c, r, t) == (60, 120, 5, 180) # r is now 5, as expected

        gemini_response_no_usage = {}
        p, c, r, t = utils.parse_token_counts('gemini', gemini_response_no_usage)
        assert (p, c, r, t) == (0, 0, 0, 0)
