# tests/test_utils.py
"""
Unit tests for the utility functions in aicli/utils.py.
These tests focus on data manipulation, file processing, and message formatting.
"""

import argparse
import base64
import io
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aicli import config, personas, utils
from aicli.engine import AIEngine


# Helper to create a standard mock args namespace for config resolution tests
def create_mock_args(**kwargs):
    defaults = {
        "prompt": None,
        "engine": None,  # Default is now None
        "model": None,
        "persona": None,
        "system_prompt": None,
        "file": None,
        "exclude": None,
        "memory": False,
        "session_name": None,
        "stream": None,
        "max_tokens": None,
        "debug": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveConfigPrecedence:
    """Tests the complex config resolution logic."""

    @pytest.fixture(autouse=True)
    def setup_config_mocks(self, mocker, fake_fs):
        # Mock settings for predictable defaults
        mocker.patch.dict(
            utils.settings,
            {
                "default_engine": "gemini",
                "stream": True,
                "memory_enabled": True,
                "default_max_tokens": 1024,
                "default_openai_chat_model": "gpt-4o-mini",
                "default_gemini_model": "gemini-1.5-flash",
            },
        )
        # Mock persona loading
        self.mock_load_persona = mocker.patch("aicli.personas.load_persona")

        # Create a mock persona object
        self.mock_persona_obj = MagicMock(spec=personas.Persona)
        self.mock_persona_obj.engine = "openai"
        self.mock_persona_obj.model = "gpt-4-persona"
        self.mock_persona_obj.max_tokens = 2048
        self.mock_persona_obj.stream = False

    def test_cli_args_have_highest_precedence(self):
        """CLI arguments should override any persona or default settings."""
        self.mock_load_persona.return_value = self.mock_persona_obj
        args = create_mock_args(
            engine="openai",
            model="gpt-cli",
            max_tokens=4096,
            stream=True,
            persona="test_persona",
        )

        config_params = utils.resolve_config_precedence(args)

        assert config_params["engine_name"] == "openai"
        assert config_params["model"] == "gpt-cli"
        assert config_params["max_tokens"] == 4096
        assert config_params["stream"] is True

    def test_persona_overrides_defaults(self):
        """Persona settings should override default settings."""
        self.mock_load_persona.return_value = self.mock_persona_obj
        args = create_mock_args(persona="test_persona")

        config_params = utils.resolve_config_precedence(args)

        assert config_params["engine_name"] == "openai"  # From persona
        assert config_params["model"] == "gpt-4-persona"  # From persona
        assert config_params["max_tokens"] == 2048  # From persona
        assert config_params["stream"] is False  # From persona

    def test_defaults_are_used_when_no_overrides(self):
        """Default settings should be used when no CLI or persona settings apply."""
        self.mock_load_persona.return_value = None
        args = create_mock_args()

        config_params = utils.resolve_config_precedence(args)

        assert config_params["engine_name"] == "gemini"  # Default
        assert config_params["model"] == "gemini-1.5-flash"  # Default
        assert config_params["max_tokens"] == 1024  # Default
        assert config_params["stream"] is True  # Default
        assert config_params["memory_enabled"] is True  # Default

    def test_memory_toggle(self):
        """The --memory flag should toggle the default memory setting."""
        # Test case 1: Default is True, flag should make it False
        args_with_toggle = create_mock_args(memory=True)
        config1 = utils.resolve_config_precedence(args_with_toggle)
        assert not config1["memory_enabled"]

        # Test case 2: Single-shot mode should always disable memory
        args_single_shot = create_mock_args(prompt="A question", memory=True)
        config2 = utils.resolve_config_precedence(args_single_shot)
        assert not config2["memory_enabled"]

    def test_persona_model_only_applies_if_engine_matches(self):
        """A persona's model should not be used if the engine is overridden by CLI."""
        self.mock_load_persona.return_value = self.mock_persona_obj
        # Persona is openai/gpt-4-persona, but CLI forces gemini
        args = create_mock_args(engine="gemini", persona="test_persona")

        config_params = utils.resolve_config_precedence(args)

        assert config_params["engine_name"] == "gemini"
        # Model should fall back to the default for Gemini, not the persona's OpenAI model
        assert config_params["model"] == "gemini-1.5-flash"


class TestSelectModel:
    @pytest.fixture(autouse=True)
    def setup_select_model_mocks(self, mocker):
        self.mock_engine = MagicMock(spec=AIEngine)
        self.mock_engine.name = "openai"
        self.mock_engine.fetch_available_models.return_value = [
            "model-a",
            "model-b",
            "model-c",
        ]
        mocker.patch.dict(
            utils.settings,
            {
                "default_openai_chat_model": "gpt-4o-mini",
                "default_openai_image_model": "dall-e-3",
            },
        )
        self.mock_prompt = mocker.patch("aicli.utils.prompt")

    def test_select_model_default_selection_chat(self):
        """Tests that select_model returns the default chat model if user selects 'y'."""
        self.mock_prompt.return_value = "y"
        model = utils.select_model(self.mock_engine, "chat")
        assert model == "gpt-4o-mini"
        self.mock_engine.fetch_available_models.assert_not_called()

    def test_select_model_default_selection_image(self):
        """Tests that select_model returns the default image model if user presses Enter."""
        self.mock_prompt.return_value = ""  # Enter is default 'y'
        model = utils.select_model(self.mock_engine, "image")
        assert model == "dall-e-3"
        self.mock_engine.fetch_available_models.assert_not_called()

    def test_select_model_user_selection(self):
        """Tests that a valid numeric user choice for a model is returned."""
        self.mock_prompt.side_effect = ["n", "2"]
        model = utils.select_model(self.mock_engine, "chat")
        assert model == "model-b"

    def test_select_model_invalid_selection_falls_back_to_default(self, capsys):
        """Tests that invalid input for model selection falls back to the default."""
        self.mock_prompt.side_effect = ["n", "99"]
        model = utils.select_model(self.mock_engine, "chat")
        assert model == "gpt-4o-mini"
        output = capsys.readouterr().out
        assert "Invalid selection. Using default: gpt-4o-mini" in output

    def test_select_model_no_available_models_falls_back_to_default(self, capsys):
        """Tests behavior when fetch_available_models returns an empty list."""
        self.mock_engine.fetch_available_models.return_value = []
        self.mock_prompt.side_effect = ["n"]

        model = utils.select_model(self.mock_engine, "chat")
        assert model == "gpt-4o-mini"
        output = capsys.readouterr().out
        assert "Fetching available models..." in output
        assert "Using default: gpt-4o-mini" in output


class TestUtils:
    """Test suite for utility functions."""

    def test_sanitize_filename(self):
        """Tests that filenames are correctly sanitized."""
        assert utils.sanitize_filename("  a test file!? ") == "a_test_file"
        assert (
            utils.sanitize_filename("another-test_with_numbers_123")
            == "another_test_with_numbers_123"
        )
        assert utils.sanitize_filename("") == "unnamed_log"
        assert utils.sanitize_filename("...///") == "unnamed_log"
        assert (
            utils.sanitize_filename("fïlëñåmë_wïth_ünîcödë") == "fïlëñåmë_wïth_ünîcödë"
        )  # Keep Unicode as per current implementation's regex

    @pytest.mark.parametrize(
        "filename, expected",
        [
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
        ],
    )
    def test_is_supported_text_file(self, filename, expected):
        """Tests the is_supported_text_file logic for various filenames."""
        path = Path(f"/test/{filename}")
        assert utils.is_supported_text_file(path) == expected

    def test_construct_user_message_openai(self):
        """Tests user message construction for the OpenAI engine."""
        msg = utils.construct_user_message("openai", "test prompt", [])
        assert msg == {
            "role": "user",
            "content": [{"type": "text", "text": "test prompt"}],
        }

    def test_construct_user_message_gemini_with_image(self):
        """Tests user message construction for the Gemini engine with image data."""
        image_data = [{"mime_type": "image/png", "data": "base64_string"}]
        msg = utils.construct_user_message("gemini", "test prompt", image_data)
        assert msg["role"] == "user"
        assert len(msg["parts"]) == 2
        assert msg["parts"][0] == {"text": "test prompt"}
        assert msg["parts"][1] == {
            "inline_data": {"mime_type": "image/png", "data": "base64_string"}
        }

    def test_translate_history(self):
        """Tests translation of conversation history between engine formats."""
        openai_history = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi there"},
        ]
        gemini_history = utils.translate_history(openai_history, "gemini")
        assert gemini_history[0] == {"role": "user", "parts": [{"text": "Hello"}]}
        assert gemini_history[1] == {"role": "model", "parts": [{"text": "Hi there"}]}

        gemini_history_back_to_openai = utils.translate_history(
            gemini_history, "openai"
        )
        assert gemini_history_back_to_openai[0] == {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }
        assert gemini_history_back_to_openai[1] == {
            "role": "assistant",
            "content": "Hi there",
        }

    def test_process_files_with_fake_fs(self, fake_fs):
        """Tests file and directory processing using a fake filesystem."""
        # Setup fake directories and files
        config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        fake_fs.create_file(
            config.PERSISTENT_MEMORY_FILE, contents="persistent memory data"
        )

        project_dir = Path("/fake_project")
        fake_fs.create_dir(project_dir / "src")
        fake_fs.create_file(project_dir / "src/main.py", contents="print('hello')")
        fake_fs.create_file(project_dir / "README.md", contents="# Read Me")
        fake_fs.create_file(project_dir / "ignore.log", contents="log data")

        # The exclusion logic resolves paths. In a test with a fake filesystem,
        # providing a relative path like "ignore.log" would be resolved against the
        # real filesystem's CWD, causing a mismatch. We provide an absolute path
        # within the fake filesystem to ensure the test works correctly.
        exclusion_path = str((project_dir / "ignore.log").resolve())

        # Test processing a directory
        mem, attachments, images = utils.process_files(
            paths=[str(project_dir)], use_memory=True, exclusions=[exclusion_path]
        )

        assert mem == "persistent memory data"

        main_py_path = (project_dir / "src/main.py").resolve()
        readme_md_path = (project_dir / "README.md").resolve()
        ignore_log_path = (project_dir / "ignore.log").resolve()

        assert main_py_path in attachments
        assert attachments[main_py_path] == "print('hello')"

        assert readme_md_path in attachments
        assert attachments[readme_md_path] == "# Read Me"

        assert (
            ignore_log_path not in attachments
        )  # Check that the excluded file is not in attachments
        assert images == []
        assert (
            len(attachments) == 2
        )  # Only two files should have been processed into attachments

    def test_process_files_single_text_file(self, fake_fs):
        """Tests processing a single text file."""
        file_path = Path("/my_script.py")
        file_path.write_text("def func(): pass")

        mem, attachments, images = utils.process_files(
            [str(file_path)], use_memory=False, exclusions=[]
        )

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
        (root_dir / "ignore_dir/secret.txt").write_text(
            "sensitive info"
        )  # Excluded dir
        (root_dir / "config.json").write_text("{}")
        (root_dir / "image.jpg").write_bytes(
            b"fake_jpg_data"
        )  # Should be collected as image

        mem, attachments, images = utils.process_files(
            paths=[str(root_dir)],
            use_memory=False,
            exclusions=[str(root_dir / "ignore_dir")],
        )

        assert len(attachments) == 3  # app.py, README.md, config.json
        assert (root_dir / "src/app.py").resolve() in attachments
        assert (root_dir / "docs/README.md").resolve() in attachments
        assert (root_dir / "config.json").resolve() in attachments
        assert (root_dir / "ignore_dir/secret.txt").resolve() not in attachments
        assert len(images) == 1
        assert images[0]["mime_type"] == "image/jpeg"

    def test_process_files_with_images(self, fake_fs):
        """Tests processing image files."""
        img_path = Path("/my_image.png")
        img_path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            )
        )  # A tiny transparent PNG

        mem, attachments, images = utils.process_files(
            [str(img_path)], use_memory=False, exclusions=[]
        )

        assert mem is None
        assert not attachments
        assert len(images) == 1
        assert images[0]["type"] == "image"
        assert images[0]["mime_type"] == "image/png"
        assert (
            images[0]["data"]
            == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )

    def test_process_files_with_zip_archive(self, fake_fs):
        """Tests processing text files within a zip archive."""
        zip_path = Path("/my_archive.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("code/script.py", "print('zip code')")
            zf.writestr("docs/notes.md", "Zip notes.")
            zf.writestr(
                "images/pic.png", b"binary_data"
            )  # Not text, should be ignored by zip processing
            zf.writestr(
                "excluded/temp.txt", "should be excluded"
            )  # excluded by exclusion list

        mem, attachments, images = utils.process_files(
            paths=[str(zip_path)],
            use_memory=False,
            exclusions=["temp.txt"],  # Exclude by name
        )

        assert mem is None
        assert (
            zip_path in attachments
        )  # The entire formatted content of the zip is stored under the zip's path
        assert (
            "--- FILE (from my_archive.zip): code/script.py ---\nprint('zip code')"
            in attachments[zip_path]
        )
        assert (
            "--- FILE (from my_archive.zip): docs/notes.md ---\nZip notes."
            in attachments[zip_path]
        )
        assert "images/pic.png" not in attachments[zip_path]  # Binary should be ignored
        assert "excluded/temp.txt" not in attachments[zip_path]  # Excluded by name
        assert not images  # Images in zip files are not supported this way for now

    def test_process_files_with_tar_archive(self, fake_fs):
        """Tests processing text files within a .tar.gz archive."""
        tar_path = Path("/my_archive.tar.gz")

        # Create an in-memory tar.gz file
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add a supported text file
            py_content = b"print('hello from tar')"
            py_info = tarfile.TarInfo(name="code/script.py")
            py_info.size = len(py_content)
            tar.addfile(py_info, io.BytesIO(py_content))

            # Add an unsupported file type
            png_content = b"fake_png_data"
            png_info = tarfile.TarInfo(name="images/pic.png")
            png_info.size = len(png_content)
            tar.addfile(png_info, io.BytesIO(png_content))

            # Add a file to be excluded by name
            log_content = b"excluded log content"
            log_info = tarfile.TarInfo(name="excluded.log")
            log_info.size = len(log_content)
            tar.addfile(log_info, io.BytesIO(log_content))

        # Write the buffer to the fake filesystem
        fake_fs.create_file(tar_path, contents=tar_buffer.getvalue())

        mem, attachments, images = utils.process_files(
            paths=[str(tar_path)],
            use_memory=False,
            exclusions=["excluded.log"],  # Exclude by name
        )

        assert mem is None
        assert tar_path in attachments
        # Check that the python script was included
        assert (
            "--- FILE (from my_archive.tar.gz): code/script.py ---\nprint('hello from tar')"
            in attachments[tar_path]
        )
        # Check that the unsupported file was ignored
        assert "images/pic.png" not in attachments[tar_path]
        # Check that the excluded file was ignored
        assert "excluded.log" not in attachments[tar_path]
        # Check that there is only one file included
        assert attachments[tar_path].count("--- FILE") == 1
        assert not images

    def test_process_files_no_memory_enabled(self, fake_fs):
        """Tests that persistent memory content is not loaded when use_memory is False."""
        # Removed: fake_fs.create_dir(config.DATA_DIR) - already created by conftest fixture
        # Ensure log directory is created for persistent memory if needed
        config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        fake_fs.create_file(
            config.PERSISTENT_MEMORY_FILE, contents="persistent memory data"
        )

        mem, attachments, images = utils.process_files(
            paths=[], use_memory=False, exclusions=[]
        )

        assert mem is None  # Should be None because use_memory is False

    def test_extract_text_from_message_openai_str(self):
        """Extracts text from a simple OpenAI string content."""
        msg = {"role": "assistant", "content": "Hello there."}
        assert utils.extract_text_from_message(msg) == "Hello there."

    def test_extract_text_from_message_openai_list(self):
        """Extracts text from an OpenAI list-of-dict content."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's up?"},
                {"type": "image_url", "url": "data:image/png;base64,..."},
            ],
        }
        assert utils.extract_text_from_message(msg) == "What's up?"

    def test_extract_text_from_message_gemini(self):
        """Extracts text from a Gemini-style message with 'parts'."""
        msg = {"role": "model", "parts": [{"text": "How can I help?"}]}
        assert utils.extract_text_from_message(msg) == "How can I help?"

        msg_with_image = {
            "role": "user",
            "parts": [
                {"text": "Describe this."},
                {"inline_data": {"mime_type": "image/jpeg", "data": "abc"}},
            ],
        }
        assert utils.extract_text_from_message(msg_with_image) == "Describe this."

    def test_extract_text_from_message_empty_or_malformed(self):
        """Tests extraction from empty or malformed messages."""
        assert utils.extract_text_from_message({"role": "user"}) == ""
        assert utils.extract_text_from_message({"role": "user", "content": []}) == ""
        assert (
            utils.extract_text_from_message(
                {"role": "user", "content": [{"type": "image_url"}]}
            )
            == ""
        )
        assert utils.extract_text_from_message({}) == ""
