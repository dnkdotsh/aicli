# tests/test_personas.py
"""
Tests for the persona management module in aicli/personas.py.
"""

import pytest
import json
import shutil

# Import the module to be tested
from aicli import personas
from aicli import config

def create_fake_persona(fs, filename: str, content: dict):
    """Helper to create a persona file in the correct directory on the fake filesystem."""
    persona_path = config.PERSONAS_DIRECTORY / filename
    fs.create_file(persona_path, contents=json.dumps(content))


class TestPersonas:
    """Test suite for persona management."""

    def test_ensure_personas_directory_and_default_creates_all(self, fake_fs):
        """
        Tests that the function creates the directory and the default file when neither exist.
        """
        # The fake_fs fixture pre-creates this dir; remove it for this specific test.
        shutil.rmtree(config.PERSONAS_DIRECTORY)
        assert not config.PERSONAS_DIRECTORY.exists()

        personas.ensure_personas_directory_and_default()

        assert config.PERSONAS_DIRECTORY.is_dir()
        default_persona_path = config.PERSONAS_DIRECTORY / personas.DEFAULT_PERSONA_FILENAME
        assert default_persona_path.is_file()

        # Check content of default persona
        with open(default_persona_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data['name'] == "AICLI Assistant"
        assert "expert knowledge of the AICLI tool" in data['description']
        assert "You are a versatile and helpful general-purpose AI assistant." in data['system_prompt']
        assert "## AICLI Tool Documentation" in data['system_prompt']

    def test_ensure_personas_directory_does_not_overwrite(self, fake_fs):
        """
        Tests that the function does not overwrite an existing default persona file.
        """
        custom_content = {"name": "Custom Default", "system_prompt": "Do custom things."}
        create_fake_persona(fake_fs, personas.DEFAULT_PERSONA_FILENAME, custom_content)

        personas.ensure_personas_directory_and_default()

        default_persona_path = config.PERSONAS_DIRECTORY / personas.DEFAULT_PERSONA_FILENAME
        with open(default_persona_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data['name'] == "Custom Default"
        assert data['system_prompt'] == "Do custom things."

    def test_load_persona_success(self, fake_fs):
        """Tests successfully loading a valid persona."""
        persona_content = {
            "name": "Test Coder",
            "description": "A coding persona.",
            "system_prompt": "You are a coder.",
            "engine": "openai",
            "model": "gpt-4o",
            "max_tokens": 8192,
            "stream": False
        }
        create_fake_persona(fake_fs, "coder.json", persona_content)

        # Test with extension
        p = personas.load_persona("coder.json")
        assert p is not None
        assert p.name == "Test Coder"
        assert p.filename == "coder.json"
        assert p.description == "A coding persona."
        assert p.system_prompt == "You are a coder."
        assert p.engine == "openai"
        assert p.model == "gpt-4o"
        assert p.max_tokens == 8192
        assert p.stream is False

        # Test without extension
        p_no_ext = personas.load_persona("coder")
        assert p_no_ext is not None
        assert p_no_ext.name == "Test Coder"

    def test_load_persona_file_not_found(self, fake_fs):
        """Tests that loading a non-existent persona returns None."""
        assert personas.load_persona("non_existent") is None

    def test_load_persona_invalid_json(self, fake_fs):
        """Tests that loading a malformed JSON file returns None."""
        malformed_path = config.PERSONAS_DIRECTORY / "malformed.json"
        # single quotes are invalid JSON, this will cause a JSONDecodeError
        fake_fs.create_file(malformed_path, contents="{ 'name': 'bad json', }")

        assert personas.load_persona("malformed.json") is None

    @pytest.mark.parametrize("missing_key", ["name", "system_prompt"])
    def test_load_persona_missing_required_keys(self, fake_fs, missing_key):
        """Tests that loading a persona missing required fields returns None."""
        persona_content = {
            "name": "Test Persona",
            "system_prompt": "A prompt."
        }
        del persona_content[missing_key]
        create_fake_persona(fake_fs, "incomplete.json", persona_content)

        assert personas.load_persona("incomplete.json") is None

    def test_list_personas(self, fake_fs):
        """Tests listing multiple personas, ensuring valid ones are returned and sorted."""
        # Create valid personas
        create_fake_persona(fake_fs, "coder.json", {"name": "Coder", "system_prompt": "p1"})
        create_fake_persona(fake_fs, "writer.json", {"name": "Writer", "system_prompt": "p2"})
        create_fake_persona(fake_fs, "analyst.json", {"name": "Analyst", "system_prompt": "p3"})

        # Create invalid/irrelevant files that should be ignored
        create_fake_persona(fake_fs, "incomplete.json", {"name": "Incomplete"}) # Missing system_prompt
        fake_fs.create_file(config.PERSONAS_DIRECTORY / "not_a_persona.txt", contents="hello")
        fake_fs.create_file(config.PERSONAS_DIRECTORY / "bad.json", contents="not json")

        persona_list = personas.list_personas()

        assert len(persona_list) == 3
        # Check for correct sorting by name
        assert persona_list[0].name == "Analyst"
        assert persona_list[1].name == "Coder"
        assert persona_list[2].name == "Writer"

    def test_list_personas_no_directory(self, fake_fs):
        """Tests that list_personas returns an empty list if the directory doesn't exist."""
        # The fake_fs fixture pre-creates this dir; remove it for this specific test.
        shutil.rmtree(config.PERSONAS_DIRECTORY)
        assert not config.PERSONAS_DIRECTORY.exists()
        assert personas.list_personas() == []
