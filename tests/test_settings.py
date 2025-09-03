# tests/test_settings.py
"""
Tests for the settings management in aicli/settings.py.
"""

import json

from aicli import config
from aicli import settings as app_settings


class TestSettings:
    """Test suite for settings management."""

    def test_save_setting_bool(self, fake_fs):
        """Tests saving a boolean value from a string."""
        success, message = app_settings.save_setting("stream", "false")
        assert success is True
        assert "Setting 'stream' updated to 'False'" in message
        with open(config.SETTINGS_FILE) as f:
            data = json.load(f)
        assert data["stream"] is False

    def test_save_setting_int(self, fake_fs):
        """Tests saving an integer value from a string."""
        success, message = app_settings.save_setting("api_timeout", "120")
        assert success is True
        assert "Setting 'api_timeout' updated to '120'" in message
        with open(config.SETTINGS_FILE) as f:
            data = json.load(f)
        assert data["api_timeout"] == 120

    def test_save_setting_unknown_key(self):
        """Tests that an unknown key is handled gracefully."""
        success, message = app_settings.save_setting("non_existent_key", "some_value")
        assert success is False
        assert "Unknown setting: 'non_existent_key'" in message
