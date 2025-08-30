# tests/test_settings.py
"""
Tests for the settings management in aicli/settings.py.
"""

import pytest
import json

from aicli import settings as app_settings
from aicli import config

class TestSettings:
    """Test suite for settings management."""

    def test_save_setting_bool(self, fake_fs):
        """Tests saving a boolean value from a string."""
        app_settings.save_setting('stream', 'false')
        with open(config.SETTINGS_FILE, 'r') as f:
            data = json.load(f)
        assert data['stream'] is False

    def test_save_setting_int(self, fake_fs):
        """Tests saving an integer value from a string."""
        app_settings.save_setting('api_timeout', '120')
        with open(config.SETTINGS_FILE, 'r') as f:
            data = json.load(f)
        assert data['api_timeout'] == 120

    def test_save_setting_unknown_key(self, capsys):
        """Tests that an unknown key is handled gracefully."""
        app_settings.save_setting('non_existent_key', 'some_value')
        captured = capsys.readouterr()
        assert "Unknown setting" in captured.out
