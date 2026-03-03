# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0303
import os
import tomllib
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest


from linden.config.configuration import (
    Configuration, ConfigManager, GroqConfig,
    OllamaConfig, OpenAIConfig, AnthropicConfig, MemoryConfig
)


class TestConfigManagerEdgeCases:
    def setup_method(self):
        """Reset ConfigManager before each test."""
        ConfigManager.reset()

    def teardown_method(self):
        """Reset ConfigManager after each test."""
        ConfigManager.reset()

    def test_initialize_with_path_object(self, temp_config_file):
        """Test initializing ConfigManager with Path object."""
        path = Path(temp_config_file)
        ConfigManager.initialize(path)

        assert ConfigManager.is_initialized()
        assert ConfigManager._config_path == str(path)

    def test_get_with_path_object(self, temp_config_file):
        """Test getting configuration with Path object."""
        path = Path(temp_config_file)
        config = ConfigManager.get(path)

        assert ConfigManager.is_initialized()
        assert isinstance(config, Configuration)

    def test_default_config_paths_search_order(self, temp_config_file):
        """Test that default config paths are searched in the correct order."""
        # Reset ConfigManager and save original default paths
        original_paths = ConfigManager._default_config_paths
        test_config_path = None

        try:
            # Create a temp file in a path we control
            with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as temp:
                test_config_path = temp.name
                temp.write(b"""
[models]
dec = "test-model"
tool = "test-model"
extractor = "test-model"
speaker = "test-model"

[groq]
base_url = "url"
api_key = "key"
timeout = 60

[ollama]
timeout = 30

[openai]
api_key = "key"
timeout = 60

[anthropic]
api_key = "anthropic-key"
max_tokens = 4096
timeout = 60

[google]
api_key = "google-key"
timeout = 60

[memory]
path = "/tmp/path"
collection_name= "test_memories"
""")

            # Override the default paths to include our temp file
            ConfigManager._default_config_paths = ["non_existent.toml", test_config_path]

            # Reset the manager
            ConfigManager.reset()

            # Get config without explicit path (should find our temp file)
            config = ConfigManager.get()

            # Check that it loaded our test configuration
            assert config is not None

            # Check that it used the second path
            assert ConfigManager._config_path == test_config_path

        finally:
            # Restore original paths
            ConfigManager._default_config_paths = original_paths
            ConfigManager.reset()

            # Clean up temp file if it was created
            if test_config_path and os.path.exists(test_config_path):
                os.unlink(test_config_path)


class TestConfigTOMLErrors:
    def test_invalid_toml_syntax(self, temp_invalid_config_file):
        """Test that invalid TOML syntax raises a ParseError."""
        with pytest.raises(tomllib.TOMLDecodeError):
            Configuration.from_file(temp_invalid_config_file)
    