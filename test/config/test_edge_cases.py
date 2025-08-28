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
    Configuration, ConfigManager, ModelsConfig, GroqConfig, 
    OllamaConfig, OpenAIConfig, MemoryConfig
)


class TestConfigurationEdgeCases:
    def test_openai_api_key_empty(self, temp_config_file):
        """Test that empty OpenAI API key is replaced with a default value."""
        # Modify the config file to have an empty API key
        with open(temp_config_file, "rb") as f:
            data = tomllib.load(f)
        
        with open(temp_config_file, "wb") as f:
            # Set empty API key
            data_str = """
[models]
dec = "gpt-4o"
tool = "gpt-4-turbo"
extractor = "gpt-3.5-turbo"
speaker = "claude-3-opus"

[groq]
base_url = "https://api.groq.com/openai/v1"
api_key = "groq-test-key"
timeout = 60

[ollama]
timeout = 30

[openai]
api_key = ""
timeout = 60

[memory]
path = "/tmp/linden-memory"
collection_name= "test_memories"
"""
            f.write(data_str.encode('utf-8'))
        
        # Load the configuration
        config = Configuration.from_file(temp_config_file)
        
        # Check that the API key was replaced with the default value
        assert config.openai.api_key == "api-key"
        assert os.environ.get('OPENAI_API_KEY') == "api-key"
    
    def test_openai_api_key_none(self, temp_config_file):
        """Test that None OpenAI API key is replaced with a default value."""
        # Modify the config file to have None as the API key (using empty string since TOML doesn't support null)
        with open(temp_config_file, "wb") as f:
            data_str = """
[models]
dec = "gpt-4o"
tool = "gpt-4-turbo"
extractor = "gpt-3.5-turbo"
speaker = "claude-3-opus"

[groq]
base_url = "https://api.groq.com/openai/v1"
api_key = "groq-test-key"
timeout = 60

[ollama]
timeout = 30

[openai]
api_key = ""
timeout = 60

[memory]
path = "/tmp/linden-memory"
collection_name= "test_memories"
"""
            f.write(data_str.encode('utf-8'))
        
        # Load the configuration
        config = Configuration.from_file(temp_config_file)
        
        # Check that the API key was replaced with the default value
        assert config.openai.api_key == "api-key"
        assert os.environ.get('OPENAI_API_KEY') == "api-key"


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
            assert config.models.dec == "test-model"
            
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
    
    def test_missing_required_sections(self, temp_incomplete_config_file):
        """Test that missing required sections raise KeyError."""
        with pytest.raises(KeyError):
            Configuration.from_file(temp_incomplete_config_file)
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"")
    def test_empty_config_file(self, mock_file):
        """Test that an empty config file raises a KeyError because there are no sections."""
        with pytest.raises(KeyError):
            Configuration.from_file("mock_config.toml")
