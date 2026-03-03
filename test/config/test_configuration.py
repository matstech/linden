# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0303
import os
import tempfile
import pytest

from linden.config.configuration import (
    Configuration, ConfigManager, GroqConfig, 
    OllamaConfig, OpenAIConfig, AnthropicConfig, MemoryConfig
)


class TestConfiguration:
    def test_from_file_loads_valid_config(self, temp_config_file):
        """Test that Configuration.from_file correctly loads a valid config file."""
        config = Configuration.from_file(temp_config_file)
        
        # Check that all sections are loaded
        assert isinstance(config.groq, GroqConfig)
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.openai, OpenAIConfig)
        assert isinstance(config.anthropic, AnthropicConfig)
        assert isinstance(config.memory, MemoryConfig)
        
        # Check specific values
        assert config.groq.base_url == "https://api.groq.com/openai/v1"
        assert config.groq.api_key == "groq-test-key"
        assert config.openai.api_key == "openai-test-key"
        assert config.anthropic.api_key == "anthropic-test-key"
        assert config.anthropic.max_tokens == 4096
        assert config.memory.path == "/tmp/linden-memory"
    
    def test_from_file_with_invalid_path(self):
        """Test that Configuration.from_file raises FileNotFoundError for invalid file path."""
        with pytest.raises(FileNotFoundError):
            Configuration.from_file("non_existent_config_file.toml")

    def test_env_var_overrides_toml_api_key(self, temp_config_file, monkeypatch):
        """Test that an environment variable for an API key overrides the value in the TOML file."""
        monkeypatch.setenv("OPENAI_API_KEY", "key_from_env")
        config = Configuration.from_file(temp_config_file)
        assert config.openai.api_key == "key_from_env"

    def test_env_var_loads_api_key_if_missing_in_toml(self, monkeypatch):
        """Test that an API key is loaded from an env var if it's not in the TOML file."""
        # Create a config file without an OpenAI API key
        toml_content = b"""
[models]
dec = "test"
tool = "test"
extractor = "test"
speaker = "test"
[openai]
timeout = 30
[groq]
[ollama]
timeout = 30
[anthropic]
[google]
        """
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as temp:
            temp.write(toml_content)
            temp_path = temp.name

        try:
            monkeypatch.setenv("OPENAI_API_KEY", "only_from_env")
            config = Configuration.from_file(temp_path)
            assert config.openai.api_key == "only_from_env"
        finally:
            os.unlink(temp_path)

    def test_config_loads_without_memory_section(self):
        """Test that the configuration loads successfully when the [memory] section is missing."""
        toml_content = b"""
[models]
dec = "test"
tool = "test"
extractor = "test"
speaker = "test"
[openai]
timeout = 30
[groq]
[ollama]
timeout = 30
[anthropic]
[google]
        """
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as temp:
            temp.write(toml_content)
            temp_path = temp.name

        try:
            config = Configuration.from_file(temp_path)
            assert config.memory is None
        finally:
            os.unlink(temp_path)



class TestConfigManager:
    def setup_method(self):
        """Reset ConfigManager before each test."""
        ConfigManager.reset()
    
    def teardown_method(self):
        """Reset ConfigManager after each test."""
        ConfigManager.reset()
    
    def test_initialize(self, temp_config_file):
        """Test that ConfigManager.initialize correctly loads a configuration."""
        ConfigManager.initialize(temp_config_file)
        
        # Check that the manager is initialized
        assert ConfigManager.is_initialized() is True
        
        # Get the configuration
        config = ConfigManager.get()
        
        # Check that it's a valid Configuration instance
        assert isinstance(config, Configuration)
        assert config.openai.api_key == "openai-test-key"
    
    def test_get_with_explicit_path(self, temp_config_file):
        """Test that ConfigManager.get with an explicit path initializes and returns config."""
        # Manager is not initialized yet
        assert ConfigManager.is_initialized() is False
        
        # Get with explicit path
        config = ConfigManager.get(temp_config_file)
        
        # Check that it's initialized and returns a valid config
        assert ConfigManager.is_initialized() is True
        assert isinstance(config, Configuration)
    
    def test_get_with_default_path(self, temp_dir_with_config):
        """Test that ConfigManager.get finds a config file in default locations."""
        # Change to the temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir_with_config)
        
        try:
            # Get without explicit path (should find config.toml in current directory)
            config = ConfigManager.get()
            
            # Check that it's a valid Configuration instance
            assert ConfigManager.is_initialized() is True
            assert isinstance(config, Configuration)
        
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def test_get_without_config_file(self, monkeypatch):
        """Test that ConfigManager.get raises RuntimeError when no config file is found."""
        # Create a temporary directory without a config file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to the temp directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Patch default config paths to avoid finding any real config files
                monkeypatch.setattr(ConfigManager, '_default_config_paths', 
                                   ["non_existent_config.toml"])
                
                # Get without explicit path should raise RuntimeError
                with pytest.raises(RuntimeError) as excinfo:
                    ConfigManager.get()
                
                # Check the error message
                assert "ConfigManager not initialized" in str(excinfo.value)
                assert "non_existent_config.toml" in str(excinfo.value)
            
            finally:
                # Restore original directory
                os.chdir(original_dir)
    
    def test_reload(self, temp_config_file):
        """Test that ConfigManager.reload reloads the configuration."""
        # Initialize with a config file
        ConfigManager.initialize(temp_config_file)
        
        # Modify the config file
        with open(temp_config_file, "wb") as f:
            f.write(b"""
[models]
dec = "modified-model"
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
api_key = "modified-key"
timeout = 60

[anthropic]
api_key = "modified-anthropic-key"
max_tokens = 8192
timeout = 120

[google]
api_key = "modified-google-key"
timeout = 60

[memory]
path = "/tmp/linden-memory"
collection_name= "test-memories"
""")
        
        # Reload the config
        ConfigManager.reload()
        
        # Check that the configuration was updated
        config = ConfigManager.get()
        assert config.openai.api_key == "modified-key"
        assert config.anthropic.api_key == "modified-anthropic-key"
        assert config.anthropic.max_tokens == 8192
    
    def test_reload_without_initialization(self):
        """Test that ConfigManager.reload raises RuntimeError when not initialized."""
        with pytest.raises(RuntimeError) as excinfo:
            ConfigManager.reload()
        
        assert "No configuration file specified" in str(excinfo.value)
    
    def test_reset(self, temp_config_file):
        """Test that ConfigManager.reset clears the configuration."""
        # Initialize with a config file
        ConfigManager.initialize(temp_config_file)
        
        # Check that it's initialized
        assert ConfigManager.is_initialized() is True
        
        # Reset
        ConfigManager.reset()
        
        # Check that it's no longer initialized
        assert ConfigManager.is_initialized() is False
