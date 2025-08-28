# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0303

class TestConfigImports:
    """Test that all modules can be imported correctly."""
    
    def test_import_configuration(self):
        """Test importing the Configuration class."""
        from linden.config.configuration import Configuration
        assert Configuration is not None
    
    def test_import_config_manager(self):
        """Test importing the ConfigManager class."""
        from linden.config.configuration import ConfigManager
        assert ConfigManager is not None
    
    def test_import_dataclasses(self):
        """Test importing all dataclasses."""
        from linden.config.configuration import (
            ModelsConfig, GroqConfig, OllamaConfig, OpenAIConfig, MemoryConfig
        )
        assert ModelsConfig is not None
        assert GroqConfig is not None
        assert OllamaConfig is not None
        assert OpenAIConfig is not None
        assert MemoryConfig is not None
    
    def test_import_all_from_config(self):
        """Test importing all from linden.config."""
        import linden.config
        assert linden.config is not None
        
        # Test importing __init__ worked correctly
        from linden.config import Configuration, ConfigManager
        assert Configuration is not None
        assert ConfigManager is not None
