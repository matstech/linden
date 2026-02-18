"""Module defininf neede configuration model"""
import os
import tomllib
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelsConfig:
    """
    Configuration for AI models used in the application.
    
    Attributes:
        dec: Model used for decision making
        tool: Model used for tool execution
        extractor: Model used for data extraction
        speaker: Model used for text generation or conversation
    """
    dec: str
    tool: str
    extractor: str
    speaker: str


@dataclass
class GroqConfig:
    """
    Configuration for Groq API client.
    
    Attributes:
        base_url: Base URL for Groq API
        api_key: Authentication key for Groq API
        timeout: Request timeout in seconds
    """
    base_url: str
    api_key: str
    timeout: int

@dataclass
class OllamaConfig:
    """
    Configuration for Ollama client.
    
    Attributes:
        timeout: Request timeout in seconds
    """
    timeout: int


@dataclass
class OpenAIConfig:
    """
    Configuration for OpenAI API client.
    
    Attributes:
        api_key: Authentication key for OpenAI API
        timeout: Request timeout in seconds
    """
    api_key: str
    timeout: int
    
@dataclass
class AnthropicConfig:
    """
    Configuration for Anthropic API client.
    
    Attributes:
        api_key: Authentication key for Anthropic API
        timeout: Request timeout in seconds
    """
    api_key: str
    max_tokens: int
    timeout: int

@dataclass
class MemoryConfig:
    """
    Configuration for agent memory storage.
    
    Attributes:
        path: File path for memory storage
        collection_name: Name of the memory collection
    """
    path: str
    collection_name: str


@dataclass
class Configuration:
    """
    Main configuration class that contains all settings for the application.
    
    This class aggregates all configuration components including models,
    API clients (Groq, Ollama, OpenAI), and memory settings.
    
    Attributes:
        models: Configuration for AI models
        groq: Configuration for Groq API
        ollama: Configuration for Ollama
        openai: Configuration for OpenAI API
        memory: Configuration for agent memory storage
    """
    models: ModelsConfig
    groq: GroqConfig
    ollama: OllamaConfig
    openai: OpenAIConfig
    anthropic: AnthropicConfig
    memory: MemoryConfig

    @classmethod
    def from_file(cls, file_path: str | Path) -> 'Configuration':
        """
        Create a Configuration instance from a TOML file.
        
        Args:
            file_path: Path to the TOML configuration file
            
        Returns:
            Configuration: A new configuration instance with values from the file
        """
        with open(file_path, 'rb') as f:
            data = tomllib.load(f)

        openai_config = OpenAIConfig(**data['openai'])
        if openai_config.api_key is None or openai_config.api_key == '':
            openai_config.api_key = 'api-key'
        os.environ['OPENAI_API_KEY'] = openai_config.api_key

        return cls(
            models=ModelsConfig(**data['models']),
            groq=GroqConfig(**data['groq']),
            ollama=OllamaConfig(**data['ollama']),
            openai=openai_config,
            anthropic=AnthropicConfig(**data['anthropic']),
            memory=MemoryConfig(**data['memory'])
        )


class ConfigManager:
    """
    Singleton manager for application configuration.
    
    Provides centralized access to configuration settings and handles
    configuration initialization, retrieval, and reloading.
    
    Attributes:
        _instance: Internal storage for the singleton Configuration instance
        _config_path: Path to the configuration file
        _default_config_paths: List of default paths to search for config files
    """
    _instance: Optional['Configuration'] = None
    _config_path: Optional[str] = None
    _default_config_paths = ["config.toml", "config/config.toml", "settings.toml", "../config.toml"]

    @classmethod
    def initialize(cls, config_path: str | Path) -> None:
        """
        Initialize the ConfigManager with a configuration file.
        
        Args:
            config_path: Path to the configuration file
        """
        cls._instance = Configuration.from_file(config_path)
        cls._config_path = str(config_path)

    @classmethod
    def get(cls, config_path: Optional[str | Path] = None) -> 'Configuration':
        """
        Get the configuration instance, initializing it if necessary.
        
        If the ConfigManager is not initialized, this method will attempt to initialize it
        using the provided config_path or by searching for a config file in default locations.
        
        Args:
            config_path: Optional path to the configuration file
            
        Returns:
            Configuration: The configuration instance
            
        Raises:
            RuntimeError: If no configuration file is found and no config_path is provided
        """
        if cls._instance is None:
            if config_path:
                cls.initialize(config_path)
            else:
                for default_path in cls._default_config_paths:
                    if Path(default_path).exists():
                        cls.initialize(default_path)
                        break
                else:
                    raise RuntimeError(
                        "ConfigManager not initialized and no configuration file "
                        f"found in: {', '.join(cls._default_config_paths)}. "
                        "Call initialize() explicitly or specify config_path."
                    )
        return cls._instance

    @classmethod
    def reload(cls) -> None:
        """
        Reload the configuration from the previously used file.
        
        Raises:
            RuntimeError: If no configuration file has been specified previously
        """
        if cls._config_path is None:
            raise RuntimeError("No configuration file specified")
        cls._instance = Configuration.from_file(cls._config_path)

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if the ConfigManager is initialized.
        
        Returns:
            bool: True if the ConfigManager has been initialized, False otherwise
        """
        return cls._instance is not None

    @classmethod
    def reset(cls) -> None:
        """
        Reset the ConfigManager by clearing the current configuration instance and path.
        
        This is useful for testing or when you want to reinitialize with a different configuration.
        """
        cls._instance = None
        cls._config_path = None
