"""Module defininf neede configuration model"""
import os
import tomllib
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroqConfig:
    """
    Configuration for Groq API client.
    
    Attributes:
        base_url: Base URL for Groq API
        api_key: Authentication key for Groq API
        timeout: Request timeout in seconds
    """
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = 60

@dataclass
class OllamaConfig:
    """
    Configuration for Ollama client.
    
    Attributes:
        timeout: Request timeout in seconds
    """
    timeout: Optional[int] = 30


@dataclass
class OpenAIConfig:
    """
    Configuration for OpenAI API client.
    
    Attributes:
        api_key: Authentication key for OpenAI API
        timeout: Request timeout in seconds
    """
    api_key: Optional[str] = None
    timeout: Optional[int] = 60
    
@dataclass
class AnthropicConfig:
    """
    Configuration for Anthropic API client.
    
    Attributes:
        api_key: Authentication key for Anthropic API
        timeout: Request timeout in seconds
    """
    api_key: Optional[str] = None
    max_tokens: int = 1024
    timeout: int = 60


@dataclass
class GoogleConfig:
    """
    Configuration for Google (Gemini) API client.
    
    Attributes:
        api_key: Authentication key for Google GenAI
        timeout: Request timeout in seconds
    """
    api_key: Optional[str] = None
    timeout: Optional[int] = 60


@dataclass
class MemoryConfig:
    """
    Configuration for agent memory storage.
    
    Attributes:
        path: File path for memory storage
        collection_name: Name of the memory collection
        llm_provider: Provider for the LLM (e.g., 'openai', 'ollama')
        llm_model: Model name for the LLM
        embedder_provider: Provider for the embedder
        embedder_model: Model name for the embedder
    """
    path: str
    collection_name: str
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    embedder_provider: str = "openai"
    embedder_model: str = "text-embedding-3-small"
    summarization_threshold_chars: int = 1500


@dataclass
class Configuration:
    """
    Main configuration class that contains all settings for the application.
    
    This class aggregates all configuration components including models,
    API clients (Groq, Ollama, OpenAI), and memory settings.
    
    Attributes:
        groq: Configuration for Groq API
        ollama: Configuration for Ollama
        openai: Configuration for OpenAI API
        anthropic: Configuration for Anthropic API
        google: Configuration for Google GenAI API
        memory: Configuration for agent memory storage
    """
    groq: GroqConfig
    ollama: OllamaConfig
    openai: OpenAIConfig
    anthropic: AnthropicConfig
    google: GoogleConfig
    memory: Optional[MemoryConfig] = None

    @classmethod
    def from_file(cls, file_path: str | Path) -> 'Configuration':
        """
        Create a Configuration instance from a TOML file.
        This method also loads API keys from standard environment variables 
        (e.g., OPENAI_API_KEY), which take precedence over values in the TOML file.
        
        Args:
            file_path: Path to the TOML configuration file
            
        Returns:
            Configuration: A new configuration instance with values from the file
        """
        with open(file_path, 'rb') as f:
            data = tomllib.load(f)

        # Load configurations, giving priority to environment variables for API keys
        openai_data = data.get('openai', {})
        openai_config = OpenAIConfig(
            api_key=os.getenv('OPENAI_API_KEY') or openai_data.get('api_key'),
            timeout=openai_data.get('timeout', 60)
        )

        groq_data = data.get('groq', {})
        groq_config = GroqConfig(
            api_key=os.getenv('GROQ_API_KEY') or groq_data.get('api_key'),
            base_url=groq_data.get('base_url'),
            timeout=groq_data.get('timeout', 60)
        )

        anthropic_data = data.get('anthropic', {})
        anthropic_config = AnthropicConfig(
            api_key=os.getenv('ANTHROPIC_API_KEY') or anthropic_data.get('api_key'),
            max_tokens=anthropic_data.get('max_tokens', 1024),
            timeout=anthropic_data.get('timeout', 60)
        )

        google_data = data.get('google', {})
        google_config = GoogleConfig(
            api_key=os.getenv('GOOGLE_API_KEY') or google_data.get('api_key'),
            timeout=google_data.get('timeout', 60)
        )

        ollama_data = data.get('ollama', {})
        ollama_config = OllamaConfig(timeout=ollama_data.get('timeout', 30))
        
        # Load memory configuration if it exists
        memory_data = data.get('memory')
        memory_config = MemoryConfig(**memory_data) if memory_data else None

        return cls(
            groq=groq_config,
            ollama=ollama_config,
            openai=openai_config,
            anthropic=anthropic_config,
            google=google_config,
            memory=memory_config
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
