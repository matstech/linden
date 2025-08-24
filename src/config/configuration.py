import os
import tomllib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelsConfig:
    dec: str
    tool: str
    extractor: str
    speaker: str


@dataclass
class GroqConfig:
    base_url: str
    api_key: str
    timeout: int
    
@dataclass
class OllamaConfig:
    timeout: int


@dataclass
class OpenAIConfig:
    api_key: str
    timeout: int
    
@dataclass
class MemoryConfig:
    path: str


@dataclass
class Configuration:
    models: ModelsConfig
    groq: GroqConfig
    ollama: OllamaConfig
    openai: OpenAIConfig
    memory: MemoryConfig
    
    @classmethod
    def from_file(cls, file_path: str | Path) -> 'Configuration':
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
            memory=MemoryConfig(**data['memory'])
        )


class ConfigManager:
    _instance: Optional['Configuration'] = None
    _config_path: Optional[str] = None
    _default_config_paths = ["config.toml", "config/config.toml", "settings.toml", "../config.toml"]
    
    @classmethod
    def initialize(cls, config_path: str | Path) -> None:
        cls._instance = Configuration.from_file(config_path)
        cls._config_path = str(config_path)
    
    @classmethod
    def get(cls, config_path: Optional[str | Path] = None) -> 'Configuration':
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
        if cls._config_path is None:
            raise RuntimeError("No configuration file specified")
        cls._instance = Configuration.from_file(cls._config_path)
    
    @classmethod
    def is_initialized(cls) -> bool:
        return cls._instance is not None
    
    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._config_path = None