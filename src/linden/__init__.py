"""
Linden - A Python framework for building AI agents with multi-provider LLM support.

This package provides tools for creating AI agents with:
- Multi-provider LLM support (OpenAI, Groq, Ollama)
- Persistent memory capabilities
- Function calling support
- Configurable agent behaviors
"""

__version__ = "0.4.0"
__author__ = "matstech"
__email__ = "matteo.stabile2@gmail.com"

# Import core components for easy access
from .core import AgentRunner
from .memory import AgentMemory
from .config import Configuration

__all__ = [
    "AgentRunner",
    "AgentMemory",
    "Configuration",
]
