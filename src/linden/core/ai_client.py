"""Module wrap interface of AiClient"""
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel

class Provider(Enum):
    """Enum defining usable AI provider"""
    OLLAMA = 1
    GROQ = 2
    OPENAI = 3

class AiClient(ABC):
    """Interface for AI client"""
    @abstractmethod
    def query_llm(self, prompt: str, stream: bool = False, output_format: BaseModel = None):
        """
        Query the language model with a prompt.
        
        Args:
            prompt: The input text to send to the LLM
            stream: Whether to stream the response
            output_format: Optional Pydantic model to validate and format the output
            
        Returns:
            The LLM's response, either as a string or formatted according to output_format
        """
        return
