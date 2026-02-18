"""Module wrap interface of AiClient"""
from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import Callable, Generator, Optional

from pydantic import BaseModel, TypeAdapter

from ..core import model
from ..memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

class Provider(Enum):
    """Enum defining usable AI provider"""
    OLLAMA = 1
    GROQ = 2
    OPENAI = 3
    ANTHROPIC = 4
    GOOGLE = 5

class AiClient(ABC):
    """Interface for AI client"""
    @abstractmethod
    def query_llm(self, prompt: str, memory: AgentMemory, stream: bool = False, output_format: BaseModel = None):
        """
        Query the language model with a prompt.
        
        Args:
            prompt: The input text to send to the LLM
            memory: AgentMemory object to provide context for the query
            stream: Whether to stream the response (default: False)
            output_format: Optional Pydantic model to validate and format the output
            
        Returns:
            The LLM's response, either as a string or formatted according to output_format
        """
        return

class BaseChatClient(AiClient):
    """Shared utilities for chat-based providers."""
    def _stream_response(
        self,
        memory: AgentMemory,
        response: Generator,
        content_extractor: Callable[[object], Optional[str]],
    ) -> Generator[str, None, None]:
        def stream_generator():
            full_response = []
            try:
                for chunk in response:
                    content = content_extractor(chunk)
                    if content:
                        full_response.append(content)
                        yield content
            except Exception as e:
                logger.error("Error in stream_generator: %s", e)
                raise
            finally:
                if full_response:
                    memory.record({"role": "assistant", "content": "".join(full_response)})
        return stream_generator()

    def _normalize_tool_calls(self, tool_calls: Optional[list]) -> Optional[list[model.ToolCall]]:
        if not tool_calls:
            return None
        tool_calls_dicts = [tc.model_dump() if hasattr(tc, "model_dump") else dict(tc) for tc in tool_calls]
        tool_calls_adapter = TypeAdapter(list[model.ToolCall])
        return tool_calls_adapter.validate_python(tool_calls_dicts)

    def _record_assistant_message(self, memory: AgentMemory, content: Optional[str]) -> None:
        memory.record({"role": "assistant", "content": content})

    def _finalize_response(self, memory: AgentMemory, content: Optional[str], tool_calls: Optional[list]) -> tuple[str, list]:
        normalized_tool_calls = self._normalize_tool_calls(tool_calls)
        if normalized_tool_calls:
            return (content, normalized_tool_calls)
        self._record_assistant_message(memory, content)
        return (content, None)
