"""Module wrap interface of AiClient"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Generator
import logging

from pydantic import BaseModel
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
    """Base class for chat-oriented providers.

    Provides common helpers for streaming responses and recording
    assistant messages into the shared memory abstraction.
    """

    def _stream_response(
        self,
        memory: AgentMemory,
        response: Generator[Any, None, None],
        content_extractor: Callable[[Any], str | None],
    ) -> Generator[str, None, None]:
        """Yield stream chunks while safely aggregating and persisting them.

        Args:
            memory: Agent memory to persist the final assistant message
            response: Streaming generator returned by the provider SDK
            content_extractor: Callable that extracts text from a response chunk

        Yields:
            str: The extracted text chunks from the streaming response
        """
        def stream_generator():
            full_response: list[str] = []
            try:
                for chunk in response:
                    try:
                        content = content_extractor(chunk) if content_extractor else None
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.error("Error extracting stream content: %s", exc)
                        raise

                    if content:
                        full_response.append(content)
                        yield content
            except Exception:
                # Propagate upstream after logging in extractor/consumer if needed
                raise
            finally:
                if full_response:
                    complete_content = "".join(full_response)
                    self._record_assistant_message(memory, complete_content)

        return stream_generator()

    def _record_assistant_message(self, memory: AgentMemory, content: str | None) -> None:
        """Record the assistant response in memory if available."""
        if content:
            memory.record({"role": "assistant", "content": content})
