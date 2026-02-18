# pylint: disable=C0301
"""Module wrap GROQ provider interaction"""
import logging
from typing import Generator
from pydantic import BaseModel, TypeAdapter
from groq import Groq, Stream
from groq.types.chat import ChatCompletion, ChatCompletionChunk

from ..core import model
from ..memory.agent_memory import AgentMemory
from .ai_client import BaseChatClient
from ..config.configuration import ConfigManager

logger = logging.getLogger(__name__)

class GroqClient(BaseChatClient):
    """Defining GROQ integration"""
    def __init__(self, model: str, temperature: float, tools =  None):
        self.model = model
        self.temperature = temperature
        self.tools = tools
        groq_config = ConfigManager.get().groq
        self.client = Groq(timeout=groq_config.timeout, api_key=groq_config.api_key,base_url=groq_config.base_url)
        super().__init__()

    def query_llm(self, prompt: str, memory: AgentMemory, stream: bool = False, output_format: BaseModel = None) -> Generator[str, None, None] | tuple[str, list]:
        """Query the Groq LLM with proper error handling and response management.
        
        Args:
            memory (AgentMemory): Message history for chat mode (AgentMemory memory object)
            input (str): The input text or prompt
            stream (bool, optional): Whether to stream the response. Defaults to False.
            format (BaseModel, optional): Optional Pydantic model for response validation. Defaults to None.
            
        Returns:
            Generator[str, None, None] | tuple[str, list]: Either a generator of text chunks (if stream=True)
            or a tuple of (content, tool_calls) where content is the model's output and 
            tool_calls is a list of tool calls (or None) (if stream=False).
        """
        try:

            conversation = memory.get_conversation(user_input=prompt)

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                stream=stream,
                messages=conversation,
                tools=self.tools,
                response_format={"type": "json_object"} if output_format else None)

            if not stream:
                return self._build_final_response(memory=memory, response=response)
            else:
                return self._generate_stream(memory=memory, response=response)

        except Exception as e:
            logger.error("Error in Groq query: %s", str(e))
            raise

    def _extract_stream_content(self, chunk: ChatCompletionChunk) -> str | None:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            return delta.content
        return None

    def _generate_stream(self, memory: AgentMemory, response: Stream[ChatCompletionChunk]) -> Generator[str, None, None]:
        """Handle streaming response with proper cleanup and error handling.
        
        Note: This method only handles text content since streaming is disabled when tools are present.
        
        Args:
            memory (AgentMemory): Memory object to record the assistant response
            response (Stream[ChatCompletionChunk]): The streaming response
            
        Yields:
            str: Chunks of text content from the model response
        """
        return self._stream_response(memory=memory, response=response, content_extractor=self._extract_stream_content)

    def _normalize_tool_calls(self, tool_calls: list | None) -> list | None:
        if not tool_calls:
            return None
        tool_calls_dicts = [tc.model_dump() if hasattr(tc, "model_dump") else dict(tc) for tc in tool_calls]
        tool_calls_adapter = TypeAdapter(list[model.ToolCall])
        return tool_calls_adapter.validate_python(tool_calls_dicts)

    def _build_final_response(self, memory: AgentMemory, response: ChatCompletion) -> tuple[str, list]:
        """Processes a complete (non-streaming) response and updates memory.

        Args:
            memory (AgentMemory): Memory object to record the assistant response
            response (ChatCompletion): The HTTP response object containing the full response.

        Returns:
            tuple[str, list]: A tuple containing (content, tool_calls) where content is the model's output 
            and tool_calls is a list of tool calls (or None if no tools were called).
        """
        content = None
        tool_calls = None

        if not hasattr(response, 'choices'):
            content = ''
        elif len(response.choices) > 0 and hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
            message = response.choices[0].message
            content = message.content
            tool_calls = message.tool_calls

        return self._finalize_response(memory=memory, content=content, tool_calls=tool_calls)

