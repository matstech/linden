# pylint: disable=C0301
"""Module wrap OPENAI provider interaction"""
import logging
from typing import Generator
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, TypeAdapter
from .ai_client import AiClient
from ..core import model
from ..memory.agent_memory import AgentMemory


from ..config.configuration import ConfigManager

logger = logging.getLogger(__name__)

class OpenAiClient(AiClient):
    """Defining OPENAI integration"""
    def __init__(self, model: str, temperature: float, tools =  None):
        self.model = model
        self.temperature = temperature
        self.tools = tools
        openai_config = ConfigManager.get().openai
        self.client = OpenAI(timeout=openai_config.timeout, api_key=openai_config.api_key)
        super().__init__()

    def query_llm(self, prompt: str, memory: AgentMemory, stream: bool = False, output_format: BaseModel = None) -> Generator[str, None, None] | tuple[str, list]:
        """Query the OpenAI LLM with proper error handling and response management.
        
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
            logger.error("Error in OpenAI query: %s", str(e))
            raise

    def _generate_stream(self, memory: AgentMemory, response: Stream[ChatCompletionChunk]) -> Generator[str, None, None]:
        """Handle streaming response with proper cleanup and error handling.
        
        Note: This method only handles text content since streaming is disabled when tools are present.
        
        Args:
            memory (AgentMemory): Memory object to record the assistant response
            response (Stream[ChatCompletionChunk]): The streaming response
            
        Yields:
            str: Chunks of text content from the model response
        """
        def stream_generator():
            full_response = []
            try:
                for chunk in response:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        content = delta.content

                        if content:
                            full_response.append(content)
                            yield content
            except Exception as e:
                logger.error("Error in stream_generator: %s", e)
                raise
            finally:
                # Record the complete response in memory
                if full_response:
                    complete_content = "".join(full_response)
                    memory.record({"role": "assistant", "content": complete_content})
        return stream_generator()

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
            tool_calls = None
        elif len(response.choices) > 0 and hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
            content = response.choices[0].message.content
            tool_calls = response.choices[0].message.tool_calls

        tc = None
        if tool_calls:
            # tool_calls is a list of ChatCompletionMessageToolCall objects
            # First convert them to dict (if they're not already)
            tool_calls_dicts = [tc.model_dump() if hasattr(tc, "model_dump") else dict(tc) for tc in tool_calls]
            # Then use TypeAdapter to validate the list as ToolCalls
            tool_calls_adapter = TypeAdapter(list[model.ToolCall])
            tc = tool_calls_adapter.validate_python(tool_calls_dicts)
            # For tool calls, don't save to memory (result will be saved by AgentRunner)
        else:
            # Save to memory only normal text responses, not tool calls
            memory.record({"role": "assistant", "content": content})

        return (content, tc)
