# pylint: disable=C0301
"""Module wrap ANTHROPIC provider interaction"""
import logging
from typing import Generator
from anthropic import Anthropic, Stream
from anthropic.types import Message, RawMessageStreamEvent
from .ai_client import AiClient
from ..core import model
from ..memory.agent_memory import AgentMemory

from ..config.configuration import ConfigManager

logger = logging.getLogger(__name__)

class AnthropicClient(AiClient):
    """Defining ANTHROPIC integration"""
    def __init__(self, model: str, temperature: float, tools =  None):
        self.model = model
        self.temperature = temperature
        self.tools = tools
        anthropic_config = ConfigManager.get().anthropic
        self.client = Anthropic(timeout=anthropic_config.timeout, api_key=anthropic_config.api_key)
        super().__init__()

    def query_llm(self, prompt: str, memory: AgentMemory, stream: bool = False, output_format: bool = False) -> Generator[str, None, None] | tuple[str, list]:
        """Query the Anthropic LLM with proper error handling and response management.
        
        Args:
            stream (AgentMemory): Message history for chat mode (AgentMemory memory object)
            prompt (str): The input text or prompt
            output_format (bool, optional): Whether to stream the response. Defaults to False.
            format (BaseModel, optional): Optional Pydantic model for response validation. Defaults to None.
            
        Returns:
            Generator[str, None, None] | tuple[str, list]: Either a generator of text chunks (if stream=True)
            or a tuple of (content, tool_calls) where content is the model's output and 
            tool_calls is a list of tool calls (or None) (if stream=False).
        """
        try:

            response = self.client.messages.create(
                model=self.model,
                max_tokens=ConfigManager.get().anthropic.max_tokens,
                temperature=self.temperature,
                stream=stream,
                messages=memory.get_conversation(user_input=prompt)[1:],
                system=[self._inject_output_format_info(memory.get_system_prompt(), output_format)],
                tools=self.tools  if self.tools and len(self.tools) > 0 else [])

            if stream:
                return self._generate_stream(memory=memory, response=response)
            else:
                return self._build_final_response(memory=memory, response=response)

        except Exception as e:
            logger.error("Error in Anthropic query: %s", str(e))
            raise

    def _generate_stream(self, memory: AgentMemory, response: Stream[RawMessageStreamEvent]) -> Generator[str, None, None]:
        """Handle streaming response with proper cleanup and error handling.
        
        Note: This method only handles text content since streaming is disabled when tools are present.
        
        Args:
            memory (AgentMemory): Memory object to record the assistant response
            response (Stream[RawMessageStreamEvent]): The streaming response
            
        Yields:
            str: Chunks of text content from the model response
        """
        def stream_generator():
            full_response = []
            try:
                for event in response:
                    # Handle text content chunks only (tools are not available in streaming mode)
                    if event.type == 'content_block_delta':
                        if hasattr(event.delta, 'text'):
                            content = event.delta.text
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

    def _build_final_response(self, memory: AgentMemory, response: Message) -> tuple[str, list]:
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

        if not hasattr(response, 'content'):
            content = ''
            tool_calls = None
        elif len(response.content) > 0:
            tool_calls = None
            for c in response.content:
                if c.type == 'tool_use':
                    tool_calls = c
            if not tool_calls:
                content = response.content[0].text

        tc = []
        if tool_calls:
            tool_call = model.ToolCall(id=tool_calls.id,
                                       type=tool_calls.type,
                                       function=model.Function(name=tool_calls.name, arguments=tool_calls.input))
            tc.append(tool_call)
        else:
            # Save to memory only normal text responses, not tool calls
            memory.record({"role": "assistant", "content": content})

        return (content, tc)

    def _inject_output_format_info(self, system_prompt: str, output_format: bool):
        text = system_prompt['content']
        if output_format:
            text = f"""{text}.\nThe output format must be follow the structure:\n{output_format.model_json_schema()}."""
        return {"type":"text", "text": text}
