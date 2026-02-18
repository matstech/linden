import logging
from typing import Generator

from pydantic import BaseModel
from .ai_client import BaseChatClient
from ..core import model
from ..memory.agent_memory import AgentMemory
from ..config.configuration import ConfigManager
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GoogleClient(BaseChatClient):
    
    """Defining GOOGLE integration"""
    def __init__(self, model: str, temperature: float, tools =  None):
        self.model = model
        self.temperature = temperature
        self.tools = tools
        google_config = ConfigManager.get().google
        self.client = genai.Client(timeout=google_config.timeout, api_key=google_config.api_key)
        super().__init__()


    def query_llm(self, prompt: str, memory: AgentMemory, stream: bool = False, output_format: BaseModel = None) -> Generator[str, None, None] | tuple[str, list]:
        """Query the Google LLM with proper error handling and response management.
        
        Args:
            memory (AgentMemory): Message history for chat mode (AgentMemory memory object)
            input (str): The input text or prompt
            stream (bool, optional): Whether to stream the response. Defaults to False.
            output_format (BaseModel, optional): Optional Pydantic model for response validation. Defaults to None.
            
        Returns:
            Generator[str, None, None] | tuple[str, list]: Either a generator of text chunks (if stream=True)
            or a tuple of (content, tool_calls) where content is the model's output and 
            tool_calls is a list of tool calls (or None) (if stream=False).
        """
        try:

            conversation = memory.get_conversation(user_input=prompt)
            system_prompt = memory.get_system_prompt()

            message = prompt
            history = []
            if conversation:
                last_message = conversation[-1]
                if last_message.get("role") == "user":
                    message = last_message.get("content", "")
                    history = self._build_contents(conversation[:-1])
                else:
                    history = self._build_contents(conversation)

            config = types.GenerateContentConfig(
                temperature=self.temperature,
                tools=self.tools,
                system_instruction=(system_prompt.get("content") if isinstance(system_prompt, dict) else system_prompt),
                # It depends on Pydantic usage or not
                response_mime_type="application/json" if output_format else None,
                response_schema=output_format.model_json_schema() if output_format else None,
            )

            chat = self.client.chats.create(
                model=self.model,
                config=config,
                history=history)

            if stream:
                response = chat.send_message_stream(message)
                return self._generate_stream(memory=memory, response=response)

            response = chat.send_message(message)
            return self._build_final_response(memory=memory, response=response)

        except Exception as e:
            logger.error("Error in Google query: %s", str(e))
            raise

    def _extract_stream_content(self, chunk: types.GenerateContentResponse) -> str | None:
        """Extract text content from a streaming GenerateContentResponse chunk."""
        if hasattr(chunk, "text"):
            return chunk.text
        return None

    def _generate_stream(self, memory: AgentMemory, response: Generator[types.GenerateContentResponse, None, None]) -> Generator[str, None, None]:
        """Handle streaming response with proper cleanup and error handling.
        
        Note: This method only handles text content since streaming is disabled when tools are present.
        
        Args:
            memory (AgentMemory): Memory object to record the assistant response
            response (Generator[types.GenerateContentResponse, None, None]): The streaming response
            
        Yields:
            str: Chunks of text content from the model response
        """
        return self._stream_response(memory=memory, response=response, content_extractor=self._extract_stream_content)

    def _build_final_response(self, memory: AgentMemory, response: types.GenerateContentResponse) -> tuple[str, list]:
        """Processes a complete (non-streaming) response and updates memory.

        Args:
            memory (AgentMemory): Memory object to record the assistant response
            response (types.GenerateContentResponse): The HTTP response object containing the full response.

        Returns:
            tuple[str, list]: A tuple containing (content, tool_calls) where content is the model's output 
            and tool_calls is a list of tool calls (or None if no tools were called).
        """
        content = response.text if hasattr(response, "text") else None
        tool_calls = None

        if hasattr(response, "function_calls") and response.function_calls:
            tool_calls = []
            for call in response.function_calls:
                tool_calls.append(
                    model.ToolCall(
                        id=call.id,
                        type="function",
                        function=model.Function(name=call.name, arguments=call.args),
                    )
                )

        if tool_calls:
            return (content, tool_calls)

        self._record_assistant_message(memory, content)
        return (content, None)

    def _build_contents(self, conversation: list[dict]) -> list[types.Content]:
        """Convert OpenAI-style conversation messages into Google GenAI content objects."""
        contents = []
        for message in conversation:
            if message.get("role") == "system":
                continue
            contents.append(
                types.Content(
                    role=message.get("role"),
                    parts=[types.Part(text=message.get("content", ""))],
                )
            )
        return contents