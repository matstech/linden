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
        self.tools = self._normalize_tools(tools)
        google_config = ConfigManager.get().google
        self.client = genai.Client(http_options=types.HttpOptions(timeout=google_config.timeout*1000), api_key=google_config.api_key)
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

    def _normalize_tools(self, tools):
        """Convert tool definitions to google.genai Tool objects."""
        if not tools:
            return None
        normalized = []
        for t in tools:
            if isinstance(t, types.Tool):
                normalized.append(t)
                continue
            if isinstance(t, dict):
                fdecls = t.get("function_declarations") or t.get("functionDeclarations")
                if fdecls:
                    fd_objs = []
                    for fd in fdecls:
                        if not isinstance(fd, dict):
                            continue
                        fd_objs.append(
                            types.FunctionDeclaration(
                                name=fd.get("name"),
                                description=fd.get("description"),
                                parameters=fd.get("parameters"),
                            )
                        )
                    normalized.append(types.Tool(function_declarations=fd_objs))
        return normalized if normalized else None

    def _extract_stream_content(self, chunk: types.GenerateContentResponse) -> str | None:
        """Extract text content from a streaming GenerateContentResponse chunk."""
        if hasattr(chunk, "candidates") and chunk.candidates:
            parts = getattr(chunk.candidates[0].content, "parts", [])
            return "".join(part.text for part in parts if hasattr(part, "text") and part.text)
        if hasattr(chunk, "parts") and chunk.parts:
            return "".join(part.text for part in chunk.parts if hasattr(part, "text") and part.text)
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
        content = None
        tool_calls = None

        if hasattr(response, "candidates") and response.candidates:
            parts = getattr(response.candidates[0].content, "parts", [])
            content = "".join(part.text for part in parts if hasattr(part, "text") and part.text) or None
        elif hasattr(response, "parts") and response.parts:
            content = "".join(part.text for part in response.parts if hasattr(part, "text") and part.text) or None

        allowed_tool_names = set()
        if self.tools:
            for tool in self.tools:
                if isinstance(tool, types.Tool) and getattr(tool, "function_declarations", None):
                    for declaration in tool.function_declarations:
                        name = getattr(declaration, "name", None)
                        if name:
                            allowed_tool_names.add(name)
                elif isinstance(tool, dict) and ("functionDeclarations" in tool or "function_declarations" in tool):
                    decls = tool.get("functionDeclarations") or tool.get("function_declarations")
                    for declaration in decls:
                        name = declaration.get("name") if isinstance(declaration, dict) else None
                        if name:
                            allowed_tool_names.add(name)

        disallowed_calls = []
        if hasattr(response, "function_calls") and response.function_calls:
            tool_calls = []
            for call in response.function_calls:
                if allowed_tool_names and call.name not in allowed_tool_names:
                    disallowed_calls.append(call)
                    continue
                tool_calls.append(
                    model.ToolCall(
                        id=call.id,
                        type="function",
                        function=model.Function(name=call.name, arguments=call.args),
                    )
                )
            if not tool_calls:
                tool_calls = None

        if tool_calls is None and disallowed_calls and content is None:
            # If the model tried to call an unknown tool but provided code in args, surface that code
            for call in disallowed_calls:
                args = getattr(call, "args", None)
                if isinstance(args, dict) and "code" in args and isinstance(args["code"], str):
                    content = args["code"]
                    break
            if content is None:
                content = "\n".join(
                    f"Unsupported tool call: {call.name} args: {call.args}" for call in disallowed_calls
                ) or None

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
            role = message.get("role")
            if role == "assistant":
                role = "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=message.get("content", ""))],
                )
            )
        return contents