import logging
from typing import Generator
from ollama import ChatResponse, Options, GenerateResponse, Client
from pydantic import BaseModel, TypeAdapter
from ..core import model
from ..memory.agent_memory import AgentMemory
from ..core.ai_client import AiClient
from ..config.configuration import ConfigManager

logger = logging.getLogger(__name__)

class Ollama(AiClient):
    def __init__(self, model:str, temperature: float,  tools = None):
        self.model = model
        self.temperature = temperature
        self.tools = tools
        ollama_config = ConfigManager.get().ollama
        self.client = Client(timeout=ollama_config.timeout)
        super().__init__()
    
    def query_llm(self, input: str, memory: AgentMemory, stream: bool = False, format: BaseModel = None) -> Generator[str, None, None] | tuple[str, list]:
        """ Interaction with local LLM by official Ollama client

        Args:
            input (str): The user input or prompt
            history (list[object]): The conversation history
            stream (bool, optional): Whether to stream the response. Defaults to False.
            format (BaseModel, optional): Optional Pydantic model for response validation. Defaults to None.

        Returns:
            Generator[str, None, None] | tuple[str, list]: Either a generator of text chunks (if stream=True)
            or a tuple of (content, tool_calls) where content is the model's output and 
            tool_calls is a list of tool calls (or None) (if stream=False).
        """
        try:

            conversation = memory.get_conversation(user_input=input)
            
            response: ChatResponse = self.client.chat(
                model=self.model,
                options=Options(temperature=self.temperature),
                messages=conversation,
                stream=stream,
                tools=self.tools,
                format=format.model_json_schema() if format and not stream else None)

            if not stream:
                return self._build_final_response(memory=memory, response=response)
            else:
                return self._generate_stream(memory=memory, response=response)
        except Exception as e:
            logger.error(f"Error in query_llm: {str(e)}")
            raise

    def _generate_stream(self, memory: AgentMemory, response: ChatResponse) -> Generator[str, None, None]:
        """ Handles a streaming response by yielding content and updating memory.

            Args:
                memory (AgentMemory): Memory object to record the assistant response
                response (ChatResponse): The streaming response generator

            Yields:
                str: Chunks of text content from the model response
        """
        def stream_generator():
            full_response = []
            tool_calls = None
            try:
                for chunk in response:
                    if hasattr(chunk, 'message') and chunk.message:
                        content = chunk.message.content if hasattr(chunk.message, 'content') else str(chunk.message)
                        # tool_calls can be None or a list
                        if hasattr(chunk.message, 'tool_calls') and chunk.message.tool_calls:
                            tool_calls = chunk.message.tool_calls
                    else:
                        continue
                    if content:
                        full_response.append(content)
                        yield content
            except Exception as e:
                logger.error(f"Error in stream_generator: {str(e)}")
                raise
            finally:
                # After streaming, record in persistent memory only if there are no tool_calls
                # (tool results are saved by AgentRunner)
                if full_response:
                    complete_content = "".join(full_response)
                    history_entry = {"role": "assistant", "content": complete_content}
                    if tool_calls:
                        history_entry["tool_calls"] = tool_calls
                    
                    if not tool_calls:
                        memory.record(history_entry)
        return stream_generator()
    
    def _build_final_response(self, memory: AgentMemory, response: ChatResponse|GenerateResponse) -> tuple[str, list]:
        """ Processes a complete (non-streaming) response and updates memory.

            Args:
                memory (AgentMemory): Memory object to record the assistant response
                response (ChatResponse | GenerateResponse): The HTTP response object containing the full response.

            Returns:
                tuple[str, list]: A tuple containing (content, tool_calls) where content is the model's output 
                and tool_calls is a list of tool calls (or None if no tools were called).
        """
        # Extract content based on response type
        if not hasattr(response, 'message') and hasattr(response, 'response'):
            content = response.response
            tool_calls = None
        else:
            content = response.message.content
            # Extract tool calls if present
            tool_calls = getattr(response.message, 'tool_calls', None)
            
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

