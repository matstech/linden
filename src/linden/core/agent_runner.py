""" Module wrap an agent """
import logging
import json
from typing import Callable, Generator
from pydantic import BaseModel, ValidationError
from requests import RequestException
from ..memory.agent_memory import AgentMemory
from .ai_client import Provider
from ..utils.doc_string_parser import parse_google_docstring
from ..provider.groq import GroqClient
from .model import ToolCall, ToolError, ToolNotFound
from ..provider.ollama import Ollama
from ..provider.openai import OpenAiClient

PATTERN = r"%#(.*?)#%"

logger = logging.getLogger(__name__)

class AgentRunner:
    """ Define Agent features """
    def __init__(self, 
                 name: str,
                 model: str,
                 temperature: int,
                 system_prompt: str = None,
                 tools: list[Callable[[], None]] = None,
                 output_type:BaseModel = None,
                 client: Provider = Provider.OLLAMA,
                 retries=3):
        
        self.name = name
        self.model = model
        self.temperature = temperature
        self.tools = tools if tools is not None else []
        self.retries = retries
        self.output_type = output_type
        self.history = []
        self.tool_desc = self._parse_tools()
        self._set_system_prompt(system_prompt)
        self._set_client(client=client)
        self.memory = AgentMemory(agent_id=self.name, system_prompt=self.system_prompt, history=self.history)
        
        logger.info("Init agent %s", name)

    def ask_to_llm(self, input: str, stream: bool = False, format: BaseModel = None) -> Generator[str, None, None] | tuple[str, list]:
        """Query the LLM client with the given input
        
        Args:
            input (str): The input text or prompt
            stream (bool, optional): Whether to stream the response. Defaults to False.
            format (BaseModel, optional): Optional Pydantic model for response validation. Defaults to None.
            
        Returns:
            Generator[str, None, None] | tuple[str, list]: Either a generator of text chunks (if stream=True)
            or a tuple of (content, tool_calls) where content is the model's output and 
            tool_calls is a list of tool calls (or None) (if stream=False).
        """
        return self.client.query_llm(memory=self.memory, input=input,stream=stream, format=format)

    def run(self, user_question: str, stream: bool = False) -> Generator[str, None, None] | BaseModel | str | list:
        """ Execute agent query on LLM
        
        Args:
            user_question (str): The user's question or input to process
            stream (bool, optional): Whether to stream the response. Defaults to False.
            
        Returns:
            Generator[str, None, None] | BaseModel | str | list: One of the following:
            - A generator of text chunks (if stream=True)
            - A Pydantic model instance (if output_type is set and no tool is called)
            - A string with the raw output (if output_type is None and no tool is called)
            - A list or other return value from a tool (if a tool is called)
        """
        # ensure client has all the tools set
        if self.client.tools is not None and len(self.client.tools) is not len(self.tools):
            self.client.tools = self.tool_desc
        input = user_question
        for turn in range(0, self.retries+1):                
            logger.info("Turn %d", turn)
            self.memory.record({"role": "user", "content": input})
            turn+=1
            try:
                data = self.ask_to_llm(input=input, stream=stream, format=self.output_type)
                if stream is True:
                    # return the stream generator
                    return data
                logger.debug(f"Agent {self.name} response: {data}")
                if data[0] is not None and (data[1] is None or len(data[1]) == 0):
                        if self.output_type is not None:
                            # return output_type obj or raise ValidationError exception
                            return self.output_type.model_validate_json(data[0]) 
                        else:
                            # return raw output
                            return data[0] 
                else:
                    # return tool output directly
                    return self.tool_call(tool_actions=data[1]) 
            except (ValueError, json.JSONDecodeError, RequestException, ToolError, ToolNotFound) as exc:
                if isinstance(exc, RequestException):
                    err = f"error in calling model: {exc.args}"
                elif isinstance(exc, (ValueError,json.JSONDecodeError, ValidationError)):
                    err = f"invalid response from model: {exc}"
                elif isinstance(exc, ToolError):
                    err = f"{exc.message} {exc.tool_name} with input {exc.tool_input}"
                    # excluding error tool from the list of available tool
                    temp_tools = []
                    for tool in self.tool_desc:
                        if tool['function']['name'] != exc.tool_name:
                            temp_tools.append(tool)
                    self.client.tools = temp_tools
                elif isinstance(exc, ToolNotFound):
                    err = exc.message
                logger.warning("Error during agent execution: %s", err)
                input = err

    def reset(self):
        """ Clean the history of the agent leaving just the first message in memory (system prompt)"""
        logger.info("Agent %s history reset", self.name)
        self.memory.reset()

    def tool_call(self, tool_actions: list[ToolCall]):
        """ Tool call router """
        if len(self.tools) == 0:
            raise ToolNotFound("no tool found for the agent")
        try:
            for action in tool_actions:
                for tool in self.tools:
                    if getattr(tool, '__name__', 'Unknown') == action.function.name:
                        args = action.function.arguments
                        if isinstance(args, str):
                            args = json.loads(args)
                        elif isinstance(args, dict) and "params" in args and isinstance(args["params"], dict):
                            merged_args = {k: v for k, v in args.items() if k != "params"}
                            merged_args.update(args["params"])
                            return tool(merged_args)
                        return tool(**args)
            raise ToolNotFound("no tool found to execute specified actions")
        except Exception as exc:
            raise ToolError(message="invalid tool call", 
                            tool_name=action.function.name,
                            tool_input=action.function.arguments) from exc

    def add_to_context(self, content: str, persist: bool = False):
        self.memory.record(content, persist)
    
    # -------- UTILITIES
    def _parse_doc_string(self, doc_string: str):
        """ Parse doc_string of method tool"""
        doc_string_dict = {}
        if doc_string != "": 
            doc_string_dict = json.loads(doc_string.strip().replace('\n',''))
        return doc_string_dict
    
    def _set_system_prompt(self, system_prompt: str) :
        if self.output_type:
            system_prompt = f"{system_prompt}.\nThe JSON object must use the schema: {json.dumps(self.output_type.model_json_schema(), indent=2)}"
        self.system_prompt = {"role": "system", "content": system_prompt if system_prompt is not None else ""}
        
        
    def _parse_tools(self):
        if self.tools and len(self.tools) > 0:
            tool_desc = []
            for tool in self.tools:
                doc_string = parse_google_docstring(docstring=tool.__doc__, func_name=tool.__name__)
                tool_desc.append({"type":"function", "function":doc_string}) 
            return tool_desc
        
    def _set_client(self, client: Provider):
        match client:
            case Provider.GROQ:
                self.client = GroqClient(model=self.model, temperature=self.temperature, tools=self.tool_desc)    
            case Provider.OPENAI:
                self.client = OpenAiClient(model=self.model, temperature=self.temperature, tools=self.tool_desc)
            case _:
                self.client = Ollama(model=self.model, temperature=self.temperature, tools=self.tool_desc)

