# pylint: disable=C0301
""" Module wrap an agent """
import logging
import json
from typing import Callable, Generator, Any
import uuid
from pydantic import BaseModel, Field, ValidationError
from requests import RequestException

from linden.provider.anthropic import AnthropicClient
from ..memory.agent_memory import AgentMemory
from ..provider.ai_client import Provider
from ..utils.doc_string_parser import parse_google_docstring
from ..provider.groq import GroqClient
from .model import ToolCall, ToolError, ToolNotFound
from ..provider.ollama import Ollama
from ..provider.openai import OpenAiClient

logger = logging.getLogger(__name__)

class AgentConfiguration(BaseModel):
    """
    Configuration class for an AI agent.
    This class defines the configuration parameters for an AI agent including
    the language model to use, prompt settings, tools, and other operational
    parameters.
    Attributes:
        user_id (str): Unique identifier for the user
        name (str): Name of the agent, defaults to a random UUID4 string
        model (str): LLM to use to run the agent
        temperature (float): Temperature for the LLM, between 0 and 1
        system_prompt (str): System prompt for the agent as a simple text
        tools (list[Callable[..., any]]): List of callable tools available to the agent, defaults to empty list
        output_type (BaseModel | None): Optional Pydantic model for the output format of the agent
        client (Provider): AI provider to use for this agent, defaults to Provider.OLLAMA
        retries (int): Number of retry attempts for failed requests, defaults to 3
    """
    
    user_id: str = Field(..., description="Unique identifier for the user")
    name: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Name of the agent")
    model: str = Field(..., description="LLM to use to run the agent")
    temperature: float = Field(ge=0, le=1, description="Temperature for the LLM, between 0 and 1")
    system_prompt: str = Field(..., description="System prompt for the agent as a simple text")
    tools: list[Callable[..., any]] = Field(default_factory=list, description="List of callable tools available to the agent")
    output_type: Any = Field(default=None, description="Pydantic model for the output format of the agent")
    client: Provider = Field(default=Provider.OLLAMA, description="AI provider to use for this agent", enum=Provider)
    retries: int = Field(default=3, description="Number of retry attempts for failed requests")
    
    model_config = {
        "extra": "forbid"  # Reject any parameters not defined in the model
    }


class AgentRunner:
    """ Define Agent features """
    def __init__(self, config: AgentConfiguration | None = None, **kwargs):
        """
        Initialize an AgentRunner either from an AgentConfiguration instance
        or from keyword arguments.

        Args:
            config: Optional AgentConfiguration instance. If provided, kwargs are ignored.
            **kwargs: Configuration parameters that will be used to create
                    an AgentConfiguration instance.
        
        Raises:
            ValidationError: If any parameter is invalid or not recognized
                            by the AgentConfiguration model.
        """
        # If config is provided, use it; otherwise create from kwargs
        if config is None:
            config = AgentConfiguration(**kwargs)  # ValidationError for invalid params
        elif kwargs:
            logger.warning("Both config and kwargs provided - kwargs will be ignored")
        
        # initialize fields from configuration
        self.user_id = config.user_id
        self.name = config.name
        self.model = config.model
        self.temperature = config.temperature
        self.tools = config.tools  # already defaulted to [] in AgentConfiguration
        self.retries = config.retries
        self.output_type = config.output_type

        # conversation history and tool descriptions
        self.history = []
        self.tool_desc = self._parse_tools(provider=config.client)

        # system prompt and client initialization
        self._set_system_prompt(config.system_prompt)
        self._set_client(client=config.client)

        # memory manager
        self.memory = AgentMemory(agent_id=self.name, user_id=self.user_id, system_prompt=self.system_prompt, history=self.history)

        logger.info("Init agent %s", self.name)

    def ask_to_llm(self, prompt: str, stream: bool = False, output_format: BaseModel = None) -> Generator[str, None, None] | tuple[str, list]:
        """Query the LLM client with the given input
        
        Args:
            prompt (str): The input text or prompt
            stream (bool, optional): Whether to stream the response. Defaults to False.
            output_format (BaseModel, optional): Optional Pydantic model for response validation. Defaults to None.
            
        Returns:
            Generator[str, None, None] | tuple[str, list]: Either a generator of text chunks (if stream=True)
            or a tuple of (content, tool_calls) where content is the model's output and 
            tool_calls is a list of tool calls (or None) (if stream=False).
        """
        return self.client.query_llm(prompt=prompt, memory=self.memory, stream=stream, output_format=output_format)

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
            stream = False # stream must be set to False if there are tools set
        u_input = user_question
        for turn in range(0, self.retries+1):
            logger.info("Turn %d", turn)
            self.memory.record({"role": "user", "content": u_input})
            turn+=1
            try:
                data = self.ask_to_llm(prompt=u_input, stream=stream, output_format=self.output_type)
                if stream is True:
                    # return the stream generator
                    return data
                logger.debug("Agent %s response: %s", self.name, data)
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
                else:
                    err = exc.message
                logger.warning("Error during agent execution: %s", err)
                u_input = err

    def reset(self):
        """
        Reset the agent's conversation history.
        
        Clears all messages from memory except for the system prompt,
        effectively starting a new conversation while keeping the agent's configuration.
        """
        logger.info("Agent %s history reset", self.name)
        self.memory.reset()

    def tool_call(self, tool_actions: list[ToolCall]):
        """
        Execute tool calls requested by the LLM.
        
        Matches tool calls from the LLM with available tools and executes them with
        the provided arguments, handling various formats of tool arguments.
        
        Args:
            tool_actions: List of tool calls to execute
            
        Returns:
            The result of the executed tool
            
        Raises:
            ToolNotFound: If no matching tool is found for an action
            ToolError: If there's an error executing the tool
        """
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
        """
        Add content to the agent's memory context.
        
        Args:
            content: The content to add to memory
            persist: Whether to persist the content to long-term memory
        """
        self.memory.record(content, persist)

    # -------- UTILITIES
    def _parse_doc_string(self, doc_string: str):
        """
        Parse the docstring of a tool method into a dictionary.
        
        Args:
            doc_string: The docstring to parse
            
        Returns:
            dict: A dictionary representation of the docstring
        """
        doc_string_dict = {}
        if doc_string != "":
            doc_string_dict = json.loads(doc_string.strip().replace('\n',''))
        return doc_string_dict

    def _set_system_prompt(self, system_prompt: str) :
        """
        Configure the system prompt, including output schema if required.
        
        If an output_type is specified, appends the JSON schema to the prompt to guide the model.
        
        Args:
            system_prompt: The base system prompt
        """
        if self.output_type:
            system_prompt = f"{system_prompt}.\nThe JSON object must use the schema: {json.dumps(self.output_type.model_json_schema(), indent=2)}"
        self.system_prompt = {"role": "system", "content": system_prompt if system_prompt is not None else ""}

    def _parse_tools(self, provider: Provider):
        """
        Parse the docstrings of available tools into a format suitable for LLM function calling.
        
        Extracts function descriptions, parameters, and other metadata from tool docstrings
        using the Google docstring format.
        
        Returns:
            list: A list of tool descriptions in the format expected by LLM providers
        """
        if self.tools and len(self.tools) > 0:
            tool_desc = []
            for tool in self.tools:
                doc_string = parse_google_docstring(
                    docstring=tool.__doc__,
                    func_name=tool.__name__,
                    include_returns= False if provider == Provider.ANTHROPIC else True,
                    provider=provider)
                if provider == Provider.ANTHROPIC:
                    tool_desc.append(doc_string)
                else:
                    tool_desc.append({"type":"function", "function":doc_string})
            return tool_desc

    def _set_client(self, client: Provider):
        """
        Initialize the appropriate AI client based on the specified provider.
        
        Creates an instance of GroqClient, OpenAiClient, or Ollama based on the provider enum,
        with the agent's model, temperature, and tool descriptions.
        
        Args:
            client: The provider enum indicating which client to use
        """
        match client:
            case Provider.GROQ:
                self.client = GroqClient(model=self.model, temperature=self.temperature, tools=self.tool_desc)
            case Provider.OPENAI:
                self.client = OpenAiClient(model=self.model, temperature=self.temperature, tools=self.tool_desc)
            case Provider.ANTHROPIC:
                self.client = AnthropicClient(model=self.model, temperature=self.temperature, tools=self.tool_desc)
            case _:
                self.client = Ollama(model=self.model, temperature=self.temperature, tools=self.tool_desc)
