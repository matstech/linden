"""
Tests for the agent_runner module which contains the AgentRunner class.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call
import json
import inspect
from typing import Dict, Any, List, Callable

from linden.core.agent_runner import AgentRunner
from linden.core.model import ToolCall, Function, ToolCalls, ToolError, ToolNotFound
from linden.core.ai_client import AiClient, Provider
from linden.memory.agent_memory import AgentMemory


class TestToolFunction:
    """Class for testing tool functions."""
    
    def __init__(self, return_value=None, is_async=False):
        self.return_value = return_value
        self.is_async = is_async
        self.call_count = 0
        self.call_args = []
    
    async def async_call(self, *args, **kwargs):
        self.call_count += 1
        self.call_args.append((args, kwargs))
        return self.return_value
    
    def sync_call(self, *args, **kwargs):
        self.call_count += 1
        self.call_args.append((args, kwargs))
        return self.return_value
    
    def __call__(self, *args, **kwargs):
        if self.is_async:
            return self.async_call(*args, **kwargs)
        return self.sync_call(*args, **kwargs)


@pytest.mark.asyncio
async def test_agent_runner_initialization(mock_ai_client):
    """Test AgentRunner initialization."""
    # Create a sample tool
    def sample_tool(arg1, arg2):
        """A sample tool for testing.
        
        Args:
            arg1: First argument
            arg2: Second argument
            
        Returns:
            str: The result string
        """
        return f"Result: {arg1}, {arg2}"
    
    # Create an agent runner
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[sample_tool]
    )
    
    # Verify initialization
    assert agent.name == "test_agent"
    assert agent.model == "gpt-4"
    assert agent.temperature == 0.7
    assert sample_tool in agent.tools
    assert isinstance(agent.memory, AgentMemory)


@pytest.mark.asyncio
async def test_agent_runner_with_memory(mock_ai_client, mock_agent_memory):
    """Test AgentRunner with pre-initialized memory."""
    # Create an agent runner with default memory
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[]
    )
    
    # Verify that memory is initialized with correct agent_id and system_prompt
    assert agent.memory.agent_id == "test_agent"
    assert "You are a test assistant." in agent.memory.system_prompt["content"]


@pytest.mark.asyncio
async def test_agent_runner_add_to_context(mock_agent_runner):
    """Test add_to_context method."""
    # Add a message to the context
    mock_agent_runner.add_to_context({"role": "user", "content": "Hello agent"})
    
    # Verify the memory was updated
    mock_agent_runner.memory.record.assert_called_once()
    
    # Reset the mock for the next call
    mock_agent_runner.memory.record.reset_mock()
    
    # Add another message with persistence
    mock_agent_runner.add_to_context({"role": "assistant", "content": "Hello user"}, persist=True)
    
    # Verify the memory was updated with persistence flag
    mock_agent_runner.memory.record.assert_called_once_with(
        {"role": "assistant", "content": "Hello user"}, True
    )


@pytest.mark.asyncio
async def test_agent_runner_tool_call_sync():
    """Test tool_call method with sync tool."""
    # Create a sample tool
    def test_tool(param1, param2):
        """A test tool.
        
        Args:
            param1: First parameter
            param2: Second parameter
            
        Returns:
            str: The result
        """
        return f"Result: {param1}, {param2}"
    
    # Create an agent runner with the tool
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[test_tool]
    )
    
    # Create tool calls list
    function = Function(name="test_tool", arguments={"param1": "value1", "param2": 42})
    tool_call = ToolCall(id="call_123", function=function)
    tool_calls = [tool_call]
    
    # Call the tool
    result = agent.tool_call(tool_actions=tool_calls)
    
    # Verify the result
    assert result == "Result: value1, 42"


@pytest.mark.asyncio
async def test_agent_runner_tool_call_async():
    """Test tool_call method with async tool."""
    # Create a sample async tool
    async def async_tool(param1, param2):
        """An async test tool.
        
        Args:
            param1: First parameter
            param2: Second parameter
            
        Returns:
            str: The async result
        """
        return f"Async Result: {param1}, {param2}"
    
    # Create an agent runner with the async tool
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[async_tool]
    )
    
    # Create tool calls list
    function = Function(name="async_tool", arguments={"param1": "async_value", "param2": 99})
    tool_call = ToolCall(id="call_456", function=function)
    tool_calls = [tool_call]
    
    # Call the tool
    result = await agent.tool_call(tool_actions=tool_calls)
    
    # Verify the result
    assert result == "Async Result: async_value, 99"


@pytest.mark.asyncio
async def test_agent_runner_tool_call_not_found():
    """Test tool_call method with non-existent tool."""
    # Create an agent runner with no tools
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[]
    )
    
    # Create a tool call for a non-existent tool
    function = Function(name="non_existent_tool", arguments={"param": "value"})
    tool_call = ToolCall(id="call_789", function=function)
    tool_calls = [tool_call]
    
    # Call the tool and expect ToolNotFound error
    with pytest.raises(ToolNotFound) as exc_info:
        await agent.tool_call(tool_actions=tool_calls)
    
    # Verify error message
    assert "no tool found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_runner_tool_call_error():
    """Test tool_call method with tool that raises an error."""
    # Create a tool that raises an error
    def error_tool(param):
        """A tool that raises an error.
        
        Args:
            param: A parameter
            
        Raises:
            ValueError: Always raises this error
        """
        raise ValueError("Tool execution error")
    
    # Create an agent runner with the error tool
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[error_tool]
    )
    
    # Create a tool call
    function = Function(
        name="error_tool",
        arguments={"param": "value"}
    )
    
    tool_call = ToolCall(
        id="call_error",
        function=function
    )
    
    # Create tool calls list
    tool_calls = [tool_call]
    
    # Execute the tool and expect ToolError
    with pytest.raises(ToolError) as exc_info:
        agent.tool_call(tool_actions=tool_calls)
    
    # Verify error details
    assert exc_info.value.tool_name == "error_tool" 
    assert exc_info.value.message == "invalid tool call"


@pytest.mark.asyncio
async def test_agent_runner_run():
    """Test run method."""
    # Create a test tool
    def test_tool(param):
        """A test tool.
        
        Args:
            param: Input parameter
            
        Returns:
            str: The result
        """
        return f"Tool result: {param}"
    
    # Create a mock AI client
    mock_client = MagicMock()
    mock_client.query_llm.return_value = ("Response content", [
        ToolCall(id="call_123", function=Function(name="test_tool", arguments={"param": "test_value"}))
    ])
    
    # Patch the client creation
    with patch('linden.core.agent_runner.Ollama', return_value=mock_client):
        # Create agent
        agent = AgentRunner(
            name="test_agent",
            model="gpt-4",
            temperature=0.7,
            system_prompt="You are a test assistant.",
            tools=[test_tool]
        )
        
        # Run the agent
        response = agent.run(user_question="Run the test tool")
        
        # Verify that query_llm was called
        mock_client.query_llm.assert_called_once()
        
        # Verify the response is the tool result
        assert response == "Tool result: test_value"


@pytest.mark.asyncio
async def test_agent_runner_run_no_tools_used():
    """Test run method when no tools are used."""
    # Create a mock AI client with no tool calls
    mock_client = MagicMock()
    mock_client.query_llm.return_value = ("No tools needed response", None)
    
    # Patch the client creation
    with patch('linden.core.agent_runner.Ollama', return_value=mock_client):
        # Create agent
        agent = AgentRunner(
            name="test_agent",
            model="gpt-4",
            temperature=0.7,
            system_prompt="You are a test assistant.",
            tools=[]
        )
        
        # Run the agent
        response = agent.run(user_question="Just respond without tools")
        
        # Verify the response
        assert response == "No tools needed response"


def test_agent_runner_with_system_prompt():
    """Test AgentRunner with custom system prompt."""
    # Create an agent with a custom system prompt
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a custom test assistant.",
        tools=[]
    )
    
    # Verify the system prompt was set correctly
    assert agent.system_prompt["role"] == "system"
    assert "You are a custom test assistant." in agent.system_prompt["content"]
    



@pytest.mark.asyncio
async def test_agent_runner_multiple_tool_calls():
    """Test tool_call method with multiple tool calls."""
    # Create test tools
    def tool1(param):
        """First test tool.
        
        Args:
            param: A parameter
            
        Returns:
            str: The result
        """
        return f"Tool 1 result: {param}"
    
    def tool2(param):
        """Second test tool.
        
        Args:
            param: A parameter
            
        Returns:
            str: The result
        """
        return f"Tool 2 result: {param}"
    
    # Create an agent with both tools
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[tool1, tool2]
    )
    
    # Create tool calls
    function1 = Function(name="tool1", arguments={"param": "value1"})
    tool_call1 = ToolCall(id="call_1", function=function1)
    function2 = Function(name="tool2", arguments={"param": "value2"})
    tool_call2 = ToolCall(id="call_2", function=function2)
    
    # Test tool_call with multiple tool calls
    result1 = agent.tool_call(tool_actions=[tool_call1])
    assert result1 == "Tool 1 result: value1"
    
    result2 = agent.tool_call(tool_actions=[tool_call2])
    assert result2 == "Tool 2 result: value2"


def test_agent_runner_parse_tools():
    """Test _parse_tools method."""
    # Create test tools with docstrings
    def tool1(param1: str, param2: int) -> str:
        """
        Test tool 1.
        
        Args:
            param1 (str): First parameter
            param2 (int): Second parameter
            
        Returns:
            str: The result
        """
        return f"Result: {param1}, {param2}"
    
    async def tool2(param: dict) -> dict:
        """Test tool 2.
        
        Args:
            param (dict): Parameter dictionary
            
        Returns:
            dict: Result dictionary
        """
        return {"result": param}
    
    # Create agent with tools
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[tool1, tool2]
    )
    
    # Verify that tool_desc was created correctly
    tool_desc = agent.tool_desc
    assert len(tool_desc) == 2
    
    # Each tool should be a dictionary with type and function
    for tool in tool_desc:
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool


def test_agent_runner_reset():
    """Test reset method."""
    # Mock the memory
    mock_memory = MagicMock(spec=AgentMemory)
    
    # Create a mock AI client
    mock_client = MagicMock()
    
    # Patch the client creation and memory creation
    with patch('linden.core.agent_runner.AgentMemory', return_value=mock_memory), \
         patch('linden.core.agent_runner.Ollama', return_value=mock_client):
        
        # Create agent
        agent = AgentRunner(
            name="test_agent",
            model="gpt-4",
            temperature=0.7,
            system_prompt="You are a test assistant.",
            tools=[]
        )
        
        # Call reset
        agent.reset()
        
        # Verify that memory's reset method was called
        mock_memory.reset.assert_called_once()
    



def test_agent_runner_with_output_type():
    """Test AgentRunner with output_type parameter."""
    from pydantic import BaseModel
    
    # Define a custom output model
    class TestOutputModel(BaseModel):
        result: str
        confidence: float
    
    # Create agent with output_type
    agent = AgentRunner(
        name="test_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant.",
        tools=[],
        output_type=TestOutputModel
    )
    
    # Verify the system prompt has been updated to include the schema
    assert "schema" in agent.system_prompt["content"].lower()
    assert "result" in agent.system_prompt["content"]
    assert "confidence" in agent.system_prompt["content"]
