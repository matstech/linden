"""
Integration tests for the core module components.
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from linden.core.agent_runner import AgentRunner
from linden.core.ai_client import AiClient, Provider
from linden.core.model import ToolCall, Function
from linden.memory.agent_memory import AgentMemory
from linden.provider.openai import OpenAiClient


def test_full_conversation_flow():
    """Test a full conversation flow with multiple interactions."""
    # Mock the OpenAiClient creation
    with patch("linden.core.agent_runner.OpenAiClient") as mock_openai_class:
        # Setup the OpenAI mock
        mock_openai = MagicMock(spec=OpenAiClient)
        
        # Define tool functions
        def weather_tool(location: str):
            """Get the weather for a location."""
            return f"The weather in {location} is sunny and 75°F"
        
        def calculator(operation: str, a: float, b: float):
            """Perform a mathematical operation."""
            if operation == "multiply":
                return a * b
            return None
        
        # Create an agent runner with tools
        agent = AgentRunner(
            name="test_agent",
            model="gpt-4",
            temperature=0.7,
            system_prompt="You are a helpful assistant.",
            tools=[weather_tool, calculator],
            client=Provider.OPENAI
        )
        
        # Mock the query_llm method to handle different queries
        mock_openai.query_llm = MagicMock(side_effect=[
            # First call: Weather request with tool call
            ("", [
                ToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="weather_tool",
                        arguments='{"location": "New York"}'
                    )
                )
            ]),
            # Second call: Calculation with result
            ("The result is 35", None)
        ])
        
        # Replace the agent's client with our mock
        agent.client = mock_openai
        
        # First interaction
        first_response = agent.run(user_question="What's the weather in New York?")
        assert "weather" in str(first_response).lower() and "New York" in str(first_response)
        
        # Second interaction
        second_response = agent.run(user_question="What is 5 times 7?")
        assert second_response == "The result is 35"
        
        # Verify our mock was called twice
        assert mock_openai.query_llm.call_count == 2
        
        # Create memory
        memory = AgentMemory(messages=[])
        
        # Define tool functions
        def weather_tool(location: str):
            """Get the weather for a location.
            
            Args:
                location (str): The location to get weather for
                
            Returns:
                str: The weather forecast
            """
            return f"The weather in {location} is sunny and 75°F"
        
        def calculator(operation: str, a: float, b: float):
            """Perform a mathematical operation.
            
            Args:
                operation (str): One of 'add', 'subtract', 'multiply', 'divide'
                a (float): First number
                b (float): Second number
                
            Returns:
                float: The result of the operation
            """
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            elif operation == "multiply":
                return a * b
            elif operation == "divide":
                return a / b
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        # Create agent with tools
        agent = AgentRunner(
            client=ai_client,
            tools={
                "weather": weather_tool,
                "calculator": calculator
            },
            memory=memory
        )
        
        # Mock AI client responses for first interaction
        function1 = Function(name="weather", arguments={"location": "San Francisco"})
        mock_first_tool_call = ToolCall(
            id="call_1", 
            function=function1
        )
        ai_client.get_tool_calls = AsyncMock(
            side_effect=[
                ToolCalls(tool_calls=[mock_first_tool_call]),  # First call uses weather
                ToolCalls(tool_calls=[])  # Second call doesn't use tools
            ]
        )
        
        ai_client.get_completion = AsyncMock(
            return_value="The weather in San Francisco is sunny and 75°F. Is there anything else you'd like to know?"
        )
        
        # First user message
        first_response = await agent.run(
            input_text="What's the weather in San Francisco?",
            system="You are a helpful assistant with access to tools."
        )
        
        # Verify first response
        assert "sunny" in first_response
        assert "75°F" in first_response
        
        # Verify memory has 4 messages (system, user, tool, assistant)
        assert len(memory.messages) == 4
        assert memory.messages[0]["role"] == "system"
        assert memory.messages[1]["role"] == "user"
        assert "San Francisco" in memory.messages[1]["content"]
        assert memory.messages[2]["role"] == "tool"
        assert memory.messages[3]["role"] == "assistant"
        
        # Mock AI client responses for second interaction
        function2 = Function(
            name="calculator", 
            arguments={
                "operation": "multiply",
                "a": 5,
                "b": 7
            }
        )
        mock_second_tool_call = ToolCall(
            id="call_2", 
            function=function2
        )
        
        ai_client.get_tool_calls = AsyncMock(
            side_effect=[
                ToolCalls(tool_calls=[mock_second_tool_call]),  # Uses calculator
                ToolCalls(tool_calls=[])  # No more tools
            ]
        )
        
        ai_client.get_completion = AsyncMock(
            return_value="5 multiplied by 7 equals 35."
        )
        
        # Second user message
        second_response = await agent.run(input_text="What is 5 times 7?")
        
        # Verify second response
        assert "35" in second_response
        
        # Verify memory now has 7 messages (previous 4 + user, tool, assistant)
        assert len(memory.messages) == 7
        assert memory.messages[4]["role"] == "user"
        assert memory.messages[5]["role"] == "tool"
        assert memory.messages[6]["role"] == "assistant"


def test_integration_with_error_recovery():
    """Test integration with error recovery in tools."""
    # Mock the provider client
    with patch("linden.core.agent_runner.OpenAiClient") as mock_openai_class:
        # Setup the OpenAI mock
        mock_openai = MagicMock(spec=OpenAiClient)
        
        # Define tool functions
        def working_tool(param: str):
            """A tool that works correctly."""
            return f"Successfully processed {param}"
        
        def failing_tool(param: str):
            """A tool that always raises an error."""
            raise ValueError(f"Error processing {param}")
        
        # Create an agent runner with both tools
        agent = AgentRunner(
            name="error_recovery_agent",
            model="gpt-4",
            temperature=0.7,
            system_prompt="You are a helpful assistant.",
            tools=[working_tool, failing_tool],
            client=Provider.OPENAI
        )
        
        # Mock the query_llm method to simulate tool calls
        mock_openai.query_llm = MagicMock(side_effect=[
            # First call: Will try the failing tool
            ("", [
                ToolCall(
                    id="call_fail",
                    type="function",
                    function=Function(
                        name="failing_tool",
                        arguments='{"param": "test-data"}'
                    )
                )
            ]),
            # Second call: After error, tries the working tool
            ("", [
                ToolCall(
                    id="call_work",
                    type="function",
                    function=Function(
                        name="working_tool",
                        arguments='{"param": "recovery-data"}'
                    )
                )
            ]),
            # Final call: Returns a response
            ("Recovery successful", None)
        ])
        
        # Replace the agent's client with our mock
        agent.client = mock_openai
        
        # Run the agent
        response = agent.run(user_question="Try using the tools")
        
        # Verify the response
        assert "Successfully processed recovery-data" == response
        
        # Verify the mock was called 3 times (initial call with error, retry with working tool, final result)
        assert mock_openai.query_llm.call_count == 3
        mock_openai_class.return_value = mock_openai
        
        # Create the AI client
        ai_client = AiClient(provider=Provider.OPENAI, model="gpt-4")
        
        # Define tools with one that fails
        def failing_tool(param: str):
            """A tool that will fail.
            
            Args:
                param (str): A parameter
                
            Returns:
                str: Never returns successfully
            """
            raise ValueError("This tool always fails")
        
        def backup_tool(param: str):
            """A backup tool that works.
            
            Args:
                param (str): A parameter
                
            Returns:
                str: Success message
            """
            return f"Backup tool succeeded with {param}"
        
        # Create agent with tools
        agent = AgentRunner(
            client=ai_client,
            tools={
                "failing_tool": failing_tool,
                "backup_tool": backup_tool
            },
            memory=None
        )
        
        # Mock AI client responses
        # First it will try the failing tool
        function1 = Function(
            name="failing_tool", 
            arguments={"param": "test_value"}
        )
        mock_first_tool_call = ToolCall(
            id="call_1", 
            function=function1
        )
        
        # After failure, it will try the backup tool
        function2 = Function(
            name="backup_tool", 
            arguments={"param": "recovery_value"}
        )
        mock_second_tool_call = ToolCall(
            id="call_2", 
            function=function2
        )
        
        ai_client.get_tool_calls = AsyncMock(
            side_effect=[
                ToolCalls(tool_calls=[mock_first_tool_call]),
                ToolCalls(tool_calls=[mock_second_tool_call]),
                ToolCalls(tool_calls=[])
            ]
        )
        
        ai_client.get_completion = AsyncMock(
            return_value="The first tool failed, but the backup tool worked."
        )
        
        # Run the agent
        response = await agent.run(input_text="Try using the tools")
        
        # Verify response
        assert "backup tool worked" in response
        
        # Verify messages show error and recovery
        assert len(agent.messages) == 5  # user, assistant, tool (error), tool (success), assistant
        
        error_message = agent.messages[2]
        assert error_message["role"] == "tool"
        assert "error" in error_message["content"].lower()
        assert "fails" in error_message["content"].lower()
        
        success_message = agent.messages[3]
        assert success_message["role"] == "tool"
        assert "succeeded" in success_message["content"]
        assert "recovery_value" in success_message["content"]


def test_provider_integration():
    """Test integration with different providers."""
    # We'll test that the agent can be created with different providers
    
    # First create an agent with OpenAI provider
    openai_agent = AgentRunner(
        name="openai_agent",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        tools=[],
        client=Provider.OPENAI
    )
    
    # Test that the client is properly set
    assert openai_agent is not None
    
    # Create an agent with Groq provider
    groq_agent = AgentRunner(
        name="groq_agent",
        model="llama3-8b-8192",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        tools=[],
        client=Provider.GROQ
    )
    
    # Test that the client is properly set
    assert groq_agent is not None
    
    # Create an agent with Ollama provider
    ollama_agent = AgentRunner(
        name="ollama_agent",
        model="llama3",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        tools=[],
        client=Provider.OLLAMA
    )
    
    # Test that the client is properly set
    assert ollama_agent is not None


def test_memory_persistence_with_agent():
    """Test memory persistence through multiple agent sessions."""
    # Create a memory object
    memory = AgentMemory(agent_id="test_agent_memory", system_prompt="You are a helpful assistant.")
    
    # First, verify the memory starts with just a system prompt
    conversation = memory.get_conversation()
    assert len(conversation) == 1
    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"] == "You are a helpful assistant."
    
    # Add some messages
    memory.record({"role": "user", "content": "Hello, how are you?"})
    memory.record({"role": "assistant", "content": "I'm doing well, thanks for asking!"})
    
    # Verify messages were added
    conversation = memory.get_conversation()
    assert len(conversation) == 3
    assert conversation[1]["role"] == "user"
    assert conversation[2]["role"] == "assistant"
    
    # Create an agent with the existing memory
    with patch("linden.core.agent_runner.OpenAiClient") as mock_openai_class:
        # Setup a mock client
        mock_openai = MagicMock(spec=OpenAiClient)
        mock_openai.query_llm = MagicMock(return_value=("I remember our previous conversation!", None))
        mock_openai_class.return_value = mock_openai
        
        # Create agent with the existing memory
        agent = AgentRunner(
            name="test_agent_memory",
            model="gpt-4",
            temperature=0.7,
            system_prompt=None,  # Let it use the system prompt from memory
            tools=[],
            client=Provider.OPENAI
        )
        
        # Replace the agent's default memory with our existing memory
        agent.memory = memory
        
        # Test the agent
        response = agent.run(user_question="Do you remember our conversation?")
        
        # Verify response
        assert response == "I remember our previous conversation!"
        
        # Verify memory now has 5 messages (original 3 + new user + new assistant)
        conversation = memory.get_conversation()
        assert len(conversation) == 5
