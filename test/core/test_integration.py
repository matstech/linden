# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0303
from unittest.mock import MagicMock

from linden.core import AgentRunner, AgentConfiguration
from linden.core.model import ToolCall, Function
from linden.memory.agent_memory import AgentMemory
from linden.provider.ai_client import Provider


def test_full_conversation_flow():
    """Test a full conversation flow with multiple interactions."""
    # Define tool functions
    def weather_tool(location: str):
        """Get the weather for a location."""
        return f"The weather in {location} is sunny and 75°F"
    
    def calculator(operation: str, a: float, b: float):
        """Perform a mathematical operation."""
        if operation == "multiply":
            return a * b
        return None
    
    # Create a mock client
    mock_client = MagicMock()
    mock_client.tools = None
    
    # Mock the query_llm method to handle different queries
    mock_client.query_llm = MagicMock(side_effect=[
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
    
    # Create an agent runner with tools using AgentConfiguration
    config = AgentConfiguration(
        user_id="test_user",
        name="test_agent",
        model="llama3",  # Use a model that works with Ollama for testing
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        tools=[weather_tool, calculator],
        client=Provider.OLLAMA
    )
    agent = AgentRunner(config=config)
    
    # Replace the agent's client with our mock
    agent.client = mock_client
    
    # First interaction
    first_response = agent.run(user_question="What's the weather in New York?")
    assert "weather" in str(first_response).lower() and "New York" in str(first_response)
    
    # Second interaction
    second_response = agent.run(user_question="What is 5 times 7?")
    assert second_response == "The result is 35"
    
    # Verify our mock was called twice
    assert mock_client.query_llm.call_count == 2


def test_integration_with_error_recovery():
    """Test integration with error recovery in tools."""
    # Define tools
    def working_tool(param: str):
        """A tool that works properly."""
        return f"Successfully processed {param}"
    
    def failing_tool(param: str):
        """A tool that always raises an error."""
        raise ValueError(f"Error processing {param}")
    
    # Create mock client
    mock_client = MagicMock()
    mock_client.tools = None
    mock_client.query_llm = MagicMock(return_value=("Tool executed successfully", None))
    
    # Create an agent runner with both tools (legacy style)
    agent = AgentRunner(
        user_id="test_user",
        name="error_recovery_agent",
        model="llama3",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        tools=[working_tool, failing_tool],
        client=Provider.OLLAMA
    )
    
    # Replace with mock
    agent.client = mock_client
    
    # Run the agent
    response = agent.run(user_question="Try using the tools")
    
    # Verify response
    assert "Tool executed successfully" in response


def test_provider_integration():
    """Test integration with different providers."""
    # We'll test that the agent can be created with Ollama provider (least dependencies)
    
    # Create an agent with Ollama provider using AgentConfiguration
    config = AgentConfiguration(
        user_id="test_user",
        name="ollama_agent",
        model="llama3",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        tools=[],
        client=Provider.OLLAMA
    )
    
    # Create the agent
    ollama_agent = AgentRunner(config=config)
    
    # Test that the client is properly set
    assert ollama_agent is not None
    assert ollama_agent.name == "ollama_agent"
    assert ollama_agent.model == "llama3"


def test_memory_persistence_with_agent():
    """Test memory persistence through multiple agent sessions."""
    # Create a memory object
    memory = AgentMemory(agent_id="test_agent_memory", user_id="test_user", client=MagicMock(), config=MagicMock(), system_prompt="You are a helpful assistant.", history_max_messages=20)
    
    # Test basic memory functionality
    assert memory.agent_id == "test_agent_memory"
    assert memory.user_id == "test_user"
    assert memory.system_prompt == "You are a helpful assistant."
    
    # Create a mock client
    mock_client = MagicMock()
    mock_client.tools = None
    mock_client.query_llm = MagicMock(return_value=("I remember our previous conversation!", None))
    
    # Create agent with the existing memory using AgentConfiguration
    config = AgentConfiguration(
        user_id="test_user",
        name="test_agent_memory",
        model="llama3",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        client=Provider.OLLAMA
    )
    agent = AgentRunner(config=config)
    
    # Replace agent memory and client with our mocks
    agent.memory = memory
    agent.client = mock_client
    
    # Make a query
    response = agent.run(user_question="Do you remember what we talked about before?")
    
    # Verify the response
    assert "remember" in response
    assert mock_client.query_llm.call_count == 1