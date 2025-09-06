"""
Fixtures for core tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from linden.core import AgentRunner, AgentConfiguration
from linden.provider.ai_client import AiClient, Provider
from linden.core.model import ToolCall, Function, ToolError, ToolNotFound
from linden.memory.agent_memory import AgentMemory
from linden.config.configuration import ConfigManager

# Setup a mock for ConfigManager.get()
@pytest.fixture(autouse=True)
def mock_config():
    mock_config = MagicMock()
    mock_config.ollama.url = "http://localhost:11434"
    mock_config.openai.api_key = "test_key"
    mock_config.openai.timeout = 60
    mock_config.groq.api_key = "test_key"
    mock_config.groq.timeout = 60
    
    with patch.object(ConfigManager, 'get', return_value=mock_config):
        yield mock_config


class TestOutputModel(BaseModel):
    """Test output model for testing."""
    result: str
    confidence: float


@pytest.fixture
def mock_tool():
    """Create a mock tool function."""
    def sample_tool(param1: str, param2: int = 0) -> dict:
        """A sample tool function for testing.
        
        Args:
            param1 (str): A string parameter.
            param2 (int, optional): An integer parameter. Defaults to 0.
            
        Returns:
            dict: A result dictionary.
        """
        return {"result": f"Processed {param1} with value {param2}"}
    
    return sample_tool


@pytest.fixture
def mock_tool_with_error():
    """Create a mock tool that raises an error."""
    def error_tool(param: str) -> None:
        """A tool that always raises an error.
        
        Args:
            param (str): A parameter that will be ignored.
            
        Raises:
            ValueError: Always raises this error.
        """
        raise ValueError("Tool execution failed")
    
    return error_tool


@pytest.fixture
def mock_ai_client():
    """Create a mock AI client."""
    # Create a concrete implementation of AiClient for testing
    class TestAiClient(AiClient):
        def query_llm(self, input, memory, stream=False, format=None):
            return "Test response", None
    
    # Return an instance of the concrete implementation
    return TestAiClient()


@pytest.fixture
def mock_agent_memory():
    """Create a mock agent memory."""
    memory = MagicMock(spec=AgentMemory)
    memory.get_conversation.return_value = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ]
    
    return memory


@pytest.fixture
def mock_agent_runner(mock_ai_client):
    """Create a mock agent runner with dependencies patched."""
    # Patch the memory class
    mock_memory = MagicMock(spec=AgentMemory)
    
    with patch('linden.core.agent_runner.AgentMemory', return_value=mock_memory), \
         patch('linden.core.agent_runner.Ollama', return_value=mock_ai_client):
        
        # Create the agent runner with configuration
        config = AgentConfiguration(
            user_id="test_user",
            name="test_agent",
            model="test-model",
            temperature=0.7,
            system_prompt="You are a test assistant."
        )
        runner = AgentRunner(config=config)
        
        yield runner


@pytest.fixture
def mock_tool_call():
    """Create a mock tool call."""
    return ToolCall(
        id="call_123",
        type="function",
        function=Function(
            name="sample_tool",
            arguments='{"param1": "test", "param2": 42}'
        )
    )


@pytest.fixture
def mock_invalid_tool_call():
    """Create an invalid mock tool call."""
    return ToolCall(
        id="call_456",
        type="function",
        function=Function(
            name="nonexistent_tool",
            arguments='{"param": "value"}'
        )
    )
