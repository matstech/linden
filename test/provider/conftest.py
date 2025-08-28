"""
Fixtures for provider tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from linden.provider import OpenAiClient, GroqClient, Ollama
from linden.provider.ai_client import AiClient
from linden.core.model import ToolCall, Function
from linden.memory.agent_memory import AgentMemory


@pytest.fixture
def mock_config_manager():
    """Mock for the ConfigManager."""
    with patch('linden.provider.openai.ConfigManager') as mock_openai_cm, \
         patch('linden.provider.groq.ConfigManager') as mock_groq_cm, \
         patch('linden.provider.ollama.ConfigManager') as mock_ollama_cm:
        
        # Mock for OpenAI config
        mock_openai_config = MagicMock()
        mock_openai_config.timeout = 30
        mock_openai_config.api_key = "test-api-key"
        mock_openai_cm.get.return_value.openai = mock_openai_config
        
        # Mock for Groq config
        mock_groq_config = MagicMock()
        mock_groq_config.timeout = 30
        mock_groq_config.api_key = "test-api-key"
        mock_groq_config.base_url = "https://api.groq.com/v1"
        mock_groq_cm.get.return_value.groq = mock_groq_config
        
        # Mock for Ollama config
        mock_ollama_config = MagicMock()
        mock_ollama_config.timeout = 30
        mock_ollama_cm.get.return_value.ollama = mock_ollama_config
        
        yield {
            "openai": mock_openai_cm,
            "groq": mock_groq_cm,
            "ollama": mock_ollama_cm
        }


@pytest.fixture
def mock_openai_client(mock_config_manager):
    """Mock for the OpenAI client."""
    with patch('linden.provider.openai.OpenAI') as mock_openai:
        mock_instance = mock_openai.return_value
        # Create a mock for chat completions
        mock_instance.chat.completions.create = MagicMock()
        client = OpenAiClient(model="gpt-4", temperature=0.7)
        yield client, mock_instance


@pytest.fixture
def mock_groq_client(mock_config_manager):
    """Mock for the Groq client."""
    with patch('linden.provider.groq.Groq') as mock_groq:
        mock_instance = mock_groq.return_value
        # Create a mock for chat completions
        mock_instance.chat.completions.create = MagicMock()
        client = GroqClient(model="llama3-8b-8192", temperature=0.7)
        yield client, mock_instance


@pytest.fixture
def mock_ollama_client(mock_config_manager):
    """Mock for the Ollama client."""
    with patch('linden.provider.ollama.Client') as mock_ollama:
        mock_instance = mock_ollama.return_value
        # Create a mock for chat
        mock_instance.chat = MagicMock()
        client = Ollama(model="llama3", temperature=0.7)
        yield client, mock_instance


@pytest.fixture
def mock_memory():
    """Mock for agent memory."""
    memory = MagicMock(spec=AgentMemory)
    memory.get_conversation.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    return memory


@pytest.fixture
def mock_openai_response():
    """Mock for OpenAI response."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_resp.choices = [mock_choice]
    return mock_resp


@pytest.fixture
def mock_groq_response():
    """Mock for Groq response."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_resp.choices = [mock_choice]
    return mock_resp


@pytest.fixture
def mock_ollama_response():
    """Mock for Ollama response."""
    mock_resp = MagicMock()
    mock_resp.message = MagicMock()
    mock_resp.message.content = "Test response"
    mock_resp.message.tool_calls = None
    return mock_resp


@pytest.fixture
def mock_tool_call():
    """Create a mock tool call."""
    return ToolCall(
        id="call_123",
        type="function",
        function=Function(
            name="test_function",
            arguments='{"param1": "value1"}'
        )
    )
