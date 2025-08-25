"""
Tests for the ai_client module which contains the AiClient class and Provider enum.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json

from linden.core.ai_client import AiClient, Provider, AgentKind
from linden.memory.agent_memory import AgentMemory


def test_provider_enum():
    """Test the Provider enum values."""
    # Test all enum values
    assert isinstance(Provider.OPENAI.value, int)
    assert isinstance(Provider.GROQ.value, int)
    assert isinstance(Provider.OLLAMA.value, int)
    assert Provider.OPENAI.value != Provider.GROQ.value
    assert Provider.GROQ.value != Provider.OLLAMA.value
    assert Provider.OPENAI.value != Provider.OLLAMA.value
    
    # Test enum conversion to string
    assert str(Provider.OPENAI) == "Provider.OPENAI"
    assert str(Provider.GROQ) == "Provider.GROQ"
    assert str(Provider.OLLAMA) == "Provider.OLLAMA"


def test_agent_kind_enum():
    """Test the AgentKind enum values."""
    # Test enum values
    assert isinstance(AgentKind.GENERATE.value, int)
    assert isinstance(AgentKind.CHAT.value, int)
    assert AgentKind.GENERATE.value != AgentKind.CHAT.value
    
    # Test enum conversion to string
    assert str(AgentKind.GENERATE) == "AgentKind.GENERATE"
    assert str(AgentKind.CHAT) == "AgentKind.CHAT"


def test_ai_client_is_abstract():
    """Test that AiClient is an abstract class."""
    with pytest.raises(TypeError):
        client = AiClient()  # Should raise TypeError because it's an abstract class


class TestImplementation(AiClient):
    """Test implementation of the abstract AiClient class."""
    
    def query_llm(self, input, memory, stream=False, format=None):
        """Test implementation of the query_llm method."""
        return f"Response to: {input}", None


def test_ai_client_implementation():
    """Test a concrete implementation of AiClient."""
    # Create a concrete implementation
    client = TestImplementation()
    
    # Test the query_llm method
    memory = MagicMock(spec=AgentMemory)
    response, _ = client.query_llm("Test input", memory)
    
    assert response == "Response to: Test input"


def test_query_llm_with_stream():
    """Test query_llm method with stream=True."""
    # Create a concrete implementation
    client = TestImplementation()
    
    # Test the query_llm method with stream=True
    memory = MagicMock(spec=AgentMemory)
    response, _ = client.query_llm("Test input", memory, stream=True)
    
    assert response == "Response to: Test input"


def test_query_llm_with_format():
    """Test query_llm method with a format specified."""
    # Create a custom implementation that respects format
    class FormattedClient(AiClient):
        def query_llm(self, input, memory, stream=False, format=None):
            if format:
                return format(result="Formatted response", confidence=0.9), None
            return f"Unformatted: {input}", None
    
    from pydantic import BaseModel
    
    class TestFormat(BaseModel):
        result: str
        confidence: float
    
    client = FormattedClient()
    memory = MagicMock(spec=AgentMemory)
    
    # Test with format
    response, _ = client.query_llm("Test input", memory, format=TestFormat)
    
    assert isinstance(response, TestFormat)
    assert response.result == "Formatted response"
    assert response.confidence == 0.9


def test_multiple_query_calls():
    """Test multiple calls to query_llm."""
    client = TestImplementation()
    memory = MagicMock(spec=AgentMemory)
    
    # Make multiple calls to query_llm
    responses = []
    for i in range(3):
        response, _ = client.query_llm(f"Input {i}", memory)
        responses.append(response)
    
    # Verify responses
    assert responses[0] == "Response to: Input 0"
    assert responses[1] == "Response to: Input 1"
    assert responses[2] == "Response to: Input 2"


def test_subclasses_requirements():
    """Test that subclasses must implement required methods."""
    # Define a subclass that doesn't implement query_llm
    class IncompleteClient(AiClient):
        pass
    
    # Trying to instantiate it should raise TypeError
    with pytest.raises(TypeError):
        client = IncompleteClient()


def test_custom_agent_kind():
    """Test a client with a specified agent kind."""
    class KindAwareClient(AiClient):
        def __init__(self, kind=AgentKind.GENERATE):
            self.kind = kind
            
        def query_llm(self, input, memory, stream=False, format=None):
            if self.kind == AgentKind.GENERATE:
                return f"Generated: {input}", None
            elif self.kind == AgentKind.CHAT:
                return f"Chat: {input}", None
            return f"Unknown kind: {input}", None
    
    # Test with different kinds
    generate_client = KindAwareClient(kind=AgentKind.GENERATE)
    chat_client = KindAwareClient(kind=AgentKind.CHAT)
    
    memory = MagicMock(spec=AgentMemory)
    
    generate_response, _ = generate_client.query_llm("Test", memory)
    chat_response, _ = chat_client.query_llm("Test", memory)
    
    assert generate_response == "Generated: Test"
    assert chat_response == "Chat: Test"
