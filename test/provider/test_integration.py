"""
Tests for provider interchangeability and integration.
"""

import pytest
from unittest.mock import MagicMock, patch
from linden.provider import OpenAiClient, GroqClient, Ollama
from linden.core.ai_client import AiClient


class TestProviderIntegration:
    def test_provider_interface_compliance(self, mock_config_manager):
        """Test that all providers comply with the AiClient interface."""
        # Initialize one instance of each provider
        openai_client = OpenAiClient(model="gpt-4", temperature=0.7)
        groq_client = GroqClient(model="llama3-8b-8192", temperature=0.7)
        ollama_client = Ollama(model="llama3", temperature=0.7)
        
        # Verify all clients are instances of AiClient
        assert isinstance(openai_client, AiClient)
        assert isinstance(groq_client, AiClient)
        assert isinstance(ollama_client, AiClient)
        
        # Verify all clients have the required methods
        for client in [openai_client, groq_client, ollama_client]:
            assert hasattr(client, 'query_llm')
    
    def test_client_interchangeability(self, mock_memory, mock_config_manager):
        """Test that clients can be used interchangeably."""
        # Create mocks for the client API calls
        with patch('linden.provider.openai.OpenAI') as mock_openai_api, \
             patch('linden.provider.groq.Groq') as mock_groq_api, \
             patch('linden.provider.ollama.Client') as mock_ollama_api:
            
            # Set up common mock behavior for each API
            mock_openai_instance = mock_openai_api.return_value
            mock_groq_instance = mock_groq_api.return_value
            mock_ollama_instance = mock_ollama_api.return_value
            
            # Common mock response structure
            def create_api_mock_response(text="API response"):
                mock_resp = MagicMock()
                mock_choice = MagicMock()
                mock_message = MagicMock()
                mock_message.content = text
                mock_message.tool_calls = None
                mock_choice.message = mock_message
                mock_resp.choices = [mock_choice]
                return mock_resp
            
            # Set up OpenAI mock
            mock_openai_instance.chat.completions.create.return_value = create_api_mock_response("OpenAI response")
            
            # Set up Groq mock
            mock_groq_instance.chat.completions.create.return_value = create_api_mock_response("Groq response")
            
            # Set up Ollama mock
            mock_ollama_instance.chat.return_value = MagicMock(message=MagicMock(content="Ollama response"))
            
            # Initialize clients
            openai_client = OpenAiClient(model="gpt-4", temperature=0.7)
            groq_client = GroqClient(model="llama3-8b-8192", temperature=0.7)
            ollama_client = Ollama(model="llama3", temperature=0.7)
            
            # Create a function that accepts any AiClient and uses it
            def process_with_any_client(client: AiClient, input_text: str):
                result, _ = client.query_llm(input=input_text, memory=mock_memory)
                return result
            
            # Test that the function works with all client types
            assert process_with_any_client(openai_client, "Hello") == "OpenAI response"
            assert process_with_any_client(groq_client, "Hello") == "Groq response"
            assert process_with_any_client(ollama_client, "Hello") == "Ollama response"

    def test_client_inheritance(self):
        """Test that provider clients inherit correctly from AiClient."""
        assert issubclass(OpenAiClient, AiClient)
        assert issubclass(GroqClient, AiClient)
        assert issubclass(Ollama, AiClient)

    def test_provider_api_consistency(self, mock_memory, mock_config_manager):
        """Test that all providers implement the same API pattern."""
        # Create mocks for the client API calls
        with patch('linden.provider.openai.OpenAI') as mock_openai_api, \
             patch('linden.provider.groq.Groq') as mock_groq_api, \
             patch('linden.provider.ollama.Client') as mock_ollama_api:
            
            # Initialize clients
            openai_client = OpenAiClient(model="gpt-4", temperature=0.7)
            groq_client = GroqClient(model="llama3-8b-8192", temperature=0.7)
            ollama_client = Ollama(model="llama3", temperature=0.7)
            
            # Test tools parameter is properly stored
            tools = [{"type": "function", "function": {"name": "test_fn", "description": "test"}}]
            
            openai_with_tools = OpenAiClient(model="gpt-4", temperature=0.7, tools=tools)
            groq_with_tools = GroqClient(model="llama3-8b-8192", temperature=0.7, tools=tools)
            ollama_with_tools = Ollama(model="llama3", temperature=0.7, tools=tools)
            
            assert openai_with_tools.tools == tools
            assert groq_with_tools.tools == tools
            assert ollama_with_tools.tools == tools
            
            # Test each client exposes the same constructor parameters
            assert hasattr(openai_client, 'model')
            assert hasattr(openai_client, 'temperature')
            assert hasattr(openai_client, 'tools')
            
            assert hasattr(groq_client, 'model')
            assert hasattr(groq_client, 'temperature')
            assert hasattr(groq_client, 'tools')
            
            assert hasattr(ollama_client, 'model')
            assert hasattr(ollama_client, 'temperature')
            assert hasattr(ollama_client, 'tools')
