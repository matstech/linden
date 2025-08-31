# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0303
from unittest.mock import patch
from linden.provider import OpenAiClient, GroqClient, Ollama
from linden.provider.ai_client import AiClient


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
    
    def test_client_interchangeability(self, mock_memory, mock_openai_client, mock_groq_client, mock_ollama_client):
        """Test that clients can be used interchangeably."""
        # Get the client instances from the fixtures
        openai_client, _ = mock_openai_client
        groq_client, _ = mock_groq_client
        ollama_client, _ = mock_ollama_client
        
        # Set up responses for each client
        with patch.object(openai_client, 'query_llm', return_value=("OpenAI response", None)), \
             patch.object(groq_client, 'query_llm', return_value=("Groq response", None)), \
             patch.object(ollama_client, 'query_llm', return_value=("Ollama response", None)):
            
            # Create a function that accepts any AiClient and uses it
            def process_with_any_client(client: AiClient, input_text: str):
                result, _ = client.query_llm(prompt=input_text, memory=mock_memory)
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

    def test_provider_api_consistency(self, mock_openai_client, mock_groq_client, mock_ollama_client):
        """Test that all providers implement the same API pattern."""
        # Get the client instances from the fixtures
        openai_client, _ = mock_openai_client
        groq_client, _ = mock_groq_client
        ollama_client, _ = mock_ollama_client
        
        # Test tools parameter is properly stored
        tools = [{"type": "function", "function": {"name": "test_fn", "description": "test"}}]
        
        # Mock the ConfigManager for these new instances
        with patch('linden.provider.openai.ConfigManager'), \
             patch('linden.provider.groq.ConfigManager'), \
             patch('linden.provider.ollama.ConfigManager'):
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
