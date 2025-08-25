"""
Tests for the Ollama client.
"""

import pytest
from unittest.mock import MagicMock, patch
import json
from linden.provider.ollama import Ollama
from linden.core.model import ToolCall, Function
from pydantic import BaseModel


class TestOllamaClient:
    def test_init(self, mock_config_manager):
        """Test client initialization."""
        client = Ollama(model="llama3", temperature=0.7)
        assert client.model == "llama3"
        assert client.temperature == 0.7
        assert client.tools is None
    
    def test_init_with_tools(self, mock_config_manager):
        """Test client initialization with tools."""
        tools = [{"type": "function", "function": {"name": "test_fn", "description": "A test function"}}]
        client = Ollama(model="llama3", temperature=0.7, tools=tools)
        assert client.tools == tools

    def test_query_llm_non_streaming(self, mock_ollama_client, mock_memory, mock_ollama_response):
        """Test querying the LLM in non-streaming mode."""
        client, mock_instance = mock_ollama_client
        
        # Setup response
        mock_instance.chat.return_value = mock_ollama_response
        
        # Call method
        content, tool_calls = client.query_llm(input="Hello", memory=mock_memory, stream=False)
        
        # Verify
        mock_memory.get_conversation.assert_called_once_with(user_input="Hello")
        mock_instance.chat.assert_called_once()
        assert content == "Test response"
        assert tool_calls is None
        mock_memory.record.assert_called_once()
    
    def test_query_llm_with_format(self, mock_ollama_client, mock_memory, mock_ollama_response):
        """Test querying the LLM with a format."""
        client, mock_instance = mock_ollama_client
        
        # Setup response
        mock_instance.chat.return_value = mock_ollama_response
        
        # Define a Pydantic model for format validation
        class TestFormat(BaseModel):
            result: str
            
        # Expected schema
        expected_schema = TestFormat.model_json_schema()
        
        # Call method
        client.query_llm(input="Hello", memory=mock_memory, format=TestFormat)
        
        # Verify format was passed
        call_kwargs = mock_instance.chat.call_args.kwargs
        assert call_kwargs["format"] == expected_schema

    def test_query_llm_with_tools(self, mock_ollama_client, mock_memory, mock_ollama_response):
        """Test querying the LLM with tools."""
        client, mock_instance = mock_ollama_client
        
        # Add tools
        tools = [{"type": "function", "function": {"name": "test_fn", "description": "A test function"}}]
        client.tools = tools
        
        # Setup response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "test_fn"
        mock_tool_call.function.arguments = '{"param": "value"}'
        mock_tool_call.model_dump = MagicMock(return_value={
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "test_fn",
                "arguments": '{"param": "value"}'
            }
        })
        
        mock_ollama_response.message.tool_calls = [mock_tool_call]
        mock_instance.chat.return_value = mock_ollama_response
        
        # Call method
        content, tool_calls = client.query_llm(input="Hello", memory=mock_memory)
        
        # Verify
        assert content == "Test response"
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].function.name == "test_fn"
        # Memory should not be recorded when there are tool calls
        assert not mock_memory.record.called

    def test_streaming_response(self, mock_ollama_client, mock_memory):
        """Test streaming response handling."""
        client, mock_instance = mock_ollama_client
        
        # Patch the stream_generator function
        with patch.object(client, '_generate_stream') as mock_generate_stream:
            # Create a simple generator that yields content
            def fake_generator():
                yield "Hello"
                yield " world"
                # Record what should have been recorded
                mock_memory.record({"role": "assistant", "content": "Hello world"})
                
            # Set the return value
            mock_generate_stream.return_value = fake_generator()
            
            # Call method
            stream_gen = client.query_llm(input="Tell me a greeting", memory=mock_memory, stream=True)
            
            # Verify streaming content
            result = []
            for chunk in stream_gen:
                result.append(chunk)
                
            assert result == ["Hello", " world"]
            
            # Verify our fake generator was called
            mock_generate_stream.assert_called_once()
            
            # Memory.record should have been called by our fake generator
            mock_memory.record.assert_called_once_with({"role": "assistant", "content": "Hello world"})

    def test_error_handling(self, mock_ollama_client, mock_memory):
        """Test error handling in the client."""
        client, mock_instance = mock_ollama_client
        
        # Setup to raise an exception
        mock_instance.chat.side_effect = Exception("API Error")
        
        # Verify exception is raised
        with pytest.raises(Exception, match="API Error"):
            client.query_llm(input="Hello", memory=mock_memory)

    def test_build_final_response_generate_response(self, mock_ollama_client, mock_memory):
        """Test _build_final_response method with a GenerateResponse."""
        client, _ = mock_ollama_client
        
        # Create mock GenerateResponse
        mock_resp = MagicMock()
        mock_resp.response = "Generated text"
        
        # Ensure it doesn't have message attribute
        if hasattr(mock_resp, 'message'):
            delattr(mock_resp, 'message')
        
        content, tool_calls = client._build_final_response(memory=mock_memory, response=mock_resp)
        
        assert content == "Generated text"
        assert tool_calls is None
        mock_memory.record.assert_called_once()

    def test_streaming_with_tool_calls(self, mock_ollama_client, mock_memory):
        """Test streaming response with tool calls."""
        client, mock_instance = mock_ollama_client
        
        # Create mock chunk with tool calls
        mock_chunk = MagicMock()
        mock_chunk.message = MagicMock()
        mock_chunk.message.content = "Processing with tools"
        
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "test_fn"
        mock_tool_call.function.arguments = '{"param": "value"}'
        
        mock_chunk.message.tool_calls = [mock_tool_call]
        
        # Set up mock to return an iterable with one chunk
        mock_instance.chat.return_value = [mock_chunk]
        
        # Call method
        stream_gen = client.query_llm(input="Use tools", memory=mock_memory, stream=True)
        
        # Collect the streaming results
        result = []
        for chunk in stream_gen:
            result.append(chunk)
        
        assert result == ["Processing with tools"]
        
        # Tool calls should be included in history entry, but memory.record shouldn't be called
        # because we have tool calls
        assert not mock_memory.record.called
