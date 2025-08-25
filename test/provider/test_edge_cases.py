"""
Tests for edge cases and error handling in provider clients.
"""

import pytest
from unittest.mock import MagicMock, patch
from linden.provider import OpenAiClient, GroqClient, Ollama
from linden.core.model import ToolCall, Function
from pydantic import BaseModel, ValidationError


class TestProviderEdgeCases:
    def test_empty_response_openai(self, mock_openai_client, mock_memory):
        """Test handling of empty responses for OpenAI."""
        client, mock_instance = mock_openai_client
        
        # Mock an empty response
        mock_response = MagicMock()
        # No choices attribute
        if hasattr(mock_response, 'choices'):
            delattr(mock_response, 'choices')
        
        mock_instance.chat.completions.create.return_value = mock_response
        
        # Should handle gracefully
        content, tool_calls = client.query_llm(input="Hello", memory=mock_memory)
        assert content == ''
        assert tool_calls is None

    def test_empty_response_groq(self, mock_groq_client, mock_memory):
        """Test handling of empty responses for Groq."""
        client, mock_instance = mock_groq_client
        
        # Mock an empty response
        mock_response = MagicMock()
        # No choices attribute
        if hasattr(mock_response, 'choices'):
            delattr(mock_response, 'choices')
        
        mock_instance.chat.completions.create.return_value = mock_response
        
        # Should handle gracefully
        content, tool_calls = client.query_llm(input="Hello", memory=mock_memory)
        assert content == ''
        assert tool_calls is None

    def test_empty_response_ollama(self, mock_ollama_client, mock_memory):
        """Test handling of empty responses for Ollama."""
        client, mock_instance = mock_ollama_client
        
        # Mock an empty response without message attribute
        mock_response = MagicMock()
        if hasattr(mock_response, 'message'):
            delattr(mock_response, 'message')
            
        # Set the response attribute to empty string explicitly
        mock_response.response = ""
            
        mock_instance.chat.return_value = mock_response
        
        # Should handle gracefully
        content, tool_calls = client.query_llm(input="Hello", memory=mock_memory)
        # For ollama, content comes from the response attribute directly in this case
        assert content == ""
        assert tool_calls is None

    def test_malformed_tool_calls(self, mock_openai_client, mock_memory):
        """Test handling of malformed tool calls."""
        client, mock_instance = mock_openai_client
        
        # Create a response with malformed tool calls
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Test response"
        
        # Tool call missing required fields
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        # Missing function.name
        mock_tool_call.function.arguments = '{"param": "value"}'
        mock_tool_call.model_dump = MagicMock(return_value={
            "id": "call_123",
            "type": "function",
            "function": {
                # Name is missing
                "arguments": '{"param": "value"}'
            }
        })
        
        mock_message.tool_calls = [mock_tool_call]
        mock_choice.message = mock_message
        mock_resp.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_resp
        
        # Should raise a validation error when processing tool calls
        with pytest.raises(Exception):
            client.query_llm(input="Hello", memory=mock_memory)

    def test_invalid_json_in_tool_arguments(self, mock_groq_client, mock_memory):
        """Test handling of invalid JSON in tool arguments."""
        client, mock_instance = mock_groq_client
        
        # Create a response with invalid JSON in tool arguments
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Test response"
        
        # Create a tool call with invalid JSON that will cause a JSON decode error
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "test_fn"
        mock_tool_call.function.arguments = '{"param": invalid json}'  # Invalid JSON
        
        # Model_dump should cause problems with validation
        def model_dump_raising():
            # This will cause a validation error later
            return {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_fn", 
                    "arguments": '{"param": invalid json}'  # Invalid JSON
                }
            }
        
        # Mock the model_dump method to return a problematic dict
        mock_tool_call.model_dump = model_dump_raising
        
        # Add tool calls to the message
        mock_message.tool_calls = [mock_tool_call]
        mock_choice.message = mock_message
        mock_resp.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_resp
        
        # Patch TypeAdapter to raise an exception when validating
        with patch('linden.provider.groq.TypeAdapter') as mock_adapter:
            mock_adapter_instance = MagicMock()
            mock_adapter.return_value = mock_adapter_instance
            # Make validate_python raise an exception
            mock_adapter_instance.validate_python.side_effect = Exception("Invalid JSON")
            
            # Now the test should pass as we expect an exception
            with pytest.raises(Exception):
                client.query_llm(input="Hello", memory=mock_memory)

    def test_streaming_with_errors(self, mock_ollama_client, mock_memory):
        """Test streaming with errors mid-stream."""
        client, mock_instance = mock_ollama_client
        
        # Create a generator that will raise an exception mid-stream
        def error_generator():
            yield MagicMock(message=MagicMock(content="Start of response"))
            raise Exception("Stream error")
            yield MagicMock(message=MagicMock(content=" end of response"))
        
        mock_instance.chat.return_value = error_generator()
        
        # Call method
        stream_gen = client.query_llm(input="Tell me a greeting", memory=mock_memory, stream=True)
        
        # Should raise the exception
        with pytest.raises(Exception, match="Stream error"):
            list(stream_gen)

    def test_streaming_empty_chunks(self, mock_openai_client, mock_memory):
        """Test streaming with empty chunks."""
        client, mock_instance = mock_openai_client
        
        # Create a mock streaming response with empty chunks
        mock_empty_chunk = MagicMock()
        mock_empty_chunk.choices = [MagicMock()]
        mock_empty_chunk.choices[0].delta.content = None  # Empty content
        
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Valid content"
        
        # Set up mock to return an iterable with one empty chunk and one valid chunk
        mock_instance.chat.completions.create.return_value = [mock_empty_chunk, mock_chunk]
        
        # Call method
        stream_gen = client.query_llm(input="Tell me something", memory=mock_memory, stream=True)
        
        # Should only yield the valid content
        result = []
        for chunk in stream_gen:
            result.append(chunk)
            
        assert result == ["Valid content"]
        mock_memory.record.assert_called_once_with({"role": "assistant", "content": "Valid content"})


class TestFormatValidation:
    """Tests for format validation in responses."""

    class TestFormat(BaseModel):
        """Test pydantic model for validation."""
        result: str
        count: int

    def test_openai_format_validation(self, mock_openai_client, mock_memory):
        """Test format validation for OpenAI."""
        client, mock_instance = mock_openai_client
        
        # Mock a response with valid JSON for the format
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"result": "test", "count": 42}'
        mock_choice.message = mock_message
        mock_resp.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_resp
        
        # Call with format parameter
        content, _ = client.query_llm(input="Give me a result", memory=mock_memory, format=self.TestFormat)
        
        # Verify the format parameter was passed
        call_kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        # Content should be JSON string
        assert content == '{"result": "test", "count": 42}'

    def test_groq_format_validation(self, mock_groq_client, mock_memory):
        """Test format validation for Groq."""
        client, mock_instance = mock_groq_client
        
        # Mock a response with valid JSON for the format
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"result": "test", "count": 42}'
        mock_choice.message = mock_message
        mock_resp.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_resp
        
        # Call with format parameter
        content, _ = client.query_llm(input="Give me a result", memory=mock_memory, format=self.TestFormat)
        
        # Verify the format parameter was passed
        call_kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        # Content should be JSON string
        assert content == '{"result": "test", "count": 42}'

    def test_ollama_format_validation(self, mock_ollama_client, mock_memory):
        """Test format validation for Ollama."""
        client, mock_instance = mock_ollama_client
        
        # Mock a response with valid JSON for the format
        mock_resp = MagicMock()
        mock_resp.message = MagicMock()
        mock_resp.message.content = '{"result": "test", "count": 42}'
        mock_instance.chat.return_value = mock_resp
        
        # Call with format parameter
        content, _ = client.query_llm(input="Give me a result", memory=mock_memory, format=self.TestFormat)
        
        # Verify the format parameter was passed
        call_kwargs = mock_instance.chat.call_args.kwargs
        assert call_kwargs["format"] == self.TestFormat.model_json_schema()
        # Content should be JSON string
        assert content == '{"result": "test", "count": 42}'
