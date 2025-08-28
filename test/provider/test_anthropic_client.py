# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0303
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
import pytest
from linden.provider.anthropic import AnthropicClient


@pytest.fixture
def mock_anthropic_config_manager():
    """Mock for the ConfigManager specific to Anthropic."""
    with patch('linden.provider.anthropic.ConfigManager') as mock_cm:
        mock_anthropic_config = MagicMock()
        mock_anthropic_config.timeout = 30
        mock_anthropic_config.api_key = "test-api-key"
        mock_anthropic_config.max_tokens = 4096
        mock_cm.get.return_value.anthropic = mock_anthropic_config
        yield mock_cm


@pytest.fixture
def mock_anthropic_client(mock_anthropic_config_manager):
    """Mock for the Anthropic client."""
    with patch('linden.provider.anthropic.Anthropic') as mock_anthropic:
        mock_instance = mock_anthropic.return_value
        mock_instance.messages.create = MagicMock()
        client = AnthropicClient(model="claude-3-sonnet-20240229", temperature=0.7)
        yield client, mock_instance


@pytest.fixture
def mock_memory():
    """Mock for agent memory."""
    memory = MagicMock()
    memory.get_conversation.return_value = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    memory.get_system_prompt.return_value = "You are a helpful assistant"
    return memory


@pytest.fixture
def mock_anthropic_response():
    """Mock for Anthropic response."""
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "Test response"
    mock_resp.content = [mock_content]
    return mock_resp


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock for Anthropic response with tool call."""
    mock_resp = MagicMock()
    mock_tool_content = MagicMock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.id = "call_123"
    mock_tool_content.name = "test_function"
    mock_tool_content.input = {"param1": "value1"}
    mock_resp.content = [mock_tool_content]
    return mock_resp


@pytest.fixture
def mock_anthropic_stream():
    """Mock for Anthropic streaming response."""
    def create_event(event_type, delta_text=None):
        event = MagicMock()
        event.type = event_type
        if event_type == 'content_block_delta' and delta_text:
            event.delta = MagicMock()
            event.delta.text = delta_text
        return event
    
    return [
        create_event('content_block_start'),
        create_event('content_block_delta', 'Hello'),
        create_event('content_block_delta', ' world'),
        create_event('content_block_stop')
    ]


class TestAnthropicClient:
    def test_init(self, mock_anthropic_config_manager):
        """Test client initialization."""
        client = AnthropicClient(model="claude-3-sonnet-20240229", temperature=0.7)
        assert client.model == "claude-3-sonnet-20240229"
        assert client.temperature == 0.7
        assert client.tools is None
    
    def test_init_with_tools(self, mock_anthropic_config_manager):
        """Test client initialization with tools."""
        tools = [{"type": "function", "function": {"name": "test_fn", "description": "A test function"}}]
        client = AnthropicClient(model="claude-3-sonnet-20240229", temperature=0.7, tools=tools)
        assert client.tools == tools

    def test_query_llm_non_streaming(self, mock_anthropic_client, mock_memory, mock_anthropic_response):
        """Test querying the LLM in non-streaming mode."""
        client, mock_instance = mock_anthropic_client
        
        # Setup response
        mock_instance.messages.create.return_value = mock_anthropic_response
        
        # Call method
        content, tool_calls = client.query_llm(prompt="Hello", memory=mock_memory, stream=False)
        
        # Verify
        mock_memory.get_conversation.assert_called_once_with(user_input="Hello")
        mock_instance.messages.create.assert_called_once()
        assert content == "Test response"
        assert tool_calls == []  # Returns empty list when no tool calls
        mock_memory.record.assert_called_once()
    
    def test_query_llm_with_format(self, mock_anthropic_client, mock_memory, mock_anthropic_response):
        """Test querying the LLM with a format."""
        client, mock_instance = mock_anthropic_client
        
        # Setup response
        mock_instance.messages.create.return_value = mock_anthropic_response
        
        # Define a Pydantic model for format validation
        class TestFormat(BaseModel):
            result: str
        
        # Call method
        client.query_llm(prompt="Hello", memory=mock_memory, output_format=TestFormat)
        
        # Verify the system prompt was modified to include format info
        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert "output format must be follow the structure" in call_kwargs["system"]

    def test_query_llm_with_tools(self, mock_anthropic_client, mock_memory, mock_anthropic_tool_response):
        """Test querying the LLM with tools."""
        client, mock_instance = mock_anthropic_client
        
        # Add tools
        tools = [{"type": "function", "function": {"name": "test_fn", "description": "A test function"}}]
        client.tools = tools
        
        # Setup response with tool calls
        mock_instance.messages.create.return_value = mock_anthropic_tool_response
        
        # Call method
        content, tool_calls = client.query_llm(prompt="Hello", memory=mock_memory)
        
        # Verify
        assert content is None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].function.name == "test_function"
        # Memory should not be recorded when there are tool calls
        assert not mock_memory.record.called

    def test_streaming_response(self, mock_anthropic_client, mock_memory, mock_anthropic_stream):
        """Test streaming response handling."""
        client, mock_instance = mock_anthropic_client
        
        # Set up mock to return an iterable for streaming
        mock_instance.messages.create.return_value = mock_anthropic_stream
        
        # Call method
        stream_gen = client.query_llm(prompt="Tell me a greeting", memory=mock_memory, stream=True)
        
        # Verify streaming content
        result = []
        for chunk in stream_gen:
            result.append(chunk)
            
        assert result == ["Hello", " world"]
        
        # After streaming completes, the complete content should be recorded
        mock_memory.record.assert_called_once_with({"role": "assistant", "content": "Hello world"})

    def test_error_handling(self, mock_anthropic_client, mock_memory):
        """Test error handling in the client."""
        client, mock_instance = mock_anthropic_client
        
        # Setup to raise an exception
        mock_instance.messages.create.side_effect = Exception("API Error")
        
        # Verify exception is raised
        with pytest.raises(Exception, match="API Error"):
            client.query_llm(prompt="Hello", memory=mock_memory)

    def test_build_final_response_without_content(self, mock_anthropic_client, mock_memory):
        """Test _build_final_response method when response has no content."""
        client, _ = mock_anthropic_client
        
        # Create mock response with no content attribute
        mock_resp = MagicMock()
        delattr(mock_resp, 'content')
        
        content, tool_calls = client._build_final_response(memory=mock_memory, response=mock_resp)
        
        assert content == ''
        assert tool_calls == []

    def test_build_final_response_empty_content(self, mock_anthropic_client, mock_memory):
        """Test _build_final_response method when response has empty content."""
        client, _ = mock_anthropic_client
        
        # Create mock response with empty content
        mock_resp = MagicMock()
        mock_resp.content = []
        
        content, tool_calls = client._build_final_response(memory=mock_memory, response=mock_resp)
        
        assert tool_calls == []

    def test_build_final_response_with_mixed_content(self, mock_anthropic_client, mock_memory):
        """Test _build_final_response method when response has both text and tool content."""
        client, _ = mock_anthropic_client
        
        # Create mock response with both text and tool content
        mock_resp = MagicMock()
        
        # Text content
        mock_text_content = MagicMock()
        mock_text_content.type = "text"
        mock_text_content.text = "Here's the result:"
        
        # Tool content
        mock_tool_content = MagicMock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.id = "call_456"
        mock_tool_content.name = "get_weather"
        mock_tool_content.input = {"location": "New York"}
        
        mock_resp.content = [mock_text_content, mock_tool_content]
        
        content, tool_calls = client._build_final_response(memory=mock_memory, response=mock_resp)
        
        # Should return the tool call, not the text content
        assert content is None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_456"
        assert tool_calls[0].function.name == "get_weather"

    def test_inject_output_format_info_with_format(self, mock_anthropic_client):
        """Test _inject_output_format_info method with a format."""
        client, _ = mock_anthropic_client
        
        # Define a Pydantic model
        class TestFormat(BaseModel):
            name: str
            age: int
        
        system_prompt = "You are a helpful assistant"
        result = client._inject_output_format_info(system_prompt, TestFormat)
        
        assert "You are a helpful assistant" in result
        assert "output format must be follow the structure" in result
        # The method calls model_json_schema() which returns the actual schema, not the string
        assert "'name'" in result  # Check that the schema content is present

    def test_inject_output_format_info_without_format(self, mock_anthropic_client):
        """Test _inject_output_format_info method without a format."""
        client, _ = mock_anthropic_client
        
        system_prompt = "You are a helpful assistant"
        result = client._inject_output_format_info(system_prompt, False)
        
        assert result == system_prompt

    def test_generate_stream_error_handling(self, mock_anthropic_client, mock_memory):
        """Test error handling in _generate_stream method."""
        client, _ = mock_anthropic_client
        
        # Create a mock stream that raises an exception
        def error_stream():
            mock_event = MagicMock()
            mock_event.type = 'content_block_delta'
            mock_event.delta = MagicMock()
            mock_event.delta.text = "Hello"
            yield mock_event
            raise Exception("Stream error")
        
        mock_response = error_stream()
        
        # Call the method and verify exception is raised
        with pytest.raises(Exception):
            stream_gen = client._generate_stream(memory=mock_memory, response=mock_response)
            list(stream_gen)  # Consume the generator to trigger the exception

    def test_streaming_with_no_text_content(self, mock_anthropic_client, mock_memory):
        """Test streaming response with events that have no text content."""
        client, mock_instance = mock_anthropic_client
        
        # Create events without text content  
        mock_event1 = MagicMock()
        mock_event1.type = 'content_block_start'
        
        mock_event2 = MagicMock()
        mock_event2.type = 'content_block_delta' 
        # Create a delta without text attribute by using spec
        mock_event2.delta = MagicMock(spec=[])  # Empty spec means no attributes
        
        mock_event3 = MagicMock()
        mock_event3.type = 'content_block_stop'
        
        mock_events = [mock_event1, mock_event2, mock_event3]
        
        mock_instance.messages.create.return_value = mock_events
        
        # Call method
        stream_gen = client.query_llm(prompt="Hello", memory=mock_memory, stream=True)
        
        # Consume the generator
        result = list(stream_gen)
        
        # Should be empty since no text content was yielded
        assert result == []
        
        # Memory should still be recorded with empty content since full_response will be empty
        mock_memory.record.assert_not_called()  # No recording should happen for empty response

    def test_tools_parameter_handling(self, mock_anthropic_client, mock_memory, mock_anthropic_response):
        """Test that tools parameter is handled correctly in API call."""
        client, mock_instance = mock_anthropic_client
        
        # Test with no tools
        client.tools = None
        mock_instance.messages.create.return_value = mock_anthropic_response
        
        client.query_llm(prompt="Hello", memory=mock_memory)
        
        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == []
        
        # Test with empty tools list
        client.tools = []
        mock_instance.messages.create.reset_mock()
        
        client.query_llm(prompt="Hello", memory=mock_memory)
        
        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == []
        
        # Test with actual tools
        tools = [{"type": "function", "function": {"name": "test_fn"}}]
        client.tools = tools
        mock_instance.messages.create.reset_mock()
        
        client.query_llm(prompt="Hello", memory=mock_memory)
        
        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == tools