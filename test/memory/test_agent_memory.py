# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0303
from unittest.mock import patch, MagicMock
from linden.memory.agent_memory import AgentMemory


class TestAgentMemory:
    """Test suite for the AgentMemory class."""
    

    def test_init_with_system_prompt(self, test_agent_id, test_system_prompt):
        """Test AgentMemory initialization with system prompt."""
        with patch('linden.memory.agent_memory.MemoryManager'):
            mock_client = MagicMock()
            mock_config = MagicMock()
            memory = AgentMemory(agent_id=test_agent_id, user_id="test", client=mock_client, config=mock_config, system_prompt=test_system_prompt)
            
            assert memory.agent_id == test_agent_id
            assert memory.system_prompt == test_system_prompt
            assert isinstance(memory.history, list)
            # History should contain the system prompt
            assert memory.history == [test_system_prompt]
    
    def test_init_with_history(self, test_agent_id, test_system_prompt):
        """Test AgentMemory initialization with history."""
        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        with patch('linden.memory.agent_memory.MemoryManager'):
            mock_client = MagicMock()
            mock_config = MagicMock()
            memory = AgentMemory(
                agent_id=test_agent_id,
                user_id="test",
                client=mock_client,
                config=mock_config,
                system_prompt=test_system_prompt,
                history=history
            )
            
            # Should override the history with system prompt
            assert memory.history == [test_system_prompt]
    
    def test_memory_property(self, agent_memory_with_mocked_manager, mock_mem0_memory):
        """Test memory property returns the Memory instance from the manager."""
        memory = agent_memory_with_mocked_manager
        assert memory.memory is mock_mem0_memory
    
    def test_record_dict_message(self, agent_memory_with_mocked_manager):
        """Test recording a message as dict."""
        memory = agent_memory_with_mocked_manager
        message = {"role": "user", "content": "Hello"}
        
        memory.record(message)
        
        # Check that message is added to history
        assert message in memory.history
    
    def test_record_string_message(self, agent_memory_with_mocked_manager):
        """Test recording a message as string."""
        memory = agent_memory_with_mocked_manager
        message = "Hello"
        
        memory.record(message)
        
        # Check that message is converted to dict and added to history
        expected_message = {"role": "user", "content": message}
        assert expected_message in memory.history
    
    def test_record_with_persist(self, agent_memory_with_mocked_manager, mock_mem0_memory):
        """Test recording a message with persist=True."""
        memory = agent_memory_with_mocked_manager
        message = "Hello"
        
        memory.record(message, persist=True)
        
        # Check that message is added to persistent memory
        mock_mem0_memory.add.assert_called_once()
        # User ID should be "mat"
        assert mock_mem0_memory.add.call_args[1]['user_id'] == "test"
    
    def test_record_with_persist_error(self, agent_memory_with_mocked_manager, mock_mem0_memory):
        """Test recording a message with persist=True when add fails."""
        memory = agent_memory_with_mocked_manager
        message = "Hello"
        mock_mem0_memory.add.side_effect = Exception("Test error")
        
        # Should not raise an exception
        memory.record(message, persist=True)
        
        # Check that message is still added to history
        expected_message = {"role": "user", "content": message}
        assert expected_message in memory.history
    
    def test_get_conversation_no_memories(self, agent_memory_with_mocked_manager, mock_mem0_memory):
        """Test get_conversation with no memories found."""
        memory = agent_memory_with_mocked_manager
        user_input = "What is the meaning of life?"
        
        mock_mem0_memory.search.return_value = {"results": []}
        
        # Add some history
        memory.record("Hello")
        memory.record({"role": "assistant", "content": "Hi there!"})
        
        result = memory.get_conversation(user_input)
        
        # Search should be called
        mock_mem0_memory.search.assert_called_once()
        assert mock_mem0_memory.search.call_args[1]['query'] == user_input
        assert mock_mem0_memory.search.call_args[1]['user_id'] == "test"
        
        # Since no memories were found, history should remain unchanged
        assert len(result) == 3  # System prompt + 2 messages
    
    def test_get_conversation_with_memories(self, agent_memory_with_mocked_manager, mock_mem0_memory):
        """Test get_conversation with memories found."""
        memory = agent_memory_with_mocked_manager
        user_input = "What is the meaning of life?"
        
        # Mock search to return some memories
        mock_mem0_memory.search.return_value = {
            "results": [
                {"memory": "The meaning of life is 42."},
                {"memory": "Life's meaning is subjective."}
            ]
        }
        
        result = memory.get_conversation(user_input)
        
        # Search should be called
        mock_mem0_memory.search.assert_called_once()
        
        # The last message should contain the memories
        assert len(result) == 2  # System prompt + new user message with memories
        assert "relevant context" in result[-1]["content"].lower()
        assert "meaning of life is 42" in result[-1]["content"]
        assert "subjective" in result[-1]["content"]
    
    def test_get_conversation_search_error(self, agent_memory_with_mocked_manager, mock_mem0_memory):
        """Test get_conversation when search fails."""
        memory = agent_memory_with_mocked_manager
        user_input = "What is the meaning of life?"
        
        # Mock search to raise an exception
        mock_mem0_memory.search.side_effect = Exception("Test error")
        
        # Add some history
        memory.record("Hello")
        memory.record({"role": "assistant", "content": "Hi there!"})
        
        result = memory.get_conversation(user_input)
        
        # Should return history without adding memories
        assert len(result) == 3  # System prompt + 2 messages
    
    def test_reset(self, agent_memory_with_mocked_manager, test_system_prompt):
        """Test reset clears history and sets system prompt."""
        memory = agent_memory_with_mocked_manager
        
        # Add some history
        memory.record("Hello")
        memory.record({"role": "assistant", "content": "Hi there!"})
        assert len(memory.history) > 1
        
        memory.reset()
        
        # History should only contain system prompt
        assert memory.history == [test_system_prompt]

    def test_get_conversation_with_summarization_above_threshold(self, agent_memory_with_mocked_manager, mock_mem0_memory):
        """Test summarization is triggered when memory fragments exceed the character threshold."""
        memory = agent_memory_with_mocked_manager
        user_input = "Tell me about my project."
        
        # Mock search to return long memory fragments to trigger summarization
        long_fragment = "This is a very long memory fragment designed to exceed the default character threshold for summarization. " * 50
        mock_mem0_memory.search.return_value = {
            "results": [
                {"memory": "Project 'Phoenix' is about rebranding."},
                {"memory": long_fragment}
            ]
        }
        
        # Mock the client's response for summarization
        summary_text = "Project 'Phoenix' is a rebranding effort and has a long description."
        memory.client.query_llm.return_value = (summary_text, None)
        
        # Call the method under test
        conversation = memory.get_conversation(user_input)
        
        # Verify that the client's query_llm was called for summarization
        memory.client.query_llm.assert_called_once()
        
        # Verify that the final conversation contains the summarized context
        final_prompt = conversation[-1]['content']
        assert summary_text in final_prompt
        assert long_fragment not in final_prompt

    def test_get_conversation_without_summarization_below_threshold(self, agent_memory_with_mocked_manager, mock_mem0_memory):
        """Test summarization is skipped when memory fragments are below the character threshold."""
        memory = agent_memory_with_mocked_manager
        user_input = "Tell me about my project."
        
        # Mock search to return short memories that won't trigger summarization
        short_fragment_1 = "Project 'Phoenix' is about rebranding."
        short_fragment_2 = "The deadline is Q4."
        mock_mem0_memory.search.return_value = {
            "results": [
                {"memory": short_fragment_1},
                {"memory": short_fragment_2}
            ]
        }
        
        # Call the method under test
        conversation = memory.get_conversation(user_input)
        
        # Verify that the client's query_llm was NOT called
        memory.client.query_llm.assert_not_called()
        
        # Verify that the final conversation contains the raw, formatted memories
        final_prompt = conversation[-1]['content']
        assert f"- {short_fragment_1}" in final_prompt
        assert f"- {short_fragment_2}" in final_prompt
