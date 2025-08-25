import pytest
from unittest.mock import patch, MagicMock
import threading

from linden.memory.agent_memory import MemoryManager
from linden.config.configuration import ConfigManager


class TestMemoryManager:
    """Test suite for the MemoryManager singleton."""
    
    def setup_method(self):
        """Reset the MemoryManager singleton before each test."""
        MemoryManager._instance = None
    
    def teardown_method(self):
        """Reset the MemoryManager singleton after each test."""
        MemoryManager._instance = None
    
    def test_singleton_pattern(self):
        """Test that MemoryManager follows the singleton pattern."""
        manager1 = MemoryManager()
        manager2 = MemoryManager()
        
        # Both variables should reference the same instance
        assert manager1 is manager2
        assert id(manager1) == id(manager2)
    
    def test_thread_safe_creation(self):
        """Test that MemoryManager creation is thread-safe."""
        # This test verifies that multiple threads creating the singleton simultaneously
        # will all get the same instance back.
        
        results = []
        
        def create_manager():
            manager = MemoryManager()
            results.append(id(manager))
        
        # Create multiple threads that all try to instantiate the manager
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_manager)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All threads should have received the same instance (same ID)
        assert len(results) == 5
        assert len(set(results)) == 1
    
    @patch('linden.memory.agent_memory.Memory')
    @patch('linden.memory.agent_memory.ConfigManager')
    def test_create_memory(self, mock_config_manager, mock_memory_class):
        """Test that _create_memory correctly initializes a Memory instance."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.openai.api_key = "test-api-key"
        mock_config.memory.path = "/tmp/test-memory"
        mock_config_manager.get.return_value = mock_config
        
        # Setup Memory mock
        mock_memory_instance = MagicMock()
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        # Create memory manager and get memory
        manager = MemoryManager()
        memory = manager._create_memory()
        
        # Verify Memory.from_config was called with expected config
        mock_memory_class.from_config.assert_called_once()
        config_arg = mock_memory_class.from_config.call_args[0][0]
        
        # Check that config contains expected values
        assert config_arg["llm"]["config"]["api_key"] == "test-api-key"
        assert config_arg["embedder"]["config"]["api_key"] == "test-api-key"
        assert config_arg["vector_store"]["config"]["path"] == "/tmp/test-memory"
        
        # Verify the returned memory instance is our mock
        assert memory is mock_memory_instance
    
    def test_get_memory_creates_instance_once(self, mock_mem0_memory):
        """Test that get_memory creates the memory instance only once."""
        with patch('linden.memory.agent_memory.MemoryManager._create_memory', return_value=mock_mem0_memory) as mock_create:
            manager = MemoryManager()
            
            # First call should create memory
            memory1 = manager.get_memory()
            mock_create.assert_called_once()
            
            # Second call should reuse existing memory
            mock_create.reset_mock()
            memory2 = manager.get_memory()
            mock_create.assert_not_called()
            
            # Both calls should return the same instance
            assert memory1 is memory2
            assert memory1 is mock_mem0_memory
    
    def test_reset_memory(self, mock_mem0_memory):
        """Test that reset_memory clears the memory instance."""
        with patch('linden.memory.agent_memory.MemoryManager._create_memory', return_value=mock_mem0_memory):
            manager = MemoryManager()
            
            # First get memory (initializes the instance)
            memory1 = manager.get_memory()
            assert manager._memory is memory1
            
            # Reset memory
            manager.reset_memory()
            assert manager._memory is None
            
            # Get memory again (should re-create the instance)
            with patch('linden.memory.agent_memory.MemoryManager._create_memory', return_value=MagicMock()) as mock_create:
                memory2 = manager.get_memory()
                mock_create.assert_called_once()
                assert memory1 is not memory2
    
    def test_get_all_agent_memories_with_agent_id(self, mock_memory_manager):
        """Test get_all_agent_memories with specific agent ID."""
        mock_mem0_memory = mock_memory_manager.get_memory()
        mock_mem0_memory.get_all.return_value = [{"id": 1, "content": "test"}]
        
        result = mock_memory_manager.get_all_agent_memories(agent_id="test-agent")
        
        mock_mem0_memory.get_all.assert_called_once_with(agent_id="test-agent")
        assert result == [{"id": 1, "content": "test"}]
    
    def test_get_all_agent_memories_without_agent_id(self, mock_memory_manager):
        """Test get_all_agent_memories without agent ID."""
        mock_mem0_memory = mock_memory_manager.get_memory()
        mock_mem0_memory.get_all.return_value = [{"id": 1, "content": "test"}]
        
        result = mock_memory_manager.get_all_agent_memories()
        
        # Should try to get all memories using user_id="*"
        mock_mem0_memory.get_all.assert_called_once_with(user_id="*")
        assert result == [{"id": 1, "content": "test"}]
    
    def test_get_all_agent_memories_fallback(self, mock_memory_manager):
        """Test get_all_agent_memories fallback when get_all raises exception."""
        mock_mem0_memory = mock_memory_manager.get_memory()
        mock_mem0_memory.get_all.side_effect = Exception("Test error")
        
        result = mock_memory_manager.get_all_agent_memories()
        
        # Should return empty list on error
        assert result == []
