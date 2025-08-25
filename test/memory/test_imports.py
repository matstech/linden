import pytest


class TestMemoryImports:
    """Test importing the memory module components."""
    
    def test_import_agent_memory(self):
        """Test importing AgentMemory class."""
        from linden.memory import AgentMemory
        assert AgentMemory is not None
    
    def test_import_memory_manager(self):
        """Test importing MemoryManager class."""
        from linden.memory.agent_memory import MemoryManager
        assert MemoryManager is not None
    
    def test_import_memory_package(self):
        """Test importing the memory package."""
        import linden.memory
        assert linden.memory is not None
        
        # AgentMemory should be exported from the package
        assert hasattr(linden.memory, 'AgentMemory')
