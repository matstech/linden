import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from pathlib import Path

from linden.memory.agent_memory import MemoryManager, AgentMemory


@pytest.fixture
def mock_mem0_memory():
    """Mock for mem0 Memory instance."""
    mock_memory = MagicMock()
    mock_memory.add.return_value = None
    mock_memory.search.return_value = {"results": []}
    mock_memory.get_all.return_value = []
    return mock_memory


@pytest.fixture
def mock_memory_manager(mock_mem0_memory):
    """Mock for MemoryManager."""
    with patch('linden.memory.agent_memory.MemoryManager._create_memory', return_value=mock_mem0_memory):
        # Reset the singleton instance
        MemoryManager._instance = None
        manager = MemoryManager()
        yield manager
        # Reset the singleton instance after the test
        MemoryManager._instance = None


@pytest.fixture
def test_agent_id():
    """Unique agent ID for testing."""
    return "test-agent-id"





@pytest.fixture
def test_system_prompt():
    """System prompt for testing."""
    return {"role": "system", "content": "You are a helpful assistant."}


@pytest.fixture
def agent_memory_with_mocked_manager(mock_memory_manager, test_agent_id, test_system_prompt):
    """AgentMemory instance with mocked Memory manager."""
    agent_mem = AgentMemory(agent_id=test_agent_id, system_prompt=test_system_prompt)
    return agent_mem


@pytest.fixture
def temp_memory_path():
    """Create a temporary directory for memory storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a path for the memory
        memory_path = Path(temp_dir) / "test_memory"
        memory_path.mkdir(exist_ok=True)
        yield str(memory_path)
        # Cleanup happens automatically when the context manager exits


@pytest.fixture
def temp_config_with_memory_path(temp_memory_path):
    """Create a temporary config file with memory path configured."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as temp:
        # Write a minimal valid configuration
        config_content = f"""
[models]
dec = "gpt-4o"
tool = "gpt-4-turbo"
extractor = "gpt-3.5-turbo"
speaker = "claude-3-opus"

[groq]
base_url = "https://api.groq.com/openai/v1"
api_key = "groq-test-key"
timeout = 60

[ollama]
timeout = 30

[openai]
api_key = "test-openai-key"
timeout = 60

[memory]
path = "{temp_memory_path}"
collection_name = "test_memories"
"""
        temp.write(config_content.encode('utf-8'))
        temp_path = temp.name
    
    # Yield the path to the temporary file
    yield temp_path
    
    # Clean up the temporary file after the test
    os.unlink(temp_path)
