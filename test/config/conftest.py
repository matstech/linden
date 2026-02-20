import os
import sys
import pytest
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def temp_config_file():
    """Create a temporary config file with valid TOML content."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as temp:
        # Write a minimal valid configuration
        temp.write(b"""
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
api_key = "openai-test-key"
timeout = 60

[anthropic]
api_key = "anthropic-test-key"
max_tokens = 4096
timeout = 60

[google]
api_key = "google-test-key"
timeout = 60

[memory]
path = "/tmp/linden-memory"
collection_name = "test_memories"
""")
        temp_path = temp.name
    
    # Yield the path to the temporary file
    yield temp_path
    
    # Clean up the temporary file after the test
    os.unlink(temp_path)


@pytest.fixture
def temp_empty_config_file():
    """Create a temporary empty config file."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as temp:
        temp_path = temp.name
    
    yield temp_path
    
    os.unlink(temp_path)


@pytest.fixture
def temp_invalid_config_file():
    """Create a temporary config file with invalid TOML content."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as temp:
        # Write invalid TOML content
        temp.write(b"""
[models]
dec = "gpt-4o"
# Missing required fields
# Invalid TOML structure
this is not valid TOML
""")
        temp_path = temp.name
    
    yield temp_path
    
    os.unlink(temp_path)


@pytest.fixture
def temp_incomplete_config_file():
    """Create a temporary config file with incomplete configuration."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as temp:
        # Write incomplete configuration (missing some required sections)
        temp.write(b"""
[models]
dec = "gpt-4o"
tool = "gpt-4-turbo"
extractor = "gpt-3.5-turbo"
speaker = "claude-3-opus"

[openai]
api_key = "openai-test-key"
timeout = 60
""")
        temp_path = temp.name
    
    yield temp_path
    
    os.unlink(temp_path)


@pytest.fixture
def temp_dir_with_config():
    """Create a temporary directory with a config file."""
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "config.toml"
    
    # Write a valid configuration
    with open(config_path, "wb") as f:
        f.write(b"""
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
api_key = "openai-test-key"
timeout = 60

[anthropic]
api_key = "anthropic-test-key"
max_tokens = 4096
timeout = 60

[google]
api_key = "google-test-key"
timeout = 60

[memory]
path = "/tmp/linden-memory"
collection_name = "test_memories"
""")
    
    yield temp_dir
    
    # Clean up
    os.unlink(config_path)
    os.rmdir(temp_dir)
