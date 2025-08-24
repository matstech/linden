# Linden

<div align="center">
  <img src="doc/logo.png" alt="Linden Logo" width="200"/>
</div>

<div align="center">
  <p><em>A Python framework for building AI agents with multi-provider LLM support, persistent memory, and function calling capabilities.</em></p>
</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Basic Agent Setup](#basic-agent-setup)
  - [Agent with Function Calling](#agent-with-function-calling)
  - [Streaming Responses](#streaming-responses)
  - [Structured Output with Pydantic](#structured-output-with-pydantic)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [Architecture](#architecture)
  - [Core Components](#core-components)
  - [Memory Architecture](#memory-architecture)
  - [Function Tool Definition](#function-tool-definition)
- [Advanced Usage](#advanced-usage)
  - [Multi-Turn Conversations](#multi-turn-conversations)
  - [Error Handling and Retries](#error-handling-and-retries)
  - [Memory Management](#memory-management)
  - [Provider-Specific Features](#provider-specific-features)
- [API Reference](#api-reference)
  - [AgentRunner](#agentrunner)
  - [Memory Classes](#memory-classes)
  - [Configuration](#configuration-1)
- [Error Types](#error-types)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

Linden is a comprehensive AI agent framework that provides a unified interface for interacting with multiple Large Language Model (LLM) providers including OpenAI, Groq, and Ollama. It features persistent conversation memory, automatic tool/function calling, and robust error handling for building production-ready AI applications.

## Features

- **Multi-Provider LLM Support**: Seamless integration with OpenAI, Groq, and Ollama
- **Persistent Memory**: Long-term conversation memory using FAISS vector storage and embeddings
- **Function Calling**: Automatic parsing and execution of tools with Google-style docstring support
- **Streaming Support**: Real-time response streaming for interactive applications
- **Thread-Safe Memory**: Concurrent agent support with isolated memory per agent
- **Configuration Management**: Flexible TOML-based configuration with environment variable support
- **Type Safety**: Full Pydantic model support for structured outputs
- **Error Handling**: Comprehensive error handling with retry mechanisms

## Installation

```bash
pip install linden
```

## Requirements

- Python >= 3.9
- Dependencies automatically installed:
  - `openai` - OpenAI API client
  - `groq` - Groq API client  
  - `ollama` - Ollama local LLM client
  - `pydantic` - Data validation and serialization
  - `mem0` - Memory management
  - `docstring_parser` - Function documentation parsing

## Quick Start

### Basic Agent Setup

```python
from linden.core import AgentRunner, Provider

# Create a simple agent
agent = AgentRunner(
    name="assistant",
    model="gpt-4",
    temperature=0.7,
    system_prompt="You are a helpful AI assistant.",
    client=Provider.OPENAI
)

# Ask a question
response = agent.run("What is the capital of France?")
print(response)
```

### Agent with Function Calling

```python
def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather for a location.
    
    Args:
        location (str): The city name or location
        units (str, optional): Temperature units (celsius/fahrenheit). Defaults to celsius.
        
    Returns:
        str: Weather information
    """
    return f"The weather in {location} is 22°{units[0].upper()}"

# Create agent with tools
agent = AgentRunner(
    name="weather_bot",
    model="gpt-4",
    temperature=0.7,
    system_prompt="You are a weather assistant.",
    tools=[get_weather],
    client=Provider.OPENAI
)

response = agent.run("What's the weather in Paris?")
print(response)
```

### Streaming Responses

```python
# Stream responses for real-time interaction
for chunk in agent.run("Tell me a story", stream=True):
    print(chunk, end="", flush=True)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel

class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

agent = AgentRunner(
    name="extractor",
    model="gpt-4",
    temperature=0.1,
    system_prompt="Extract person information from text.",
    output_type=PersonInfo,
    client=Provider.OPENAI
)

result = agent.run("John Smith is a 30-year-old software engineer.")
print(f"Name: {result.name}, Age: {result.age}")
```

## Configuration

Create a `config.toml` file in your project root:

```toml
[models]
dec = "gpt-4"
tool = "gpt-4"
extractor = "gpt-3.5-turbo"
speaker = "gpt-4"

[openai]
api_key = "your-openai-api-key"
timeout = 30

[groq]
base_url = "https://api.groq.com/openai/v1"
api_key = "your-groq-api-key" 
timeout = 30

[ollama]
timeout = 60

[memory]
path = "./memory_db"
```

### Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

## Architecture

### Core Components

#### AgentRunner
The main agent orchestrator that handles:
- LLM interaction and response processing
- Tool calling and execution
- Memory management
- Error handling and retries
- Streaming and non-streaming responses

#### Memory System
- **AgentMemory**: Per-agent conversation history and semantic search
- **MemoryManager**: Thread-safe singleton for shared vector storage
- **Persistent Storage**: FAISS-based vector database for long-term memory

#### AI Clients
Abstract interface with concrete implementations:
- **OpenAiClient**: OpenAI GPT models
- **GroqClient**: Groq inference API
- **Ollama**: Local LLM execution

#### Function Calling
- Automatic parsing of Google-style docstrings
- JSON Schema generation for tool descriptions
- Type-safe argument parsing and validation
- Error handling for tool execution

### Memory Architecture

The memory system uses a shared FAISS vector store with agent isolation:

```python
# Each agent has isolated memory
agent1 = AgentRunner(name="agent1", ...)
agent2 = AgentRunner(name="agent2", ...)

# Memories are automatically isolated by agent_id
agent1.run("Remember I like coffee")
agent2.run("Remember I like tea")

# Each agent only retrieves its own memories
```

### Function Tool Definition

Functions must use Google-style docstrings for automatic parsing:

```python
def search_database(query: str, limit: int = 10, filters: dict = None) -> list:
    """Search the knowledge database.
    
    Args:
        query (str): The search query string
        limit (int, optional): Maximum results to return. Defaults to 10.
        filters (dict, optional): Additional search filters:
            category (str): Filter by category
            date_range (str): Date range in ISO format
            
    Returns:
        list: List of search results with metadata
    """
    # Implementation here
    pass
```

## Advanced Usage

### Multi-Turn Conversations

```python
agent = AgentRunner(name="chat_bot", model="gpt-4", temperature=0.7)

# Conversation maintains context automatically
agent.run("My name is Alice")
agent.run("What's my name?")  # Will remember "Alice"
agent.run("Tell me about my previous question")  # Has full context
```

### Error Handling and Retries

```python
agent = AgentRunner(
    name="robust_agent",
    model="gpt-4", 
    temperature=0.7,
    retries=3  # Retry failed calls up to 3 times
)

try:
    response = agent.run("Complex query that might fail")
except ToolError as e:
    print(f"Tool execution failed: {e.message}")
except ToolNotFound as e:
    print(f"Tool not found: {e.message}")
```

### Memory Management

```python
# Reset agent memory
agent.reset()

# Add context without user interaction
agent.add_to_context("Important context information", persist=True)

# Get conversation history
history = agent.memory.get_conversation("Current query")
```

### Provider-Specific Features

```python
# Use local Ollama models
local_agent = AgentRunner(
    name="local_agent",
    model="llama2",
    client=Provider.OLLAMA
)

# Use Groq for fast inference
fast_agent = AgentRunner(
    name="fast_agent", 
    model="mixtral-8x7b-32768",
    client=Provider.GROQ
)
```

## API Reference

### AgentRunner

#### Constructor Parameters
- `name` (str): Unique agent identifier
- `model` (str): LLM model name
- `temperature` (int): Response randomness (0-1)
- `system_prompt` (str, optional): System instruction
- `tools` (list[Callable], optional): Available functions
- `output_type` (BaseModel, optional): Structured output schema
- `client` (Provider): LLM provider selection
- `retries` (int): Maximum retry attempts

#### Methods
- `run(user_question: str, stream: bool = False)`: Execute agent query
- `reset()`: Clear conversation history
- `add_to_context(content: str, persist: bool = False)`: Add contextual information

### Memory Classes

#### AgentMemory
- `record(message: str, persist: bool = False)`: Store message
- `get_conversation(user_input: str)`: Retrieve relevant context
- `reset()`: Clear agent memory

#### MemoryManager (Singleton)
- `get_memory()`: Access shared memory instance
- `get_all_agent_memories(agent_id: str = None)`: Retrieve stored memories

### Configuration

#### ConfigManager
- `initialize(config_path: str | Path)`: Load configuration file
- `get(config_path: Optional[str | Path] = None)`: Get configuration instance
- `reload()`: Refresh configuration from file

## Error Types

- `ToolNotFound`: Requested function not available
- `ToolError`: Function execution failed
- `ValidationError`: Pydantic model validation failed
- `RequestException`: HTTP/API communication error

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- GitHub Issues: [https://github.com/matstech/linden/issues](https://github.com/matstech/linden/issues)
- Documentation: [https://github.com/matstech/linden](https://github.com/matstech/linden)