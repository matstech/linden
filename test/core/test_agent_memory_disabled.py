# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
import pytest
from linden.core.agent_runner import AgentRunner, AgentConfiguration
from unittest.mock import MagicMock, patch

def test_agent_initialization_with_memory_disabled():
    """
    Verify that an agent with memory disabled initializes correctly
    and does not create a long-term memory backend.
    """
    # No API keys for memory should be needed for this test, thanks to the mock_config fixture
    config = AgentConfiguration(
        user_id="test_user_mem_disabled",
        model="test-model",
        temperature=0.5,
        system_prompt="System prompt.",
        enable_memory=False  # The key part of this test
    )
    agent = AgentRunner(config=config)

    # 1. Check that the AgentMemory object itself exists and is configured correctly
    assert agent.memory is not None
    assert agent.memory.long_term_memory_enabled is False

    # 2. Check that the long-term memory backend (the 'mem0' object) was NOT created
    assert agent.memory.memory is None

@pytest.mark.asyncio
async def test_agent_preserves_short_term_history_with_memory_disabled():
    """
    Verify that an agent with memory disabled still maintains
    short-term conversational history.
    """
    config = AgentConfiguration(
        user_id="test_user_hist_disabled",
        model="test-model",
        temperature=0.5,
        system_prompt="You are a parrot.",
        enable_memory=False
    )
    
    # Mock the AI client to control its responses and inspect its inputs
    mock_client = MagicMock()
    
    with patch('linden.core.agent_runner.Ollama', return_value=mock_client):
        agent = AgentRunner(config=config)

        # --- First turn ---
        # Mock the LLM response for the first message
        mock_client.query_llm.return_value = ("Hello back!", None)
        agent.run("Hello")

        # The 'run' method calls 'record' which populates the history.
        # Let's check the state of the history inside the memory object.
        # History should contain system prompt and the first user message.
        assert len(agent.memory.history) == 2
        assert agent.memory.history[0]['role'] == 'system'
        assert agent.memory.history[1]['content'] == 'Hello'

        # --- Second turn ---
        # The 'run' method will call record again for the assistant's response.
        # But this happens inside the provider, which we mocked.
        # So we manually record the assistant response to simulate the full loop.
        agent.memory.record({"role": "assistant", "content": "Hello back!"})

        mock_client.query_llm.return_value = ("I am fine, thank you!", None)
        agent.run("How are you?")

        # History should now contain all 4 messages
        assert len(agent.memory.history) == 4
        assert agent.memory.history[0]['role'] == 'system'
        assert agent.memory.history[1]['content'] == 'Hello'
        assert agent.memory.history[2]['role'] == 'assistant'
        assert agent.memory.history[2]['content'] == 'Hello back!'
        assert agent.memory.history[3]['content'] == 'How are you?'

def test_add_to_context_persist_does_not_fail_with_memory_disabled():
    """
    Verify that calling add_to_context with persist=True does not
    raise an error when long-term memory is disabled.
    """
    config = AgentConfiguration(
        user_id="test_user_persist_disabled",
        model="test-model",
        temperature=0.5,
        system_prompt="System prompt.",
        enable_memory=False
    )
    agent = AgentRunner(config=config)

    # This call should do nothing and not raise an AttributeError
    try:
        agent.add_to_context("some fact", persist=True)
    except AttributeError:
        pytest.fail("add_to_context with persist=True raised an AttributeError with memory disabled.")
