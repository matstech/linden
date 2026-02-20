"""Module containing helper functions for AI providers."""
from ..memory.agent_memory import AgentMemory


def prepare_conversation(prompt: str, memory: AgentMemory | None) -> list[dict]:
    """
    Prepares the conversation history list for an LLM call.

    If memory is provided, it retrieves the full conversation context.
    If memory is None (e.g., for summarization calls), it creates
    a simple list containing only the user prompt.
    """
    if memory:
        return memory.get_conversation(user_input=prompt)

    # This is a call without memory (e.g., summarization), create a simple conversation
    return [{"role": "user", "content": prompt}]
