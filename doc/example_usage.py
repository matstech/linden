"""
Example usage of the Linden framework.

This example shows how to use the library with the new import structure.
"""

# Import the main components from linden
from linden.core import AgentRunner, Provider

def main():
    """Example of how to use Linden with the new import structure."""
    
    # Create an agent
    agent = AgentRunner(
        user_id="user123",  # User identifier for memory isolation
        name="example_agent",
        model="gpt-3.5-turbo",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        client=Provider.OPENAI
    )
    
    # Use the agent
    response = agent.ask_to_llm("Hello, how are you?")
    print(f"Agent response: {response}")

if __name__ == "__main__":
    main()
