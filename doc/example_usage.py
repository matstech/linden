"""
Example usage of the Linden framework.

This script provides an interactive command-line loop to chat with a Linden agent.
"""
import os
from linden.core import AgentRunner, AgentConfiguration
from linden.provider.ai_client import Provider

def main():
    """
    Initializes a Linden agent and starts an interactive chat session.
    """
    
    # Ensure the OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running this script: export OPENAI_API_KEY='your-key'")
        return

    # Create agent configuration
    # This agent will have short-term memory but no long-term persistence
    # because enable_memory defaults to True, but no [memory] config is loaded.
    config = AgentConfiguration(
        user_id="user123",
        name="interactive_agent",
        model="gpt-4o-mini",
        temperature=0.7,
        system_prompt="You are a helpful assistant.",
        client=Provider.OPENAI,
        enable_memory=True # Explicitly enabling memory (short-term is active)
    )
    
    # Create an agent
    agent = AgentRunner(config=config)
    
    print("Linden agent is ready. Type 'exit' or 'quit' to end the session.")
    print("-" * 20)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting agent session.")
                break

            # Use agent.run() for the full agent experience (including memory)
            # and stream=True for interactive responses.
            response_generator = agent.run(user_input, stream=True)
            
            print("Agent: ", end="", flush=True)
            # Check if the response is a generator (streaming)
            if hasattr(response_generator, '__iter__') and not isinstance(response_generator, (str, dict, list)):
                for chunk in response_generator:
                    print(chunk, end="", flush=True)
                print()  # Newline after the full response
            else:
                # Handle non-streaming responses (e.g., from tool calls)
                print(response_generator)

        except KeyboardInterrupt:
            print("\n\nExiting agent session.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # Optional: break the loop on error, or continue
            # break

if __name__ == "__main__":
    main()
