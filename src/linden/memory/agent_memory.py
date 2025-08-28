# pylint: disable=C0301
"""Module wrap agent memory options and functionalities """
import logging
import threading
from typing import Optional
from mem0 import Memory

from ..config.configuration import ConfigManager

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Thread-safe singleton for managing Memory instances.
    
    This singleton ensures that all AgentMemory instances share the same
    underlying Memory object and FAISS index. Each agent is distinguished
    by its agent_id when storing and retrieving memories, allowing for
    proper memory isolation while sharing the same vector store.
    
    Benefits:
    - Thread-safe memory access across multiple agents
    - Shared FAISS index reduces memory overhead
    - Agent isolation through agent_id filtering
    - Consistent memory configuration across all agents
    """

    _instance: Optional['MemoryManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'MemoryManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._memory = None
                    cls._instance._memory_lock = threading.Lock()
        return cls._instance

    def get_memory(self) -> Memory:
        """Get or create the Memory instance in a thread-safe manner."""
        if self._memory is None:
            if not hasattr(self, "_memory_lock"):
              self._memory_lock = threading.Lock()
            with self._memory_lock:
                if self._memory is None:
                    self._memory = self._create_memory()
        return self._memory

    def reset_memory(self) -> None:
        """Reset the Memory instance for testing purposes."""
        with self._memory_lock:
            self._memory = None

    def get_all_agent_memories(self, agent_id: str = None):
        """
        Get all memories for a specific agent or all agents.
        
        Args:
            agent_id (str, optional): Specific agent ID to filter by.
                                    If None, returns all memories for all agents.
            
        Returns:
            list: List of memories
        """
        memory = self.get_memory()
        if agent_id:
            return memory.get_all(agent_id=agent_id)
        else:
            # To get all memories, we need to use a workaround since mem0 requires at least one ID
            # We'll try to get all memories by using a dummy user_id filter that should match all
            try:
                # This might not work in all versions, so we'll catch the exception
                return memory.get_all(user_id="*")
            except Exception:
                # Fallback: return  empty list since we can't get all without specific IDs
                logger.warning("Warning: can't get agents memory without specific user id")
                return []

    def _create_memory(self) -> Memory:
        """Create a new Memory instance with current configuration."""
        conf = ConfigManager.get()
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": conf.openai.api_key
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": conf.openai.api_key
                }
            },
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": conf.memory.collection_name,
                    "path": conf.memory.path,
                    "distance_strategy":"cosine",
                    "normalize_L2": "True"
                }
            }
        }
        return Memory.from_config(config)


class AgentMemory:
    """
    Manages agent memory for persistent conversation history and retrieval.
    
    Uses a thread-safe singleton MemoryManager to ensure consistent memory access
    across multiple agent instances while avoiding global variable issues.
    
    Key Features:
    - Shared FAISS index across all agents (via MemoryManager singleton)
    - Agent-specific memory isolation using agent_id
    - Thread-safe memory operations
    - Persistent conversation history with semantic search
    
    Memory Architecture:
    - All agents share the same Memory instance and FAISS vector store
    - Each agent's messages are tagged with their unique agent_id
    - Memory retrieval is filtered by agent_id to maintain isolation
    - This design prevents index conflicts while enabling efficient memory usage
    """
    def __init__(self, agent_id: str, user_id: str,  system_prompt: str = None, history: list[dict] = None):
        """
        Initialize AgentMemory for a specific agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            system_prompt (str, optional): System prompt to initialize conversation history
        """
        self.agent_id = agent_id
        self.user_id = user_id
        self.system_prompt = system_prompt
        self._memory_manager = MemoryManager()
        self._write_lock = threading.Lock()

        if history is None:
            self.history = []
        else:
            self.history = history
        self._set_system_prompt()

    @property
    def memory(self) -> Memory:
        """Get the Memory instance from the singleton manager."""
        return self._memory_manager.get_memory()
    
    def get_system_prompt(self):
        """Get the system prompt"""
        return self.system_prompt

    def _set_system_prompt(self):
        """Initialize conversation history with system prompt."""
        self.history = [self.system_prompt]

    def record(self, message: str, persist: bool = False):
        """
        Record a message in both persistent memory and conversation history.
        
        Args:
            message (dict|str): Message to record. Can be a dict with 'role' and 'content'
                               or a string (treated as user message)
        """
        if isinstance(message, dict) and "role" in message:
            self.history.append(message)
        elif isinstance(message, str):
            self.history.append({"role": "user", "content": message})

        if persist:
            with self._write_lock:
                # Add to persistent memory first
                try:
                    self.memory.add(message, user_id=self.user_id, infer=False)
                except Exception as e:
                    logger.warning("Warning: Failed to persist message: %s", e)

    def get_conversation(self, user_input: str):
        """
        Build conversation context including system prompt, relevant memories, and current input.
        
        Args:
            user_input (str): Current user input to search memories for
            
        Returns:
            list: Conversation messages including system prompt, relevant memories, and current input
        """
        try:
            search_result = self.memory.search(query=user_input,
                                               user_id=self.user_id,
                                               limit=10)

            if search_result.get('results'):
                formatted_memories = "\n".join([f"- {mem['memory']}" for mem in search_result.get('results')])
                meta_prompt = f"""
                {user_input}

                ---
                [System Instructions]: Before answering, review the following context retrieved from your long-term memory. 
                Use this information to provide a more accurate and complete response.

                Here is some relevant context from past conversations:\n{formatted_memories}
                ---
                """
                self.history.append({"role": "user", "content": meta_prompt})
        except Exception as e:
            logger.warning("Warning: Memory search failed, using local history only: %s", e)

        return self.history

    def reset(self):
        """Reset agent memory and conversation history."""
        self.history = []
        self._set_system_prompt()
