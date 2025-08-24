from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable
from pydantic import BaseModel
from ..memory.agent_memory import AgentMemory


class AgentKind(Enum):
    GENERATE = 1
    CHAT = 2
class Provider(Enum):
    OLLAMA = 1
    GROQ = 2
    OPENAI = 3
    
class AiClient(ABC):
    @abstractmethod
    def query_llm(self, input: str, memory:AgentMemory, stream: bool = False,format: BaseModel = None):
        return
    