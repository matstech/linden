"""Module defining needed model in agent flow"""
from typing import Optional, List, Union
from pydantic import BaseModel

class ToolNotFound(Exception):
    """ Exception for tools not declared"""
    def __init__(self, message):
        """
        Initialize a ToolNotFound exception.
        
        Args:
            message: The error message
        """
        self.message = message

class ToolError(Exception):
    """ Tool exception """
    def __init__(self, message, tool_name=None, tool_input=None):
        """
        Initialize a ToolError exception.
        
        Args:
            message: The error message
            tool_name: Name of the tool that caused the error
            tool_input: Input provided to the tool that caused the error
        """
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.tool_input = tool_input

class Function(BaseModel):
    """ Class defining function call descriptor """
    name: str
    arguments: Union[str, dict]

    def __str__(self) :
        """
        Get string representation of the Function.
        
        Returns:
            str: String representation showing the function name and arguments
        """
        return f"{self.name}( with arguments ){self.arguments})"

class ToolCall(BaseModel):
    """ Class definining tool call output """
    id: Optional[str] = None
    type: Optional[str] = None
    function: Function

    def __str__(self):
        return f"ToolCall {self.id} of type {self.type}, function: {self.function}"

class ToolCalls(BaseModel):
    """ Class representing client llm response message """
    tool_calls: Optional[List[ToolCall]]
