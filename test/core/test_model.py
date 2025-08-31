# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0303
from linden.core.model import ToolCall, ToolError, ToolNotFound, Function, ToolCalls


def test_tool_call_initialization():
    """Test ToolCall initialization."""
    # Test with valid parameters
    function = Function(name="test_tool", arguments={"arg1": "value1"})
    tool_call = ToolCall(id="test_id", function=function)
    
    assert tool_call.id == "test_id"
    assert tool_call.function.name == "test_tool"
    assert tool_call.function.arguments == {"arg1": "value1"}
    assert isinstance(tool_call, ToolCall)


def test_tool_call_methods():
    """Test ToolCall methods."""
    function = Function(name="test_tool", arguments={"arg1": "value1", "arg2": 2})
    tool_call = ToolCall(id="test_id", function=function)
    
    # Test str representation
    assert "ToolCall" in str(tool_call)
    assert "test_id" in str(tool_call)
    assert "test_tool" in str(tool_call)
    
    # Test model_dump method (pydantic's equivalent to to_dict)
    model_dict = tool_call.model_dump()
    assert model_dict["id"] == "test_id"
    assert model_dict["function"]["name"] == "test_tool"
    assert model_dict["function"]["arguments"] == {"arg1": "value1", "arg2": 2}


def test_function_initialization():
    """Test Function initialization."""
    arguments = {
        "name": "test_value",
        "age": 30
    }
    
    function = Function(
        name="test_function",
        arguments=arguments
    )
    
    assert function.name == "test_function"
    assert function.arguments == arguments
    assert isinstance(function, Function)


def test_function_to_dict():
    """Test Function model_dump method."""
    arguments = {
        "name": "test_value"
    }
    
    function = Function(
        name="test_function",
        arguments=arguments
    )
    
    expected_dict = {
        "name": "test_function",
        "arguments": arguments
    }
    
    assert function.model_dump() == expected_dict


def test_tool_calls_initialization():
    """Test ToolCalls initialization."""
    # Test with empty list
    tool_calls = ToolCalls(tool_calls=[])
    assert len(tool_calls.tool_calls) == 0
    
    # Test with list of tool calls
    function1 = Function(name="tool1", arguments={"arg1": "val1"})
    function2 = Function(name="tool2", arguments={"arg2": "val2"})
    
    tool_call1 = ToolCall(id="id1", function=function1)
    tool_call2 = ToolCall(id="id2", function=function2)
    
    tool_calls = ToolCalls(tool_calls=[tool_call1, tool_call2])
    assert len(tool_calls.tool_calls) == 2
    assert tool_calls.tool_calls[0].id == "id1"
    assert tool_calls.tool_calls[1].function.name == "tool2"


def test_tool_calls_model_dump():
    """Test ToolCalls model_dump method."""
    function1 = Function(name="tool1", arguments={"arg1": "val1"})
    function2 = Function(name="tool2", arguments={"arg2": "val2"})
    
    tool_call1 = ToolCall(id="id1", function=function1)
    tool_call2 = ToolCall(id="id2", function=function2)
    
    tool_calls = ToolCalls(tool_calls=[tool_call1, tool_call2])
    
    # Get model_dump output
    dump_dict = tool_calls.model_dump()
    
    # Verify structure
    assert "tool_calls" in dump_dict
    assert len(dump_dict["tool_calls"]) == 2
    assert dump_dict["tool_calls"][0]["id"] == "id1"
    assert dump_dict["tool_calls"][0]["function"]["name"] == "tool1"
    assert dump_dict["tool_calls"][1]["id"] == "id2"
    assert dump_dict["tool_calls"][1]["function"]["name"] == "tool2"


def test_tool_error_initialization():
    """Test ToolError initialization."""
    # Simple initialization
    error = ToolError("Test error message")
    assert error.message == "Test error message"
    
    # With tool name and input
    error_with_tool = ToolError("Error in tool", tool_name="test_tool", tool_input={"arg": "value"})
    assert error_with_tool.message == "Error in tool"
    assert error_with_tool.tool_name == "test_tool"
    assert error_with_tool.tool_input == {"arg": "value"}


def test_tool_not_found_error():
    """Test ToolNotFound error."""
    # Simple initialization
    error = ToolNotFound("test_tool")
    assert error.message == "test_tool"
