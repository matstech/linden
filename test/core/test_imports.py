# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0303
def test_import_core_package():
    """Test that the core package can be imported."""
    import linden.core
    assert linden.core is not None
    
    # Test accessing exported symbols
    assert hasattr(linden.core, "AgentRunner")
    assert hasattr(linden.core, "ToolCall")
    assert hasattr(linden.core, "ToolError")
    assert hasattr(linden.core, "ToolNotFound")


def test_import_submodules():
    """Test that core submodules can be imported."""
    import linden.core.agent_runner
    import linden.provider.ai_client
    import linden.core.model
    
    assert hasattr(linden.core.agent_runner, "AgentRunner")
    assert hasattr(linden.provider.ai_client, "AiClient")
    assert hasattr(linden.provider.ai_client, "Provider")
    assert hasattr(linden.core.model, "ToolCall")


def test_direct_imports():
    """Test direct imports of core classes."""
    from linden.core import AgentRunner, ToolCall, ToolError, ToolNotFound
    
    assert AgentRunner.__name__ == "AgentRunner"
    assert issubclass(ToolCall, object)
    assert issubclass(ToolError, Exception)
    assert issubclass(ToolNotFound, Exception)


def test_core_init_has_all():
    """Test that __all__ is properly defined in the __init__.py file."""
    import linden.core
    assert hasattr(linden.core, "__all__")
    assert set(linden.core.__all__) == {
        "AgentRunner", 
        "ToolCall",
        "ToolError",
        "ToolNotFound"
    }
