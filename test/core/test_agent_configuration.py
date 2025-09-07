# pylint: disable=C0303
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
import pytest
from pydantic import ValidationError

from linden.core.agent_runner import AgentConfiguration, AgentRunner


def test_agent_configuration_with_unexpected_parameter():
    """Test that AgentConfiguration rejects unexpected parameters."""
    # Test that AgentConfiguration with extra='forbid' rejects unknown parameters
    with pytest.raises(ValidationError) as exc_info:
        AgentConfiguration(
            user_id="test_user",
            name="test_agent", 
            model="gpt-4",
            temperature=0.7,
            system_prompt="You are a test assistant.",
            unexpected_param="this should cause an error"  # This should be rejected
        )
    
    # Verify that the error mentions the unexpected parameter
    error_str = str(exc_info.value)
    assert "unexpected_param" in error_str
    assert "Extra inputs are not permitted" in error_str


def test_agent_runner_with_unexpected_parameter_via_config():
    """Test that AgentRunner with AgentConfiguration rejects unexpected parameters."""
    # First create a valid configuration
    valid_config = AgentConfiguration(
        user_id="test_user",
        name="test_agent",
        model="gpt-4", 
        temperature=0.7,
        system_prompt="You are a test assistant."
    )
    
    # This should work fine
    agent = AgentRunner(config=valid_config)
    assert agent.name == "test_agent"
    assert agent.model == "gpt-4"
    
    # Now test that creating config with unexpected parameter fails
    with pytest.raises(ValidationError) as exc_info:
        AgentConfiguration(
            user_id="test_user",
            name="test_agent",
            model="gpt-4",
            temperature=0.7,
            system_prompt="You are a test assistant.",
            invalid_parameter="should_fail"
        )
    
    # Verify the error message
    error_str = str(exc_info.value)
    assert "invalid_parameter" in error_str
    assert "Extra inputs are not permitted" in error_str


def test_agent_runner_with_unexpected_parameter_via_kwargs():
    """Test that AgentRunner rejects unexpected parameters when passed via kwargs."""
    # Test that when AgentRunner creates AgentConfiguration from kwargs,
    # it also rejects unexpected parameters
    with pytest.raises(ValidationError) as exc_info:
        AgentRunner(
            user_id="test_user",
            name="test_agent",
            model="gpt-4",
            temperature=0.7,
            system_prompt="You are a test assistant.",
            unknown_param="this_should_fail"
        )
    
    # Verify error details
    error_str = str(exc_info.value)
    assert "unknown_param" in error_str
    assert "Extra inputs are not permitted" in error_str


def test_agent_configuration_valid_parameters():
    """Test that AgentConfiguration accepts all valid parameters."""
    def sample_tool():
        """A sample tool."""
        return "result"
    
    # Test with all valid parameters
    config = AgentConfiguration(
        user_id="test_user",
        name="test_agent",
        model="gpt-4",
        temperature=0.5,
        system_prompt="You are a helpful assistant.",
        tools=[sample_tool],
        output_type=None,
        retries=5
    )
    
    # Verify all fields are set correctly
    assert config.user_id == "test_user"
    assert config.name == "test_agent"
    assert config.model == "gpt-4"
    assert config.temperature == 0.5
    assert config.system_prompt == "You are a helpful assistant."
    assert len(config.tools) == 1
    assert config.tools[0] is sample_tool
    assert config.output_type is None
    assert config.retries == 5


def test_agent_configuration_default_values():
    """Test that AgentConfiguration uses correct default values."""
    config = AgentConfiguration(
        user_id="test_user",
        model="gpt-4",
        temperature=0.7,
        system_prompt="You are a test assistant."
    )
    
    # Check defaults
    assert config.name is not None  # Should get a UUID
    assert len(config.name) > 0
    assert config.tools == []  # Default empty list
    assert config.output_type is None  # Default None
    assert config.retries == 3  # Default 3


def test_agent_configuration_temperature_validation():
    """Test that AgentConfiguration validates temperature range."""
    # Test valid temperature
    config = AgentConfiguration(
        user_id="test_user",
        model="gpt-4",
        temperature=0.5,
        system_prompt="Test prompt"
    )
    assert config.temperature == 0.5
    
    # Test invalid temperature (too high)
    with pytest.raises(ValidationError) as exc_info:
        AgentConfiguration(
            user_id="test_user",
            model="gpt-4", 
            temperature=1.5,  # Invalid: > 1
            system_prompt="Test prompt"
        )
    
    error_str = str(exc_info.value)
    assert "less than or equal to 1" in error_str
    
    # Test invalid temperature (negative)
    with pytest.raises(ValidationError) as exc_info:
        AgentConfiguration(
            user_id="test_user",
            model="gpt-4",
            temperature=-0.1,  # Invalid: < 0
            system_prompt="Test prompt"
        )
    
    error_str = str(exc_info.value)
    assert "greater than or equal to 0" in error_str
