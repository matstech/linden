"""
Fixtures for utils tests.
"""

import pytest
from docstring_parser import parse


@pytest.fixture
def simple_docstring():
    """Fixture providing a simple docstring."""
    return """
    A simple function.

    This is a longer description.

    Args:
        param1 (str): A string parameter.
        param2 (int): An integer parameter.

    Returns:
        bool: A boolean return value.
    """


@pytest.fixture
def complex_docstring():
    """Fixture providing a complex docstring with nested parameters."""
    return """
    A function with complex parameters.

    Args:
        config (dict): Configuration object:
            api_key (str): API key for authentication.
            timeout (int, optional): Timeout in seconds.
        data (list): Input data list.

    Returns:
        dict: Result dictionary.
    """


@pytest.fixture
def parsed_simple_docstring(simple_docstring):
    """Fixture providing a parsed simple docstring."""
    return parse(simple_docstring)


@pytest.fixture
def parsed_complex_docstring(complex_docstring):
    """Fixture providing a parsed complex docstring."""
    return parse(complex_docstring)
