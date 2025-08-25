"""
Tests for doc_string_parser module.
"""

import pytest
from linden.utils.doc_string_parser import (
    parse_google_docstring, 
    _build_description, 
    _convert_type, 
    _extract_constraints,
    _parse_nested_parameters
)
from docstring_parser import parse


class TestDocStringParser:
    def test_parse_google_docstring_with_basic_docstring(self):
        """Test parsing a basic Google-style docstring."""
        docstring = """
        This is a short description.

        This is a longer description that spans
        multiple lines.

        Args:
            param1 (str): A string parameter.
            param2 (int): An integer parameter.

        Returns:
            bool: A boolean return value.
        """

        result = parse_google_docstring(docstring, func_name="test_func")
        
        # Check basic structure
        assert result["name"] == "test_func"
        assert "This is a short description" in result["description"]
        assert "longer description" in result["description"]
        
        # Check parameters
        assert "param1" in result["parameters"]["properties"]
        assert "param2" in result["parameters"]["properties"]
        assert result["parameters"]["properties"]["param1"]["type"] == "string"
        assert result["parameters"]["properties"]["param2"]["type"] == "integer"
        assert "param1" in result["parameters"]["required"]
        assert "param2" in result["parameters"]["required"]
        
        # Check returns
        assert result["returns"]["type"] == "boolean"
        assert "boolean return value" in result["returns"]["description"]

    def test_parse_google_docstring_with_optional_parameter(self):
        """Test parsing a docstring with optional parameters."""
        docstring = """
        Test function with optional parameter.

        Args:
            param1 (str): A required parameter.
            param2 (int, optional): An optional parameter.

        Returns:
            str: A string return value.
        """

        result = parse_google_docstring(docstring, func_name="test_func")
        
        assert "param1" in result["parameters"]["required"]
        assert "param2" not in result["parameters"]["required"]
        assert result["parameters"]["properties"]["param2"]["type"] == "integer"

    def test_parse_google_docstring_with_nested_parameters(self):
        """Test parsing a docstring with nested parameters."""
        docstring = """
        Test function with nested parameters.

        Args:
            config (dict): Configuration object with the following fields:
                api_key (str): The API key for authentication.
                timeout (int, optional): Request timeout in seconds.
                retries (int): Number of retries. Minimum: 1, Maximum: 5.

        Returns:
            bool: Success flag.
        """

        result = parse_google_docstring(docstring, func_name="test_func")
        
        config_prop = result["parameters"]["properties"]["config"]
        assert config_prop["type"] == "object"
        assert "api_key" in config_prop["properties"]
        assert "timeout" in config_prop["properties"]
        assert "retries" in config_prop["properties"]
        assert config_prop["properties"]["api_key"]["type"] == "string"
        assert config_prop["properties"]["timeout"]["type"] == "integer"
        assert "api_key" in config_prop["required"]
        assert "timeout" not in config_prop["required"]
        assert "retries" in config_prop["required"]

    def test_parse_google_docstring_with_constraints(self):
        """Test parsing a docstring with constraints in description."""
        docstring = """
        Test function with constraints.

        Args:
            count (int): Number of items. Minimum: 1, Maximum: 100, Default: 10
            name (str): Name with no constraints.

        Returns:
            dict: Result object.
        """

        result = parse_google_docstring(docstring, func_name="test_func")
        
        count_param = result["parameters"]["properties"]["count"]
        assert count_param["minimum"] == 1
        assert count_param["maximum"] == 100
        assert count_param["default"] == 10

    def test_parse_google_docstring_with_special_formats(self):
        """Test parsing a docstring with special formats like UUID and ISO timestamps."""
        docstring = """
        Test function with special formats.

        Args:
            id (str): User ID in UUID format.
            timestamp (str): Event time in ISO format.

        Returns:
            dict: Result object.
        """

        result = parse_google_docstring(docstring, func_name="test_func")
        
        assert result["parameters"]["properties"]["id"]["format"] == "uuid"
        assert result["parameters"]["properties"]["timestamp"]["format"] == "date-time"

    def test_parse_google_docstring_with_array_return(self):
        """Test parsing a docstring with array return type."""
        docstring = """
        Test function returning an array.

        Args:
            count (int): Number of items.

        Returns:
            list: A list of strings.
        """

        result = parse_google_docstring(docstring, func_name="test_func")
        
        assert result["returns"]["type"] == "array"
        assert result["returns"]["items"]["type"] == "string"

    def test_parse_google_docstring_without_returns(self):
        """Test parsing a docstring without including returns section."""
        docstring = """
        Test function without including returns.

        Args:
            param (str): A parameter.

        Returns:
            None: This function returns nothing.
        """

        result = parse_google_docstring(docstring, func_name="test_func", include_returns=False)
        
        assert "returns" not in result

    def test_parse_empty_docstring(self):
        """Test parsing an empty docstring."""
        result = parse_google_docstring("", func_name="test_func")
        assert result == {}
        
        result = parse_google_docstring(None, func_name="test_func")
        assert result == {}

    def test_build_description(self):
        """Test the _build_description helper function."""
        parsed_docstring = parse("""
        Short description.
        
        Longer description
        that spans multiple lines.
        """)
        
        description = _build_description(parsed_docstring)
        assert "Short description" in description
        assert "Longer description" in description
        assert "spans multiple lines" in description

    def test_convert_type(self):
        """Test the _convert_type helper function."""
        assert _convert_type("int") == "integer"
        assert _convert_type("integer") == "integer"
        assert _convert_type("float") == "number"
        assert _convert_type("number") == "number"
        assert _convert_type("bool") == "boolean"
        assert _convert_type("boolean") == "boolean"
        assert _convert_type("list") == "array"
        assert _convert_type("array") == "array"
        assert _convert_type("dict") == "object"
        assert _convert_type("object") == "object"
        assert _convert_type("str") == "string"
        assert _convert_type("string") == "string"
        assert _convert_type("unknown_type") == "string"
        assert _convert_type(None) == "string"

    def test_extract_constraints(self):
        """Test the _extract_constraints helper function."""
        description = "A parameter with constraints. Minimum: 1, Maximum: 100, Default: 10"
        constraints = _extract_constraints(description)
        
        assert constraints["minimum"] == 1
        assert constraints["maximum"] == 100
        assert constraints["default"] == 10
        
        # Test with no constraints
        assert _extract_constraints("No constraints here") == {}
        assert _extract_constraints(None) == {}

    def test_parse_nested_parameters(self):
        """Test the _parse_nested_parameters helper function."""
        description = """Configuration object with:
        api_key (str): The API key for authentication.
        timeout (int, optional): Request timeout in seconds.
        retries (int): Number of retries.
        """
        
        nested_params = _parse_nested_parameters(description)
        
        assert nested_params["type"] == "object"
        assert "api_key" in nested_params["properties"]
        assert "timeout" in nested_params["properties"]
        assert "retries" in nested_params["properties"]
        assert nested_params["properties"]["api_key"]["type"] == "string"
        assert nested_params["properties"]["timeout"]["type"] == "integer"
        assert "api_key" in nested_params["required"]
        assert "timeout" not in nested_params["required"]
        assert "retries" in nested_params["required"]
        
        # Test with no nested parameters
        assert _parse_nested_parameters("Just a regular description") is None
