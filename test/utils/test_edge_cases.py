"""
Test edge cases for the doc_string_parser module.
"""

import pytest
from linden.utils.doc_string_parser import parse_google_docstring


class TestDocStringParserEdgeCases:
    def test_complex_nested_parameters(self):
        """Test parsing a complex docstring with deeply nested parameters."""
        docstring = """
        A function with complex nested parameters.

        Args:
            config (dict): Configuration with complex structure:
                api (dict): API configuration:
                    key (str): API key.
                    version (str, optional): API version.
                options (dict, optional): Optional settings:
                    timeout (int): Timeout in seconds.
                    retries (int, optional): Number of retry attempts.

        Returns:
            dict: Response data.
        """

        result = parse_google_docstring(docstring, func_name="complex_func")
        
        # Verify the config parameter is properly captured
        config_props = result["parameters"]["properties"]["config"]
        assert config_props["type"] == "object"
        
        # The current implementation doesn't support multi-level nesting beyond one level
        # The nested parameters are flattened at the first level
        assert "api" in config_props["properties"]
        assert "options" in config_props["properties"]

    def test_malformed_docstring(self):
        """Test parsing a malformed docstring."""
        docstring = """
        A function with a malformed docstring.

        Args:
            param1: Missing type.
            param2 (invalid type): Invalid type.
            param3 (int) Missing colon.
            param4 (int, Invalid constraint: 100): Invalid constraint format.

        Returns:
            Something.
        """

        result = parse_google_docstring(docstring, func_name="malformed_func")
        
        # Should still return a valid schema even with malformed docstring
        assert "name" in result
        assert "parameters" in result
        assert "properties" in result["parameters"]

    def test_empty_sections(self):
        """Test docstring with empty sections."""
        docstring = """
        A function with empty sections.

        Args:

        Returns:

        """

        result = parse_google_docstring(docstring, func_name="empty_sections_func")
        
        # Should have basic structure but empty sections
        assert result["name"] == "empty_sections_func"
        assert "parameters" in result
        assert "properties" in result["parameters"]
        assert result["parameters"]["properties"] == {}
        assert "returns" not in result or result["returns"] is None

    def test_only_description_no_sections(self):
        """Test docstring with only description and no sections."""
        docstring = """
        A function with only description and no Args or Returns sections.
        This should still parse correctly.
        """

        result = parse_google_docstring(docstring, func_name="description_only_func")
        
        assert result["name"] == "description_only_func"
        assert "description" in result
        assert "A function with only description" in result["description"]
        assert "parameters" in result
        assert "properties" in result["parameters"]
        assert result["parameters"]["properties"] == {}

    def test_mixed_type_specifications(self):
        """Test parsing with mixed type specifications."""
        docstring = """
        A function with mixed type specifications.

        Args:
            param1 (Union[str, int]): A mixed type parameter.
            param2 (List[str]): A list of strings.
            param3 (Dict[str, Any]): A dictionary with string keys.
            param4 (Optional[bool]): An optional boolean.

        Returns:
            Tuple[int, str]: A tuple of int and string.
        """

        result = parse_google_docstring(docstring, func_name="mixed_types_func")
        
        # Currently the parser handles complex types simply
        # The parser extracts the first type from Union/type hints
        # For Union[str, int], the parser might extract 'int' if it appears first in the docstring
        assert result["parameters"]["properties"]["param1"]["type"] in ["string", "integer"]
        assert result["parameters"]["properties"]["param2"]["type"] == "array"  # List detected as array
        assert result["parameters"]["properties"]["param3"]["type"] == "object"  # Dict detected as object
        assert result["parameters"]["properties"]["param4"]["type"] == "boolean"  # Optional type is extracted

    def test_with_raises_section(self):
        """Test parsing a docstring that includes a Raises section."""
        docstring = """
        A function that may raise exceptions.

        Args:
            param1 (str): A parameter.

        Raises:
            ValueError: If param1 is empty.
            TypeError: If param1 is not a string.

        Returns:
            int: A result code.
        """

        result = parse_google_docstring(docstring, func_name="raises_func")
        
        # The parser currently doesn't extract Raises sections
        assert "name" in result
        assert "parameters" in result
        assert "returns" in result
        assert result["returns"]["type"] == "integer"
        # No 'raises' field is expected in current implementation

    def test_with_examples_section(self):
        """Test parsing a docstring that includes Examples section."""
        docstring = """
        A function with examples section.

        Args:
            param1 (str): A parameter.

        Returns:
            int: A result code.

        Examples:
            >>> my_func("test")
            42
            >>> my_func("another test")
            84
        """

        result = parse_google_docstring(docstring, func_name="examples_func")
        
        # The parser currently doesn't specifically handle Examples sections
        assert "name" in result
        assert "parameters" in result
        assert "returns" in result
        # Examples become part of the long description
        # or are ignored if after a recognized section
