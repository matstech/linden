# pylint: disable=C0301
"""Module wrap doc string parsing flow"""
import re
from docstring_parser import parse

from linden.provider.ai_client import Provider

def parse_google_docstring(docstring: str, func_name: str = "function_name", include_returns: bool = True, provider: Provider = Provider.OLLAMA) -> dict:
    """
    Parse Google-style docstring using docstring-parser library.
    
    Args:
        docstring: Google-style docstring
        func_name: Name of the function (optional)
        
    Returns:
        Dictionary in OpenAI function format
    """
    if not docstring or not docstring.strip():
        return {}

    parsed = parse(docstring)

    # Build basic structure
    param_name = "parameters"

    if provider == Provider.ANTHROPIC:
        param_name = "input_schema"

    result = {
        "name": func_name,
        "description": _build_description(parsed),
        f"{param_name}": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    # Add parameters
    for param in parsed.params:
        param_schema = {
            "type": _convert_type(param.type_name),
            "description": param.description or ""
        }

        # Check if this is a complex parameter with nested structure
        nested_params = _parse_nested_parameters(param.description)
        if nested_params and param.type_name and "dict" in param.type_name.lower():
            # This is a nested object parameter
            param_schema = nested_params
            # Keep the main description but clean it up
            main_desc_lines = param.description.split('\n')
            if main_desc_lines:
                # Take first meaningful line as description
                for line in main_desc_lines:
                    if line.strip() and not ':' in line:
                        param_schema["description"] = line.strip()
                        break
        else:
            # Regular parameter processing
            # Add special formats
            if "uuid" in param.description.lower():
                param_schema["format"] = "uuid"
            elif "iso" in param.description.lower() or "timestamp" in param.description.lower():
                param_schema["format"] = "date-time"

            # Extract constraints from description
            param_schema.update(_extract_constraints(param.description))

        result[param_name]["properties"][param.arg_name] = param_schema

        # Add to required if not optional
        if not param.is_optional:
            result[param_name]["required"].append(param.arg_name)

    # Add returns section if present and requested
    if include_returns and parsed.returns and parsed.returns.description:
        return_type = parsed.returns.type_name or ""
        result["returns"] = {
            "type": _convert_type(return_type),
            "description": parsed.returns.description
        }

        # Handle array returns
        if "list" in return_type.lower():
            result["returns"]["items"] = {"type": "string"}

    return result

def _build_description(parsed) -> str:
    """Build description from parsed docstring."""
    parts = []
    if parsed.short_description:
        parts.append(parsed.short_description.strip())
    if parsed.long_description:
        # Remove excessive newlines and clean up formatting
        clean_long = parsed.long_description.strip().replace('\n\n', ' ').replace('\n', ' ')
        parts.append(clean_long)
    return " ".join(parts)

def _convert_type(type_name: str) -> str:
    """Convert Python type to JSON Schema type."""
    if not type_name:
        return "string"

    type_name = type_name.lower()

    if "int" in type_name:
        return "integer"
    elif "float" in type_name or "number" in type_name:
        return "number"
    elif "bool" in type_name:
        return "boolean"
    elif "list" in type_name or "array" in type_name:
        return "array"
    elif "dict" in type_name or "object" in type_name:
        return "object"
    else:
        return "string"

def _extract_constraints(description: str) -> dict:
    """Extract numeric constraints from parameter description."""
    constraints = {}
    if not description:
        return constraints

    desc_lower = description.lower()

    # Extract default value
    default_match = re.search(r'default:\s*(\d+)', desc_lower)
    if default_match:
        constraints["default"] = int(default_match.group(1))

    # Extract minimum
    min_match = re.search(r'minimum:\s*(\d+)', desc_lower)
    if min_match:
        constraints["minimum"] = int(min_match.group(1))

    # Extract maximum
    max_match = re.search(r'maximum:\s*(\d+)', desc_lower)
    if max_match:
        constraints["maximum"] = int(max_match.group(1))

    return constraints

def _parse_nested_parameters(description: str) -> dict:
    """Parse nested parameters from a parameter description."""

    # Pattern to match nested parameter definitions like:
    # session_id (str): Session identifier in UUID format.
    # user_id (str, optional): User identifier in UUID format.
    param_pattern = r'(\w+)\s*\(([^)]+)\)\s*:\s*([^.]+\.?)'

    properties = {}
    required = []

    matches = re.findall(param_pattern, description)

    for param_name, param_type, param_desc in matches:
        # Check if parameter is optional
        is_optional = 'optional' in param_type.lower()

        # Clean up type (remove optional keyword)
        clean_type = re.sub(r',?\s*optional', '', param_type, flags=re.IGNORECASE).strip()

        # Create parameter schema
        param_schema = {
            "type": _convert_type(clean_type),
            "description": param_desc.strip().rstrip('.')  # Remove trailing period
        }

        # Add special formats
        if "uuid" in param_desc.lower():
            param_schema["format"] = "uuid"
        elif "iso" in param_desc.lower() or "timestamp" in param_desc.lower():
            param_schema["format"] = "date-time"

        properties[param_name] = param_schema

        # Add to required if not optional
        if not is_optional:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    } if properties else None
