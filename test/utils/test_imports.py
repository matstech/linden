"""
Test importability of utils modules.
"""

import pytest


def test_import_utils_package():
    """Test that the utils package can be imported."""
    import linden.utils
    assert linden.utils is not None
    assert hasattr(linden.utils, "parse_google_docstring")


def test_import_parse_google_docstring():
    """Test that parse_google_docstring can be imported directly."""
    from linden.utils import parse_google_docstring
    assert callable(parse_google_docstring)


def test_utils_init_has_all():
    """Test that __all__ is properly defined in the __init__.py file."""
    import linden.utils
    assert hasattr(linden.utils, "__all__")
    assert set(linden.utils.__all__) == {"parse_google_docstring"}


def test_utils_modules_can_be_imported_directly():
    """Test that utils modules can be imported directly."""
    import linden.utils.doc_string_parser
    assert hasattr(linden.utils.doc_string_parser, "parse_google_docstring")
    assert callable(linden.utils.doc_string_parser.parse_google_docstring)
