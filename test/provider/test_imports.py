# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0303

def test_import_provider_package():
    """Test that the provider package can be imported."""
    import linden.provider
    assert hasattr(linden.provider, "OpenAiClient")
    assert hasattr(linden.provider, "GroqClient")
    assert hasattr(linden.provider, "Ollama")
    assert hasattr(linden.provider, "GoogleClient")


def test_import_openai_client():
    """Test that the OpenAI client can be imported."""
    from linden.provider import OpenAiClient
    assert OpenAiClient.__name__ == "OpenAiClient"


def test_import_groq_client():
    """Test that the Groq client can be imported."""
    from linden.provider import GroqClient
    assert GroqClient.__name__ == "GroqClient"


def test_import_ollama_client():
    """Test that the Ollama client can be imported."""
    from linden.provider import Ollama
    assert Ollama.__name__ == "Ollama"


def test_import_google_client():
    """Test that the Google client can be imported."""
    from linden.provider import GoogleClient
    assert GoogleClient.__name__ == "GoogleClient"


def test_provider_init_has_all():
    """Test that __all__ is properly defined in the __init__.py file."""
    import linden.provider
    assert hasattr(linden.provider, "__all__")
    assert set(linden.provider.__all__) == {"OpenAiClient", "GroqClient", "Ollama", "GoogleClient", "Provider"}


def test_provider_modules_can_be_imported_directly():
    """Test that provider modules can be imported directly."""
    import linden.provider.openai
    import linden.provider.groq
    import linden.provider.ollama
    import linden.provider.google
    
    assert hasattr(linden.provider.openai, "OpenAiClient")
    assert hasattr(linden.provider.groq, "GroqClient")
    assert hasattr(linden.provider.ollama, "Ollama")
    assert hasattr(linden.provider.google, "GoogleClient")
