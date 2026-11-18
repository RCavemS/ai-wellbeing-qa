from src.config import Config
from src.llm import MockLLMClient, get_llm_client


def test_mock_llm_empathetic_response():
    """Verify mock LLM returns an empathetic check-in reply within word limit (FR2.3)."""
    client = MockLLMClient()
    response = client.generate("Estou a ter um dia difícil e sinto-me sobrecarregado.")
    assert "compreendo" in response.lower() or "partilhares" in response.lower()
    assert len(response.split()) <= 120


def test_mock_llm_off_topic_redirect():
    """Verify mock LLM redirects off-topic questions back to the check-in (FR1.3)."""
    client = MockLLMClient()
    response = client.generate("Qual é a capital de França?")
    assert "check-in de bem-estar" in response.lower()


def test_get_llm_client_factory():
    """Verify factory returns MockLLMClient for mock provider configuration."""
    config = Config(provider="mock")
    client = get_llm_client(config)
    assert isinstance(client, MockLLMClient)


def test_gemini_client_requires_configured_model_name():
    """Verify Gemini client raises ValueError if model name is not configured."""
    import pytest
    from src.llm import GeminiLLMClient

    client = GeminiLLMClient(api_key="fake-key", model_name=None)
    with pytest.raises(ValueError, match="Model name must be configured"):
        client.generate("Hello")
