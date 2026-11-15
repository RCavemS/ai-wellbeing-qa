from src.config import Config
from src.llm import MockLLMClient, get_llm_client


def test_mock_llm_portuguese_response():
    client = MockLLMClient()
    response = client.generate("Estou a ter um dia difícil e sinto-me sobrecarregado.")
    assert "compreendo" in response.lower() or "partilhares" in response.lower()
    # Concise check (FR2.3)
    assert len(response.split()) <= 120


def test_mock_llm_english_response():
    client = MockLLMClient()
    response = client.generate("I am feeling really exhausted from work today.")
    assert "thank you for sharing" in response.lower() or "support" in response.lower()
    assert len(response.split()) <= 120


def test_mock_llm_off_topic_redirect():
    client = MockLLMClient()
    response_pt = client.generate("Qual é a capital de França?")
    assert "check-in de bem-estar" in response_pt.lower()

    response_en = client.generate("What is the capital of France?")
    assert "wellbeing check-in" in response_en.lower()


def test_get_llm_client_factory():
    config = Config(provider="mock")
    client = get_llm_client(config)
    assert isinstance(client, MockLLMClient)
