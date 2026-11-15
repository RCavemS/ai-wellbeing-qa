"""LLM client abstractions and provider implementations."""

import abc
from typing import Optional
from src.config import Config


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from LLM given prompt and optional system instructions."""
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Mock synthetic LLM client for offline development, CI, and deterministic testing."""

    def __init__(self, model_name: str = "mock-model", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Return empathetic synthetic response for check-in."""
        lower = prompt.lower()

        # Off-topic redirect (FR1.3)
        if any(w in lower for w in ["capital of", "weather in", "write code", "calculate", "qual é a capital", "programar"]):
            return (
                "Estou aqui especificamente para acompanhar o teu check-in de bem-estar. "
                "Como te estás a sentir neste momento?"
            )

        # Standard empathetic check-in response (FR1.2, FR2)
        return (
            "Obrigado por partilhares isso comigo. Compreendo perfeitamente que estejas a sentir esse peso, "
            "e é totalmente legítimo sentires-te assim. Faz uma pausa e respira fundo. "
            "Há alguma pequena coisa que te possa trazer algum alívio hoje?"
        )


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini API client using standard HTTP requests."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", temperature: float = 0.7):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"Instructions:\n{system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 250,
            },
        }

        response = httpx.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def get_llm_client(config: Config) -> BaseLLMClient:
    """Factory function to build LLM client based on configuration."""
    provider = config.provider.lower().strip()

    if provider == "gemini" and config.api_key:
        return GeminiLLMClient(
            api_key=config.api_key,
            model_name=config.model_name,
            temperature=config.temperature,
        )

    # Default to mock/synthetic client for safe offline execution
    return MockLLMClient(
        model_name=config.model_name,
        temperature=config.temperature,
    )
