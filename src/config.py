"""Configuration management for AI Wellbeing Assistant."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file if present in working directory
load_dotenv()

_UNSET = object()


def _parse_float_env(var_name: str, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


class Config:
    """Application configuration with environment variable defaults."""

    def __init__(
        self,
        provider: object = _UNSET,
        model_name: object = _UNSET,
        temperature: object = _UNSET,
        api_key: object = _UNSET,
        classifier_provider: object = _UNSET,
        classifier_model: object = _UNSET,
        classifier_timeout: object = _UNSET,
    ):
        self.provider: str = (
            os.getenv("LLM_PROVIDER") or "mock" if provider is _UNSET else str(provider)
        )
        self.model_name: Optional[str] = (
            os.getenv("LLM_MODEL") if model_name is _UNSET else model_name  # type: ignore[assignment]
        )
        self.temperature: float = (
            _parse_float_env("LLM_TEMPERATURE", 0.7)
            if temperature is _UNSET
            else float(temperature)  # type: ignore[arg-type]
        )
        self.api_key: Optional[str] = (
            os.getenv("LLM_API_KEY") if api_key is _UNSET else api_key  # type: ignore[assignment]
        )
        self.classifier_provider: str = (
            os.getenv("CLASSIFIER_PROVIDER") or "mock"
            if classifier_provider is _UNSET
            else str(classifier_provider)
        )
        self.classifier_model: Optional[str] = (
            os.getenv("CLASSIFIER_MODEL")
            if classifier_model is _UNSET
            else classifier_model  # type: ignore[assignment]
        )
        self.classifier_timeout: float = (
            _parse_float_env("CLASSIFIER_TIMEOUT", 10.0)
            if classifier_timeout is _UNSET
            else float(classifier_timeout)  # type: ignore[arg-type]
        )


_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Retrieve or initialize application configuration singleton."""
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = Config()
    return _config_instance
