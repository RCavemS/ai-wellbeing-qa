"""Configuration management for AI Wellbeing Assistant."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env file if present in working directory
load_dotenv()


@dataclass
class Config:
    """Application configuration with environment variable defaults."""

    provider: str = "mock"
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    api_key: Optional[str] = None

    def __post_init__(self):
        # Override with environment variables if not explicitly customized or set to defaults
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider and self.provider == "mock":
            self.provider = env_provider

        env_model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL")
        if env_model and self.model_name == "gemini-1.5-flash":
            self.model_name = env_model

        env_temp = os.getenv("LLM_TEMPERATURE")
        if env_temp is not None:
            try:
                self.temperature = float(env_temp)
            except ValueError:
                pass

        env_api_key = os.getenv("LLM_API_KEY")
        if env_api_key and self.api_key is None:
            self.api_key = env_api_key


_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Retrieve or initialize application configuration singleton."""
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = Config()
    return _config_instance
