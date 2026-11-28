import os
from unittest.mock import patch
from src.config import Config, get_config


def test_config_default_values():
    """Verify default configuration values when no env vars are provided."""
    with patch.dict(os.environ, {}, clear=True):
        config = Config()
        assert config.provider == "mock"
        assert config.model_name is None
        assert config.temperature == 0.7
        assert config.api_key is None
        assert config.classifier_provider == "mock"
        assert config.classifier_model is None
        assert config.classifier_timeout == 30.0


def test_config_env_overrides():
    """Verify configuration loads properly from environment variables."""
    test_env = {
        "LLM_PROVIDER": "gemini",
        "LLM_MODEL": "gemini-2.0-flash",
        "LLM_TEMPERATURE": "0.3",
        "LLM_API_KEY": "test-secret-key",
        "CLASSIFIER_PROVIDER": "gemini",
        "CLASSIFIER_MODEL": "gemini-3.5-flash-lite",
        "CLASSIFIER_TIMEOUT": "15.5",
    }
    with patch.dict(os.environ, test_env, clear=True):
        config = Config()
        assert config.provider == "gemini"
        assert config.model_name == "gemini-2.0-flash"
        assert config.temperature == 0.3
        assert config.api_key == "test-secret-key"
        assert config.classifier_provider == "gemini"
        assert config.classifier_model == "gemini-3.5-flash-lite"
        assert config.classifier_timeout == 15.5


def test_get_config_cached():
    """Verify get_config returns a valid Config instance."""
    config = get_config()
    assert isinstance(config, Config)


def test_config_explicit_mock_not_overridden_by_env():
    """Verify explicit Config(classifier_provider='mock') is preserved even if env sets gemini."""
    test_env = {
        "LLM_PROVIDER": "gemini",
        "CLASSIFIER_PROVIDER": "gemini",
        "CLASSIFIER_TIMEOUT": "25.0",
    }
    with patch.dict(os.environ, test_env, clear=True):
        config = Config(provider="mock", classifier_provider="mock", classifier_timeout=10.0)
        assert config.provider == "mock"
        assert config.classifier_provider == "mock"
        assert config.classifier_timeout == 10.0
