"""Shared pytest fixtures for the test suite."""

import pytest
import src.config


@pytest.fixture(autouse=True)
def force_mock_providers(monkeypatch: pytest.MonkeyPatch):
    """Ensure every test defaults to mock providers even if local .env configures gemini."""
    monkeypatch.setenv("CLASSIFIER_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    src.config._config_instance = None
    yield
    src.config._config_instance = None
