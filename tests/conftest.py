"""Shared pytest fixtures for the test suite."""

from pathlib import Path
import pytest
import src.config


@pytest.fixture(autouse=True)
def force_mock_providers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Ensure every test defaults to mock providers and writes audit JSONL to a temporary directory."""
    monkeypatch.setenv("CLASSIFIER_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("CLASSIFIER_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    src.config._config_instance = None
    yield
    src.config._config_instance = None
