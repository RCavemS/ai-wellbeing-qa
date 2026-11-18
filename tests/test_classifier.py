"""Unit tests for Stage 2 LLM risk classifier (ADR-001, Issue #31)."""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest
from src.config import Config
from src.classifier import (
    BaseRiskClassifier,
    ClassifierError,
    DEFAULT_CLASSIFIER_TIMEOUT,
    DEFAULT_THINKING_LEVEL,
    GeminiRiskClassifier,
    LABEL_CLINICAL,
    LABEL_CRISIS,
    LABEL_NONE,
    MockRiskClassifier,
    RISK_CLASSIFIER_PROMPT,
    get_risk_classifier,
)


def _make_gemini_http_response(status_code: int = 200, text_content: str = '{"label": "NONE"}') -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text_content}],
                }
            }
        ]
    }
    return resp


def test_mock_risk_classifier_default_and_sequence():
    """Verify MockRiskClassifier returns NONE by default and supports scripted test responses."""
    mock_clf = MockRiskClassifier()
    assert isinstance(mock_clf, BaseRiskClassifier)
    assert mock_clf.classify("Olá, tudo bem?") == LABEL_NONE
    assert mock_clf.call_count == 1

    scripted = MockRiskClassifier(responses=[LABEL_CRISIS, LABEL_CLINICAL])
    assert scripted.classify("msg 1") == LABEL_CRISIS
    assert scripted.classify("msg 2") == LABEL_CLINICAL
    assert scripted.classify("msg 3") == LABEL_NONE
    assert scripted.call_count == 3


def test_risk_classifier_prompt_instructs_crisis_when_unsure():
    """Verify the classifier prompt instructs the model to choose CRISIS when uncertain."""
    prompt_lower = RISK_CLASSIFIER_PROMPT.lower()
    assert "crisis" in prompt_lower
    assert "clinical" in prompt_lower
    assert "none" in prompt_lower
    assert "unsure" in prompt_lower or "uncertain" in prompt_lower or "ambiguous" in prompt_lower


def test_gemini_classifier_sends_strict_json_schema_temp_zero_and_default_thinking_level():
    """Verify GeminiRiskClassifier sends temperature=0, JSON schema enum, thinkingLevel='minimal', and default timeout=30s."""
    classifier = GeminiRiskClassifier(
        api_key="fake-api-key",
        model_name="gemini-3.5-flash-lite",
    )
    mock_resp = _make_gemini_http_response(200, '{"label": "CRISIS"}')

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        label = classifier.classify("acho q hoje é o último check-in q faço")

    assert label == LABEL_CRISIS
    assert mock_post.call_count == 1
    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 30.0
    assert DEFAULT_CLASSIFIER_TIMEOUT == 30.0

    payload = kwargs["json"]
    gen_config = payload["generationConfig"]
    assert gen_config["temperature"] == 0.0
    assert gen_config["responseMimeType"] == "application/json"
    assert gen_config["responseSchema"]["properties"]["label"]["enum"] == [
        LABEL_CRISIS,
        LABEL_CLINICAL,
        LABEL_NONE,
    ]
    assert gen_config["thinkingConfig"] == {"thinkingLevel": DEFAULT_THINKING_LEVEL}
    assert DEFAULT_THINKING_LEVEL == "minimal"


def test_gemini_classifier_sends_api_key_in_header_not_in_url():
    """Verify GeminiRiskClassifier sends the API key in x-goog-api-key header and never in the URL."""
    secret_key = "secret-gemini-api-key-999"
    classifier = GeminiRiskClassifier(
        api_key=secret_key,
        model_name="gemini-3.5-flash-lite",
    )
    mock_resp = _make_gemini_http_response(200, '{"label": "NONE"}')

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        label = classifier.classify("Hoje estou tranquilo.")

    assert label == LABEL_NONE
    assert mock_post.call_count == 1
    args, kwargs = mock_post.call_args
    called_url = args[0] if args else kwargs["url"]

    assert secret_key not in called_url
    assert "key=" not in called_url
    assert "?" not in called_url
    assert kwargs["headers"]["x-goog-api-key"] == secret_key


def test_gemini_classifier_retries_on_429_then_succeeds():
    """429 then success -> retried with backoff and succeeds."""
    classifier = GeminiRiskClassifier(
        api_key="fake-api-key",
        model_name="gemini-3.5-flash-lite",
        backoff_base=0.25,
    )
    resp_429 = _make_gemini_http_response(429)
    resp_200 = _make_gemini_http_response(200, '{"label": "CLINICAL"}')

    with patch("httpx.post", side_effect=[resp_429, resp_200]) as mock_post, patch("time.sleep") as mock_sleep:
        label = classifier.classify("esqueci-me da sertralina ontem, tomo 2 hj p compensar?")

    assert label == LABEL_CLINICAL
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(0.25)


def test_gemini_classifier_retries_on_5xx_then_succeeds():
    """5xx then success -> retried with backoff and succeeds."""
    classifier = GeminiRiskClassifier(
        api_key="fake-api-key",
        model_name="gemini-3.5-flash-lite",
        backoff_base=0.1,
    )
    resp_503 = _make_gemini_http_response(503)
    resp_200 = _make_gemini_http_response(200, '{"label": "CRISIS"}')

    with patch("httpx.post", side_effect=[resp_503, resp_200]) as mock_post, patch("time.sleep") as mock_sleep:
        label = classifier.classify("passei na ponte e fiquei lá parado mt tempo a olhar p baixo")

    assert label == LABEL_CRISIS
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(0.1)


def test_gemini_classifier_429_three_times_raises_classifier_error():
    """429 three times -> stops at 3 attempts and raises ClassifierError."""
    classifier = GeminiRiskClassifier(
        api_key="fake-api-key",
        model_name="gemini-3.5-flash-lite",
        backoff_base=0.1,
    )
    resp_429 = _make_gemini_http_response(429)

    with patch("httpx.post", side_effect=[resp_429, resp_429, resp_429]) as mock_post, patch("time.sleep") as mock_sleep:
        with pytest.raises(ClassifierError) as exc_info:
            classifier.classify("sensitive user text")

    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2
    assert "429" in exc_info.value.error_type
    assert "sensitive user text" not in str(exc_info.value)


@pytest.mark.parametrize(
    "bad_text,expected_error_type",
    [
        ("not valid json", "InvalidJSONError"),
        ('{"wrong_key": "CRISIS"}', "InvalidJSONError"),
        ('{"label": "UNKNOWN_LABEL"}', "InvalidLabelError"),
        ('{"label": "ELEVATED_CONCERN"}', "InvalidLabelError"),
    ],
)
def test_gemini_classifier_invalid_json_or_enum_raises_classifier_error(bad_text, expected_error_type):
    """Invalid JSON or label outside enum raises ClassifierError without raw text."""
    classifier = GeminiRiskClassifier(
        api_key="fake-api-key",
        model_name="gemini-3.5-flash-lite",
    )
    mock_resp = _make_gemini_http_response(200, bad_text)

    with patch("httpx.post", return_value=mock_resp):
        with pytest.raises(ClassifierError) as exc_info:
            classifier.classify("my private message content")

    assert exc_info.value.error_type == expected_error_type
    assert "my private message content" not in str(exc_info.value)


def test_get_risk_classifier_startup_fails_fast_when_gemini_missing_key_or_model():
    """Startup fails fast with provider 'gemini' and no key or model (or -latest alias)."""
    config_no_model = Config(
        classifier_provider="gemini",
        classifier_model="",
        api_key="valid-key",
    )
    with pytest.raises(ValueError, match="CLASSIFIER_MODEL is not set"):
        get_risk_classifier(config_no_model)

    config_no_key = Config(
        classifier_provider="gemini",
        classifier_model="gemini-3.5-flash-lite",
        api_key="",
    )
    with pytest.raises(ValueError, match="LLM_API_KEY is not set"):
        get_risk_classifier(config_no_key)

    config_latest_alias = Config(
        classifier_provider="gemini",
        classifier_model="gemini-flash-lite-latest",
        api_key="valid-key",
    )
    with pytest.raises(ValueError, match="-latest"):
        get_risk_classifier(config_latest_alias)
