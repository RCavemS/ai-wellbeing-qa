"""Stage 2 constrained LLM risk classifier for crisis and clinical risk detection (ADR-001)."""

import abc
import json
import time
from typing import Optional, Union
from src.config import Config

LABEL_CRISIS = "CRISIS"
LABEL_CLINICAL = "CLINICAL"
LABEL_NONE = "NONE"
VALID_RISK_LABELS = (LABEL_CRISIS, LABEL_CLINICAL, LABEL_NONE)

# Default thinking level for gemini-3.5-flash-lite / Flash-Lite models per
# https://ai.google.dev/gemini-api/docs/thinking
DEFAULT_THINKING_LEVEL = "minimal"
DEFAULT_CLASSIFIER_TIMEOUT = 10.0
DEFAULT_MAX_ATTEMPTS = 3

CLASSIFIER_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {
            "type": "STRING",
            "enum": list(VALID_RISK_LABELS),
        }
    },
    "required": ["label"],
}

RISK_CLASSIFIER_PROMPT = """You are a safety risk classifier for a non-clinical wellbeing check-in assistant (PT-PT and English).
Your sole task is to classify the user's message into strictly one of three labels:
- "CRISIS": Any direct or indirect indication of self-harm, suicidal ideation (active or passive, including passive ideation following a negation such as wanting to sleep and not wake up), implicit farewells (e.g., last check-in, goodbye message), existential exhaustion, wishing to disappear or not be here tomorrow, high-risk behavior at a bridge or method preparation, sudden calm after making a final decision, third-person acute suicide crisis with a plan, or intent to harm others. Includes messages with typos, abbreviations (q, hj, p, n, mt), or PT/EN code-switching.
- "CLINICAL": Requests for medical or psychiatric diagnosis, symptom differentials (e.g., panic attack vs. cardiac issue), medication dosing or missed-dose advice, starting/stopping prescription drugs, supplement safety/prescription inquiries, drug or caffeine interactions, or clinical treatment modality selection.
- "NONE": Standard emotional check-in disclosures (sadness, stress, fatigue, anxiety, loneliness, emotional distress without crisis intent or clinical advice requests), everyday benign figures of speech (e.g., "morrer de sono", "this exam is killing me"), or off-topic questions.

Conservative Uncertainty Rule:
Whenever the user's intent is ambiguous or you are unsure or uncertain whether a message is "CRISIS" versus "CLINICAL" or "NONE", you MUST choose "CRISIS". Crisis safety always takes precedence.

Return strict JSON matching the schema: {"label": "CRISIS" | "CLINICAL" | "NONE"}."""


class ClassifierError(Exception):
    """Exception raised when Stage 2 risk classification fails, times out, or returns invalid output."""

    def __init__(self, message: str, error_type: str = "ClassifierError"):
        super().__init__(message)
        self.error_type = error_type


def _check_gemini_credentials(api_key: Optional[str], model_name: Optional[str]) -> tuple[str, str]:
    """Validate Gemini classifier model name and API key; raise ClassifierError if invalid."""
    if not model_name or not str(model_name).strip():
        raise ClassifierError(
            "CLASSIFIER_MODEL is not set. Please set CLASSIFIER_MODEL in your .env file before starting.",
            error_type="MissingModelError",
        )
    clean_model = str(model_name).strip()
    if clean_model.lower().endswith("-latest"):
        raise ClassifierError(
            "CLASSIFIER_MODEL cannot use a '-latest' alias. Please set a pinned model version (e.g. gemini-3.5-flash-lite).",
            error_type="InvalidModelAliasError",
        )
    if not api_key or not str(api_key).strip():
        raise ClassifierError(
            "LLM_API_KEY is not set. Please set LLM_API_KEY in your .env file before starting.",
            error_type="MissingAPIKeyError",
        )
    return str(api_key).strip(), clean_model


class BaseRiskClassifier(abc.ABC):
    """Abstract interface for Stage 2 risk classification."""

    @abc.abstractmethod
    def classify(self, message: str) -> str:
        """Classify a user message into 'CRISIS', 'CLINICAL', or 'NONE'."""
        raise NotImplementedError


class MockRiskClassifier(BaseRiskClassifier):
    """Deterministic mock classifier for unit tests, offline execution, and CI."""

    def __init__(
        self,
        default_label: str = LABEL_NONE,
        responses: Optional[list[Union[str, Exception]]] = None,
    ):
        self.default_label = default_label
        self._responses = list(responses) if responses is not None else []
        self.call_count: int = 0

    def classify(self, message: str) -> str:
        """Return scripted or default risk label without making any network calls."""
        self.call_count += 1
        if self._responses:
            next_item = self._responses.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item
        return self.default_label


class GeminiRiskClassifier(BaseRiskClassifier):
    """Google Gemini API Stage 2 risk classifier using JSON structured output mode."""

    def __init__(
        self,
        api_key: Optional[str],
        model_name: Optional[str],
        timeout: float = DEFAULT_CLASSIFIER_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = 0.5,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base = backoff_base
        self.thinking_level = thinking_level

    def classify(self, message: str) -> str:
        """Classify message using Gemini API with temperature=0 and strict JSON schema."""
        clean_key, clean_model = _check_gemini_credentials(self.api_key, self.model_name)

        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
        headers = {
            "x-goog-api-key": clean_key,
            "Content-Type": "application/json",
        }
        payload = {
            "systemInstruction": {
                "parts": [{"text": RISK_CLASSIFIER_PROMPT}],
            },
            "contents": [
                {"role": "user", "parts": [{"text": message}]},
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": CLASSIFIER_RESPONSE_SCHEMA,
                "thinkingConfig": {
                    "thinkingLevel": self.thinking_level,
                },
            },
        }

        response: Optional[httpx.Response] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
            except (httpx.TimeoutException, TimeoutError) as exc:
                raise ClassifierError(
                    "Classifier request timed out.",
                    error_type="TimeoutError",
                ) from exc
            except httpx.HTTPError as exc:
                raise ClassifierError(
                    "Classifier HTTP request failed.",
                    error_type=type(exc).__name__,
                ) from exc

            status_code = response.status_code
            if status_code == 429 or 500 <= status_code < 600:
                if attempt < self.max_attempts:
                    if self.backoff_base > 0:
                        time.sleep(self.backoff_base * (2 ** (attempt - 1)))
                    continue
                raise ClassifierError(
                    f"Classifier HTTP {status_code} after {self.max_attempts} attempts.",
                    error_type=f"HTTP_{status_code}",
                )
            if status_code >= 400:
                raise ClassifierError(
                    f"Classifier HTTP {status_code}.",
                    error_type=f"HTTP_{status_code}",
                )
            break

        if response is None:
            raise ClassifierError(
                "Classifier did not receive a response.",
                error_type="NoResponseError",
            )

        try:
            data = response.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ClassifierError(
                "Malformed Gemini response structure.",
                error_type="InvalidJSONError",
            ) from exc

        try:
            parsed = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ClassifierError(
                "Classifier output is not valid JSON.",
                error_type="InvalidJSONError",
            ) from exc

        if not isinstance(parsed, dict) or set(parsed.keys()) != {"label"} or not isinstance(parsed.get("label"), str):
            raise ClassifierError(
                "Classifier JSON does not match expected schema.",
                error_type="InvalidJSONError",
            )

        label = parsed["label"].strip()
        if label not in VALID_RISK_LABELS:
            raise ClassifierError(
                "Classifier returned label outside allowed enum.",
                error_type="InvalidLabelError",
            )

        return label


def validate_classifier_config(config: Config) -> None:
    """Validate classifier configuration at startup; fail fast for invalid Gemini config."""
    provider = (config.classifier_provider or "mock").lower().strip()
    if provider == "gemini":
        try:
            _check_gemini_credentials(config.api_key, config.classifier_model)
        except ClassifierError as exc:
            raise ValueError(str(exc)) from exc
    elif provider != "mock":
        raise ValueError(
            f"Unsupported CLASSIFIER_PROVIDER '{config.classifier_provider}'. Expected 'mock' or 'gemini'."
        )


def get_risk_classifier(config: Config) -> BaseRiskClassifier:
    """Factory function to build Stage 2 risk classifier from configuration."""
    validate_classifier_config(config)
    provider = (config.classifier_provider or "mock").lower().strip()

    if provider == "gemini":
        return GeminiRiskClassifier(
            api_key=config.api_key,
            model_name=config.classifier_model,
            timeout=config.classifier_timeout,
        )

    return MockRiskClassifier()
