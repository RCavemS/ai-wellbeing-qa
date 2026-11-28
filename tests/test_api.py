"""Integration tests for FastAPI endpoint (FR6) and persistent JSONL audit logging (FR5, Issue #5)."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import uuid
from fastapi.testclient import TestClient
import pytest

from src.api import create_app
from src.assistant import WellbeingAssistant
from src.classifier import ClassifierError, LABEL_CLINICAL, LABEL_CRISIS, MockRiskClassifier
from src.config import Config
from src.llm import BaseLLMClient
from src.safety import (
    AuditLogger,
    CATEGORY_CRISIS,
    CATEGORY_OUT_OF_SCOPE,
    CATEGORY_STANDARD,
    CATEGORY_SYSTEM_ERROR,
    CRISIS_RESPONSE_MESSAGE,
    OUT_OF_SCOPE_REFERRAL_MESSAGE,
    RULE_CLASSIFIER_CLINICAL,
    RULE_CLASSIFIER_CRISIS,
    RULE_CLASSIFIER_FAIL_SAFE,
    RULE_CLINICAL_ADVICE,
    RULE_CRISIS_HARM,
    RULE_LLM_FAIL_SAFE,
    RULE_NONE,
    SAFE_FALLBACK_MESSAGE,
    SafetyLayer,
)


def _read_jsonl(path: Path) -> list[dict]:
    assert path.is_file(), f"Expected JSONL audit file at {path}"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_api_standard_checkin_logs_audit_and_returns_metadata(tmp_path: Path):
    """1. Standard check-in: returns 200 OK with reply, full metadata, and JSONL audit record."""
    audit_path = tmp_path / "standard_audit.jsonl"
    config = Config(
        provider="mock",
        model_name=None,
        classifier_provider="mock",
        classifier_model="gemini-3.5-flash-lite",
        audit_log_path=str(audit_path),
    )
    assistant = WellbeingAssistant(
        config=config,
        audit_logger=AuditLogger(log_path=audit_path),
    )
    client = TestClient(create_app(assistant=assistant))

    session_uuid = str(uuid.uuid4())
    response = client.post(
        "/api/v1/check-in",
        json={
            "message": "Hoje tive um dia longo no trabalho e sinto-me um pouco cansado.",
            "session_id": session_uuid,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["reply"], str) and len(body["reply"]) > 0

    meta = body["metadata"]
    uuid.UUID(meta["interaction_id"])
    assert meta["timestamp"].endswith("Z")
    assert meta["status"] == "OK"
    assert meta["category"] == CATEGORY_STANDARD
    assert meta["rule_triggered"] == RULE_NONE
    assert meta["stage_decided"] == "STAGE_2"
    assert meta["llm_model"] == "mock"
    assert meta["classifier_model"] == "gemini-3.5-flash-lite"
    assert isinstance(meta["classifier_latency_ms"], float)
    assert isinstance(meta["latency_ms"], float)
    assert meta["classifier_fail_safe"] is False
    assert meta["error_type"] is None

    records = _read_jsonl(audit_path)
    assert len(records) == 1
    assert records[0]["interaction_id"] == meta["interaction_id"]
    assert records[0]["category"] == CATEGORY_STANDARD
    assert records[0]["llm_model"] == "mock"
    assert records[0]["classifier_model"] == "gemini-3.5-flash-lite"


def test_api_crisis_decided_by_stage1(tmp_path: Path):
    """2. Crisis decided by Stage 1: skips Stage 2 and conversational LLM, returns deterministic crisis response."""
    audit_path = tmp_path / "crisis_s1.jsonl"
    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_clf = MockRiskClassifier()
    config = Config(
        provider="mock",
        model_name="gemini-2.5-flash",
        classifier_provider="mock",
        classifier_model="gemini-3.5-flash-lite",
        audit_log_path=str(audit_path),
    )
    assistant = WellbeingAssistant(
        config=config,
        llm_client=mock_llm,
        safety_layer=SafetyLayer(classifier=mock_clf),
        audit_logger=AuditLogger(log_path=audit_path),
    )
    client = TestClient(create_app(assistant=assistant))

    response = client.post(
        "/api/v1/check-in",
        json={"message": "Quero matar-me hoje, não aguento mais viver assim."},
    )

    assert response.status_code == 200
    assert mock_clf.call_count == 0
    assert not mock_llm.generate.called

    body = response.json()
    assert body["reply"] == CRISIS_RESPONSE_MESSAGE
    meta = body["metadata"]
    assert meta["status"] == "OK"
    assert meta["category"] == CATEGORY_CRISIS
    assert meta["rule_triggered"] == RULE_CRISIS_HARM
    assert meta["stage_decided"] == "STAGE_1"
    assert meta["llm_model"] == "NONE"
    assert meta["classifier_model"] == "NONE"
    assert meta["classifier_fail_safe"] is False
    assert meta["error_type"] is None


def test_api_crisis_decided_by_stage2_mock(tmp_path: Path):
    """3. Crisis decided by Stage 2 (mock): bypasses conversational LLM and records classifier model & STAGE_2."""
    audit_path = tmp_path / "crisis_s2.jsonl"
    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_clf = MockRiskClassifier(responses=[LABEL_CRISIS])
    config = Config(
        provider="mock",
        model_name="gemini-2.5-flash",
        classifier_provider="mock",
        classifier_model="gemini-3.5-flash-lite",
        audit_log_path=str(audit_path),
    )
    assistant = WellbeingAssistant(
        config=config,
        llm_client=mock_llm,
        safety_layer=SafetyLayer(classifier=mock_clf),
        audit_logger=AuditLogger(log_path=audit_path),
    )
    client = TestClient(create_app(assistant=assistant))

    response = client.post(
        "/api/v1/check-in",
        json={"message": "Deixei uma carta na mesa da sala para a minha família ler amanhã."},
    )

    assert response.status_code == 200
    assert mock_clf.call_count == 1
    assert not mock_llm.generate.called

    body = response.json()
    assert body["reply"] == CRISIS_RESPONSE_MESSAGE
    meta = body["metadata"]
    assert meta["status"] == "OK"
    assert meta["category"] == CATEGORY_CRISIS
    assert meta["rule_triggered"] == RULE_CLASSIFIER_CRISIS
    assert meta["stage_decided"] == "STAGE_2"
    assert meta["llm_model"] == "NONE"
    assert meta["classifier_model"] == "gemini-3.5-flash-lite"
    assert meta["classifier_fail_safe"] is False
    assert meta["error_type"] is None


def test_api_clinical_message_stage1_and_stage2(tmp_path: Path):
    """4. Clinical message: returns deterministic FR4 clinical referral and logs OUT_OF_SCOPE_CLINICAL."""
    audit_path = tmp_path / "clinical.jsonl"
    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_clf = MockRiskClassifier(responses=["NONE", LABEL_CLINICAL])
    config = Config(
        provider="mock",
        model_name="gemini-2.5-flash",
        classifier_provider="mock",
        classifier_model="gemini-3.5-flash-lite",
        audit_log_path=str(audit_path),
    )
    assistant = WellbeingAssistant(
        config=config,
        llm_client=mock_llm,
        safety_layer=SafetyLayer(classifier=mock_clf),
        audit_logger=AuditLogger(log_path=audit_path),
    )
    client = TestClient(create_app(assistant=assistant))

    # Stage 1 clinical match (verified as non-crisis by Stage 2)
    resp_s1 = client.post(
        "/api/v1/check-in",
        json={"message": "Podes diagnosticar-me com perturbação de pânico?"},
    )
    assert resp_s1.status_code == 200
    body_s1 = resp_s1.json()
    assert body_s1["reply"] == OUT_OF_SCOPE_REFERRAL_MESSAGE
    assert body_s1["metadata"]["category"] == CATEGORY_OUT_OF_SCOPE
    assert body_s1["metadata"]["rule_triggered"] == RULE_CLINICAL_ADVICE
    assert body_s1["metadata"]["stage_decided"] == "STAGE_1"
    assert body_s1["metadata"]["llm_model"] == "NONE"
    assert body_s1["metadata"]["classifier_model"] == "gemini-3.5-flash-lite"

    # Stage 2 clinical match (nuanced clinical question that does not match Stage 1 regex)
    resp_s2 = client.post(
        "/api/v1/check-in",
        json={"message": "A melatonina vendida na ervanária cria habituação se usar todas as semanas?"},
    )
    assert resp_s2.status_code == 200
    body_s2 = resp_s2.json()
    assert body_s2["reply"] == OUT_OF_SCOPE_REFERRAL_MESSAGE
    assert body_s2["metadata"]["category"] == CATEGORY_OUT_OF_SCOPE
    assert body_s2["metadata"]["rule_triggered"] == RULE_CLASSIFIER_CLINICAL
    assert body_s2["metadata"]["stage_decided"] == "STAGE_2"
    assert body_s2["metadata"]["llm_model"] == "NONE"
    assert body_s2["metadata"]["classifier_model"] == "gemini-3.5-flash-lite"
    assert not mock_llm.generate.called


def test_api_classifier_failure_triggers_fail_safe(tmp_path: Path):
    """5. Classifier failure (path a): logs status ERROR, category SYSTEM_ERROR, rule CLASSIFIER_FAIL_SAFE, returns safe fallback."""
    audit_path = tmp_path / "clf_fail.jsonl"
    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_clf = MockRiskClassifier(
        responses=[ClassifierError("Classifier request timed out.", error_type="TimeoutError")]
    )
    config = Config(
        provider="mock",
        model_name="gemini-2.5-flash",
        classifier_provider="mock",
        classifier_model="gemini-3.5-flash-lite",
        audit_log_path=str(audit_path),
    )
    assistant = WellbeingAssistant(
        config=config,
        llm_client=mock_llm,
        safety_layer=SafetyLayer(classifier=mock_clf),
        audit_logger=AuditLogger(log_path=audit_path),
    )
    client = TestClient(create_app(assistant=assistant))

    response = client.post(
        "/api/v1/check-in",
        json={"message": "Hoje o dia foi bastante corrido na faculdade."},
    )

    assert not mock_llm.generate.called
    body = response.json()
    assert body["reply"] == SAFE_FALLBACK_MESSAGE
    assert "112" in body["reply"]
    assert "808 24 24 24" in body["reply"]
    assert "213 544 545" in body["reply"]

    meta = body["metadata"]
    assert meta["status"] == "ERROR"
    assert meta["category"] == CATEGORY_SYSTEM_ERROR
    assert meta["rule_triggered"] == RULE_CLASSIFIER_FAIL_SAFE
    assert meta["stage_decided"] == "STAGE_2"
    assert meta["llm_model"] == "NONE"
    assert meta["classifier_model"] == "gemini-3.5-flash-lite"
    assert meta["classifier_fail_safe"] is True
    assert meta["error_type"] == "TimeoutError"

    records = _read_jsonl(audit_path)
    assert len(records) == 1
    assert records[0]["status"] == "ERROR"
    assert records[0]["category"] == CATEGORY_SYSTEM_ERROR
    assert records[0]["rule_triggered"] == RULE_CLASSIFIER_FAIL_SAFE
    assert records[0]["classifier_fail_safe"] is True


def test_api_conversation_llm_failure_returns_503_and_logs_system_error(tmp_path: Path):
    """6. Conversation LLM failure (path b, AC-FR6.5): logs status ERROR, category SYSTEM_ERROR and returns HTTP 503 with safe fallback."""
    audit_path = tmp_path / "llm_fail.jsonl"
    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate.side_effect = RuntimeError("Downstream LLM connection reset")
    mock_clf = MockRiskClassifier(default_label="NONE")
    config = Config(
        provider="mock",
        model_name="gemini-2.5-flash",
        classifier_provider="mock",
        classifier_model="gemini-3.5-flash-lite",
        audit_log_path=str(audit_path),
    )
    assistant = WellbeingAssistant(
        config=config,
        llm_client=mock_llm,
        safety_layer=SafetyLayer(classifier=mock_clf),
        audit_logger=AuditLogger(log_path=audit_path),
    )
    client = TestClient(create_app(assistant=assistant))

    response = client.post(
        "/api/v1/check-in",
        json={"message": "Sinto-me um pouco nervoso com a reunião de amanhã."},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["reply"] == SAFE_FALLBACK_MESSAGE
    assert "112" in body["reply"]
    assert "808 24 24 24" in body["reply"]
    assert "213 544 545" in body["reply"]

    meta = body["metadata"]
    assert meta["status"] == "ERROR"
    assert meta["category"] == CATEGORY_SYSTEM_ERROR
    assert meta["rule_triggered"] == RULE_LLM_FAIL_SAFE
    assert meta["stage_decided"] == "STAGE_2"
    assert meta["llm_model"] == "gemini-2.5-flash"
    assert meta["classifier_model"] == "gemini-3.5-flash-lite"
    assert meta["classifier_fail_safe"] is False
    assert meta["error_type"] == "RuntimeError"

    records = _read_jsonl(audit_path)
    assert len(records) == 1
    assert records[0]["status"] == "ERROR"
    assert records[0]["category"] == CATEGORY_SYSTEM_ERROR
    assert records[0]["rule_triggered"] == RULE_LLM_FAIL_SAFE


@pytest.mark.parametrize(
    "invalid_payload,secret_marker",
    [
        ({"message": ""}, None),
        ({"message": "   \t\n  "}, None),
        ({"message": "SECRET_OVERLONG_INPUT_7711_" + ("x" * 1000)}, "SECRET_OVERLONG_INPUT_7711_"),
        (
            {"message": "SECRET_VALID_MSG_8822", "session_id": "SECRET_BAD_SESSION_UUID_9933"},
            "SECRET_BAD_SESSION_UUID_9933",
        ),
    ],
)
def test_api_invalid_input_returns_422_without_echoing_input(
    tmp_path: Path,
    invalid_payload: dict,
    secret_marker: str | None,
):
    """7. Invalid input (AC-FR6.4, Scope #5): returns HTTP 422 without echoing submitted input value."""
    audit_path = tmp_path / "invalid_input.jsonl"
    config = Config(audit_log_path=str(audit_path))
    assistant = WellbeingAssistant(config=config, audit_logger=AuditLogger(log_path=audit_path))
    client = TestClient(create_app(assistant=assistant))

    response = client.post("/api/v1/check-in", json=invalid_payload)

    assert response.status_code == 422
    raw_body = response.text
    parsed_body = response.json()
    assert "detail" in parsed_body
    for err_item in parsed_body["detail"]:
        assert "input" not in err_item
        assert "ctx" not in err_item

    if secret_marker is not None:
        assert secret_marker not in raw_body
    assert "SECRET_VALID_MSG_8822" not in raw_body


def test_no_audit_record_contains_message_text_across_all_paths(tmp_path: Path, caplog):
    """8. Privacy guardrail (AC-FR5.2): no audit record in memory, JSONL sink, or logs contains raw message text."""
    import logging

    caplog.set_level(logging.DEBUG)
    audit_path = tmp_path / "privacy_audit.jsonl"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate.side_effect = [
        "Compreendo que tenha sido um dia exigente. Respira fundo.",
        RuntimeError("LLM crashed with error"),
    ]
    mock_clf = MockRiskClassifier(
        responses=[
            "NONE",  # Turn 1: standard check-in
            LABEL_CRISIS,  # Turn 3: Stage 2 crisis
            LABEL_CLINICAL,  # Turn 4: Stage 2 clinical
            ClassifierError("timeout", error_type="TimeoutError"),  # Turn 5: classifier failure
            "NONE",  # Turn 6: conversation LLM failure
        ]
    )
    config = Config(
        provider="mock",
        model_name="mock-conv-v1",
        classifier_provider="mock",
        classifier_model="gemini-3.5-flash-lite",
        audit_log_path=str(audit_path),
    )
    assistant = WellbeingAssistant(
        config=config,
        llm_client=mock_llm,
        safety_layer=SafetyLayer(classifier=mock_clf),
        audit_logger=AuditLogger(log_path=audit_path),
    )
    client = TestClient(create_app(assistant=assistant))

    secrets = [
        ("SECRET_TURN1_STANDARD_101", "SECRET_TURN1_STANDARD_101 Hoje trabalhei até tarde."),
        ("SECRET_TURN2_STAGE1_CRISIS_202", "SECRET_TURN2_STAGE1_CRISIS_202 Quero matar-me agora."),
        ("SECRET_TURN3_STAGE2_CRISIS_303", "SECRET_TURN3_STAGE2_CRISIS_303 Despedi-me de todos hoje."),
        ("SECRET_TURN4_CLINICAL_404", "SECRET_TURN4_CLINICAL_404 Tomo dois comprimidos hoje?"),
        ("SECRET_TURN5_CLF_FAIL_505", "SECRET_TURN5_CLF_FAIL_505 Tive uma tarde difícil."),
        ("SECRET_TURN6_LLM_FAIL_606", "SECRET_TURN6_LLM_FAIL_606 Sinto-me cansado da semana."),
    ]

    for _, full_message in secrets:
        client.post("/api/v1/check-in", json={"message": full_message})

    in_memory_dump = json.dumps(assistant.audit_logger.entries)
    jsonl_raw = audit_path.read_text(encoding="utf-8")
    jsonl_records = _read_jsonl(audit_path)
    assert len(jsonl_records) == 6
    assert len(assistant.audit_logger.entries) == 6

    for marker, full_message in secrets:
        assert marker not in in_memory_dump
        assert full_message not in in_memory_dump
        assert marker not in jsonl_raw
        assert full_message not in jsonl_raw
        assert marker not in caplog.text
