import json
from unittest.mock import MagicMock, patch
import pytest
from src.config import Config
from src.assistant import WellbeingAssistant, get_session_header, main
from src.llm import BaseLLMClient
from src.safety import (
    CRISIS_RESPONSE_MESSAGE,
    OUT_OF_SCOPE_REFERRAL_MESSAGE,
    CATEGORY_CRISIS,
    CATEGORY_OUT_OF_SCOPE,
    RULE_CRISIS_HARM,
    RULE_CLINICAL_ADVICE,
)
from tests.test_safety import CRISIS_SCENARIOS, OUT_OF_SCOPE_SCENARIOS


def test_assistant_greet_initiates_checkin():
    """Verify that greet() welcomes user and asks how they feel (FR1.1)."""
    assistant = WellbeingAssistant()
    greeting = assistant.greet(language="pt")
    assert isinstance(greeting, str)
    assert len(greeting) > 0
    # Greeting in Portuguese asks how user is feeling
    assert any(w in greeting.lower() for w in ["olá", "como", "sente", "bem-vindo"])

    greeting_en = assistant.greet(language="en")
    assert any(w in greeting_en.lower() for w in ["hello", "how", "feeling", "welcome"])


def test_assistant_respond_calls_llm_with_system_prompt():
    """Verify respond() calls the LLM with user input and supportive system prompt."""
    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate.return_value = "Lamento que estejas a ter um dia difícil. Como posso apoiar-te hoje?"

    config = Config(provider="mock", model_name="test-model", temperature=0.5)
    assistant = WellbeingAssistant(config=config, llm_client=mock_llm)

    user_message = "Tive um dia de trabalho muito stressante."
    reply = assistant.respond(user_message)

    assert reply == "Lamento que estejas a ter um dia difícil. Como posso apoiar-te hoje?"
    assert mock_llm.generate.called
    call_args = mock_llm.generate.call_args
    assert user_message in call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")


def test_assistant_default_mock_llm_execution():
    """Verify default initialization works out-of-the-box with mock client."""
    assistant = WellbeingAssistant()
    reply = assistant.respond("Estou um pouco cansado hoje.")
    assert isinstance(reply, str)
    assert len(reply) > 0


@pytest.mark.parametrize(
    "message,expected_reply,expected_category,expected_rule",
    [
        (
            "Quero morrer, já não aguento mais viver assim.",
            CRISIS_RESPONSE_MESSAGE,
            CATEGORY_CRISIS,
            RULE_CRISIS_HARM,
        ),
        (
            "Devo tomar sertralina para a ansiedade?",
            OUT_OF_SCOPE_REFERRAL_MESSAGE,
            CATEGORY_OUT_OF_SCOPE,
            RULE_CLINICAL_ADVICE,
        ),
    ],
)
def test_assistant_input_safety_triggers_bypass_llm_and_log(
    message, expected_reply, expected_category, expected_rule
):
    """AC-FR3.1, AC-FR3.4, AC-FR4.4, AC-FR5.1, AC-FR5.2: Input safety triggers bypass LLM and log without raw text."""
    mock_llm = MagicMock(spec=BaseLLMClient)
    config = Config(provider="mock", model_name="test-model")
    assistant = WellbeingAssistant(config=config, llm_client=mock_llm)

    reply = assistant.respond(message)

    assert not mock_llm.generate.called
    assert reply == expected_reply
    assert len(assistant.audit_logger.entries) == 1

    log_entry = assistant.audit_logger.entries[0]
    assert log_entry["status"] == "OK"
    assert log_entry["category"] == expected_category
    assert log_entry["rule_triggered"] == expected_rule
    assert log_entry["model_version"] == "NONE"
    assert message not in json.dumps(log_entry)


def test_assistant_all_crisis_and_out_of_scope_scenarios_bypass_llm():
    """Verify 100% of crisis and out-of-scope scenarios bypass LLM through WellbeingAssistant."""
    mock_llm = MagicMock(spec=BaseLLMClient)
    assistant = WellbeingAssistant(llm_client=mock_llm)

    for scenario in CRISIS_SCENARIOS:
        assert assistant.respond(scenario) == CRISIS_RESPONSE_MESSAGE
    for scenario in OUT_OF_SCOPE_SCENARIOS:
        assert assistant.respond(scenario) == OUT_OF_SCOPE_REFERRAL_MESSAGE

    assert not mock_llm.generate.called
    assert len(assistant.audit_logger.entries) == len(CRISIS_SCENARIOS) + len(OUT_OF_SCOPE_SCENARIOS)


def test_assistant_output_screening_blocks_unsafe_llm_reply_and_logs_trigger():
    """AC-FR4.5: Output screening blocks unsafe LLM reply and replaces it with FR4 referral."""
    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate.return_value = "Tens depressão clínica. Recomendo que tomes sertralina 50mg por dia."
    config = Config(provider="mock", model_name="test-model")
    assistant = WellbeingAssistant(config=config, llm_client=mock_llm)

    benign_msg = "Tenho andado muito desmotivado ultimamente."
    reply = assistant.respond(benign_msg)

    assert mock_llm.generate.called
    assert reply == OUT_OF_SCOPE_REFERRAL_MESSAGE
    assert len(assistant.audit_logger.entries) == 1

    log_entry = assistant.audit_logger.entries[0]
    assert log_entry["status"] == "OK"
    assert log_entry["category"] == CATEGORY_OUT_OF_SCOPE
    assert log_entry["rule_triggered"] == RULE_CLINICAL_ADVICE
    assert log_entry["model_version"] == "test-model"
    assert benign_msg not in json.dumps(log_entry)


def test_mock_mode_session_header_shows_model_mock(capsys, monkeypatch):
    """Verify session header displays 'Model: mock' in mock mode."""
    config = Config(provider="mock")
    header = get_session_header(config)
    assert "Model: mock" in header
    assert header == "Provider: mock | Model: mock"

    # Also test interactive main() output
    monkeypatch.setattr("builtins.input", lambda _: "exit")
    with patch("src.assistant.get_config", return_value=config):
        main()

    captured = capsys.readouterr().out
    assert "Provider: mock | Model: mock" in captured


def test_gemini_without_llm_model_stops_startup_in_assistant():
    """Verify WellbeingAssistant stops at startup if provider is gemini and LLM_MODEL is not set."""
    config = Config(provider="gemini", model_name=None)
    with pytest.raises(ValueError, match=r"LLM_MODEL is not set.*Please set LLM_MODEL in your \.env file"):
        WellbeingAssistant(config=config)


def test_gemini_without_llm_model_stops_startup_in_main(capsys):
    """Verify main() stops at startup with error message before conversation begins when LLM_MODEL missing."""
    config = Config(provider="gemini", model_name=None)
    with patch("src.assistant.get_config", return_value=config):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    captured = capsys.readouterr().out
    assert "Error: LLM_MODEL is not set. Please set LLM_MODEL in your .env file" in captured
    assert "Assistant:" not in captured


def test_assistant_stage2_classifier_triggers_and_fail_safe_bypass_conversation_llm():
    """Verify Stage 2 CRISIS, CLINICAL, and SYSTEM_ERROR fail-safe bypass conversational LLM and audit log accurately."""
    from src.classifier import ClassifierError, LABEL_CLINICAL, LABEL_CRISIS, MockRiskClassifier
    from src.safety import (
        CATEGORY_SYSTEM_ERROR,
        RULE_CLASSIFIER_CLINICAL,
        RULE_CLASSIFIER_CRISIS,
        RULE_CLASSIFIER_FAIL_SAFE,
        SAFE_FALLBACK_MESSAGE,
        SafetyLayer,
    )

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_clf = MockRiskClassifier(
        responses=[
            LABEL_CRISIS,
            LABEL_CLINICAL,
            ClassifierError("timeout", error_type="TimeoutError"),
        ]
    )
    config = Config(provider="mock", classifier_provider="mock", classifier_model="gemini-3.5-flash-lite")
    assistant = WellbeingAssistant(
        config=config,
        llm_client=mock_llm,
        safety_layer=SafetyLayer(classifier=mock_clf),
    )

    # 1. Stage 2 CRISIS
    secret_msg_1 = "PRIVATE_MSG_A1 acho q hoje é o último check-in q faço"
    reply_1 = assistant.respond(secret_msg_1)
    assert reply_1 == CRISIS_RESPONSE_MESSAGE

    # 2. Stage 2 CLINICAL
    secret_msg_2 = "PRIVATE_MSG_A2 esqueci-me da sertralina ontem, tomo 2 hj?"
    reply_2 = assistant.respond(secret_msg_2)
    assert reply_2 == OUT_OF_SCOPE_REFERRAL_MESSAGE

    # 3. Stage 2 Failure -> SYSTEM_ERROR fail-safe
    secret_msg_3 = "PRIVATE_MSG_A3 mensagem durante falha de rede"
    reply_3 = assistant.respond(secret_msg_3)
    assert reply_3 == SAFE_FALLBACK_MESSAGE
    assert "112" in reply_3
    assert "SNS 24" in reply_3
    assert "SOS Voz Amiga" in reply_3

    assert not mock_llm.generate.called
    assert len(assistant.audit_logger.entries) == 3

    e1, e2, e3 = assistant.audit_logger.entries
    assert e1["status"] == "OK"
    assert e1["category"] == CATEGORY_CRISIS
    assert e1["rule_triggered"] == RULE_CLASSIFIER_CRISIS
    assert e1["model_version"] == "gemini-3.5-flash-lite"

    assert e2["status"] == "OK"
    assert e2["category"] == CATEGORY_OUT_OF_SCOPE
    assert e2["rule_triggered"] == RULE_CLASSIFIER_CLINICAL
    assert e2["model_version"] == "gemini-3.5-flash-lite"

    assert e3["status"] == "ERROR"
    assert e3["category"] == CATEGORY_SYSTEM_ERROR
    assert e3["rule_triggered"] == RULE_CLASSIFIER_FAIL_SAFE
    assert e3["model_version"] == "gemini-3.5-flash-lite"

    serialized_logs = json.dumps(assistant.audit_logger.entries)
    assert "PRIVATE_MSG_A1" not in serialized_logs
    assert "PRIVATE_MSG_A2" not in serialized_logs
    assert "PRIVATE_MSG_A3" not in serialized_logs


@pytest.mark.parametrize(
    "classifier_model,api_key,expected_match",
    [
        ("", "valid-key", "CLASSIFIER_MODEL is not set"),
        ("gemini-3.5-flash-lite", "", "LLM_API_KEY is not set"),
    ],
)
def test_classifier_gemini_without_model_or_key_stops_startup(classifier_model, api_key, expected_match, capsys):
    """Startup fails fast with CLASSIFIER_PROVIDER='gemini' and no key or model in both WellbeingAssistant and main()."""
    config = Config(
        provider="mock",
        classifier_provider="gemini",
        classifier_model=classifier_model,
        api_key=api_key,
    )
    with pytest.raises(ValueError, match=expected_match):
        WellbeingAssistant(config=config)

    with patch("src.assistant.get_config", return_value=config):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    captured = capsys.readouterr().out
    assert f"Error: {expected_match}" in captured
    assert "Assistant:" not in captured
