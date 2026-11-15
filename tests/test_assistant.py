import json
from unittest.mock import MagicMock
from src.config import Config
from src.assistant import WellbeingAssistant
from src.llm import BaseLLMClient
from src.safety import (
    CRISIS_RESPONSE_MESSAGE,
    OUT_OF_SCOPE_REFERRAL_MESSAGE,
    CATEGORY_CRISIS,
    CATEGORY_OUT_OF_SCOPE,
    RULE_CRISIS_HARM,
    RULE_CLINICAL_ADVICE,
    RULE_OUTPUT_CLINICAL_ADVICE,
)


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


def test_assistant_crisis_bypasses_llm_and_logs_trigger():
    """AC-FR3.1, AC-FR3.4, AC-FR5.1, AC-FR5.2: Crisis message bypasses LLM and logs trigger without raw text."""
    mock_llm = MagicMock(spec=BaseLLMClient)
    config = Config(provider="mock", model_name="test-model")
    assistant = WellbeingAssistant(config=config, llm_client=mock_llm)

    crisis_msg = "Quero morrer, já não aguento mais viver assim."
    reply = assistant.respond(crisis_msg)

    assert not mock_llm.generate.called
    assert reply == CRISIS_RESPONSE_MESSAGE
    assert len(assistant.audit_logger.entries) == 1

    log_entry = assistant.audit_logger.entries[0]
    assert log_entry["status"] == "OK"
    assert log_entry["category"] == CATEGORY_CRISIS
    assert log_entry["rule_triggered"] == RULE_CRISIS_HARM
    assert log_entry["model_version"] == "NONE"
    assert crisis_msg not in json.dumps(log_entry)


def test_assistant_out_of_scope_bypasses_llm_and_logs_trigger():
    """AC-FR4.1, AC-FR4.2, AC-FR4.4: Out-of-scope clinical query bypasses LLM and logs trigger."""
    mock_llm = MagicMock(spec=BaseLLMClient)
    config = Config(provider="mock", model_name="test-model")
    assistant = WellbeingAssistant(config=config, llm_client=mock_llm)

    clinical_msg = "Devo tomar sertralina para a ansiedade?"
    reply = assistant.respond(clinical_msg)

    assert not mock_llm.generate.called
    assert reply == OUT_OF_SCOPE_REFERRAL_MESSAGE
    assert len(assistant.audit_logger.entries) == 1

    log_entry = assistant.audit_logger.entries[0]
    assert log_entry["status"] == "OK"
    assert log_entry["category"] == CATEGORY_OUT_OF_SCOPE
    assert log_entry["rule_triggered"] == RULE_CLINICAL_ADVICE
    assert log_entry["model_version"] == "NONE"
    assert clinical_msg not in json.dumps(log_entry)


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
    assert log_entry["rule_triggered"] == RULE_OUTPUT_CLINICAL_ADVICE
    assert log_entry["model_version"] == "test-model"
    assert benign_msg not in json.dumps(log_entry)

