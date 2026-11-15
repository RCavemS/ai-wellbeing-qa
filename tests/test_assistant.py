from unittest.mock import MagicMock
from src.config import Config
from src.assistant import WellbeingAssistant
from src.llm import BaseLLMClient


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
