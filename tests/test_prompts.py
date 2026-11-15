from src.prompts import SYSTEM_PROMPT, format_user_prompt, DEFAULT_GREETING_PT, DEFAULT_GREETING_EN


def test_system_prompt_principles():
    """Verify that SYSTEM_PROMPT embodies FR1, FR2, and non-clinical principles."""
    prompt_lower = SYSTEM_PROMPT.lower()
    # Empathy and non-judgmental tone (FR2)
    assert "empathy" in prompt_lower or "empathetic" in prompt_lower
    assert "toxic positivity" in prompt_lower
    assert "concise" in prompt_lower or "120 words" in prompt_lower
    # Non-clinical boundary (FR4)
    assert "not a medical device" in prompt_lower or "never offer medical diagnoses" in prompt_lower
    # Off-topic redirection (FR1.3)
    assert "redirect" in prompt_lower
    # Languages
    assert "pt-pt" in prompt_lower and "english" in prompt_lower


def test_format_user_prompt_without_history():
    """Verify prompt formatting when no prior history is supplied."""
    msg = "Hoje sinto-me calmo."
    assert format_user_prompt(msg) == msg


def test_format_user_prompt_with_history():
    """Verify multi-turn history formatting."""
    history = [
        {"role": "user", "content": "Olá"},
        {"role": "assistant", "content": "Olá! Como te sentes?"},
    ]
    formatted = format_user_prompt("Sinto-me um pouco ansioso.", history)
    assert "User: Olá" in formatted
    assert "Assistant: Olá! Como te sentes?" in formatted
    assert "User: Sinto-me um pouco ansioso." in formatted
    assert formatted.endswith("Assistant:")
