"""Wellbeing Assistant service coordinating conversation flow and LLM interactions."""

from typing import Optional
from src.config import Config, get_config
from src.llm import BaseLLMClient, get_llm_client
from src.prompts import (
    SYSTEM_PROMPT,
    DEFAULT_GREETING_PT,
    DEFAULT_GREETING_EN,
    format_user_prompt,
)


class WellbeingAssistant:
    """Core AI Wellbeing Check-in Assistant."""

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.config = config or get_config()
        self.llm_client = llm_client or get_llm_client(self.config)

    def greet(self, language: str = "pt") -> str:
        """Initiate check-in flow with a supportive greeting (FR1.1)."""
        lang = language.lower().strip()
        if lang.startswith("en"):
            return DEFAULT_GREETING_EN
        return DEFAULT_GREETING_PT

    def respond(
        self,
        message: str,
        session_history: Optional[list[dict]] = None,
    ) -> str:
        """Process user message and generate empathetic check-in response (FR1, FR2)."""
        prompt = format_user_prompt(message=message, history=session_history)
        reply = self.llm_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        return reply


def main():
    """Run local interactive check-in session from the terminal."""
    print("=== AI Wellbeing Check-in Assistant (Local Session) ===")
    config = get_config()
    print(f"Provider: {config.provider} | Model: {config.model_name}")
    print("Type 'exit' or 'quit' to end the session.\n")

    assistant = WellbeingAssistant(config=config)
    print(f"Assistant: {assistant.greet('pt')}\n")

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "sair"]:
                print("Assistant: Cuida de ti. Até breve!")
                break

            reply = assistant.respond(user_input, session_history=history)
            print(f"\nAssistant: {reply}\n")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    main()
