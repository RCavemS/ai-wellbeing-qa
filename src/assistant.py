"""Wellbeing Assistant service coordinating conversation flow and LLM interactions."""

import time
from typing import Optional
from src.config import Config, get_config
from src.llm import BaseLLMClient, get_llm_client
from src.prompts import (
    SYSTEM_PROMPT,
    DEFAULT_GREETING_PT,
    DEFAULT_GREETING_EN,
    format_user_prompt,
)
from src.safety import (
    AuditLogger,
    SafetyLayer,
    CATEGORY_STANDARD,
    RULE_NONE,
)


class WellbeingAssistant:
    """Core AI Wellbeing Check-in Assistant."""

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_client: Optional[BaseLLMClient] = None,
        safety_layer: Optional[SafetyLayer] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.config = config or get_config()
        self.llm_client = llm_client or get_llm_client(self.config)
        self.safety_layer = safety_layer or SafetyLayer()
        self.audit_logger = audit_logger or AuditLogger()

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
        """Process user message through safety layer and LLM (FR1, FR2, FR3, FR4, FR5)."""
        start_time = time.perf_counter()

        # 1. Input screening (FR3 Crisis Escalation & FR4 Out-of-Scope Clinical)
        input_safety = self.safety_layer.screen_input(message)
        if input_safety.triggered:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.audit_logger.log(
                status="OK",
                category=input_safety.category,
                rule_triggered=input_safety.rule_triggered,
                model_version="NONE",
                latency_ms=latency_ms,
            )
            return input_safety.response or ""

        # 2. LLM generation for standard check-in (FR1, FR2)
        prompt = format_user_prompt(message=message, history=session_history)
        reply = self.llm_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # 3. Output screening guardrail (AC-FR4.5)
        output_safety = self.safety_layer.screen_output(reply)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        if output_safety.triggered:
            self.audit_logger.log(
                status="OK",
                category=output_safety.category,
                rule_triggered=output_safety.rule_triggered,
                model_version=self.config.model_name,
                latency_ms=latency_ms,
            )
            return output_safety.response or ""

        self.audit_logger.log(
            status="OK",
            category=CATEGORY_STANDARD,
            rule_triggered=RULE_NONE,
            model_version=self.config.model_name,
            latency_ms=latency_ms,
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
