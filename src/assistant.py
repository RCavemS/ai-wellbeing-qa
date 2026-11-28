"""Wellbeing Assistant service coordinating conversation flow and LLM interactions."""

from dataclasses import dataclass
import sys
import time
from typing import Any, Optional
from src.classifier import BaseRiskClassifier, validate_classifier_config
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
    CATEGORY_CRISIS,
    CATEGORY_STANDARD,
    CATEGORY_SYSTEM_ERROR,
    RULE_CLASSIFIER_FAIL_SAFE,
    RULE_LLM_FAIL_SAFE,
    RULE_NONE,
    SAFE_FALLBACK_MESSAGE,
    SafetyLayer,
)


@dataclass(frozen=True)
class AssistantTurnResult:
    """Result of a single assistant interaction turn including reply and audit metadata."""

    reply: str
    metadata: dict[str, Any]
    llm_error: bool = False


class WellbeingAssistant:
    """Core AI Wellbeing Check-in Assistant."""

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_client: Optional[BaseLLMClient] = None,
        safety_layer: Optional[SafetyLayer] = None,
        audit_logger: Optional[AuditLogger] = None,
        classifier: Optional[BaseRiskClassifier] = None,
    ):
        self.config = config or get_config()
        if self.config.provider.lower().strip() == "gemini" and not self.config.model_name:
            raise ValueError(
                "LLM_MODEL is not set. Please set LLM_MODEL in your .env file before starting."
            )
        self.llm_client = llm_client or get_llm_client(self.config)
        self.safety_layer = safety_layer or SafetyLayer(classifier=classifier, config=self.config)
        self.audit_logger = audit_logger or AuditLogger(log_path=self.config.audit_log_path)

    def greet(self, language: str = "pt") -> str:
        """Initiate check-in flow with a supportive greeting (FR1.1)."""
        lang = language.lower().strip()
        if lang.startswith("en"):
            return DEFAULT_GREETING_EN
        return DEFAULT_GREETING_PT

    def _resolve_llm_model(self, bypassed: bool) -> str:
        if bypassed:
            return "NONE"
        if self.config.model_name and str(self.config.model_name).strip():
            return str(self.config.model_name).strip()
        if (self.config.provider or "mock").lower().strip() == "mock":
            return "mock"
        return "NONE"

    def _resolve_classifier_model(self, bypassed: bool) -> str:
        if bypassed:
            return "NONE"
        if self.config.classifier_model and str(self.config.classifier_model).strip():
            return str(self.config.classifier_model).strip()
        if (self.config.classifier_provider or "mock").lower().strip() == "mock":
            return "mock"
        return "NONE"

    def respond_turn(
        self,
        message: str,
        session_history: Optional[list[dict]] = None,
    ) -> AssistantTurnResult:
        """Process user message through safety layer and LLM, returning reply and audit metadata (FR1–FR5)."""
        start_time = time.perf_counter()

        # 1. Input screening (FR3 Crisis Escalation & FR4 Out-of-Scope Clinical)
        input_safety = self.safety_layer.screen_input(message)
        stage2_bypassed = (
            input_safety.stage_decided == "STAGE_1" and input_safety.category == CATEGORY_CRISIS
        )
        classifier_model_used = self._resolve_classifier_model(bypassed=stage2_bypassed)
        clf_fail_safe_used = (
            input_safety.rule_triggered == RULE_CLASSIFIER_FAIL_SAFE
            or input_safety.error_type is not None
        )

        if input_safety.triggered:
            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            entry = self.audit_logger.log(
                status=input_safety.status,
                category=input_safety.category,
                rule_triggered=input_safety.rule_triggered,
                stage_decided=input_safety.stage_decided,
                llm_model=self._resolve_llm_model(bypassed=True),
                classifier_model=classifier_model_used,
                classifier_latency_ms=input_safety.latency_ms,
                latency_ms=total_latency_ms,
                classifier_fail_safe=clf_fail_safe_used,
                error_type=input_safety.error_type,
            )
            return AssistantTurnResult(
                reply=input_safety.response or "",
                metadata=entry,
                llm_error=False,
            )

        # 2. LLM generation for standard check-in (FR1, FR2)
        llm_model_used = self._resolve_llm_model(bypassed=False)
        prompt = format_user_prompt(message=message, history=session_history)
        try:
            reply = self.llm_client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as exc:
            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            entry = self.audit_logger.log(
                status="ERROR",
                category=CATEGORY_SYSTEM_ERROR,
                rule_triggered=RULE_LLM_FAIL_SAFE,
                stage_decided=input_safety.stage_decided,
                llm_model=llm_model_used,
                classifier_model=classifier_model_used,
                classifier_latency_ms=input_safety.latency_ms,
                latency_ms=total_latency_ms,
                classifier_fail_safe=False,
                error_type=type(exc).__name__,
            )
            return AssistantTurnResult(
                reply=SAFE_FALLBACK_MESSAGE,
                metadata=entry,
                llm_error=True,
            )

        # 3. Output screening guardrail (AC-FR4.5)
        output_safety = self.safety_layer.screen_output(reply)
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        if output_safety.triggered:
            entry = self.audit_logger.log(
                status=output_safety.status,
                category=output_safety.category,
                rule_triggered=output_safety.rule_triggered,
                stage_decided=output_safety.stage_decided,
                llm_model=llm_model_used,
                classifier_model=classifier_model_used,
                classifier_latency_ms=input_safety.latency_ms,
                latency_ms=total_latency_ms,
                classifier_fail_safe=False,
                error_type=None,
            )
            return AssistantTurnResult(
                reply=output_safety.response or "",
                metadata=entry,
                llm_error=False,
            )

        # 4. Standard check-in audit record (AC-FR5.1)
        entry = self.audit_logger.log(
            status="OK",
            category=CATEGORY_STANDARD,
            rule_triggered=RULE_NONE,
            stage_decided=input_safety.stage_decided,
            llm_model=llm_model_used,
            classifier_model=classifier_model_used,
            classifier_latency_ms=input_safety.latency_ms,
            latency_ms=total_latency_ms,
            classifier_fail_safe=False,
            error_type=None,
        )
        return AssistantTurnResult(
            reply=reply,
            metadata=entry,
            llm_error=False,
        )

    def respond(
        self,
        message: str,
        session_history: Optional[list[dict]] = None,
    ) -> str:
        """Process user message through safety layer and LLM (FR1, FR2, FR3, FR4)."""
        return self.respond_turn(message=message, session_history=session_history).reply


def get_session_header(config: Config) -> str:
    """Format session header line."""
    model_display = "mock" if config.provider == "mock" else (config.model_name or "None")
    return f"Provider: {config.provider} | Model: {model_display}"


def main():
    """Run local interactive check-in session from the terminal."""
    print("=== AI Wellbeing Check-in Assistant (Local Session) ===")
    config = get_config()
    print(get_session_header(config))

    if config.provider.lower().strip() == "gemini" and not config.model_name:
        print("Error: LLM_MODEL is not set. Please set LLM_MODEL in your .env file before starting.")
        sys.exit(1)

    try:
        validate_classifier_config(config)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

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
