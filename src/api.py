"""FastAPI application exposing the AI Wellbeing Check-in Assistant endpoint (FR5, FR6, Issue #5)."""

from typing import Optional
import uuid
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.assistant import WellbeingAssistant
from src.config import Config


class CheckInRequest(BaseModel):
    """Request payload schema for POST /api/v1/check-in (AC-FR6.2, AC-FR6.4)."""

    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def validate_not_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must not be empty or whitespace only.")
        return value

    @field_validator("session_id")
    @classmethod
    def validate_session_id_uuid(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                uuid.UUID(str(value))
            except (ValueError, TypeError) as exc:
                raise ValueError("session_id must be a valid UUID.") from exc
        return value


class CheckInMetadata(BaseModel):
    """Structured audit and decision trace metadata returned by POST /api/v1/check-in (AC-FR5.1, AC-FR6.3)."""

    interaction_id: str
    timestamp: str
    status: str
    category: str
    rule_triggered: str
    stage_decided: str
    llm_model: str
    classifier_model: str
    classifier_latency_ms: float
    latency_ms: float
    classifier_fail_safe: bool
    error_type: Optional[str] = None


class CheckInResponse(BaseModel):
    """Response payload schema for POST /api/v1/check-in (AC-FR6.3)."""

    reply: str
    metadata: CheckInMetadata


_SAFE_VALIDATION_MESSAGES = {
    "string_too_short": "String should have at least 1 character.",
    "string_too_long": "String should have at most 1000 characters.",
    "missing": "Field required.",
    "json_invalid": "Invalid JSON payload.",
    "string_type": "Input should be a valid string.",
}


def create_app(
    assistant: Optional[WellbeingAssistant] = None,
    config: Optional[Config] = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    fastapi_app = FastAPI(
        title="AI Wellbeing Check-in Assistant API",
        version="0.1.0",
    )
    fastapi_app.state.assistant = assistant
    fastapi_app.state.config = config

    def _get_assistant() -> WellbeingAssistant:
        if fastapi_app.state.assistant is None:
            fastapi_app.state.assistant = WellbeingAssistant(config=fastapi_app.state.config)
        return fastapi_app.state.assistant

    @fastapi_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return HTTP 422 without echoing submitted user input or raw values (AC-FR5.2, AC-FR6.4)."""
        sanitized_errors = []
        for err in exc.errors():
            err_type = str(err.get("type", "value_error"))
            raw_msg = str(err.get("msg", "Invalid input."))
            if err_type in _SAFE_VALIDATION_MESSAGES:
                safe_msg = _SAFE_VALIDATION_MESSAGES[err_type]
            elif "whitespace only" in raw_msg:
                safe_msg = "Message must not be empty or whitespace only."
            elif "valid UUID" in raw_msg:
                safe_msg = "session_id must be a valid UUID."
            else:
                safe_msg = "Invalid request input."
            sanitized_errors.append(
                {
                    "loc": list(err.get("loc", ())),
                    "msg": safe_msg,
                    "type": err_type,
                }
            )
        return JSONResponse(
            status_code=422,
            content={"detail": sanitized_errors},
        )

    @fastapi_app.post("/api/v1/check-in", response_model=CheckInResponse)
    def check_in(payload: CheckInRequest):
        svc = _get_assistant()
        turn = svc.respond_turn(payload.message)
        response_model = CheckInResponse(
            reply=turn.reply,
            metadata=CheckInMetadata(**turn.metadata),
        )
        if turn.llm_error:
            return JSONResponse(
                status_code=503,
                content=response_model.model_dump(),
            )
        return response_model

    return fastapi_app


app = create_app()
