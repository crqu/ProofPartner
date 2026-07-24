"""Resilience utilities — backoff, validation gates, graceful degradation."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ValidationError

from agentic_research.logging import get_logger
from agentic_research.models.agents import LLMResponse
from agentic_research.models.tools import CompilationResult, ToolStatus
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exponential backoff for Lean REPL
# ---------------------------------------------------------------------------


class ReplBackoffConfig:
    __slots__ = ("base_delay", "factor", "max_delay", "max_attempts")

    def __init__(
        self,
        base_delay: float = 1.0,
        factor: float = 2.0,
        max_delay: float = 60.0,
        max_attempts: int = 3,
    ) -> None:
        self.base_delay = base_delay
        self.factor = factor
        self.max_delay = max_delay
        self.max_attempts = max_attempts


class ResilientRepl:
    """Wraps a LeanRepl with exponential backoff on transient errors."""

    def __init__(
        self,
        repl: LeanRepl,
        backoff: ReplBackoffConfig | None = None,
    ) -> None:
        self._repl = repl
        self._backoff = backoff or ReplBackoffConfig()
        self._health_failures = 0
        self._unavailable = False

    @property
    def is_unavailable(self) -> bool:
        return self._unavailable

    def execute_with_backoff(self, code: str) -> CompilationResult:
        if self._unavailable:
            return CompilationResult(
                status=ToolStatus.UNAVAILABLE,
                error_message="Lean REPL is unavailable after repeated health check failures",
            )

        last_result: CompilationResult | None = None
        for attempt in range(1, self._backoff.max_attempts + 1):
            result = self._repl.execute(code)
            if result.status not in (ToolStatus.ERROR, ToolStatus.TIMEOUT):
                self._health_failures = 0
                return result

            last_result = result
            self._health_failures += 1

            if result.status == ToolStatus.ERROR:
                err_msg = (result.error_message or "").lower()
                is_transient = any(
                    kw in err_msg
                    for kw in ("process", "timeout", "connection", "transient", "crash")
                )
                if not is_transient:
                    return result
            if self._health_failures >= 3:
                self._unavailable = True
                log.warning(
                    "repl_unavailable",
                    health_failures=self._health_failures,
                )
                return CompilationResult(
                    status=ToolStatus.UNAVAILABLE,
                    error_message="Lean REPL marked unavailable after 3 consecutive failures",
                )

            if attempt < self._backoff.max_attempts:
                delay = min(
                    self._backoff.base_delay * (self._backoff.factor ** (attempt - 1)),
                    self._backoff.max_delay,
                )
                log.warning(
                    "repl_backoff_retry",
                    attempt=attempt,
                    delay=delay,
                    error=result.error_message,
                )
                time.sleep(delay)

        return last_result or CompilationResult(
            status=ToolStatus.ERROR,
            error_message="REPL retries exhausted",
        )

    def reset_health(self) -> None:
        self._health_failures = 0
        self._unavailable = False


# ---------------------------------------------------------------------------
# LLM output validation gates
# ---------------------------------------------------------------------------

VALID_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "tool_use"})


class ValidationError_(Exception):
    """Raised when LLM output fails a validation gate."""

    def __init__(self, gate: str, detail: str) -> None:
        self.gate = gate
        self.detail = detail
        super().__init__(f"Validation gate [{gate}] failed: {detail}")


def validate_llm_response(
    response: LLMResponse,
    schema: type[BaseModel] | None = None,
    parsed_data: dict[str, Any] | None = None,
) -> list[str]:
    """Run validation gates on an LLM response. Returns list of error strings (empty = pass)."""
    errors: list[str] = []

    if response.stop_reason and response.stop_reason not in VALID_STOP_REASONS:
        errors.append(
            f"Invalid stop_reason '{response.stop_reason}' — response may be truncated"
        )

    if not response.content.strip():
        errors.append("Empty response content")

    if schema is not None and parsed_data is not None:
        try:
            schema.model_validate(parsed_data)
        except ValidationError as exc:
            errors.append(f"Schema validation failed: {exc.error_count()} errors")

    if errors:
        log.warning("llm_validation_gate_failed", errors=errors)
    else:
        log.debug("llm_validation_gate_passed")

    return errors
