"""Base Agent class with run(context) -> AgentResult protocol.

Provides structured I/O via pydantic, retry logic, token tracking,
and structured logging of agent runs.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)

log = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base for all agents in the pipeline."""

    def __init__(self, name: str, max_retries: int = 3) -> None:
        self._name = name
        self._max_retries = max_retries
        self._cumulative_tokens = TokenUsage()

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def cumulative_tokens(self) -> TokenUsage:
        return self._cumulative_tokens

    def _accumulate_tokens(self, usage: TokenUsage) -> None:
        self._cumulative_tokens.input_tokens += usage.input_tokens
        self._cumulative_tokens.output_tokens += usage.output_tokens
        self._cumulative_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self._cumulative_tokens.cache_read_input_tokens += usage.cache_read_input_tokens

    def run(self, context: AgentContext) -> AgentResult:
        """Execute the agent with retry logic and structured logging."""
        log.info("agent_run_start", agent=self._name, task_len=len(context.task))
        start = time.monotonic()
        last_error: str | None = None

        for attempt in range(1, self._max_retries + 1):
            log.info("agent_attempt", agent=self._name, attempt=attempt)
            try:
                result = self._execute(context)
                elapsed = round(time.monotonic() - start, 4)
                result.duration_seconds = elapsed
                result.attempts = attempt
                self._accumulate_tokens(result.token_usage)
                log.info(
                    "agent_run_done",
                    agent=self._name,
                    status=result.status.value,
                    attempts=attempt,
                    duration_seconds=elapsed,
                    tokens=result.token_usage.total_tokens,
                )
                return result
            except Exception as exc:
                last_error = str(exc)
                log.warning(
                    "agent_attempt_error",
                    agent=self._name,
                    attempt=attempt,
                    error=last_error,
                )
                if attempt == self._max_retries:
                    break

        elapsed = round(time.monotonic() - start, 4)
        log.error(
            "agent_run_exhausted",
            agent=self._name,
            attempts=self._max_retries,
            last_error=last_error,
        )
        return AgentResult(
            agent_name=self._name,
            status=AgentStatus.MAX_RETRIES,
            error_message=f"Exhausted {self._max_retries} retries. Last error: {last_error}",
            attempts=self._max_retries,
            duration_seconds=elapsed,
        )

    @abstractmethod
    def _execute(self, context: AgentContext) -> AgentResult:
        """Subclasses implement this — a single attempt without retry handling."""
        ...
