"""Circuit breaker for the orchestrator — halts execution on sustained failures."""

from __future__ import annotations

import time
from collections import deque

from agentic_research.logging import get_logger

log = get_logger(__name__)


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and the operation should not proceed."""


class CircuitBreaker:
    """Tracks failures and opens the circuit when thresholds are breached.

    Two independent triggers:
      1. ``consecutive_failure_limit`` consecutive failures in a row.
      2. Error rate exceeds ``error_rate_threshold`` within the last
         ``window_seconds`` seconds (requires >=1 event in the window).
    """

    def __init__(
        self,
        consecutive_failure_limit: int = 5,
        error_rate_threshold: float = 0.5,
        window_seconds: float = 60.0,
    ) -> None:
        self._consecutive_failure_limit = consecutive_failure_limit
        self._error_rate_threshold = error_rate_threshold
        self._window_seconds = window_seconds

        self._consecutive_failures: int = 0
        self._events: deque[tuple[float, bool]] = deque()

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._events.append((time.monotonic(), True))
        self._prune()

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._events.append((time.monotonic(), False))
        self._prune()
        if self.is_open():
            log.warning(
                "circuit_breaker_opened",
                consecutive_failures=self._consecutive_failures,
                error_rate=self._error_rate(),
            )

    def is_open(self) -> bool:
        if self._consecutive_failures >= self._consecutive_failure_limit:
            return True
        if self._error_rate() > self._error_rate_threshold:
            return True
        return False

    def reset(self) -> None:
        self._consecutive_failures = 0
        self._events.clear()

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _error_rate(self) -> float:
        self._prune()
        if not self._events:
            return 0.0
        failures = sum(1 for _, ok in self._events if not ok)
        return failures / len(self._events)
