"""Cost tracker — records token usage per LLM call and monitors spend velocity."""

from __future__ import annotations

import time

from agentic_research.logging import get_logger
from agentic_research.models.session import (
    OPUS_CACHE_READ_PRICE_PER_MTOK,
    OPUS_CACHE_WRITE_PRICE_PER_MTOK,
    OPUS_INPUT_PRICE_PER_MTOK,
    OPUS_OUTPUT_PRICE_PER_MTOK,
)

log = get_logger(__name__)

DEFAULT_VELOCITY_THRESHOLD_USD_PER_MIN = 1.0
VELOCITY_WINDOW_SECONDS = 30.0
MAX_RECORDS = 1000


class CostTracker:
    """Tracks cumulative cost across all agents in a session."""

    def __init__(
        self,
        velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD_USD_PER_MIN,
        velocity_window: float = VELOCITY_WINDOW_SECONDS,
    ) -> None:
        self._velocity_threshold = velocity_threshold
        self._velocity_window = velocity_window
        self._records: list[tuple[float, float]] = []
        self._total_cost: float = 0.0

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Record token usage and return the cost of this call."""
        cost = (
            input_tokens * OPUS_INPUT_PRICE_PER_MTOK / 1_000_000
            + output_tokens * OPUS_OUTPUT_PRICE_PER_MTOK / 1_000_000
            + cache_read_tokens * OPUS_CACHE_READ_PRICE_PER_MTOK / 1_000_000
            + cache_write_tokens * OPUS_CACHE_WRITE_PRICE_PER_MTOK / 1_000_000
        )
        now = time.monotonic()
        self._records.append((now, cost))
        self._total_cost += cost
        if len(self._records) > MAX_RECORDS:
            self._records = self._records[-MAX_RECORDS:]

        vel = self.velocity()
        if vel > self._velocity_threshold:
            log.warning(
                "cost_velocity_exceeded",
                velocity_usd_per_min=round(vel, 4),
                threshold=self._velocity_threshold,
                total_cost=round(self._total_cost, 4),
            )

        return cost

    def total_cost(self) -> float:
        return self._total_cost

    def velocity(self) -> float:
        """Return spending rate in $/minute over the recent window."""
        if not self._records:
            return 0.0

        now = time.monotonic()
        cutoff = now - self._velocity_window
        window_cost = sum(cost for ts, cost in self._records if ts >= cutoff)

        elapsed = min(now - self._records[0][0], self._velocity_window)
        if elapsed <= 0:
            return 0.0

        return (window_cost / elapsed) * 60.0
