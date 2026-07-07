"""Context size monitoring — estimates token count and emits threshold warnings."""

from __future__ import annotations

from agentic_research.logging import get_logger

log = get_logger(__name__)

CHARS_PER_TOKEN = 4

WARNING_THRESHOLDS = (8_000, 16_000, 32_000)


class ContextSizeMonitor:
    """Estimates token count of current session context and warns at thresholds."""

    def __init__(self) -> None:
        self._total_chars: int = 0
        self._warnings_emitted: set[int] = set()

    def update(self, text: str) -> None:
        """Add text to the tracked context and check thresholds."""
        self._total_chars += len(text)
        self._check_thresholds()

    def set_total(self, total_chars: int) -> None:
        """Set the total character count directly (e.g. after recomputing from memory)."""
        self._total_chars = total_chars
        self._check_thresholds()

    @property
    def context_tokens_estimate(self) -> int:
        return self._total_chars // CHARS_PER_TOKEN

    def _check_thresholds(self) -> None:
        tokens = self.context_tokens_estimate
        for threshold in WARNING_THRESHOLDS:
            if tokens >= threshold and threshold not in self._warnings_emitted:
                self._warnings_emitted.add(threshold)
                log.warning(
                    "context_size_threshold_reached",
                    tokens_estimate=tokens,
                    threshold=threshold,
                )

    def reset(self) -> None:
        """Reset tracking state."""
        self._total_chars = 0
        self._warnings_emitted.clear()
