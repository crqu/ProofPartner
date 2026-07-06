"""Base Tool protocol for the Lean 4 tool layer.

All tools implement the Tool protocol: execute(input) -> ToolResult with
structured error reporting, latency tracking, and structlog integration.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Protocol, runtime_checkable

from agentic_research.logging import get_logger
from agentic_research.models.tools import ToolResult, ToolStatus

log = get_logger(__name__)


def _input_hash(input_data: Any) -> str:
    """Produce a short hash of the input for structured logging."""
    raw = str(input_data).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:12]


@runtime_checkable
class Tool(Protocol):
    """Protocol that every tool must satisfy."""

    @property
    def name(self) -> str: ...

    def execute(self, input_data: Any) -> ToolResult: ...


class BaseTool:
    """Convenience base class implementing logging, timing, and error handling."""

    _name: str = "base_tool"

    @property
    def name(self) -> str:
        return self._name

    def execute(self, input_data: Any) -> ToolResult:
        input_h = _input_hash(input_data)
        log.info("tool_execute_start", tool=self.name, input_hash=input_h)
        start = time.monotonic()
        try:
            result = self._run(input_data)
            elapsed = round(time.monotonic() - start, 4)
            result.duration_seconds = elapsed
            log.info(
                "tool_execute_done",
                tool=self.name,
                input_hash=input_h,
                status=result.status.value,
                duration_seconds=elapsed,
            )
            return result
        except Exception as exc:
            elapsed = round(time.monotonic() - start, 4)
            log.error(
                "tool_execute_error",
                tool=self.name,
                input_hash=input_h,
                error=str(exc),
                duration_seconds=elapsed,
            )
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                duration_seconds=elapsed,
                error_message=str(exc),
            )

    def _run(self, input_data: Any) -> ToolResult:
        raise NotImplementedError
