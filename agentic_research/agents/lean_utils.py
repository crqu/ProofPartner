"""Shared Lean code utilities for agents."""

from __future__ import annotations

import re

_PREAMBLE_PREFIX_RE = re.compile(
    r"^(import |set_option |open |section |namespace )", re.IGNORECASE
)


def _strip_preamble_lines(code: str) -> str:
    """Strip leading import/set_option/open/section/namespace and blank lines from LLM code.

    Stops at the first line that doesn't match a preamble pattern (e.g. theorem, def, lemma).
    Idempotent: returns code unchanged if no preamble lines are present.
    """
    lines = code.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _PREAMBLE_PREFIX_RE.match(stripped):
            start = i + 1
        else:
            break
    return "\n".join(lines[start:])
