"""Strip AI-generated comments from Lean code before intent judging.

Removes:
  - Single-line comments (``-- ...``)
  - Block comments (``/- ... -/``)
  - Trailing whitespace left by comment removal
  - Blank lines created by comment removal

Preserves:
  - ``#check``, ``#eval``, and other Lean commands
  - Doc-strings (``/-- ... -/``) are optionally kept (default: removed)
  - String literals containing ``--`` are not touched
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentic_research.logging import get_logger
from agentic_research.models.tools import CleanResult, ToolStatus
from agentic_research.tools.base import BaseTool

log = get_logger(__name__)


@dataclass
class HintCleanerConfig:
    keep_doc_strings: bool = False


_BLOCK_COMMENT_RE = re.compile(r"/--?(?:\s|$).*?-/", re.DOTALL)
_DOC_COMMENT_RE = re.compile(r"/--(?:\s|$).*?-/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def _remove_comments(code: str, keep_doc_strings: bool) -> tuple[str, int]:
    """Remove comments from Lean code, returning cleaned code and count of comments removed."""
    count = 0

    placeholders: dict[str, str] = {}
    placeholder_idx = 0

    def _save_string(m: re.Match[str]) -> str:
        nonlocal placeholder_idx
        key = f"\x00STR{placeholder_idx}\x00"
        placeholder_idx += 1
        placeholders[key] = m.group(0)
        return key

    protected = _STRING_RE.sub(_save_string, code)

    if keep_doc_strings:
        doc_placeholders: dict[str, str] = {}
        doc_idx = 0

        def _save_doc(m: re.Match[str]) -> str:
            nonlocal doc_idx
            key = f"\x00DOC{doc_idx}\x00"
            doc_idx += 1
            doc_placeholders[key] = m.group(0)
            return key

        protected = _DOC_COMMENT_RE.sub(_save_doc, protected)
        count += len(_BLOCK_COMMENT_RE.findall(protected))
        protected = _BLOCK_COMMENT_RE.sub("", protected)
    else:
        count += len(_BLOCK_COMMENT_RE.findall(protected))
        protected = _BLOCK_COMMENT_RE.sub("", protected)

    count += len(_LINE_COMMENT_RE.findall(protected))
    protected = _LINE_COMMENT_RE.sub("", protected)

    if keep_doc_strings:
        for key, val in doc_placeholders.items():
            protected = protected.replace(key, val)

    for key, val in placeholders.items():
        protected = protected.replace(key, val)

    lines = protected.splitlines()
    lines = [line.rstrip() for line in lines]
    cleaned = "\n".join(line for line in lines if line.strip())

    if cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"

    return cleaned, count


class HintCleaner(BaseTool):
    """Strip AI-generated comments from Lean code."""

    _name = "hint_cleaner"

    def __init__(self, config: HintCleanerConfig | None = None) -> None:
        if config is None:
            config = HintCleanerConfig()
        self._config = config
        log.info("hint_cleaner_init", keep_doc_strings=config.keep_doc_strings)

    def execute(self, code: str) -> CleanResult:
        result = super().execute(code)
        if isinstance(result, CleanResult):
            return result
        return CleanResult(
            status=result.status,
            original_code=code,
            cleaned_code=code,
            comments_removed=0,
        )

    def _run(self, input_data: Any) -> CleanResult:
        code = str(input_data)
        cleaned, removed_count = _remove_comments(code, self._config.keep_doc_strings)
        return CleanResult(
            status=ToolStatus.SUCCESS,
            original_code=code,
            cleaned_code=cleaned,
            comments_removed=removed_count,
        )
