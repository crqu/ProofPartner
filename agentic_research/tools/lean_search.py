"""Search Mathlib for relevant lemmas/theorems.

Supports multiple backends:
  1. LeanDojo premise retrieval (when installed)
  2. Moogle / LeanSearch web API (HTTP fallback)
  3. Mock mode for testing
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
from typing import Any

from agentic_research.logging import get_logger
from agentic_research.models.tools import (
    SearchResult,
    SearchResultEntry,
    ToolStatus,
)
from agentic_research.tools.base import BaseTool

log = get_logger(__name__)

MOOGLE_API_URL = "https://www.moogle.ai/api/search"
LEANSEARCH_API_URL = "https://leansearch.net/api/search"


class SearchBackend(str, Enum):
    LEAN_DOJO = "lean_dojo"
    MOOGLE = "moogle"
    MOCK = "mock"


@dataclass
class SearchConfig:
    backend: SearchBackend = SearchBackend.MOCK
    max_results: int = 10
    timeout_seconds: int = 30
    api_url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)


def detect_search_backend() -> SearchBackend:
    try:
        import lean_dojo  # noqa: F401
        return SearchBackend.LEAN_DOJO
    except ImportError:
        pass
    return SearchBackend.MOOGLE


class _MockSearchBackend:
    def search(self, query: str, max_results: int, timeout: int) -> SearchResult:
        entries = [
            SearchResultEntry(
                name="Nat.add_comm",
                type_signature="∀ (n m : ℕ), n + m = m + n",
                doc_string="Addition is commutative on natural numbers.",
                module="Mathlib.Data.Nat.Basic",
                relevance_score=0.95,
            ),
            SearchResultEntry(
                name="Nat.add_assoc",
                type_signature="∀ (n m k : ℕ), n + m + k = n + (m + k)",
                doc_string="Addition is associative on natural numbers.",
                module="Mathlib.Data.Nat.Basic",
                relevance_score=0.85,
            ),
        ]
        return SearchResult(
            status=ToolStatus.SUCCESS,
            query=query,
            entries=entries[:max_results],
            total_results=len(entries),
        )


class _MoogleBackend:
    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._url = config.api_url or MOOGLE_API_URL

    def search(self, query: str, max_results: int, timeout: int) -> SearchResult:
        payload = json.dumps({"query": query, "num_results": max_results}).encode()
        headers = {"Content-Type": "application/json"}
        headers.update(self._config.extra_headers)

        req = urllib.request.Request(self._url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError) as exc:
            return SearchResult(
                status=ToolStatus.ERROR,
                query=query,
                error_message=f"Moogle API error: {exc}",
            )

        entries: list[SearchResultEntry] = []
        for hit in data.get("data", [])[:max_results]:
            entries.append(
                SearchResultEntry(
                    name=hit.get("name", ""),
                    type_signature=hit.get("type", ""),
                    doc_string=hit.get("doc", ""),
                    module=hit.get("module", ""),
                    relevance_score=float(hit.get("score", 0.0)),
                )
            )

        return SearchResult(
            status=ToolStatus.SUCCESS,
            query=query,
            entries=entries,
            total_results=len(entries),
            truncated=len(entries) >= max_results,
        )


class _LeanDojoSearchBackend:
    def search(self, query: str, max_results: int, timeout: int) -> SearchResult:
        try:
            from lean_dojo import LeanGitRepo  # type: ignore[import-untyped]
        except ImportError:
            return SearchResult(
                status=ToolStatus.UNAVAILABLE,
                query=query,
                error_message="lean-dojo is not installed",
            )

        try:
            repo = LeanGitRepo(
                "https://github.com/leanprover-community/mathlib4",
                "master",
            )
            from lean_dojo import trace  # type: ignore[import-untyped]

            traced_repo = trace(repo)
            entries: list[SearchResultEntry] = []
            for thm in list(traced_repo.get_traced_theorems())[:max_results]:
                entries.append(
                    SearchResultEntry(
                        name=thm.full_name,
                        type_signature=str(getattr(thm, "type", "")),
                        module=str(getattr(thm, "file_path", "")),
                    )
                )

            return SearchResult(
                status=ToolStatus.SUCCESS,
                query=query,
                entries=entries,
                total_results=len(entries),
            )
        except Exception as exc:
            return SearchResult(
                status=ToolStatus.ERROR,
                query=query,
                error_message=str(exc),
            )


_SearchBackendImpl = _MockSearchBackend | _MoogleBackend | _LeanDojoSearchBackend

_BACKENDS: dict[SearchBackend, Callable[[SearchConfig], _SearchBackendImpl]] = {
    SearchBackend.MOCK: lambda cfg: _MockSearchBackend(),
    SearchBackend.MOOGLE: lambda cfg: _MoogleBackend(cfg),
    SearchBackend.LEAN_DOJO: lambda cfg: _LeanDojoSearchBackend(),
}


class LeanSearch(BaseTool):
    """Search Mathlib for relevant lemmas and theorems."""

    _name = "lean_search"

    def __init__(self, config: SearchConfig | None = None) -> None:
        if config is None:
            config = SearchConfig()
        self._config = config
        self._backend: _MockSearchBackend | _MoogleBackend | _LeanDojoSearchBackend = _BACKENDS[config.backend](config)
        log.info("lean_search_init", backend=config.backend.value)

    @property
    def search_backend(self) -> SearchBackend:
        return self._config.backend

    def _run(self, input_data: Any) -> SearchResult:
        query = str(input_data)
        return self._backend.search(
            query, self._config.max_results, self._config.timeout_seconds
        )
