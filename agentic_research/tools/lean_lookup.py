"""Retrieve full documentation and type signatures for Lean/Mathlib identifiers.

Supports multiple backends:
  1. LeanDojo (when installed)
  2. Mathlib4 docs web API (HTTP fallback)
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
from agentic_research.models.tools import LookupResult, ToolStatus
from agentic_research.tools.base import BaseTool

log = get_logger(__name__)

MATHLIB4_DOC_API_URL = "https://leanprover-community.github.io/mathlib4_docs"


class LookupBackend(str, Enum):
    LEAN_DOJO = "lean_dojo"
    WEB = "web"
    MOCK = "mock"


@dataclass
class LookupConfig:
    backend: LookupBackend = LookupBackend.MOCK
    timeout_seconds: int = 30
    api_url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)


def detect_lookup_backend() -> LookupBackend:
    try:
        import lean_dojo  # noqa: F401
        return LookupBackend.LEAN_DOJO
    except ImportError:
        pass
    return LookupBackend.WEB


_MOCK_DB: dict[str, dict[str, str]] = {
    "Nat.add_comm": {
        "type_signature": "∀ (n m : ℕ), n + m = m + n",
        "doc_string": "Addition is commutative on natural numbers.",
        "module": "Mathlib.Data.Nat.Basic",
    },
    "Nat.add_assoc": {
        "type_signature": "∀ (n m k : ℕ), n + m + k = n + (m + k)",
        "doc_string": "Addition is associative on natural numbers.",
        "module": "Mathlib.Data.Nat.Basic",
    },
    "List.map": {
        "type_signature": "∀ {α β : Type*}, (α → β) → List α → List β",
        "doc_string": "Map a function over a list.",
        "module": "Init.Data.List.Basic",
    },
}


class _MockLookupBackend:
    def lookup(self, identifier: str, timeout: int) -> LookupResult:
        entry = _MOCK_DB.get(identifier)
        if entry is None:
            return LookupResult(
                status=ToolStatus.SUCCESS,
                identifier=identifier,
                found=False,
            )
        return LookupResult(
            status=ToolStatus.SUCCESS,
            identifier=identifier,
            type_signature=entry.get("type_signature", ""),
            doc_string=entry.get("doc_string", ""),
            module=entry.get("module", ""),
            found=True,
        )


class _WebLookupBackend:
    def __init__(self, config: LookupConfig) -> None:
        self._config = config
        self._base_url = config.api_url or MATHLIB4_DOC_API_URL

    def lookup(self, identifier: str, timeout: int) -> LookupResult:
        path = identifier.replace(".", "/")
        url = f"{self._base_url}/find/{path}"
        headers = {"Accept": "application/json"}
        headers.update(self._config.extra_headers)

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return LookupResult(
                    status=ToolStatus.SUCCESS,
                    identifier=identifier,
                    found=False,
                )
            return LookupResult(
                status=ToolStatus.ERROR,
                identifier=identifier,
                error_message=f"HTTP {exc.code}: {exc.reason}",
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            return LookupResult(
                status=ToolStatus.ERROR,
                identifier=identifier,
                error_message=f"Lookup error: {exc}",
            )

        return LookupResult(
            status=ToolStatus.SUCCESS,
            identifier=identifier,
            type_signature=data.get("type", ""),
            doc_string=data.get("doc", ""),
            module=data.get("module", ""),
            source_url=data.get("source_url", ""),
            found=True,
        )


class _LeanDojoLookupBackend:
    def lookup(self, identifier: str, timeout: int) -> LookupResult:
        try:
            from lean_dojo import LeanGitRepo, trace  # type: ignore[import-untyped]
        except ImportError:
            return LookupResult(
                status=ToolStatus.UNAVAILABLE,
                identifier=identifier,
                error_message="lean-dojo is not installed",
            )

        try:
            repo = LeanGitRepo(
                "https://github.com/leanprover-community/mathlib4",
                "master",
            )
            traced_repo = trace(repo)
            for thm in traced_repo.get_traced_theorems():
                if thm.full_name == identifier:
                    return LookupResult(
                        status=ToolStatus.SUCCESS,
                        identifier=identifier,
                        type_signature=str(getattr(thm, "type", "")),
                        module=str(getattr(thm, "file_path", "")),
                        found=True,
                    )
            return LookupResult(
                status=ToolStatus.SUCCESS,
                identifier=identifier,
                found=False,
            )
        except Exception as exc:
            return LookupResult(
                status=ToolStatus.ERROR,
                identifier=identifier,
                error_message=str(exc),
            )


_LookupBackendImpl = _MockLookupBackend | _WebLookupBackend | _LeanDojoLookupBackend

_BACKENDS: dict[LookupBackend, Callable[[LookupConfig], _LookupBackendImpl]] = {
    LookupBackend.MOCK: lambda cfg: _MockLookupBackend(),
    LookupBackend.WEB: lambda cfg: _WebLookupBackend(cfg),
    LookupBackend.LEAN_DOJO: lambda cfg: _LeanDojoLookupBackend(),
}


class LeanLookup(BaseTool):
    """Look up documentation and type signatures for Lean/Mathlib identifiers."""

    _name = "lean_lookup"

    def __init__(self, config: LookupConfig | None = None) -> None:
        if config is None:
            config = LookupConfig()
        self._config = config
        self._backend: _MockLookupBackend | _WebLookupBackend | _LeanDojoLookupBackend = _BACKENDS[config.backend](config)
        log.info("lean_lookup_init", backend=config.backend.value)

    @property
    def lookup_backend(self) -> LookupBackend:
        return self._config.backend

    def _run(self, input_data: Any) -> LookupResult:
        identifier = str(input_data)
        return self._backend.lookup(identifier, self._config.timeout_seconds)
