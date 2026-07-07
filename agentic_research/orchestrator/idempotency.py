"""Idempotency cache for LLM calls — prevents duplicate API calls on retry/resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentic_research.logging import get_logger

log = get_logger(__name__)

DEFAULT_CACHE_DIR = Path(".agentic_research/idempotency_cache")


def make_idempotency_key(
    session_id: str,
    stage: str,
    conjecture_index: int,
    attempt_number: int,
) -> str:
    raw = f"{session_id}:{stage}:{conjecture_index}:{attempt_number}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class IdempotencyCache:
    """In-memory + optional disk cache keyed by deterministic idempotency keys."""

    def __init__(self, session_id: str = "", persist: bool = False) -> None:
        self._cache: dict[str, Any] = {}
        self._session_id = session_id
        self._persist = persist

    def _cache_dir(self) -> Path:
        return DEFAULT_CACHE_DIR / self._session_id

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            log.info("idempotency_cache_hit", key=key[:12])
            return self._cache[key]

        if self._persist and self._session_id:
            path = self._cache_dir() / f"{key}.json"
            if path.exists():
                data = json.loads(path.read_text())
                self._cache[key] = data
                log.info("idempotency_cache_hit_disk", key=key[:12])
                return data

        return None

    def put(self, key: str, value: Any) -> None:
        self._cache[key] = value
        if self._persist and self._session_id:
            cache_dir = self._cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / f"{key}.json"
            path.write_text(json.dumps(value, default=str))
        log.info("idempotency_cache_store", key=key[:12])

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
