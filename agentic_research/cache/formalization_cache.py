"""Cross-session formalization cache backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from agentic_research.logging import get_logger

log = get_logger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".proofpartner" / "formalization_cache.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS formalization_cache (
    type_name TEXT NOT NULL,
    type_signature TEXT,
    lean_code TEXT NOT NULL,
    lean_toolchain TEXT NOT NULL,
    mathlib_version TEXT,
    source_conjecture_hash TEXT,
    created_at TEXT NOT NULL,
    proved_lemmas TEXT,
    dependencies TEXT,
    reuse_count INTEGER DEFAULT 0,
    last_used_at TEXT,
    PRIMARY KEY (type_name, lean_toolchain)
);
CREATE INDEX IF NOT EXISTS idx_toolchain ON formalization_cache(lean_toolchain);
"""


class CachedFormalization(BaseModel):
    """A cached type formalization entry."""

    type_name: str
    type_signature: str = ""
    lean_code: str
    lean_toolchain: str
    mathlib_version: str | None = None
    source_conjecture_hash: str = ""
    created_at: str
    proved_lemmas: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    reuse_count: int = 0
    last_used_at: str | None = None


class FormalizationCache:
    """SQLite-backed cache for verified type definitions across sessions."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def get(self, type_name: str, lean_toolchain: str) -> CachedFormalization | None:
        """Exact-match lookup by type_name + lean_toolchain."""
        row = self._conn.execute(
            "SELECT * FROM formalization_cache WHERE type_name = ? AND lean_toolchain = ?",
            (type_name, lean_toolchain),
        ).fetchone()
        if row is None:
            log.info("formalization_cache_miss", type_name=type_name, lean_toolchain=lean_toolchain)
            return None

        self._conn.execute(
            "UPDATE formalization_cache SET reuse_count = reuse_count + 1, last_used_at = ? "
            "WHERE type_name = ? AND lean_toolchain = ?",
            (datetime.now(timezone.utc).isoformat(), type_name, lean_toolchain),
        )
        self._conn.commit()

        entry = self._row_to_entry(row)
        entry.reuse_count += 1
        log.info("formalization_cache_hit", type_name=type_name, reuse_count=entry.reuse_count)
        return entry

    def put(self, entry: CachedFormalization) -> None:
        """Insert or replace a cache entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO formalization_cache "
            "(type_name, type_signature, lean_code, lean_toolchain, mathlib_version, "
            "source_conjecture_hash, created_at, proved_lemmas, dependencies, reuse_count, last_used_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.type_name,
                entry.type_signature,
                entry.lean_code,
                entry.lean_toolchain,
                entry.mathlib_version,
                entry.source_conjecture_hash,
                entry.created_at,
                json.dumps(entry.proved_lemmas),
                json.dumps(entry.dependencies),
                entry.reuse_count,
                entry.last_used_at,
            ),
        )
        self._conn.commit()
        log.info("formalization_cache_store", type_name=entry.type_name, lean_toolchain=entry.lean_toolchain)

    def invalidate_toolchain(self, lean_toolchain: str) -> int:
        """Delete all entries for a given toolchain. Returns count deleted."""
        cursor = self._conn.execute(
            "DELETE FROM formalization_cache WHERE lean_toolchain = ?",
            (lean_toolchain,),
        )
        self._conn.commit()
        count = cursor.rowcount
        log.info("formalization_cache_invalidated", lean_toolchain=lean_toolchain, count=count)
        return count

    def list_entries(self) -> list[CachedFormalization]:
        """Return all cache entries."""
        rows = self._conn.execute("SELECT * FROM formalization_cache").fetchall()
        return [self._row_to_entry(row) for row in rows]

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CachedFormalization:
        proved_lemmas = json.loads(row["proved_lemmas"]) if row["proved_lemmas"] else []
        dependencies = json.loads(row["dependencies"]) if row["dependencies"] else []
        return CachedFormalization(
            type_name=row["type_name"],
            type_signature=row["type_signature"] or "",
            lean_code=row["lean_code"],
            lean_toolchain=row["lean_toolchain"],
            mathlib_version=row["mathlib_version"],
            source_conjecture_hash=row["source_conjecture_hash"] or "",
            created_at=row["created_at"],
            proved_lemmas=proved_lemmas,
            dependencies=dependencies,
            reuse_count=row["reuse_count"],
            last_used_at=row["last_used_at"],
        )
