"""Research Session Memory — persists across a full research session.

Tracks tried conjectures, partial results, promising directions, and
user preferences. Supports save/load from disk for session resume.

Implements tiered storage to manage context size:
  Hot tier:  last 3 conjectures — full detail
  Warm tier: next 10 most recent — summaries
  Cold tier: all older — compressed entries (hash + outcome + stage)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from agentic_research.logging import get_logger
from agentic_research.models.session import (
    ConjectureOutcome,
    PartialResult,
    PromisingDirection,
    SessionMemoryData,
    TriedConjecture,
    UserPreference,
)
from agentic_research.models.research import Conjecture

log = get_logger(__name__)

HOT_TIER_SIZE = 3
WARM_TIER_SIZE = 10
DEFAULT_MAX_CONJECTURES = 50

MAX_PARTIAL_RESULTS = 20
MAX_PROMISING_DIRECTIONS = 10
MAX_USER_PREFERENCES = 10


class WarmConjecture:
    """Summary of a conjecture for the warm tier."""

    __slots__ = ("statement", "outcome", "failure_reason")

    def __init__(self, statement: str, outcome: ConjectureOutcome, failure_reason: str) -> None:
        self.statement = statement
        self.outcome = outcome
        self.failure_reason = failure_reason

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "outcome": self.outcome.value,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_tried(cls, tc: TriedConjecture) -> WarmConjecture:
        reason = tc.failure_reason
        if reason and len(reason) > 120:
            reason = reason[:117] + "..."
        return cls(
            statement=tc.conjecture.statement,
            outcome=tc.outcome,
            failure_reason=reason,
        )

    @classmethod
    def from_dict(cls, d: dict) -> WarmConjecture:
        return cls(
            statement=d["statement"],
            outcome=ConjectureOutcome(d["outcome"]),
            failure_reason=d.get("failure_reason", ""),
        )


class ColdConjecture:
    """Compressed entry for the cold tier — hash + outcome + stage."""

    __slots__ = ("statement_hash", "outcome", "stage_reached")

    def __init__(self, statement_hash: str, outcome: ConjectureOutcome, stage_reached: str) -> None:
        self.statement_hash = statement_hash
        self.outcome = outcome
        self.stage_reached = stage_reached

    def to_dict(self) -> dict:
        return {
            "statement_hash": self.statement_hash,
            "outcome": self.outcome.value,
            "stage_reached": self.stage_reached,
        }

    @classmethod
    def from_tried(cls, tc: TriedConjecture) -> ColdConjecture:
        return cls(
            statement_hash=hashlib.sha256(tc.conjecture.statement.encode()).hexdigest()[:16],
            outcome=tc.outcome,
            stage_reached=tc.stage_reached.value,
        )

    @classmethod
    def from_warm(cls, wc: WarmConjecture) -> ColdConjecture:
        return cls(
            statement_hash=hashlib.sha256(wc.statement.encode()).hexdigest()[:16],
            outcome=wc.outcome,
            stage_reached="unknown",
        )

    @classmethod
    def from_dict(cls, d: dict) -> ColdConjecture:
        return cls(
            statement_hash=d["statement_hash"],
            outcome=ConjectureOutcome(d["outcome"]),
            stage_reached=d.get("stage_reached", "unknown"),
        )


class ResearchSessionMemory:
    """In-memory session memory with disk persistence and tiered storage."""

    def __init__(
        self,
        session_id: str | None = None,
        max_conjectures: int = DEFAULT_MAX_CONJECTURES,
    ) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._data = SessionMemoryData()
        self._max_conjectures = max_conjectures

        self._warm_conjectures: list[WarmConjecture] = []
        self._cold_conjectures: list[ColdConjecture] = []

        self._stage_summaries: list[str] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def data(self) -> SessionMemoryData:
        return self._data

    @property
    def warm_conjectures(self) -> list[WarmConjecture]:
        return self._warm_conjectures

    @property
    def cold_conjectures(self) -> list[ColdConjecture]:
        return self._cold_conjectures

    @property
    def stage_summaries(self) -> list[str]:
        return self._stage_summaries

    @property
    def total_conjecture_count(self) -> int:
        return (
            len(self._data.tried_conjectures)
            + len(self._warm_conjectures)
            + len(self._cold_conjectures)
        )

    def record_conjecture(
        self,
        conjecture: Conjecture,
        outcome: ConjectureOutcome = ConjectureOutcome.PENDING,
        *,
        lean_statement: str = "",
        proof_code: str | None = None,
        failure_reason: str = "",
        stage_reached: str = "conjecturing",
    ) -> None:
        from agentic_research.models.session import PipelineStage

        entry = TriedConjecture(
            conjecture=conjecture,
            outcome=outcome,
            lean_statement=lean_statement,
            proof_code=proof_code,
            failure_reason=failure_reason,
            stage_reached=PipelineStage(stage_reached),
        )
        self._data.tried_conjectures.append(entry)
        log.info(
            "session_memory_conjecture_recorded",
            statement=conjecture.statement[:80],
            outcome=outcome.value,
            total_tried=len(self._data.tried_conjectures),
        )
        self.compact()

    def compact(self) -> None:
        """Demote entries from hot -> warm -> cold to maintain tier sizes."""
        hot = self._data.tried_conjectures

        while len(hot) > HOT_TIER_SIZE:
            demoted = hot.pop(0)
            self._warm_conjectures.append(WarmConjecture.from_tried(demoted))

        while len(self._warm_conjectures) > WARM_TIER_SIZE:
            demoted_warm = self._warm_conjectures.pop(0)
            self._cold_conjectures.append(ColdConjecture.from_warm(demoted_warm))

        total = self.total_conjecture_count
        if total > self._max_conjectures:
            self._archive_overflow()

        log.debug(
            "session_memory_compacted",
            hot=len(self._data.tried_conjectures),
            warm=len(self._warm_conjectures),
            cold=len(self._cold_conjectures),
        )

    def _archive_overflow(self) -> None:
        """Archive oldest cold entries when total exceeds max_conjectures."""
        overflow = self.total_conjecture_count - self._max_conjectures
        if overflow <= 0:
            return

        archived = self._cold_conjectures[:overflow]
        self._cold_conjectures = self._cold_conjectures[overflow:]

        archive_data = [c.to_dict() for c in archived]
        archive_path = Path(f"session_archive_{self._session_id}.json")

        existing: list[dict] = []
        if archive_path.exists():
            existing = json.loads(archive_path.read_text())

        existing.extend(archive_data)
        archive_path.write_text(json.dumps(existing, indent=2))

        log.info(
            "session_memory_archived",
            count=len(archived),
            path=str(archive_path),
            remaining_cold=len(self._cold_conjectures),
        )

    def compress_stage_history(
        self,
        stage_name: str,
        conjectures_tried: int,
        disproved: int,
        proved: int,
        best_conjecture: str,
    ) -> None:
        """Replace full stage history with a summary after stage completion."""
        summary = (
            f"{stage_name}: {conjectures_tried} conjectures tried, "
            f"{disproved} disproved, {proved} proved, "
            f"best conjecture: {best_conjecture}"
        )
        self._stage_summaries.append(summary)
        log.info("session_memory_stage_compressed", summary=summary)

    def update_conjecture_outcome(
        self,
        statement: str,
        outcome: ConjectureOutcome,
        *,
        lean_statement: str = "",
        proof_code: str | None = None,
        failure_reason: str = "",
        stage_reached: str | None = None,
    ) -> bool:
        from agentic_research.models.session import PipelineStage

        for tc in self._data.tried_conjectures:
            if tc.conjecture.statement == statement:
                tc.outcome = outcome
                if lean_statement:
                    tc.lean_statement = lean_statement
                if proof_code is not None:
                    tc.proof_code = proof_code
                if failure_reason:
                    tc.failure_reason = failure_reason
                if stage_reached:
                    tc.stage_reached = PipelineStage(stage_reached)
                return True
        return False

    def add_partial_result(
        self,
        lemma_statement: str,
        *,
        lean_code: str = "",
        source_conjecture: str = "",
        domain: str = "",
    ) -> None:
        self._data.partial_results.append(PartialResult(
            lemma_statement=lemma_statement,
            lean_code=lean_code,
            source_conjecture=source_conjecture,
            domain=domain,
        ))
        if len(self._data.partial_results) > MAX_PARTIAL_RESULTS:
            self._data.partial_results = self._data.partial_results[-MAX_PARTIAL_RESULTS:]
        log.info(
            "session_memory_partial_result_added",
            total=len(self._data.partial_results),
        )

    def add_promising_direction(
        self,
        title: str,
        description: str = "",
        source: str = "",
        priority: float = 0.5,
    ) -> None:
        self._data.promising_directions.append(PromisingDirection(
            title=title,
            description=description,
            source=source,
            priority=priority,
        ))
        if len(self._data.promising_directions) > MAX_PROMISING_DIRECTIONS:
            self._data.promising_directions = self._data.promising_directions[-MAX_PROMISING_DIRECTIONS:]

    def add_user_preference(self, preference: str, context: str = "") -> None:
        self._data.user_preferences.append(UserPreference(
            preference=preference,
            context=context,
        ))
        if len(self._data.user_preferences) > MAX_USER_PREFERENCES:
            self._data.user_preferences = self._data.user_preferences[-MAX_USER_PREFERENCES:]

    def has_tried(self, statement: str) -> bool:
        if self._data.has_tried(statement):
            return True
        for wc in self._warm_conjectures:
            if wc.statement == statement:
                return True
        stmt_hash = hashlib.sha256(statement.encode()).hexdigest()[:16]
        for cc in self._cold_conjectures:
            if cc.statement_hash == stmt_hash:
                return True
        return False

    def get_untried_directions(self) -> list[PromisingDirection]:
        tried_titles = {tc.conjecture.statement for tc in self._data.tried_conjectures}
        tried_titles.update(wc.statement for wc in self._warm_conjectures)
        return [
            d for d in self._data.promising_directions
            if d.title not in tried_titles
        ]

    def summary(self) -> dict:
        return {
            "session_id": self._session_id,
            "total_tried": self.total_conjecture_count,
            "hot_tier": len(self._data.tried_conjectures),
            "warm_tier": len(self._warm_conjectures),
            "cold_tier": len(self._cold_conjectures),
            "proved": len(self._data.proved_conjectures()),
            "failed": len(self._data.failed_conjectures()),
            "partial_results": len(self._data.partial_results),
            "promising_directions": len(self._data.promising_directions),
            "user_preferences": len(self._data.user_preferences),
            "stage_summaries": len(self._stage_summaries),
        }

    def save(self, path: Path) -> None:
        payload = {
            "session_id": self._session_id,
            "max_conjectures": self._max_conjectures,
            "memory": self._data.model_dump(),
            "warm_conjectures": [wc.to_dict() for wc in self._warm_conjectures],
            "cold_conjectures": [cc.to_dict() for cc in self._cold_conjectures],
            "stage_summaries": self._stage_summaries,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        log.info("session_memory_saved", path=str(path), session_id=self._session_id)

    @classmethod
    def load(cls, path: Path) -> ResearchSessionMemory:
        raw = json.loads(path.read_text())
        instance = cls(
            session_id=raw.get("session_id"),
            max_conjectures=raw.get("max_conjectures", DEFAULT_MAX_CONJECTURES),
        )
        instance._data = SessionMemoryData.model_validate(raw.get("memory", {}))
        instance._warm_conjectures = [
            WarmConjecture.from_dict(d)
            for d in raw.get("warm_conjectures", [])
        ]
        instance._cold_conjectures = [
            ColdConjecture.from_dict(d)
            for d in raw.get("cold_conjectures", [])
        ]
        instance._stage_summaries = raw.get("stage_summaries", [])
        log.info("session_memory_loaded", path=str(path), session_id=instance._session_id)
        return instance
