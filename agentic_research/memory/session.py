"""Research Session Memory — persists across a full research session.

Tracks tried conjectures, partial results, promising directions, and
user preferences. Supports save/load from disk for session resume.
"""

from __future__ import annotations

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


class ResearchSessionMemory:
    """In-memory session memory with disk persistence."""

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._data = SessionMemoryData()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def data(self) -> SessionMemoryData:
        return self._data

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

    def add_user_preference(self, preference: str, context: str = "") -> None:
        self._data.user_preferences.append(UserPreference(
            preference=preference,
            context=context,
        ))

    def has_tried(self, statement: str) -> bool:
        return self._data.has_tried(statement)

    def get_untried_directions(self) -> list[PromisingDirection]:
        tried_titles = {tc.conjecture.statement for tc in self._data.tried_conjectures}
        return [
            d for d in self._data.promising_directions
            if d.title not in tried_titles
        ]

    def summary(self) -> dict:
        return {
            "session_id": self._session_id,
            "total_tried": len(self._data.tried_conjectures),
            "proved": len(self._data.proved_conjectures()),
            "failed": len(self._data.failed_conjectures()),
            "partial_results": len(self._data.partial_results),
            "promising_directions": len(self._data.promising_directions),
            "user_preferences": len(self._data.user_preferences),
        }

    def save(self, path: Path) -> None:
        payload = {
            "session_id": self._session_id,
            "memory": self._data.model_dump(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        log.info("session_memory_saved", path=str(path), session_id=self._session_id)

    @classmethod
    def load(cls, path: Path) -> ResearchSessionMemory:
        raw = json.loads(path.read_text())
        instance = cls(session_id=raw.get("session_id"))
        instance._data = SessionMemoryData.model_validate(raw.get("memory", {}))
        log.info("session_memory_loaded", path=str(path), session_id=instance._session_id)
        return instance
