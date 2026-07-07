"""Checkpoint & rollback — stage-level checkpointing with revert support.

Supports both in-memory and disk-based checkpoint persistence.
Disk checkpoints are stored as JSON in .agentic_research/checkpoints/{session_id}/.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_research.logging import get_logger
from agentic_research.memory.session import ResearchSessionMemory
from agentic_research.models.session import (
    SessionCheckpoint,
    StageTokenUsage,
)
from agentic_research.orchestrator.state import PipelineStateMachine

log = get_logger(__name__)

DEFAULT_CHECKPOINT_DIR = Path(".agentic_research/checkpoints")
MAX_CHECKPOINTS = 20


class CheckpointManager:
    """Manages stage-level checkpoints for rollback support."""

    def __init__(self, session_id: str = "", persist: bool = False) -> None:
        self._checkpoints: list[SessionCheckpoint] = []
        self._counter = 0
        self._session_id = session_id
        self._persist = persist

    @property
    def checkpoints(self) -> list[SessionCheckpoint]:
        return list(self._checkpoints)

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    def _checkpoint_dir(self) -> Path:
        return DEFAULT_CHECKPOINT_DIR / self._session_id

    def create_checkpoint(
        self,
        state_machine: PipelineStateMachine,
        memory: ResearchSessionMemory,
        stage_usages: list[StageTokenUsage] | None = None,
    ) -> SessionCheckpoint:
        self._counter += 1
        checkpoint_id = f"ckpt_{self._counter}"

        checkpoint = SessionCheckpoint(
            checkpoint_id=checkpoint_id,
            stage=state_machine.current_stage,
            session_state=state_machine.session_state.model_copy(deep=True),
            memory=memory.data.model_copy(deep=True),
            stage_token_usages=list(stage_usages) if stage_usages else [],
        )
        self._checkpoints.append(checkpoint)
        if len(self._checkpoints) > MAX_CHECKPOINTS:
            self._checkpoints = self._checkpoints[-MAX_CHECKPOINTS:]

        if self._persist and self._session_id:
            self._save_checkpoint_to_disk(checkpoint)

        log.info(
            "checkpoint_created",
            checkpoint_id=checkpoint_id,
            stage=state_machine.current_stage.value,
            total_checkpoints=len(self._checkpoints),
        )
        return checkpoint

    def _save_checkpoint_to_disk(self, checkpoint: SessionCheckpoint) -> Path:
        ckpt_dir = self._checkpoint_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{checkpoint.checkpoint_id}.json"
        path.write_text(checkpoint.model_dump_json(indent=2))
        log.info("checkpoint_saved_to_disk", path=str(path))
        return path

    @classmethod
    def load_checkpoint_from_disk(
        cls, session_id: str, checkpoint_id: str
    ) -> SessionCheckpoint | None:
        path = DEFAULT_CHECKPOINT_DIR / session_id / f"{checkpoint_id}.json"
        if not path.exists():
            log.warning("checkpoint_file_not_found", path=str(path))
            return None
        data = json.loads(path.read_text())
        return SessionCheckpoint.model_validate(data)

    def list_disk_checkpoints(self) -> list[str]:
        ckpt_dir = self._checkpoint_dir()
        if not ckpt_dir.exists():
            return []
        return sorted(
            p.stem for p in ckpt_dir.glob("ckpt_*.json")
        )

    def latest_disk_checkpoint(self) -> SessionCheckpoint | None:
        ids = self.list_disk_checkpoints()
        if not ids:
            return None
        return self.load_checkpoint_from_disk(self._session_id, ids[-1])

    def rollback(
        self,
        checkpoint_id: str,
        state_machine: PipelineStateMachine,
        memory: ResearchSessionMemory,
    ) -> bool:
        target = None
        target_idx = -1
        for i, cp in enumerate(self._checkpoints):
            if cp.checkpoint_id == checkpoint_id:
                target = cp
                target_idx = i
                break

        if target is None:
            log.warning("rollback_failed", checkpoint_id=checkpoint_id, reason="not found")
            return False

        state_machine.restore(target.session_state.model_copy(deep=True))
        memory._data = target.memory.model_copy(deep=True)

        self._checkpoints = self._checkpoints[: target_idx + 1]

        log.info(
            "rollback_completed",
            checkpoint_id=checkpoint_id,
            restored_stage=target.stage.value,
            remaining_checkpoints=len(self._checkpoints),
        )
        return True

    def rollback_to_latest(
        self,
        state_machine: PipelineStateMachine,
        memory: ResearchSessionMemory,
    ) -> bool:
        if not self._checkpoints:
            return False
        return self.rollback(
            self._checkpoints[-1].checkpoint_id,
            state_machine,
            memory,
        )

    def get_checkpoint(self, checkpoint_id: str) -> SessionCheckpoint | None:
        for cp in self._checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                return cp
        return None

    def clear(self) -> None:
        self._checkpoints.clear()
        self._counter = 0
