"""Checkpoint & rollback — stage-level checkpointing with revert support."""

from __future__ import annotations

from agentic_research.logging import get_logger
from agentic_research.memory.session import ResearchSessionMemory
from agentic_research.models.session import (
    SessionCheckpoint,
    StageTokenUsage,
)
from agentic_research.orchestrator.state import PipelineStateMachine

log = get_logger(__name__)


class CheckpointManager:
    """Manages stage-level checkpoints for rollback support."""

    def __init__(self) -> None:
        self._checkpoints: list[SessionCheckpoint] = []
        self._counter = 0

    @property
    def checkpoints(self) -> list[SessionCheckpoint]:
        return list(self._checkpoints)

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

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

        log.info(
            "checkpoint_created",
            checkpoint_id=checkpoint_id,
            stage=state_machine.current_stage.value,
            total_checkpoints=len(self._checkpoints),
        )
        return checkpoint

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
