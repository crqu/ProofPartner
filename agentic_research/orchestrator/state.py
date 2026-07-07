"""Pipeline state machine with validated transitions and event emission."""

from __future__ import annotations

from agentic_research.logging import get_logger
from agentic_research.models.session import (
    PipelineStage,
    SessionState,
    StateTransition,
    TERMINAL_STAGES,
    VALID_TRANSITIONS,
)

log = get_logger(__name__)


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""


class PipelineStateMachine:
    """Manages pipeline state with validated transitions and event logging."""

    def __init__(self, initial_state: PipelineStage = PipelineStage.EXPLORING) -> None:
        self._state = SessionState(stage=initial_state)

    @property
    def current_stage(self) -> PipelineStage:
        return self._state.stage

    @property
    def session_state(self) -> SessionState:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._state.stage in TERMINAL_STAGES

    def can_transition(self, to_stage: PipelineStage) -> bool:
        allowed = VALID_TRANSITIONS.get(self._state.stage, frozenset())
        return to_stage in allowed

    def transition(
        self,
        to_stage: PipelineStage,
        *,
        reason: str = "",
        conjecture_index: int | None = None,
    ) -> None:
        if not self.can_transition(to_stage):
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.stage.value} to {to_stage.value}"
            )

        from_stage = self._state.stage
        record = StateTransition(
            from_state=from_stage,
            to_state=to_stage,
            reason=reason,
            conjecture_index=conjecture_index,
        )
        self._state.transitions.append(record)
        self._state.stage = to_stage

        if conjecture_index is not None:
            self._state.active_conjecture_index = conjecture_index

        log.info(
            "pipeline_state_transition",
            from_state=from_stage.value,
            to_state=to_stage.value,
            reason=reason,
            conjecture_index=conjecture_index,
            transition_count=len(self._state.transitions),
        )

    def restore(self, state: SessionState) -> None:
        self._state = state.model_copy(deep=True)
        log.info("pipeline_state_restored", stage=self._state.stage.value)
