"""Pydantic models for the conjecture refinement loop (Phase 8)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from agentic_research.models.agents import TokenUsage
from agentic_research.models.research import Conjecture


class RefinementType(str, Enum):
    WEAKENING = "weakening"
    STRENGTHENING = "strengthening"
    REFORMULATION = "reformulation"
    SPECIALIZATION = "specialization"


class RefinementOutcome(str, Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    FORMALIZATION_FAILED = "formalization_failed"
    INTENT_MISMATCH = "intent_mismatch"
    PROOF_FAILED = "proof_failed"
    SKIPPED = "skipped"


class RefinementStatus(str, Enum):
    PROVED = "proved"
    EXHAUSTED = "exhausted"


class RefinementAttempt(BaseModel):
    """A single refinement attempt: original → refined, with outcome."""

    original: Conjecture
    refined: Conjecture
    refinement_type: RefinementType
    outcome: RefinementOutcome = RefinementOutcome.SKIPPED
    failure_reason: str = ""
    proof_code: str | None = None
    depth: int = Field(default=0, ge=0)


class RefinementHistory(BaseModel):
    """Full history of refinement attempts from an original idea."""

    original_idea: str = Field(description="The user's original rough idea")
    original_conjecture: Conjecture | None = None
    attempts: list[RefinementAttempt] = Field(default_factory=list)
    final_result: RefinementStatus | None = None

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

    @property
    def proved_variant(self) -> RefinementAttempt | None:
        for attempt in self.attempts:
            if attempt.outcome == RefinementOutcome.PROVED:
                return attempt
        return None


class RefinementReport(BaseModel):
    """Human-readable report of the refinement journey."""

    markdown_report: str = Field(default="", description="Formatted markdown report")
    structured_history: RefinementHistory = Field(default_factory=lambda: RefinementHistory(original_idea=""))


class RefinementResult(BaseModel):
    """End-to-end result from the refinement pipeline."""

    status: RefinementStatus
    proved_variant: Conjecture | None = None
    proof_code: str | None = None
    history: RefinementHistory = Field(default_factory=lambda: RefinementHistory(original_idea=""))
    report: RefinementReport | None = None
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    max_depth_reached: int = 0
