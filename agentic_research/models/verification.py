"""Pydantic models for the verification pipeline (Phase 6).

Covers intent verification (blind/direct/adversarial paths),
informalization (Lean→NL back-translation), and counterexample search.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class VerificationPath(str, Enum):
    BLIND = "blind"
    DIRECT = "direct"
    ADVERSARIAL = "adversarial"
    OPENAI = "openai"


class IntentVerdictType(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"


class PathVerdict(BaseModel):
    """Result from a single verification path."""

    path: VerificationPath
    verdict: IntentVerdictType
    concerns: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class IntentVerdict(BaseModel):
    """Aggregated verdict from all verification paths."""

    overall_verdict: IntentVerdictType
    path_verdicts: list[PathVerdict] = Field(default_factory=list)
    adjudication_notes: str = Field(default="")
    all_concerns: list[str] = Field(default_factory=list)

    @property
    def has_concerns(self) -> bool:
        return len(self.all_concerns) > 0


class CounterexampleStatus(str, Enum):
    DISPROVED = "disproved"
    PLAUSIBLE = "plausible"


class CounterexampleCandidate(BaseModel):
    """A single candidate counterexample."""

    description: str = Field(description="Natural language description of the counterexample")
    lean_code: str = Field(default="", description="Lean 4 code formalizing the counterexample")
    compilation_status: str = Field(default="not_attempted", description="ok/error/not_attempted")
    proves_negation: bool = Field(default=False, description="Whether this proves the negation")


class CounterexampleResult(BaseModel):
    """Result from the counterexample search."""

    status: CounterexampleStatus
    candidates_tried: list[CounterexampleCandidate] = Field(default_factory=list)
    successful_counterexample: CounterexampleCandidate | None = None
    attempts_made: int = 0

    @property
    def is_disproved(self) -> bool:
        return self.status == CounterexampleStatus.DISPROVED


class InformalizationResult(BaseModel):
    """Result from back-translating Lean code to natural language."""

    lean_input: str = Field(description="The original Lean 4 code")
    natural_language_output: str = Field(description="Natural language description")
