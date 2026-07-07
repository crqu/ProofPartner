"""Pydantic models for the type-first formalization pipeline (Phase 5)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from agentic_research.models.agents import TokenUsage


class TypeCandidate(BaseModel):
    """A type that needs to be formalized in Lean 4."""

    name: str = Field(description="Human-readable name (e.g., 'QuasiRandomGraph')")
    informal_description: str = Field(default="", description="Natural language description")
    lean_signature: str = Field(default="", description="Proposed Lean 4 type signature")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Names of other TypeCandidates this depends on",
    )
    mathlib_analog: str | None = Field(
        default=None,
        description="Closest Mathlib type if one exists",
    )
    is_in_mathlib: bool = Field(
        default=False,
        description="Whether this type already exists in Mathlib",
    )


class TypeDependencyGraph(BaseModel):
    """Dependency graph between type candidates."""

    edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Directed edges (from, to) meaning 'from' depends on 'to'",
    )
    topological_order: list[str] = Field(
        default_factory=list,
        description="Type names in topological order (dependencies first)",
    )


class TypePlan(BaseModel):
    """Output of the Type Planner — types needed for a conjecture."""

    conjecture_statement: str = Field(description="The NL conjecture being formalized")
    candidates: list[TypeCandidate] = Field(default_factory=list)
    dependency_graph: TypeDependencyGraph = Field(default_factory=TypeDependencyGraph)
    mathlib_imports: list[str] = Field(
        default_factory=list,
        description="Mathlib modules to import for existing types",
    )


class LemmaStatement(BaseModel):
    """A single auxiliary lemma statement (NL or Lean)."""

    name: str = Field(description="Lemma identifier")
    statement_nl: str = Field(description="Natural language statement")
    statement_lean: str = Field(default="", description="Lean 4 statement (filled by formalizer)")
    for_type: str = Field(description="Which TypeCandidate this lemma validates")
    is_well_known: bool = Field(default=True, description="Whether this is a known property")


class AuxiliaryLemma(BaseModel):
    """An auxiliary lemma with its proof status."""

    lemma: LemmaStatement
    lean_code: str = Field(default="", description="Complete Lean 4 code including proof")
    proved: bool = False
    proof_code: str = Field(default="", description="Just the proof term/tactic block")
    error_message: str | None = None


class TypeFormalizationCandidate(BaseModel):
    """A single candidate formalization for a type (one of k attempts)."""

    candidate_id: int = Field(description="Index among parallel candidates")
    type_name: str = Field(description="Which type this formalizes")
    lean_code: str = Field(default="", description="Complete Lean 4 type definition")
    compiles: bool = False
    auxiliary_lemmas: list[AuxiliaryLemma] = Field(default_factory=list)

    @property
    def proved_count(self) -> int:
        return sum(1 for lem in self.auxiliary_lemmas if lem.proved)

    @property
    def total_lemma_count(self) -> int:
        return len(self.auxiliary_lemmas)

    @property
    def proved_ratio(self) -> float:
        if not self.auxiliary_lemmas:
            return 0.0
        return self.proved_count / self.total_lemma_count


class AuctionScore(BaseModel):
    """Scoring breakdown for a type formalization candidate."""

    candidate_id: int
    lemma_ratio: float = Field(default=0.0, description="Proportion of proved lemmas")
    brevity_score: float = Field(default=0.0, description="Inverse of code length, normalized")
    compilation_score: float = Field(default=0.0, description="1.0 if compiles, 0.0 otherwise")
    total_score: float = Field(default=0.0, description="Weighted composite score")


class AuctionVerdict(str, Enum):
    ACCEPTED = "accepted"
    RETRY = "retry"


class AuctionResult(BaseModel):
    """Result of the best-of-k auction for a single type."""

    type_name: str
    verdict: AuctionVerdict
    winner_id: int | None = None
    scores: list[AuctionScore] = Field(default_factory=list)
    winning_candidate: TypeFormalizationCandidate | None = None
    reason: str = Field(default="", description="Why this candidate won or why retry is needed")


class TypeFormalizationResult(BaseModel):
    """Result of formalizing all types for a conjecture."""

    type_plan: TypePlan
    auction_results: list[AuctionResult] = Field(default_factory=list)
    accepted_types: list[TypeFormalizationCandidate] = Field(default_factory=list)
    all_types_accepted: bool = False
    total_proved_lemmas: int = 0
    total_failed_lemmas: int = 0


class TheoremFormalization(BaseModel):
    """Result of formalizing the theorem statement."""

    conjecture_nl: str = Field(description="Original NL conjecture")
    lean_statement: str = Field(default="", description="Lean 4 theorem statement")
    compiles: bool = False
    iterations_used: int = 0
    type_imports: list[str] = Field(
        default_factory=list,
        description="Type definitions needed before the theorem",
    )
    failure_reason: str | None = None


class ClaimCheckVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class ClaimCheckResult(BaseModel):
    """Result of verifying formalization fidelity."""

    verdict: ClaimCheckVerdict
    original_statement: str = ""
    formalized_statement: str = ""
    reason: str = Field(default="", description="Explanation of pass/fail")
    statement_preserved: bool = True


class FormalizationPipelineResult(BaseModel):
    """End-to-end result from the formalization pipeline."""

    conjecture_nl: str
    type_formalization: TypeFormalizationResult | None = None
    theorem: TheoremFormalization | None = None
    claim_check: ClaimCheckResult | None = None
    success: bool = False
    retry_count: int = 0
    failure_stage: str | None = None
    failure_reason: str | None = None
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
