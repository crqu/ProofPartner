"""Pydantic models for the orchestrator, session memory, and checkpointing (Phase 9)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from agentic_research.models.agents import TokenUsage
from agentic_research.models.refinement import RefinementHistory
from agentic_research.models.research import Conjecture


class PipelineStage(str, Enum):
    """All stages in the explore-conjecture-prove loop."""

    EXPLORING = "exploring"
    CONJECTURING = "conjecturing"
    FORMALIZING = "formalizing"
    CHECKING_INTENT = "checking_intent"
    SEARCHING_COUNTEREXAMPLE = "searching_counterexample"
    PROVING = "proving"
    REFINING = "refining"
    COMPLETE = "complete"
    FAILED = "failed"


TERMINAL_STAGES = frozenset({PipelineStage.COMPLETE, PipelineStage.FAILED})

VALID_TRANSITIONS: dict[PipelineStage, frozenset[PipelineStage]] = {
    PipelineStage.EXPLORING: frozenset({
        PipelineStage.CONJECTURING,
        PipelineStage.FAILED,
    }),
    PipelineStage.CONJECTURING: frozenset({
        PipelineStage.FORMALIZING,
        PipelineStage.EXPLORING,
        PipelineStage.FAILED,
    }),
    PipelineStage.FORMALIZING: frozenset({
        PipelineStage.CHECKING_INTENT,
        PipelineStage.REFINING,
        PipelineStage.FAILED,
    }),
    PipelineStage.CHECKING_INTENT: frozenset({
        PipelineStage.SEARCHING_COUNTEREXAMPLE,
        PipelineStage.REFINING,
        PipelineStage.FAILED,
    }),
    PipelineStage.SEARCHING_COUNTEREXAMPLE: frozenset({
        PipelineStage.PROVING,
        PipelineStage.REFINING,
        PipelineStage.FAILED,
    }),
    PipelineStage.PROVING: frozenset({
        PipelineStage.COMPLETE,
        PipelineStage.REFINING,
        PipelineStage.CONJECTURING,
        PipelineStage.FAILED,
    }),
    PipelineStage.REFINING: frozenset({
        PipelineStage.FORMALIZING,
        PipelineStage.CONJECTURING,
        PipelineStage.EXPLORING,
        PipelineStage.COMPLETE,
        PipelineStage.FAILED,
    }),
    PipelineStage.COMPLETE: frozenset(),
    PipelineStage.FAILED: frozenset(),
}


class ConjectureOutcome(str, Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    PROOF_FAILED = "proof_failed"
    FORMALIZATION_FAILED = "formalization_failed"
    INTENT_MISMATCH = "intent_mismatch"
    PENDING = "pending"


class TriedConjecture(BaseModel):
    """Record of a conjecture that was attempted."""

    conjecture: Conjecture
    outcome: ConjectureOutcome = ConjectureOutcome.PENDING
    lean_statement: str = ""
    proof_code: str | None = None
    failure_reason: str = ""
    stage_reached: PipelineStage = PipelineStage.CONJECTURING


class PartialResult(BaseModel):
    """A lemma or intermediate result proved during a failed attempt."""

    lemma_statement: str = Field(description="Natural language or Lean statement")
    lean_code: str = Field(default="", description="Lean 4 proof code")
    source_conjecture: str = Field(default="", description="Which conjecture produced this")
    domain: str = Field(default="", description="Mathematical domain")


class PromisingDirection(BaseModel):
    """A research direction identified but not yet explored."""

    title: str
    description: str = ""
    source: str = Field(default="", description="Which exploration round proposed this")
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class UserPreference(BaseModel):
    """A direction or choice the user indicated interest in."""

    preference: str = Field(description="What the user preferred")
    context: str = Field(default="", description="When/why they indicated this")


class SessionMemoryData(BaseModel):
    """Persistent memory across a research session."""

    tried_conjectures: list[TriedConjecture] = Field(default_factory=list)
    partial_results: list[PartialResult] = Field(default_factory=list)
    promising_directions: list[PromisingDirection] = Field(default_factory=list)
    user_preferences: list[UserPreference] = Field(default_factory=list)

    def has_tried(self, statement: str) -> bool:
        return any(tc.conjecture.statement == statement for tc in self.tried_conjectures)

    def proved_conjectures(self) -> list[TriedConjecture]:
        return [tc for tc in self.tried_conjectures if tc.outcome == ConjectureOutcome.PROVED]

    def failed_conjectures(self) -> list[TriedConjecture]:
        return [
            tc for tc in self.tried_conjectures
            if tc.outcome not in (ConjectureOutcome.PROVED, ConjectureOutcome.PENDING)
        ]

    def by_domain(self, domain: str) -> list[TriedConjecture]:
        return [
            tc for tc in self.tried_conjectures
            if domain.lower() in tc.conjecture.statement.lower()
        ]

    def by_outcome(self, outcome: ConjectureOutcome) -> list[TriedConjecture]:
        return [tc for tc in self.tried_conjectures if tc.outcome == outcome]


class StageTokenUsage(BaseModel):
    """Token usage for a single pipeline stage execution."""

    stage: PipelineStage
    agent_name: str = ""
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class CostEstimate(BaseModel):
    """Cost estimate based on Claude Opus 4.6 pricing."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cache_read_cost_usd: float = 0.0
    cache_write_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return (
            self.input_cost_usd
            + self.output_cost_usd
            + self.cache_read_cost_usd
            + self.cache_write_cost_usd
        )


OPUS_INPUT_PRICE_PER_MTOK = 15.0
OPUS_OUTPUT_PRICE_PER_MTOK = 75.0
OPUS_CACHE_READ_PRICE_PER_MTOK = 1.5
OPUS_CACHE_WRITE_PRICE_PER_MTOK = 18.75


def compute_cost(usage: TokenUsage) -> CostEstimate:
    """Compute cost estimate from token usage based on Claude Opus 4.6 pricing."""
    return CostEstimate(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens,
        cache_write_tokens=usage.cache_creation_input_tokens,
        input_cost_usd=usage.input_tokens * OPUS_INPUT_PRICE_PER_MTOK / 1_000_000,
        output_cost_usd=usage.output_tokens * OPUS_OUTPUT_PRICE_PER_MTOK / 1_000_000,
        cache_read_cost_usd=usage.cache_read_input_tokens * OPUS_CACHE_READ_PRICE_PER_MTOK / 1_000_000,
        cache_write_cost_usd=usage.cache_creation_input_tokens * OPUS_CACHE_WRITE_PRICE_PER_MTOK / 1_000_000,
    )


class StateTransition(BaseModel):
    """Record of a state transition in the pipeline."""

    from_state: PipelineStage
    to_state: PipelineStage
    reason: str = ""
    conjecture_index: int | None = None


class SessionState(BaseModel):
    """Current state of the orchestrator pipeline."""

    stage: PipelineStage = PipelineStage.EXPLORING
    raw_idea: str = ""
    active_conjecture_index: int | None = None
    conjectures_processed: int = 0
    refinements_attempted: int = 0
    transitions: list[StateTransition] = Field(default_factory=list)


class SessionCheckpoint(BaseModel):
    """Snapshot of session state at a point in time for rollback."""

    checkpoint_id: str = Field(description="Unique checkpoint identifier")
    stage: PipelineStage
    session_state: SessionState
    memory: SessionMemoryData = Field(default_factory=SessionMemoryData)
    stage_token_usages: list[StageTokenUsage] = Field(default_factory=list)


class OrchestratorConfig(BaseModel):
    """Configuration for the central orchestrator."""

    max_conjectures: int = Field(default=5, ge=1, description="Max conjectures to evaluate per session")
    max_refinements: int = Field(default=3, ge=0, description="Max refinement attempts per conjecture")
    budget_limit_usd: float | None = Field(default=10.0, ge=0.0, description="Cost ceiling in USD")
    auto_mode: bool = Field(default=True, description="If True, proceed without user input at decision points")
    max_exploration_rounds: int = Field(default=2, ge=1, description="Max times to loop back to Explorer")
    max_reasoning_cycles: int = Field(default=25, ge=1, description="Max stage transitions before halting")
    use_proof_critic: bool = Field(default=True, description="Enable ProofCritic for lemma decomposition review")
    use_proof_detailer: bool = Field(default=True, description="Enable ProofDetailer for proof sketch enrichment")


class ResearchSessionResult(BaseModel):
    """Final result of a complete research session."""

    session_id: str = ""
    raw_idea: str = ""
    proved_conjectures: list[TriedConjecture] = Field(default_factory=list)
    failed_conjectures: list[TriedConjecture] = Field(default_factory=list)
    partial_results: list[PartialResult] = Field(default_factory=list)
    refinement_histories: list[RefinementHistory] = Field(default_factory=list)
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_estimate: CostEstimate = Field(default_factory=CostEstimate)
    final_stage: PipelineStage = PipelineStage.FAILED
    total_conjectures_tried: int = 0
    total_refinements: int = 0
    exploration_rounds: int = 0
