"""Pydantic models for the proof search and recursive decomposition pipeline (Phase 7)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from agentic_research.models.agents import TokenUsage


class StrategyType(str, Enum):
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CASE_ANALYSIS = "case_analysis"


class ProofStrategy(BaseModel):
    """A candidate proof strategy proposed by the LLM."""

    strategy_type: StrategyType
    description: str = Field(default="", description="Natural language description of the approach")
    relevant_lemmas: list[str] = Field(default_factory=list, description="Mathlib lemmas to use")
    plausibility: float = Field(default=0.5, ge=0.0, le=1.0, description="LLM-estimated plausibility")
    key_tactics: list[str] = Field(default_factory=list, description="Primary Lean tactics to try")


class FailureType(str, Enum):
    MISSING_HYPOTHESIS = "missing_hypothesis"
    WEAK_CHILD_LEMMA = "weak_child_lemma"
    CONTRADICTORY_CHILD = "contradictory_child"
    STUCK_GOAL = "stuck_goal"
    ASSEMBLY_ERROR = "assembly_error"
    TRUNCATED = "truncated"


class FailureDiagnosis(BaseModel):
    """Structured diagnosis of why a proof node failed."""

    failure_type: FailureType
    description: str = Field(default="", description="What went wrong")
    problematic_child_id: str | None = Field(
        default=None, description="ID of the child lemma causing the issue"
    )
    suggested_fix: str = Field(default="", description="LLM suggestion for fixing")
    lean_errors: list[str] = Field(default_factory=list)


class NodeStatus(str, Enum):
    PENDING = "pending"
    PROVED = "proved"
    FAILED = "failed"
    REFORMULATED = "reformulated"


class ProofNode(BaseModel):
    """A node in the lemma decomposition tree."""

    node_id: str = Field(description="Stable identifier (e.g., 'root', 'lemma_1', 'lemma_1_1')")
    statement_nl: str = Field(default="", description="Natural language statement")
    statement_lean: str = Field(default="", description="Lean 4 statement (with sorry body)")
    proof_code: str | None = Field(default=None, description="Lean 4 proof if proved")
    depth: int = Field(default=0, ge=0)
    children: list[str] = Field(default_factory=list, description="Child node IDs")
    parent_id: str | None = Field(default=None, description="Parent node ID")
    status: NodeStatus = Field(default=NodeStatus.PENDING)
    failure_diagnosis: FailureDiagnosis | None = None
    retries_used: int = Field(default=0, ge=0)
    from_prior_work: bool = Field(default=False, description="Tagged as from published work for axiomatization")
    source_reference: str | None = Field(
        default=None,
        description="Citation for axiomatized prior work (e.g., 'Kantorovich duality, Villani 2009')",
    )
    proof_sketch_nl: str | None = Field(
        default=None,
        description="Tactic-granularity proof sketch (3-5 intermediate steps)",
    )
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)


class LemmaTree(BaseModel):
    """Tree of decomposed sub-lemmas with topological ordering."""

    root_id: str = Field(description="ID of the root node")
    nodes: dict[str, ProofNode] = Field(default_factory=dict)
    topological_order: list[str] = Field(
        default_factory=list,
        description="Node IDs in topological order (leaves first)",
    )
    decomposition_score: float | None = Field(
        default=None,
        description="MVP scoring: weighted combination of brevity and structural balance",
    )

    def get_node(self, node_id: str) -> ProofNode | None:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[ProofNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def all_children_proved(self, node_id: str) -> bool:
        return all(c.status == NodeStatus.PROVED for c in self.get_children(node_id))

    @property
    def all_proved(self) -> bool:
        return all(n.status == NodeStatus.PROVED for n in self.nodes.values())


class ProofSearchResult(BaseModel):
    """Result from the ProofSearchAgent."""

    statement: str = Field(description="The Lean 4 statement being proved")
    proved: bool = False
    proof_code: str | None = None
    strategies_tried: list[ProofStrategy] = Field(default_factory=list)
    needs_decomposition: bool = Field(default=False, description="True if direct proof failed")
    mathlib_lemmas_found: list[str] = Field(default_factory=list)
    iterations_used: int = 0
    failure_reason: str | None = None


class RecursiveProofResult(BaseModel):
    """Result from the RecursiveProver."""

    root_statement: str = Field(description="Original statement being proved")
    proved: bool = False
    lemma_tree: LemmaTree | None = None
    final_proof: str | None = Field(default=None, description="Assembled proof if all nodes proved")
    total_nodes: int = 0
    proved_nodes: int = 0
    max_depth_reached: int = 0
    failure_reason: str | None = None


class ErrorCategory(str, Enum):
    TYPE_MISMATCH = "type_mismatch"
    MISSING_IMPORT = "missing_import"
    TACTIC_FAILURE = "tactic_failure"
    UNIVERSE_LEVEL = "universe_level"
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    TIMEOUT = "timeout"
    OTHER = "other"


class ProofCorrection(BaseModel):
    """Structured correction for a failed proof attempt."""

    error_category: ErrorCategory
    error_message: str = Field(description="Original Lean error")
    suggested_tactics: list[str] = Field(default_factory=list, description="Specific tactics to try")
    revised_proof_sketch: str = Field(default="", description="Corrected proof code")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Why this fix should work")


class CritiqueIssueType(str, Enum):
    UNSTATED_HYPOTHESIS = "unstated_hypothesis"
    UNDEFINED_TERM = "undefined_term"
    HIDDEN_CASE_SPLIT = "hidden_case_split"
    SWAPPED_QUANTIFIER = "swapped_quantifier"
    UNJUSTIFIED_STEP = "unjustified_step"
    CIRCULAR_REASONING = "circular_reasoning"
    WEAK_CHILD_LEMMA = "weak_child_lemma"
    INCOMPLETE_DECOMPOSITION = "incomplete_decomposition"


class CritiqueIssue(BaseModel):
    """A soundness concern raised by the proof critic."""

    issue_type: CritiqueIssueType
    node_id: str = Field(description="Node where the issue was found")
    description: str = Field(description="Concrete question, e.g., 'Does this assume x > 0 without stating it?'")
    severity: str = Field(default="warning", description="'blocking' or 'warning'")
    suggested_fix: str = Field(default="", description="Actionable fix suggestion")
    confirmed: bool = Field(default=False, description="True after adversarial self-check confirms the issue")


class CritiqueResult(BaseModel):
    """Result of auditing a LemmaTree for soundness."""

    issues: list[CritiqueIssue] = Field(default_factory=list)
    passed: bool = Field(default=True)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class NLProofStep(BaseModel):
    """A single step in a natural language proof sketch."""

    claim: str
    reasoning: str
    sub_claims: list[str] = Field(default_factory=list)


class NLProofSketch(BaseModel):
    """Structured informal proof produced before Lean formalization."""

    proof_steps: list[NLProofStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    key_lemmas: list[str] = Field(default_factory=list)
    overall_strategy: str = ""


class ProofPipelineResult(BaseModel):
    """End-to-end result from the proof pipeline."""

    statement: str
    proved: bool = False
    final_proof: str | None = None
    search_result: ProofSearchResult | None = None
    recursive_result: RecursiveProofResult | None = None
    claim_check_passed: bool | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    backtrack_stages: list[str] = Field(default_factory=list)
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    backend: str | None = Field(default=None, description="Lean backend used (mock/subprocess/lean_dojo)")
    verified: bool = Field(default=True, description="False if proof was run in mock mode")
