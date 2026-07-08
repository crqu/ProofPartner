"""End-to-end orchestrator tests with mocked LLM responses and real Lean 4.

Exercises the full 8-stage research loop:
  EXPLORING → CONJECTURING → FORMALIZING → CHECKING_INTENT →
  SEARCHING_COUNTEREXAMPLE → PROVING → (REFINING →) COMPLETE

LLM calls are mocked with pre-scripted responses; Lean compilation
uses real subprocess backend (gated with lean_required marker).
"""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock, patch

import pytest

from agentic_research.models.agents import (
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.formalization import (
    ClaimCheckResult,
    ClaimCheckVerdict,
    FormalizationPipelineResult,
    TheoremFormalization,
)
from agentic_research.models.proof import (
    ProofPipelineResult,
    ProofSearchResult,
)
from agentic_research.models.refinement import RefinementResult, RefinementStatus
from agentic_research.models.research import (
    Conjecture,
    ConjectureSet,
    ExplorationResult,
    ResearchDirection,
)
from agentic_research.models.session import (
    OrchestratorConfig,
    PipelineStage,
)
from agentic_research.models.verification import (
    CounterexampleResult,
    CounterexampleStatus,
    IntentVerdict,
    IntentVerdictType,
    PathVerdict,
    VerificationPath,
)
from agentic_research.orchestrator.engine import ResearchOrchestrator
from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

LEAN_AVAILABLE = shutil.which("lean") is not None

lean_required = pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")

LEAN_STMT = "theorem test_add_comm : ∀ n m : Nat, n + m = m + n := by omega"
LEAN_PROOF = "theorem test_add_comm : ∀ n m : Nat, n + m = m + n := by omega"

RAW_IDEA = "Prove that addition of natural numbers is commutative"


def _token_usage() -> TokenUsage:
    return TokenUsage(input_tokens=100, output_tokens=50)


def _make_search() -> LeanSearch:
    return LeanSearch(SearchConfig(backend=SearchBackend.MOCK))


def _make_repl(real: bool = False) -> LeanRepl:
    if real and LEAN_AVAILABLE:
        return LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


def _make_mock_llm() -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    mock.model = "claude-opus-4-6-20250616"
    mock.complete.return_value = MagicMock(
        content="{}",
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=_token_usage(),
    )
    mock.extract_json.return_value = {}
    return mock


def _conjecture() -> Conjecture:
    return Conjecture(
        statement="∀ n m : Nat, n + m = m + n",
        natural_language="For all natural numbers n and m, n + m = m + n",
        confidence=0.99,
        difficulty=1,
        novelty_score=0.2,
        formalizability_score=0.95,
    )


def _exploration_result() -> AgentResult:
    return AgentResult(
        agent_name="exploration_agent",
        status=AgentStatus.SUCCESS,
        result=ExplorationResult(
            raw_idea=RAW_IDEA,
            domain="number_theory",
            directions=[
                ResearchDirection(
                    title="Commutativity of addition",
                    description="Prove n + m = m + n for natural numbers",
                    ambition_level=1,
                )
            ],
        ).model_dump(),
        token_usage=_token_usage(),
    )


def _conjecture_result() -> AgentResult:
    return AgentResult(
        agent_name="conjecture_generator",
        status=AgentStatus.SUCCESS,
        result=ConjectureSet(
            conjectures=[_conjecture()],
            ranking=[0],
        ).model_dump(),
        token_usage=_token_usage(),
    )


def _formalization_success() -> FormalizationPipelineResult:
    return FormalizationPipelineResult(
        conjecture_nl="For all natural numbers n and m, n + m = m + n",
        theorem=TheoremFormalization(
            conjecture_nl="For all natural numbers n and m, n + m = m + n",
            lean_statement=LEAN_STMT,
            compiles=True,
        ),
        claim_check=ClaimCheckResult(
            verdict=ClaimCheckVerdict.PASS,
            original_statement="n + m = m + n",
            formalized_statement=LEAN_STMT,
        ),
        success=True,
    )


def _formalization_failure() -> FormalizationPipelineResult:
    return FormalizationPipelineResult(
        conjecture_nl="For all natural numbers n and m, n + m = m + n",
        failure_stage="theorem_formalization",
        failure_reason="Could not formalize theorem",
    )


def _intent_ok() -> IntentVerdict:
    return IntentVerdict(
        overall_verdict=IntentVerdictType.CORRECT,
        path_verdicts=[
            PathVerdict(
                path=VerificationPath.BLIND,
                verdict=IntentVerdictType.CORRECT,
                confidence=0.95,
            ),
        ],
    )


def _cx_plausible() -> CounterexampleResult:
    return CounterexampleResult(
        status=CounterexampleStatus.PLAUSIBLE,
        attempts_made=3,
    )


def _proof_success() -> ProofPipelineResult:
    return ProofPipelineResult(
        statement=LEAN_STMT,
        proved=True,
        final_proof=LEAN_PROOF,
        search_result=ProofSearchResult(statement=LEAN_STMT, proved=True, proof_code=LEAN_PROOF),
        claim_check_passed=True,
    )


def _proof_failure() -> ProofPipelineResult:
    return ProofPipelineResult(
        statement=LEAN_STMT,
        proved=False,
        search_result=ProofSearchResult(
            statement=LEAN_STMT, proved=False, failure_reason="Proof search exhausted"
        ),
        failure_stage="proof_search",
        failure_reason="Proof search exhausted",
    )


def _refinement_success() -> RefinementResult:
    return RefinementResult(
        status=RefinementStatus.PROVED,
        proved_variant=_conjecture(),
        proof_code=LEAN_PROOF,
    )


def _refinement_exhausted() -> RefinementResult:
    return RefinementResult(status=RefinementStatus.EXHAUSTED)


def _make_orchestrator(
    *, real_lean: bool = False, max_conjectures: int = 1, max_refinements: int = 1,
) -> ResearchOrchestrator:
    config = OrchestratorConfig(
        max_conjectures=max_conjectures,
        max_refinements=max_refinements,
        budget_limit_usd=100.0,
        max_exploration_rounds=2,
        max_reasoning_cycles=25,
    )
    return ResearchOrchestrator(
        llm_client=_make_mock_llm(),
        lean_repl=_make_repl(real=real_lean),
        lean_search=_make_search(),
        config=config,
        session_id="e2e_test",
    )


# ---------------------------------------------------------------------------
# Happy path: all 8 stages fire in order → COMPLETE
# ---------------------------------------------------------------------------


class TestOrchestratorE2EHappyPath:
    """EXPLORING → CONJECTURING → FORMALIZING → CHECKING_INTENT →
    SEARCHING_COUNTEREXAMPLE → PROVING → COMPLETE"""

    def test_full_8_stage_loop(self):
        orch = _make_orchestrator()

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_success(),
            ),
        ):
            result = orch.run(RAW_IDEA)

        assert result.final_stage == PipelineStage.COMPLETE
        assert result.session_id == "e2e_test"
        assert len(result.proved_conjectures) == 1
        assert result.exploration_rounds == 1
        assert result.total_conjectures_tried == 1

    def test_state_transitions_recorded(self):
        orch = _make_orchestrator()

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_success(),
            ),
        ):
            result = orch.run(RAW_IDEA)

        transitions = orch.state_machine.session_state.transitions
        stages_visited = [t.to_state for t in transitions]

        assert PipelineStage.CONJECTURING in stages_visited
        assert PipelineStage.FORMALIZING in stages_visited
        assert PipelineStage.CHECKING_INTENT in stages_visited
        assert PipelineStage.SEARCHING_COUNTEREXAMPLE in stages_visited
        assert PipelineStage.PROVING in stages_visited
        assert PipelineStage.COMPLETE in stages_visited
        assert result.final_stage == PipelineStage.COMPLETE

    def test_checkpoints_created(self):
        orch = _make_orchestrator()

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_success(),
            ),
        ):
            orch.run(RAW_IDEA)

        assert orch.checkpoint_manager.checkpoint_count > 0


# ---------------------------------------------------------------------------
# Refinement: proof fails → refinement → succeeds
# ---------------------------------------------------------------------------


class TestOrchestratorE2ERefinement:
    """Proof search fails → enters REFINING → refinement pipeline proves → COMPLETE."""

    def test_refinement_proves_after_failure(self):
        orch = _make_orchestrator(max_conjectures=1, max_refinements=3)

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_failure(),
            ),
            patch(
                "agentic_research.pipelines.refinement.RefinementPipeline.run",
                return_value=_refinement_success(),
            ),
        ):
            result = orch.run(RAW_IDEA)

        assert result.final_stage == PipelineStage.COMPLETE
        assert result.total_refinements >= 1

        transitions = orch.state_machine.session_state.transitions
        stages_visited = [t.to_state for t in transitions]
        assert PipelineStage.REFINING in stages_visited

    def test_refinement_exhausted_fails(self):
        orch = _make_orchestrator(max_conjectures=1, max_refinements=1)

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_failure(),
            ),
            patch(
                "agentic_research.pipelines.refinement.RefinementPipeline.run",
                return_value=_refinement_exhausted(),
            ),
        ):
            result = orch.run(RAW_IDEA)

        assert result.final_stage == PipelineStage.FAILED


# ---------------------------------------------------------------------------
# Formalization failure → refinement path
# ---------------------------------------------------------------------------


class TestOrchestratorE2EFormalizationFailure:
    """Formalization fails → enters REFINING directly."""

    def test_formalization_failure_triggers_refinement(self):
        orch = _make_orchestrator(max_conjectures=1, max_refinements=1)

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_failure(),
            ),
            patch(
                "agentic_research.pipelines.refinement.RefinementPipeline.run",
                return_value=_refinement_exhausted(),
            ),
        ):
            result = orch.run(RAW_IDEA)

        transitions = orch.state_machine.session_state.transitions
        stages_visited = [t.to_state for t in transitions]
        assert PipelineStage.REFINING in stages_visited
        assert result.final_stage == PipelineStage.FAILED


# ---------------------------------------------------------------------------
# Real Lean 4 compilation validation (lean_required)
# ---------------------------------------------------------------------------


@lean_required
class TestOrchestratorE2ERealLean:
    """Validate that scripted Lean outputs actually compile with real Lean 4."""

    def test_lean_stmt_compiles(self):
        """The Lean statement used in E2E tests should actually compile."""
        repl = _make_repl(real=True)
        result = repl.execute(LEAN_STMT)
        assert result.compilation_status.value == "ok", (
            f"Lean compilation failed: {result.errors}"
        )
        assert result.all_goals_closed

    def test_happy_path_with_real_lean(self):
        """Full 8-stage loop using real Lean 4 for compilation checks."""
        orch = _make_orchestrator(real_lean=True)

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_success(),
            ),
        ):
            result = orch.run(RAW_IDEA)

        assert result.final_stage == PipelineStage.COMPLETE

        repl = _make_repl(real=True)
        compilation = repl.execute(LEAN_PROOF)
        assert compilation.compilation_status.value == "ok"
        assert compilation.all_goals_closed

    def test_refinement_with_real_lean(self):
        """Refinement loop with real Lean 4 — proves after initial failure."""
        orch = _make_orchestrator(real_lean=True, max_refinements=3)

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_failure(),
            ),
            patch(
                "agentic_research.pipelines.refinement.RefinementPipeline.run",
                return_value=_refinement_success(),
            ),
        ):
            result = orch.run(RAW_IDEA)

        assert result.final_stage == PipelineStage.COMPLETE
        assert result.total_refinements >= 1

        repl = _make_repl(real=True)
        compilation = repl.execute(LEAN_PROOF)
        assert compilation.compilation_status.value == "ok"


# ---------------------------------------------------------------------------
# Token tracking through full loop
# ---------------------------------------------------------------------------


class TestOrchestratorE2ETokenTracking:
    """Verify tokens accumulate across all stages."""

    def test_tokens_accumulated_after_happy_path(self):
        orch = _make_orchestrator()

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_success(),
            ),
        ):
            result = orch.run(RAW_IDEA)

        assert result.total_token_usage.input_tokens > 0
        assert result.total_token_usage.output_tokens > 0
        assert result.cost_estimate.total_cost_usd > 0


# ---------------------------------------------------------------------------
# Memory state after full loop
# ---------------------------------------------------------------------------


class TestOrchestratorE2EMemory:
    """Verify session memory is populated correctly after full runs."""

    def test_memory_records_proved_conjecture(self):
        orch = _make_orchestrator()

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_success(),
            ),
        ):
            orch.run(RAW_IDEA)

        proved = orch.memory.data.proved_conjectures()
        assert len(proved) == 1
        assert proved[0].proof_code == LEAN_PROOF

    def test_memory_records_promising_direction(self):
        orch = _make_orchestrator()

        with (
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.run",
                return_value=_exploration_result(),
            ),
            patch(
                "agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.run",
                return_value=_conjecture_result(),
            ),
            patch(
                "agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.formalization.FormalizationPipeline.run",
                return_value=_formalization_success(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.judge",
                return_value=_intent_ok(),
            ),
            patch(
                "agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search",
                return_value=_cx_plausible(),
            ),
            patch(
                "agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens",
                new_callable=lambda: property(lambda self: _token_usage()),
            ),
            patch(
                "agentic_research.pipelines.proof.ProofPipeline.run",
                return_value=_proof_success(),
            ),
        ):
            orch.run(RAW_IDEA)

        directions = orch.memory.data.promising_directions
        assert len(directions) >= 1
        assert directions[0].title == "Commutativity of addition"
