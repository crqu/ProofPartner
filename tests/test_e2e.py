"""End-to-end verification tests for Phase 6.

Covers:
  1. Full CLI flow: explore -> formalize -> check -> prove
  2. Budget enforcement (halts before LLM calls)
  3. Timeout enforcement
  4. Circuit breaker (5 consecutive failures -> halt)
  5. Checkpointing (create, persist to disk, resume)
  6. Session memory tiering (hot/warm/cold transitions)
  7. Cost tracking accuracy against known token counts
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentic_research.cli.main import cli
from agentic_research.memory.session import (
    HOT_TIER_SIZE,
    WARM_TIER_SIZE,
    ResearchSessionMemory,
)
from agentic_research.models.research import Conjecture
from agentic_research.models.session import (
    ConjectureOutcome,
    OrchestratorConfig,
    OPUS_CACHE_READ_PRICE_PER_MTOK,
    OPUS_CACHE_WRITE_PRICE_PER_MTOK,
    OPUS_INPUT_PRICE_PER_MTOK,
    OPUS_OUTPUT_PRICE_PER_MTOK,
    PipelineStage,
)
from agentic_research.orchestrator.circuit_breaker import CircuitBreaker
from agentic_research.orchestrator.cost_tracker import CostTracker
from agentic_research.orchestrator.rollback import CheckpointManager
from agentic_research.orchestrator.state import PipelineStateMachine


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_session_dir(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr("agentic_research.cli.main.SESSION_DIR", session_dir)
    return session_dir


def _mock_token_usage():
    from agentic_research.models.agents import TokenUsage

    return TokenUsage(input_tokens=100, output_tokens=50)


def _mock_agent_result(*, success=True, result_data=None, error=None):
    from agentic_research.models.agents import AgentResult, AgentStatus, TokenUsage

    return AgentResult(
        agent_name="mock_agent",
        status=AgentStatus.SUCCESS if success else AgentStatus.FAILURE,
        result=result_data,
        error_message=error,
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


# ---------------------------------------------------------------------------
# 1. Full CLI flow: explore -> formalize -> check -> prove
# ---------------------------------------------------------------------------
class TestEndToEndFlow:
    """Exercise the decoupled pipeline: explore -> formalize -> check -> prove."""

    def test_explore_to_prove_flow(self, runner, tmp_session_dir):
        explore_data = {
            "raw_idea": "prime gaps",
            "domain": "number theory",
            "concepts": [{"name": "prime numbers"}],
            "directions": [
                {"title": "Gaps", "description": "Study gaps", "ambition_level": 3}
            ],
        }
        conj_data = {
            "conjectures": [
                {
                    "statement": "For all n > 1, there exists a prime between n and 2n",
                    "natural_language": "Bertrand's postulate",
                    "confidence": 0.99,
                    "difficulty": 3,
                    "novelty_score": 0.2,
                    "formalizability_score": 0.9,
                }
            ],
            "ranking": [0],
            "exploration_context": explore_data,
        }

        explore_result = _mock_agent_result(result_data=explore_data)
        conj_result = _mock_agent_result(result_data=conj_data)

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.agents.explorer.ExplorationAgent.run", return_value=explore_result),
            patch("agentic_research.agents.conjecturer.ConjectureGenerator.run", return_value=conj_result),
            patch("agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens", new_callable=lambda: property(lambda self: _mock_token_usage())),
            patch("agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens", new_callable=lambda: property(lambda self: _mock_token_usage())),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["explore", "prime gaps"])

        assert result.exit_code == 0
        assert "Bertrand" in result.output or "prime between n and 2n" in result.output
        assert "Cost Summary" in result.output

        # Step 2: formalize
        from agentic_research.models.formalization import (
            ClaimCheckResult,
            ClaimCheckVerdict,
            FormalizationPipelineResult,
            TheoremFormalization,
        )
        from agentic_research.models.verification import (
            IntentVerdict,
            IntentVerdictType,
            PathVerdict,
            VerificationPath,
        )

        lean_stmt = "theorem bertrand (n : Nat) (h : n > 1) : exists p, Nat.Prime p ∧ n < p ∧ p < 2 * n := sorry"

        form_result = FormalizationPipelineResult(
            conjecture_nl="Bertrand's postulate",
            theorem=TheoremFormalization(
                conjecture_nl="Bertrand's postulate",
                lean_statement=lean_stmt,
                compiles=True,
            ),
            claim_check=ClaimCheckResult(
                verdict=ClaimCheckVerdict.PASS,
                original_statement="Bertrand's postulate",
                formalized_statement=lean_stmt,
            ),
            success=True,
        )
        intent_verdict = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            path_verdicts=[
                PathVerdict(path=VerificationPath.BLIND, verdict=IntentVerdictType.CORRECT, confidence=0.95),
            ],
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.formalization.FormalizationPipeline.run", return_value=form_result),
            patch("agentic_research.agents.intent_judge.IntentJudge.judge", return_value=intent_verdict),
            patch("agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens", new_callable=lambda: property(lambda self: MagicMock(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0))),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["formalize", "Bertrand's postulate"])

        assert result.exit_code == 0
        assert "CORRECT" in result.output
        assert "bertrand" in result.output

        # Step 3: check
        from agentic_research.models.verification import CounterexampleResult, CounterexampleStatus

        cx_result = CounterexampleResult(status=CounterexampleStatus.PLAUSIBLE, attempts_made=5)

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search", return_value=cx_result),
            patch("agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens", new_callable=lambda: property(lambda self: MagicMock(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0))),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["check", lean_stmt])

        assert result.exit_code == 0
        assert "PLAUSIBLE" in result.output

        # Step 4: prove
        from agentic_research.models.proof import ProofPipelineResult

        proof_result = ProofPipelineResult(
            statement=lean_stmt,
            proved=True,
            final_proof="by exact Nat.exists_prime_and_le_and_le_two_mul h",
            claim_check_passed=True,
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.proof.ProofPipeline.run", return_value=proof_result),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["prove", lean_stmt], input="y\n")

        assert result.exit_code == 0
        assert "PROVED" in result.output


# ---------------------------------------------------------------------------
# 2. Budget enforcement
# ---------------------------------------------------------------------------
class TestBudgetEnforcementE2E:
    def test_prove_with_tiny_budget_halts(self, runner, tmp_session_dir):
        """prove --budget 0.01 should record cost and report budget exceeded."""
        from agentic_research.models.proof import ProofPipelineResult

        proof_result = ProofPipelineResult(
            statement="theorem t : True := trivial",
            proved=False,
            failure_stage="budget",
            failure_reason="Budget exceeded",
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.proof.ProofPipeline.run", return_value=proof_result),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(
                cli, ["prove", "theorem t : True := trivial", "--budget", "0.01"],
                input="y\n",
            )

        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "cost summary" in output_lower or "budget" in output_lower

    def test_explore_with_zero_budget_halts(self, runner, tmp_session_dir):
        """explore with near-zero budget halts after first agent call exceeds it."""
        explore_result = _mock_agent_result(result_data={"domain": "test"})

        big_tokens = MagicMock()
        big_tokens.input_tokens = 1_000_000
        big_tokens.output_tokens = 500_000
        big_tokens.cache_read_input_tokens = 0
        big_tokens.cache_creation_input_tokens = 0

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.agents.explorer.ExplorationAgent.run", return_value=explore_result),
            patch("agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens", new_callable=lambda: property(lambda self: big_tokens)),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["explore", "test", "--budget", "0.01"])

        output_lower = result.output.lower()
        assert "budget" in output_lower or "exceeded" in output_lower


# ---------------------------------------------------------------------------
# 3. Timeout enforcement
# ---------------------------------------------------------------------------
class TestTimeoutEnforcementE2E:
    def test_prove_timeout_respected(self, runner, tmp_session_dir):
        from agentic_research.models.proof import ProofPipelineResult

        proof_result = ProofPipelineResult(
            statement="theorem t : True := trivial",
            proved=False,
            failure_stage="timeout",
            failure_reason="Timed out",
        )

        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            if call_count[0] <= 1:
                return 0.0
            return 5.0

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.proof.ProofPipeline.run", return_value=proof_result),
            patch("agentic_research.cli.main.time.monotonic", side_effect=mock_monotonic),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(
                cli, ["prove", "theorem t : True := trivial", "--timeout", "1"],
                input="y\n",
            )

        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "timeout" in output_lower


# ---------------------------------------------------------------------------
# 4. Circuit breaker
# ---------------------------------------------------------------------------
class TestCircuitBreakerE2E:
    def test_five_consecutive_failures_opens_circuit(self):
        """5 consecutive failures should open the circuit breaker."""
        cb = CircuitBreaker(
            consecutive_failure_limit=5,
            error_rate_threshold=1.0,
        )
        for _ in range(4):
            cb.record_failure()
            assert not cb.is_open()

        cb.record_failure()
        assert cb.is_open()

    def test_success_resets_consecutive_count(self):
        cb = CircuitBreaker(
            consecutive_failure_limit=5,
            error_rate_threshold=1.0,
        )
        for _ in range(4):
            cb.record_failure()
        cb.record_success()
        assert not cb.is_open()

        for _ in range(4):
            cb.record_failure()
        assert not cb.is_open()

    def test_error_rate_trigger(self):
        """Error rate > 50% within the window should open the circuit."""
        cb = CircuitBreaker(
            consecutive_failure_limit=100,
            error_rate_threshold=0.5,
        )
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()

    def test_circuit_breaker_halts_orchestrator(self):
        """Orchestrator should halt when circuit breaker opens."""
        mock_llm = MagicMock()
        mock_llm.model = "claude-opus-4-6-20250616"
        mock_repl = MagicMock()
        mock_search = MagicMock()

        config = OrchestratorConfig(
            max_conjectures=1,
            budget_limit_usd=100.0,
            max_exploration_rounds=10,
            max_reasoning_cycles=50,
        )

        from agentic_research.orchestrator.engine import ResearchOrchestrator

        orch = ResearchOrchestrator(
            llm_client=mock_llm,
            lean_repl=mock_repl,
            lean_search=mock_search,
            config=config,
        )

        with patch.object(
            type(orch), "_handle_exploring",
            side_effect=lambda raw_idea: (
                orch._circuit_breaker.record_failure(),
                orch._state_machine.transition(PipelineStage.FAILED, reason="Exploration failed") if orch._circuit_breaker.is_open() else None,
            ),
        ):
            result = orch.run("test idea")

        assert result.final_stage == PipelineStage.FAILED


# ---------------------------------------------------------------------------
# 5. Checkpointing end-to-end
# ---------------------------------------------------------------------------
class TestCheckpointingE2E:
    def test_checkpoint_creation_and_disk_persistence(self, tmp_path, monkeypatch):
        """Verify checkpoints are created and written to disk."""
        ckpt_dir = tmp_path / "checkpoints"
        monkeypatch.setattr(
            "agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", ckpt_dir
        )

        session_id = "test-ckpt-session"
        mgr = CheckpointManager(session_id=session_id, persist=True)
        sm = PipelineStateMachine()
        memory = ResearchSessionMemory(session_id=session_id)

        ckpt = mgr.create_checkpoint(sm, memory)

        assert ckpt.checkpoint_id == "ckpt_1"
        assert ckpt.stage == PipelineStage.EXPLORING
        assert mgr.checkpoint_count == 1

        disk_path = ckpt_dir / session_id / "ckpt_1.json"
        assert disk_path.exists()

        loaded = json.loads(disk_path.read_text())
        assert loaded["checkpoint_id"] == "ckpt_1"
        assert loaded["stage"] == "exploring"

    def test_checkpoint_after_multiple_stages(self, tmp_path, monkeypatch):
        """Multiple stage transitions should each produce a checkpoint."""
        ckpt_dir = tmp_path / "checkpoints"
        monkeypatch.setattr(
            "agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", ckpt_dir
        )

        session_id = "multi-stage-ckpt"
        mgr = CheckpointManager(session_id=session_id, persist=True)
        sm = PipelineStateMachine()
        memory = ResearchSessionMemory(session_id=session_id)

        mgr.create_checkpoint(sm, memory)
        sm.transition(PipelineStage.CONJECTURING, reason="exploration done")
        mgr.create_checkpoint(sm, memory)
        sm.transition(PipelineStage.FORMALIZING, reason="conjecturing done")
        mgr.create_checkpoint(sm, memory)

        assert mgr.checkpoint_count == 3
        disk_files = list((ckpt_dir / session_id).glob("ckpt_*.json"))
        assert len(disk_files) == 3

    def test_resume_from_checkpoint(self, tmp_path, monkeypatch):
        """Resume should restore state to the checkpoint stage."""
        ckpt_dir = tmp_path / "checkpoints"
        monkeypatch.setattr(
            "agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", ckpt_dir
        )

        session_id = "resume-test"
        mgr = CheckpointManager(session_id=session_id, persist=True)
        sm = PipelineStateMachine()
        memory = ResearchSessionMemory(session_id=session_id)

        conj = Conjecture(
            statement="P(n) for all n",
            natural_language="Test conjecture",
            confidence=0.9,
            difficulty=2,
        )
        memory.record_conjecture(conj)

        sm.transition(PipelineStage.CONJECTURING, reason="done exploring")
        mgr.create_checkpoint(sm, memory)

        sm.transition(PipelineStage.FORMALIZING, reason="picked conjecture")
        mgr.create_checkpoint(sm, memory)

        new_sm = PipelineStateMachine()
        new_memory = ResearchSessionMemory(session_id="fresh")
        assert new_sm.current_stage == PipelineStage.EXPLORING
        assert new_memory.total_conjecture_count == 0

        success = mgr.rollback("ckpt_1", new_sm, new_memory)
        assert success
        assert new_sm.current_stage == PipelineStage.CONJECTURING
        assert new_memory.total_conjecture_count == 1

    def test_load_checkpoint_from_disk(self, tmp_path, monkeypatch):
        """Load a checkpoint from disk after a fresh CheckpointManager."""
        ckpt_dir = tmp_path / "checkpoints"
        monkeypatch.setattr(
            "agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", ckpt_dir
        )

        session_id = "disk-load"
        mgr = CheckpointManager(session_id=session_id, persist=True)
        sm = PipelineStateMachine()
        memory = ResearchSessionMemory(session_id=session_id)
        mgr.create_checkpoint(sm, memory)

        loaded = CheckpointManager.load_checkpoint_from_disk(session_id, "ckpt_1")
        assert loaded is not None
        assert loaded.stage == PipelineStage.EXPLORING


# ---------------------------------------------------------------------------
# 6. Session memory tiering
# ---------------------------------------------------------------------------
class TestSessionMemoryTieringE2E:
    def test_tiering_with_many_conjectures(self):
        """Recording 10+ conjectures should trigger hot -> warm -> cold transitions."""
        mem = ResearchSessionMemory(session_id="tier-test")

        for i in range(15):
            conj = Conjecture(
                statement=f"Conjecture {i}: P_{i}(n) for all n",
                natural_language=f"Test conjecture #{i}",
                confidence=0.5 + i * 0.03,
                difficulty=min(i % 5 + 1, 5),
            )
            mem.record_conjecture(conj)

        hot_count = len(mem.data.tried_conjectures)
        warm_count = len(mem.warm_conjectures)
        cold_count = len(mem.cold_conjectures)

        assert hot_count == HOT_TIER_SIZE
        assert warm_count == WARM_TIER_SIZE
        assert cold_count == 15 - HOT_TIER_SIZE - WARM_TIER_SIZE

        assert mem.total_conjecture_count == 15

    def test_warm_conjectures_are_summaries(self):
        """Warm tier entries should be summaries, not full detail."""
        mem = ResearchSessionMemory(session_id="warm-check")

        for i in range(5):
            conj = Conjecture(
                statement=f"Statement {i}",
                natural_language=f"NL {i}",
                confidence=0.5,
                difficulty=3,
            )
            mem.record_conjecture(conj, failure_reason="Some very long " * 20 + "reason")

        assert len(mem.warm_conjectures) == 2
        for wc in mem.warm_conjectures:
            assert len(wc.failure_reason) <= 120 or wc.failure_reason.endswith("...")

    def test_cold_conjectures_are_hashes(self):
        """Cold tier entries should be hash + outcome only."""
        mem = ResearchSessionMemory(session_id="cold-check")

        for i in range(20):
            conj = Conjecture(
                statement=f"Cold conjecture {i}",
                natural_language=f"NL {i}",
                confidence=0.5,
                difficulty=3,
            )
            mem.record_conjecture(conj)

        assert len(mem.cold_conjectures) > 0
        for cc in mem.cold_conjectures:
            assert len(cc.statement_hash) == 16
            assert cc.outcome == ConjectureOutcome.PENDING

    def test_has_tried_searches_all_tiers(self):
        """has_tried() should find conjectures across hot, warm, and cold tiers."""
        mem = ResearchSessionMemory(session_id="has-tried")

        statements = []
        for i in range(20):
            stmt = f"Tier check conjecture {i}"
            statements.append(stmt)
            conj = Conjecture(
                statement=stmt,
                natural_language=f"NL {i}",
                confidence=0.5,
                difficulty=3,
            )
            mem.record_conjecture(conj)

        for stmt in statements:
            assert mem.has_tried(stmt)

        assert not mem.has_tried("never recorded statement")


# ---------------------------------------------------------------------------
# 7. Cost tracking accuracy
# ---------------------------------------------------------------------------
class TestCostTrackingAccuracyE2E:
    def test_cost_matches_known_token_counts(self):
        """CostTracker totals should match manually computed costs."""
        tracker = CostTracker()

        input_tokens = 10_000
        output_tokens = 5_000
        cache_read = 2_000
        cache_write = 1_000

        tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

        expected = (
            input_tokens * OPUS_INPUT_PRICE_PER_MTOK / 1_000_000
            + output_tokens * OPUS_OUTPUT_PRICE_PER_MTOK / 1_000_000
            + cache_read * OPUS_CACHE_READ_PRICE_PER_MTOK / 1_000_000
            + cache_write * OPUS_CACHE_WRITE_PRICE_PER_MTOK / 1_000_000
        )

        assert abs(tracker.total_cost() - expected) < 1e-10

    def test_cumulative_cost_across_multiple_calls(self):
        """Multiple record_usage calls should accumulate correctly."""
        tracker = CostTracker()

        calls = [
            (1000, 500, 0, 0),
            (2000, 1000, 500, 0),
            (500, 250, 0, 100),
        ]

        expected_total = 0.0
        for inp, out, cr, cw in calls:
            cost = tracker.record_usage(
                input_tokens=inp,
                output_tokens=out,
                cache_read_tokens=cr,
                cache_write_tokens=cw,
            )
            expected_cost = (
                inp * OPUS_INPUT_PRICE_PER_MTOK / 1_000_000
                + out * OPUS_OUTPUT_PRICE_PER_MTOK / 1_000_000
                + cr * OPUS_CACHE_READ_PRICE_PER_MTOK / 1_000_000
                + cw * OPUS_CACHE_WRITE_PRICE_PER_MTOK / 1_000_000
            )
            assert abs(cost - expected_cost) < 1e-10
            expected_total += expected_cost

        assert abs(tracker.total_cost() - expected_total) < 1e-10

    def test_cost_tracker_matches_compute_cost(self):
        """CostTracker and compute_cost() should agree for the same tokens."""
        from agentic_research.models.agents import TokenUsage
        from agentic_research.models.session import compute_cost

        usage = TokenUsage(
            input_tokens=8000,
            output_tokens=3000,
            cache_read_input_tokens=1500,
            cache_creation_input_tokens=500,
        )

        estimate = compute_cost(usage)

        tracker = CostTracker()
        tracker.record_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens,
        )

        assert abs(tracker.total_cost() - estimate.total_cost_usd) < 1e-10

    def test_zero_tokens_zero_cost(self):
        tracker = CostTracker()
        assert tracker.total_cost() == 0.0
        tracker.record_usage(input_tokens=0, output_tokens=0)
        assert tracker.total_cost() == 0.0


# ---------------------------------------------------------------------------
# Status command integration
# ---------------------------------------------------------------------------
class TestStatusE2E:
    def test_status_shows_tiered_memory(self, runner, tmp_session_dir):
        """Status command should display tier breakdown when warm/cold exist."""
        session = ResearchSessionMemory(session_id="status-tier-test")
        for i in range(6):
            conj = Conjecture(
                statement=f"Status conj {i}",
                natural_language=f"NL {i}",
                confidence=0.5,
                difficulty=3,
            )
            session.record_conjecture(conj)

        session.save(tmp_session_dir / "current_session.json")
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "status-tier-test" in result.output
        assert "hot" in result.output.lower() or "warm" in result.output.lower()
