"""Tests for Phase 9: Orchestrator + Research Session Memory.

All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_research.memory.session import ResearchSessionMemory
from agentic_research.models.agents import (
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.research import Conjecture, ConjectureSet, ExplorationResult
from agentic_research.models.session import (
    ConjectureOutcome,
    CostEstimate,
    OrchestratorConfig,
    PipelineStage,
    ResearchSessionResult,
    SessionMemoryData,
    SessionState,
    StageTokenUsage,
    StateTransition,
    TriedConjecture,
    VALID_TRANSITIONS,
    TERMINAL_STAGES,
    compute_cost,
)
from agentic_research.orchestrator.rollback import CheckpointManager
from agentic_research.orchestrator.state import (
    InvalidTransitionError,
    PipelineStateMachine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conj(stmt: str = "test", nl: str = "", conf: float = 0.5) -> Conjecture:
    return Conjecture(
        statement=stmt,
        natural_language=nl or f"NL: {stmt}",
        confidence=conf,
        difficulty=3,
    )


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


def _make_mock_repl():
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


def _make_mock_search():
    from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

    return LeanSearch(SearchConfig(backend=SearchBackend.MOCK))


def _make_mock_llm(responses: list[str] | None = None) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    if responses:
        side_effects = [_mock_llm_response(text) for text in responses]
        mock.complete.side_effect = side_effects
    else:
        mock.complete.return_value = _mock_llm_response("{}")
    mock.extract_json.return_value = {}
    return mock


# ===========================================================================
# models/session.py — PipelineStage, transitions, data models
# ===========================================================================


class TestPipelineStage:
    def test_all_values(self):
        assert PipelineStage.EXPLORING == "exploring"
        assert PipelineStage.CONJECTURING == "conjecturing"
        assert PipelineStage.FORMALIZING == "formalizing"
        assert PipelineStage.CHECKING_INTENT == "checking_intent"
        assert PipelineStage.SEARCHING_COUNTEREXAMPLE == "searching_counterexample"
        assert PipelineStage.PROVING == "proving"
        assert PipelineStage.REFINING == "refining"
        assert PipelineStage.COMPLETE == "complete"
        assert PipelineStage.FAILED == "failed"

    def test_terminal_stages(self):
        assert PipelineStage.COMPLETE in TERMINAL_STAGES
        assert PipelineStage.FAILED in TERMINAL_STAGES
        assert PipelineStage.EXPLORING not in TERMINAL_STAGES

    def test_valid_transitions_defined_for_all_stages(self):
        for stage in PipelineStage:
            assert stage in VALID_TRANSITIONS

    def test_terminal_stages_have_no_transitions(self):
        assert len(VALID_TRANSITIONS[PipelineStage.COMPLETE]) == 0
        assert len(VALID_TRANSITIONS[PipelineStage.FAILED]) == 0

    def test_exploring_can_go_to_conjecturing(self):
        assert PipelineStage.CONJECTURING in VALID_TRANSITIONS[PipelineStage.EXPLORING]

    def test_proving_can_complete(self):
        assert PipelineStage.COMPLETE in VALID_TRANSITIONS[PipelineStage.PROVING]

    def test_refining_can_go_back_to_exploring(self):
        assert PipelineStage.EXPLORING in VALID_TRANSITIONS[PipelineStage.REFINING]


class TestConjectureOutcome:
    def test_values(self):
        assert ConjectureOutcome.PROVED == "proved"
        assert ConjectureOutcome.DISPROVED == "disproved"
        assert ConjectureOutcome.PROOF_FAILED == "proof_failed"
        assert ConjectureOutcome.PENDING == "pending"


class TestTriedConjecture:
    def test_defaults(self):
        tc = TriedConjecture(conjecture=_conj())
        assert tc.outcome == ConjectureOutcome.PENDING
        assert tc.lean_statement == ""
        assert tc.proof_code is None

    def test_proved(self):
        tc = TriedConjecture(
            conjecture=_conj("proved_stmt"),
            outcome=ConjectureOutcome.PROVED,
            proof_code="by trivial",
        )
        assert tc.outcome == ConjectureOutcome.PROVED
        assert tc.proof_code == "by trivial"

    def test_serialization_roundtrip(self):
        tc = TriedConjecture(
            conjecture=_conj("x"),
            outcome=ConjectureOutcome.DISPROVED,
            failure_reason="counterexample found",
        )
        restored = TriedConjecture.model_validate(tc.model_dump())
        assert restored.outcome == ConjectureOutcome.DISPROVED
        assert restored.failure_reason == "counterexample found"


class TestSessionMemoryData:
    def test_empty(self):
        data = SessionMemoryData()
        assert not data.has_tried("anything")
        assert data.proved_conjectures() == []
        assert data.failed_conjectures() == []

    def test_has_tried(self):
        data = SessionMemoryData(
            tried_conjectures=[TriedConjecture(conjecture=_conj("stmt_a"))]
        )
        assert data.has_tried("stmt_a")
        assert not data.has_tried("stmt_b")

    def test_proved_and_failed(self):
        data = SessionMemoryData(
            tried_conjectures=[
                TriedConjecture(conjecture=_conj("a"), outcome=ConjectureOutcome.PROVED),
                TriedConjecture(conjecture=_conj("b"), outcome=ConjectureOutcome.DISPROVED),
                TriedConjecture(conjecture=_conj("c"), outcome=ConjectureOutcome.PENDING),
            ]
        )
        assert len(data.proved_conjectures()) == 1
        assert len(data.failed_conjectures()) == 1

    def test_by_outcome(self):
        data = SessionMemoryData(
            tried_conjectures=[
                TriedConjecture(conjecture=_conj("a"), outcome=ConjectureOutcome.PROVED),
                TriedConjecture(conjecture=_conj("b"), outcome=ConjectureOutcome.PROVED),
                TriedConjecture(conjecture=_conj("c"), outcome=ConjectureOutcome.DISPROVED),
            ]
        )
        assert len(data.by_outcome(ConjectureOutcome.PROVED)) == 2


class TestCostEstimate:
    def test_total_cost(self):
        c = CostEstimate(
            input_cost_usd=0.10,
            output_cost_usd=0.50,
            cache_read_cost_usd=0.01,
            cache_write_cost_usd=0.02,
        )
        assert abs(c.total_cost_usd - 0.63) < 1e-9


class TestComputeCost:
    def test_basic(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = compute_cost(usage)
        assert cost.input_tokens == 1000
        assert cost.output_tokens == 500
        assert cost.input_cost_usd > 0
        assert cost.output_cost_usd > 0

    def test_zero_tokens(self):
        cost = compute_cost(TokenUsage())
        assert cost.total_cost_usd == 0.0


class TestSessionState:
    def test_defaults(self):
        s = SessionState()
        assert s.stage == PipelineStage.EXPLORING
        assert s.active_conjecture_index is None
        assert s.conjectures_processed == 0
        assert s.transitions == []

    def test_serialization(self):
        s = SessionState(
            stage=PipelineStage.PROVING,
            conjectures_processed=3,
            transitions=[
                StateTransition(
                    from_state=PipelineStage.EXPLORING,
                    to_state=PipelineStage.CONJECTURING,
                )
            ],
        )
        restored = SessionState.model_validate(s.model_dump())
        assert restored.stage == PipelineStage.PROVING
        assert len(restored.transitions) == 1


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.max_conjectures == 5
        assert cfg.max_refinements == 3
        assert cfg.budget_limit_usd is None
        assert cfg.auto_mode is True

    def test_custom(self):
        cfg = OrchestratorConfig(
            max_conjectures=10,
            max_refinements=5,
            budget_limit_usd=2.50,
            auto_mode=False,
        )
        assert cfg.max_conjectures == 10
        assert cfg.budget_limit_usd == 2.50


class TestResearchSessionResult:
    def test_defaults(self):
        r = ResearchSessionResult()
        assert r.proved_conjectures == []
        assert r.failed_conjectures == []
        assert r.total_conjectures_tried == 0

    def test_serialization(self):
        r = ResearchSessionResult(
            session_id="abc123",
            raw_idea="test idea",
            total_conjectures_tried=3,
            final_stage=PipelineStage.COMPLETE,
        )
        restored = ResearchSessionResult.model_validate(r.model_dump())
        assert restored.session_id == "abc123"
        assert restored.final_stage == PipelineStage.COMPLETE


# ===========================================================================
# orchestrator/state.py — PipelineStateMachine
# ===========================================================================


class TestPipelineStateMachine:
    def test_initial_state(self):
        sm = PipelineStateMachine()
        assert sm.current_stage == PipelineStage.EXPLORING
        assert not sm.is_terminal

    def test_custom_initial_state(self):
        sm = PipelineStateMachine(initial_state=PipelineStage.CONJECTURING)
        assert sm.current_stage == PipelineStage.CONJECTURING

    def test_valid_transition(self):
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.CONJECTURING, reason="exploration done")
        assert sm.current_stage == PipelineStage.CONJECTURING
        assert len(sm.session_state.transitions) == 1

    def test_invalid_transition_raises(self):
        sm = PipelineStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition(PipelineStage.PROVING)

    def test_terminal_state(self):
        sm = PipelineStateMachine(initial_state=PipelineStage.PROVING)
        sm.transition(PipelineStage.COMPLETE, reason="done")
        assert sm.is_terminal

    def test_can_transition(self):
        sm = PipelineStateMachine()
        assert sm.can_transition(PipelineStage.CONJECTURING)
        assert not sm.can_transition(PipelineStage.PROVING)

    def test_transition_records_conjecture_index(self):
        sm = PipelineStateMachine(initial_state=PipelineStage.CONJECTURING)
        sm.transition(PipelineStage.FORMALIZING, conjecture_index=2)
        assert sm.session_state.active_conjecture_index == 2
        assert sm.session_state.transitions[-1].conjecture_index == 2

    def test_full_happy_path(self):
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.CONJECTURING)
        sm.transition(PipelineStage.FORMALIZING)
        sm.transition(PipelineStage.CHECKING_INTENT)
        sm.transition(PipelineStage.SEARCHING_COUNTEREXAMPLE)
        sm.transition(PipelineStage.PROVING)
        sm.transition(PipelineStage.COMPLETE)
        assert sm.is_terminal
        assert len(sm.session_state.transitions) == 6

    def test_refining_back_to_exploring(self):
        sm = PipelineStateMachine(initial_state=PipelineStage.REFINING)
        sm.transition(PipelineStage.EXPLORING, reason="refinement exhausted")
        assert sm.current_stage == PipelineStage.EXPLORING

    def test_restore(self):
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.CONJECTURING)
        sm.transition(PipelineStage.FORMALIZING)

        saved = sm.session_state.model_copy(deep=True)

        sm.transition(PipelineStage.CHECKING_INTENT)
        assert sm.current_stage == PipelineStage.CHECKING_INTENT

        sm.restore(saved)
        assert sm.current_stage == PipelineStage.FORMALIZING
        assert len(sm.session_state.transitions) == 2

    def test_cannot_transition_from_terminal(self):
        sm = PipelineStateMachine(initial_state=PipelineStage.PROVING)
        sm.transition(PipelineStage.COMPLETE)
        with pytest.raises(InvalidTransitionError):
            sm.transition(PipelineStage.EXPLORING)


# ===========================================================================
# memory/session.py — ResearchSessionMemory
# ===========================================================================


class TestResearchSessionMemory:
    def test_session_id(self):
        mem = ResearchSessionMemory(session_id="test123")
        assert mem.session_id == "test123"

    def test_auto_session_id(self):
        mem = ResearchSessionMemory()
        assert len(mem.session_id) == 12

    def test_record_conjecture(self):
        mem = ResearchSessionMemory()
        conj = _conj("stmt_a")
        mem.record_conjecture(conj, ConjectureOutcome.PENDING)
        assert mem.has_tried("stmt_a")
        assert not mem.has_tried("stmt_b")

    def test_update_conjecture_outcome(self):
        mem = ResearchSessionMemory()
        conj = _conj("stmt_a")
        mem.record_conjecture(conj, ConjectureOutcome.PENDING)
        updated = mem.update_conjecture_outcome(
            "stmt_a",
            ConjectureOutcome.PROVED,
            proof_code="by trivial",
        )
        assert updated
        proved = mem.data.proved_conjectures()
        assert len(proved) == 1
        assert proved[0].proof_code == "by trivial"

    def test_update_nonexistent_returns_false(self):
        mem = ResearchSessionMemory()
        assert not mem.update_conjecture_outcome("nope", ConjectureOutcome.PROVED)

    def test_add_partial_result(self):
        mem = ResearchSessionMemory()
        mem.add_partial_result("lemma1", lean_code="by simp", source_conjecture="conj_a")
        assert len(mem.data.partial_results) == 1
        assert mem.data.partial_results[0].lean_code == "by simp"

    def test_add_promising_direction(self):
        mem = ResearchSessionMemory()
        mem.add_promising_direction("direction_1", description="explore graphs")
        assert len(mem.data.promising_directions) == 1

    def test_add_user_preference(self):
        mem = ResearchSessionMemory()
        mem.add_user_preference("try combinatorial approach", context="round 2")
        assert len(mem.data.user_preferences) == 1

    def test_get_untried_directions(self):
        mem = ResearchSessionMemory()
        mem.add_promising_direction("dir_a")
        mem.add_promising_direction("dir_b")
        mem.record_conjecture(_conj("dir_a"), ConjectureOutcome.PROVED)
        untried = mem.get_untried_directions()
        assert len(untried) == 1
        assert untried[0].title == "dir_b"

    def test_summary(self):
        mem = ResearchSessionMemory(session_id="s1")
        mem.record_conjecture(_conj("a"), ConjectureOutcome.PROVED)
        mem.record_conjecture(_conj("b"), ConjectureOutcome.DISPROVED)
        mem.add_partial_result("lemma")
        s = mem.summary()
        assert s["session_id"] == "s1"
        assert s["total_tried"] == 2
        assert s["proved"] == 1
        assert s["failed"] == 1

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.json"

            mem = ResearchSessionMemory(session_id="save_test")
            mem.record_conjecture(_conj("stmt_a"), ConjectureOutcome.PROVED)
            mem.add_partial_result("lemma_x", lean_code="by simp")
            mem.add_promising_direction("dir_1", priority=0.8)
            mem.add_user_preference("prefer algebra")
            mem.save(path)

            assert path.exists()

            loaded = ResearchSessionMemory.load(path)
            assert loaded.session_id == "save_test"
            assert loaded.has_tried("stmt_a")
            assert len(loaded.data.partial_results) == 1
            assert len(loaded.data.promising_directions) == 1
            assert len(loaded.data.user_preferences) == 1
            assert loaded.data.proved_conjectures()[0].conjecture.statement == "stmt_a"

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "session.json"
            mem = ResearchSessionMemory()
            mem.save(path)
            assert path.exists()


# ===========================================================================
# orchestrator/rollback.py — CheckpointManager
# ===========================================================================


class TestCheckpointManager:
    def test_create_checkpoint(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        cp = mgr.create_checkpoint(sm, mem)
        assert cp.checkpoint_id == "ckpt_1"
        assert cp.stage == PipelineStage.EXPLORING
        assert mgr.checkpoint_count == 1

    def test_multiple_checkpoints(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        cp1 = mgr.create_checkpoint(sm, mem)
        sm.transition(PipelineStage.CONJECTURING)
        cp2 = mgr.create_checkpoint(sm, mem)

        assert cp1.checkpoint_id == "ckpt_1"
        assert cp2.checkpoint_id == "ckpt_2"
        assert mgr.checkpoint_count == 2

    def test_rollback_restores_state(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        mgr.create_checkpoint(sm, mem)
        sm.transition(PipelineStage.CONJECTURING)
        mem.record_conjecture(_conj("a"), ConjectureOutcome.PENDING)

        mgr.create_checkpoint(sm, mem)
        sm.transition(PipelineStage.FORMALIZING)
        mem.record_conjecture(_conj("b"), ConjectureOutcome.PENDING)

        assert sm.current_stage == PipelineStage.FORMALIZING
        assert len(mem.data.tried_conjectures) == 2

        success = mgr.rollback("ckpt_1", sm, mem)
        assert success
        assert sm.current_stage == PipelineStage.EXPLORING
        assert len(mem.data.tried_conjectures) == 0

    def test_rollback_trims_later_checkpoints(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        mgr.create_checkpoint(sm, mem)
        sm.transition(PipelineStage.CONJECTURING)
        mgr.create_checkpoint(sm, mem)
        sm.transition(PipelineStage.FORMALIZING)
        mgr.create_checkpoint(sm, mem)

        assert mgr.checkpoint_count == 3

        mgr.rollback("ckpt_2", sm, mem)
        assert mgr.checkpoint_count == 2

    def test_rollback_nonexistent_returns_false(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        assert not mgr.rollback("nonexistent", sm, mem)

    def test_rollback_to_latest(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        mgr.create_checkpoint(sm, mem)
        sm.transition(PipelineStage.CONJECTURING)

        success = mgr.rollback_to_latest(sm, mem)
        assert success
        assert sm.current_stage == PipelineStage.EXPLORING

    def test_rollback_to_latest_empty(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        assert not mgr.rollback_to_latest(sm, mem)

    def test_get_checkpoint(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        mgr.create_checkpoint(sm, mem)
        cp = mgr.get_checkpoint("ckpt_1")
        assert cp is not None
        assert cp.stage == PipelineStage.EXPLORING

    def test_get_checkpoint_nonexistent(self):
        mgr = CheckpointManager()
        assert mgr.get_checkpoint("nope") is None

    def test_clear(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        mgr.create_checkpoint(sm, mem)
        mgr.create_checkpoint(sm, mem)
        assert mgr.checkpoint_count == 2

        mgr.clear()
        assert mgr.checkpoint_count == 0

    def test_checkpoint_with_stage_usages(self):
        mgr = CheckpointManager()
        sm = PipelineStateMachine()
        mem = ResearchSessionMemory()

        usages = [
            StageTokenUsage(
                stage=PipelineStage.EXPLORING,
                agent_name="explorer",
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        ]
        cp = mgr.create_checkpoint(sm, mem, usages)
        assert len(cp.stage_token_usages) == 1
        assert cp.stage_token_usages[0].agent_name == "explorer"


# ===========================================================================
# orchestrator/engine.py — ResearchOrchestrator
# ===========================================================================


def _make_exploration_result(idea: str = "test idea") -> dict:
    return ExplorationResult(
        raw_idea=idea,
        domain="number_theory",
        directions=[],
    ).model_dump()


def _make_conjecture_set(stmts: list[str] | None = None) -> dict:
    stmts = stmts or ["conj_1"]
    conjs = [_conj(s) for s in stmts]
    return ConjectureSet(
        conjectures=conjs,
        ranking=list(range(len(conjs))),
    ).model_dump()


class TestOrchestratorHappyPath:
    """Test: idea -> explore -> conjecture -> formalize -> verify -> prove -> COMPLETE."""

    def test_full_loop_proof_found(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        llm = _make_mock_llm()
        repl = _make_mock_repl()
        search = _make_mock_search()
        config = OrchestratorConfig(max_conjectures=1)

        orch = ResearchOrchestrator(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            config=config,
            session_id="happy_test",
        )

        with (
            patch.object(
                orch, "_handle_exploring",
                wraps=lambda raw_idea: _mock_exploring(orch, raw_idea),
            ) as _,
            patch.object(
                orch, "_handle_conjecturing",
                wraps=lambda raw_idea: _mock_conjecturing(orch, raw_idea),
            ),
            patch.object(
                orch, "_handle_formalizing",
                wraps=lambda raw_idea: _mock_formalizing_success(orch, raw_idea),
            ),
            patch.object(
                orch, "_handle_checking_intent",
                wraps=lambda raw_idea: _mock_intent_ok(orch),
            ),
            patch.object(
                orch, "_handle_searching_counterexample",
                wraps=lambda raw_idea: _mock_cx_plausible(orch),
            ),
            patch.object(
                orch, "_handle_proving",
                wraps=lambda raw_idea: _mock_proving_success(orch),
            ),
        ):
            result = orch.run("test idea")

        assert result.final_stage == PipelineStage.COMPLETE
        assert result.session_id == "happy_test"


class TestOrchestratorFailurePaths:
    def test_exploration_failure(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        llm = _make_mock_llm()
        repl = _make_mock_repl()
        search = _make_mock_search()

        orch = ResearchOrchestrator(
            llm_client=llm, lean_repl=repl, lean_search=search
        )

        with patch.object(
            orch, "_handle_exploring",
            wraps=lambda raw_idea: orch._state_machine.transition(
                PipelineStage.FAILED, reason="Exploration failed"
            ),
        ):
            result = orch.run("bad idea")

        assert result.final_stage == PipelineStage.FAILED

    def test_max_conjectures_reached(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        llm = _make_mock_llm()
        repl = _make_mock_repl()
        search = _make_mock_search()
        config = OrchestratorConfig(max_conjectures=1)

        orch = ResearchOrchestrator(
            llm_client=llm, lean_repl=repl, lean_search=search, config=config
        )

        call_count = [0]

        def mock_conjecturing(raw_idea):
            call_count[0] += 1
            orch._state_machine.session_state.conjectures_processed = 1
            orch._active_conjecture = _conj("c1")
            orch._active_lean_statement = ""
            orch._memory.record_conjecture(_conj("c1"), ConjectureOutcome.PENDING)
            orch._state_machine.transition(PipelineStage.FORMALIZING)

        def mock_formalizing(raw_idea):
            orch._state_machine.transition(
                PipelineStage.REFINING, reason="Formalization failed"
            )

        def mock_refining(raw_idea):
            orch._state_machine.session_state.refinements_attempted += 1
            orch._state_machine.transition(
                PipelineStage.FAILED,
                reason="Max conjectures reached",
            )

        with (
            patch.object(orch, "_handle_exploring", wraps=lambda r: _mock_exploring(orch, r)),
            patch.object(orch, "_handle_conjecturing", wraps=mock_conjecturing),
            patch.object(orch, "_handle_formalizing", wraps=mock_formalizing),
            patch.object(orch, "_handle_refining", wraps=mock_refining),
        ):
            result = orch.run("idea")

        assert result.final_stage == PipelineStage.FAILED

    def test_budget_limit_stops_execution(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        llm = _make_mock_llm()
        repl = _make_mock_repl()
        search = _make_mock_search()
        config = OrchestratorConfig(budget_limit_usd=0.0)

        orch = ResearchOrchestrator(
            llm_client=llm, lean_repl=repl, lean_search=search, config=config
        )
        orch._total_tokens = TokenUsage(input_tokens=100000, output_tokens=50000)

        result = orch.run("idea")
        assert result.final_stage == PipelineStage.FAILED

    def test_max_exploration_rounds(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        llm = _make_mock_llm()
        repl = _make_mock_repl()
        search = _make_mock_search()
        config = OrchestratorConfig(max_exploration_rounds=1, max_conjectures=1)

        orch = ResearchOrchestrator(
            llm_client=llm, lean_repl=repl, lean_search=search, config=config
        )

        explore_count = [0]

        def mock_exploring(raw_idea):
            explore_count[0] += 1
            orch._exploration_rounds += 1
            if orch._exploration_rounds > config.max_exploration_rounds:
                orch._state_machine.transition(
                    PipelineStage.FAILED,
                    reason="Max exploration rounds reached",
                )
            else:
                orch._state_machine.transition(PipelineStage.CONJECTURING)

        def mock_conj(raw_idea):
            orch._state_machine.transition(
                PipelineStage.EXPLORING, reason="No conjectures"
            )

        with (
            patch.object(orch, "_handle_exploring", wraps=mock_exploring),
            patch.object(orch, "_handle_conjecturing", wraps=mock_conj),
        ):
            result = orch.run("idea")

        assert result.final_stage == PipelineStage.FAILED


class TestOrchestratorCostTracking:
    def test_tokens_accumulated(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        llm = _make_mock_llm()
        repl = _make_mock_repl()
        search = _make_mock_search()

        orch = ResearchOrchestrator(
            llm_client=llm, lean_repl=repl, lean_search=search
        )

        orch._record_stage_usage(
            PipelineStage.EXPLORING,
            "explorer",
            TokenUsage(input_tokens=1000, output_tokens=500),
        )
        orch._record_stage_usage(
            PipelineStage.CONJECTURING,
            "conjecturer",
            TokenUsage(input_tokens=2000, output_tokens=1000),
        )

        assert orch.total_tokens.input_tokens == 3000
        assert orch.total_tokens.output_tokens == 1500
        assert len(orch._stage_usages) == 2

    def test_build_result_includes_cost(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        llm = _make_mock_llm()
        repl = _make_mock_repl()
        search = _make_mock_search()

        orch = ResearchOrchestrator(
            llm_client=llm, lean_repl=repl, lean_search=search
        )
        orch._total_tokens = TokenUsage(input_tokens=10000, output_tokens=5000)
        orch._state_machine.transition(PipelineStage.FAILED, reason="test")

        result = orch._build_result("idea")
        assert result.cost_estimate.total_cost_usd > 0
        assert result.total_token_usage.input_tokens == 10000


class TestOrchestratorProperties:
    def test_session_id(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        orch = ResearchOrchestrator(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            session_id="custom_id",
        )
        assert orch.session_id == "custom_id"

    def test_auto_session_id(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        orch = ResearchOrchestrator(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
        )
        assert len(orch.session_id) == 12

    def test_properties(self):
        from agentic_research.orchestrator.engine import ResearchOrchestrator

        orch = ResearchOrchestrator(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
        )
        assert isinstance(orch.state_machine, PipelineStateMachine)
        assert isinstance(orch.memory, ResearchSessionMemory)
        assert isinstance(orch.checkpoint_manager, CheckpointManager)
        assert isinstance(orch.total_tokens, TokenUsage)


# ---------------------------------------------------------------------------
# Mock handler helpers for orchestrator tests
# ---------------------------------------------------------------------------


def _mock_exploring(orch, raw_idea: str) -> None:
    orch._exploration_rounds += 1
    orch._checkpoint_mgr.create_checkpoint(
        orch._state_machine, orch._memory, orch._stage_usages
    )
    orch._state_machine.transition(
        PipelineStage.CONJECTURING, reason="Exploration complete"
    )


def _mock_conjecturing(orch, raw_idea: str) -> None:
    conj = _conj("test_conjecture")
    orch._memory.record_conjecture(conj, ConjectureOutcome.PENDING)
    orch._active_conjecture = conj
    orch._active_lean_statement = ""
    orch._state_machine.session_state.conjectures_processed += 1
    orch._state_machine.transition(
        PipelineStage.FORMALIZING, reason="Conjecture selected"
    )


def _mock_formalizing_success(orch, raw_idea: str) -> None:
    orch._active_lean_statement = "theorem test : True := trivial"
    orch._state_machine.transition(
        PipelineStage.CHECKING_INTENT, reason="Formalization succeeded"
    )


def _mock_intent_ok(orch) -> None:
    orch._state_machine.transition(
        PipelineStage.SEARCHING_COUNTEREXAMPLE, reason="Intent verified"
    )


def _mock_cx_plausible(orch) -> None:
    orch._state_machine.transition(
        PipelineStage.PROVING, reason="No counterexample"
    )


def _mock_proving_success(orch) -> None:
    orch._memory.update_conjecture_outcome(
        orch._active_conjecture.statement,
        ConjectureOutcome.PROVED,
        proof_code="by trivial",
    )
    orch._state_machine.transition(PipelineStage.COMPLETE, reason="Proof found")
