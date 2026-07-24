"""Tests for Phase 4: Error recovery and checkpointing.

Covers:
  - Checkpoint at every stage transition
  - Disk persistence of checkpoints
  - Resume from checkpoint
  - Idempotency cache hit/miss
  - Exponential backoff timing
  - LLM output validation gates
  - Graceful degradation for REPL unavailability
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_research.memory.session import ResearchSessionMemory
from agentic_research.models.agents import (
    AgentResult,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.session import (
    OrchestratorConfig,
    PipelineStage,
)
from agentic_research.models.tools import CompilationResult, CompilationStatus, ToolStatus
from agentic_research.orchestrator.idempotency import IdempotencyCache, make_idempotency_key
from agentic_research.orchestrator.resilience import (
    ReplBackoffConfig,
    ResilientRepl,
    validate_llm_response,
)
from agentic_research.orchestrator.rollback import CheckpointManager
from agentic_research.orchestrator.state import PipelineStateMachine
from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig


# ---------------------------------------------------------------------------
# Checkpoint at every stage
# ---------------------------------------------------------------------------


class TestCheckpointAtEveryStage:
    def test_checkpoint_created_for_exploring(self) -> None:
        mgr = CheckpointManager(session_id="test-sess")
        sm = PipelineStateMachine(initial_state=PipelineStage.EXPLORING)
        mem = ResearchSessionMemory("test-sess")
        cp = mgr.create_checkpoint(sm, mem)
        assert cp.stage == PipelineStage.EXPLORING
        assert cp.checkpoint_id == "ckpt_1"

    def test_checkpoint_created_for_all_stages(self) -> None:
        mgr = CheckpointManager(session_id="test-sess")
        mem = ResearchSessionMemory("test-sess")

        stages = [
            PipelineStage.EXPLORING,
            PipelineStage.CONJECTURING,
            PipelineStage.FORMALIZING,
            PipelineStage.CHECKING_INTENT,
            PipelineStage.SEARCHING_COUNTEREXAMPLE,
            PipelineStage.PROVING,
            PipelineStage.REFINING,
        ]
        for stage in stages:
            sm = PipelineStateMachine(initial_state=stage)
            mgr.create_checkpoint(sm, mem)

        assert mgr.checkpoint_count == len(stages)
        for i, stage in enumerate(stages):
            assert mgr.checkpoints[i].stage == stage

    def test_orchestrator_checkpoints_every_stage(self) -> None:
        """The orchestrator creates a checkpoint before each stage handler runs."""
        from agentic_research.orchestrator.engine import ResearchOrchestrator
        from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

        mock_llm = MagicMock()
        mock_llm.model = "test"
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        config = OrchestratorConfig(
            budget_limit_usd=1000.0,
            max_reasoning_cycles=2,
        )

        orch = ResearchOrchestrator(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            config=config,
        )

        mock_agent_result = AgentResult(
            agent_name="explorer",
            status=AgentStatus.FAILURE,
            result=None,
        )

        with patch("agentic_research.orchestrator.engine.ExplorationAgent") as MockExplorer:
            instance = MockExplorer.return_value
            instance.run.return_value = mock_agent_result
            orch.run("test idea")

        assert orch.checkpoint_manager.checkpoint_count >= 1
        assert orch.checkpoint_manager.checkpoints[0].stage == PipelineStage.EXPLORING


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


class TestCheckpointDiskPersistence:
    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        with patch("agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", tmp_path):
            mgr = CheckpointManager(session_id="disk-test", persist=True)
            sm = PipelineStateMachine(initial_state=PipelineStage.CONJECTURING)
            sm.session_state.raw_idea = "prime gaps"
            mem = ResearchSessionMemory("disk-test")
            cp = mgr.create_checkpoint(sm, mem)

            ckpt_file = tmp_path / "disk-test" / f"{cp.checkpoint_id}.json"
            assert ckpt_file.exists()

            loaded = CheckpointManager.load_checkpoint_from_disk("disk-test", cp.checkpoint_id)
            assert loaded is not None
            assert loaded.stage == PipelineStage.CONJECTURING
            assert loaded.session_state.raw_idea == "prime gaps"

    def test_list_disk_checkpoints(self, tmp_path: Path) -> None:
        with patch("agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", tmp_path):
            mgr = CheckpointManager(session_id="list-test", persist=True)
            sm = PipelineStateMachine(initial_state=PipelineStage.EXPLORING)
            mem = ResearchSessionMemory("list-test")
            mgr.create_checkpoint(sm, mem)
            mgr.create_checkpoint(sm, mem)

            ids = mgr.list_disk_checkpoints()
            assert ids == ["ckpt_1", "ckpt_2"]

    def test_latest_disk_checkpoint(self, tmp_path: Path) -> None:
        with patch("agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", tmp_path):
            mgr = CheckpointManager(session_id="latest-test", persist=True)
            sm = PipelineStateMachine(initial_state=PipelineStage.EXPLORING)
            mem = ResearchSessionMemory("latest-test")
            mgr.create_checkpoint(sm, mem)

            sm2 = PipelineStateMachine(initial_state=PipelineStage.CONJECTURING)
            mgr.create_checkpoint(sm2, mem)

            latest = mgr.latest_disk_checkpoint()
            assert latest is not None
            assert latest.checkpoint_id == "ckpt_2"
            assert latest.stage == PipelineStage.CONJECTURING

    def test_load_nonexistent_checkpoint(self, tmp_path: Path) -> None:
        with patch("agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", tmp_path):
            loaded = CheckpointManager.load_checkpoint_from_disk("no-session", "ckpt_99")
            assert loaded is None


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------


class TestResumeFromCheckpoint:
    def test_resume_restores_state_and_continues(self) -> None:
        from agentic_research.orchestrator.engine import ResearchOrchestrator
        from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

        mock_llm = MagicMock()
        mock_llm.model = "test"
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        config = OrchestratorConfig(
            budget_limit_usd=1000.0,
            max_reasoning_cycles=3,
        )

        orch = ResearchOrchestrator(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            config=config,
            session_id="resume-test",
        )

        mock_agent_result = AgentResult(
            agent_name="explorer",
            status=AgentStatus.FAILURE,
            result=None,
        )

        with patch("agentic_research.orchestrator.engine.ExplorationAgent") as MockExplorer:
            instance = MockExplorer.return_value
            instance.run.return_value = mock_agent_result
            orch.run("test idea for resume")

        assert orch.checkpoint_manager.checkpoint_count >= 1
        first_checkpoint = orch.checkpoint_manager.checkpoints[0]

        orch2 = ResearchOrchestrator(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            config=OrchestratorConfig(
                budget_limit_usd=1000.0,
                max_reasoning_cycles=1,
            ),
            session_id="resume-test-2",
        )
        orch2._checkpoint_mgr._checkpoints.append(first_checkpoint)

        with patch("agentic_research.orchestrator.engine.ExplorationAgent") as MockExplorer:
            instance = MockExplorer.return_value
            instance.run.return_value = mock_agent_result
            result = orch2.resume_from_checkpoint(first_checkpoint.checkpoint_id)

        assert result.session_id == "resume-test-2"
        assert result.raw_idea == "test idea for resume"

    def test_resume_nonexistent_checkpoint_returns_empty_result(self) -> None:
        from agentic_research.orchestrator.engine import ResearchOrchestrator
        from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

        mock_llm = MagicMock()
        mock_llm.model = "test"
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        orch = ResearchOrchestrator(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            session_id="bad-resume",
        )
        result = orch.resume_from_checkpoint("ckpt_nonexistent")
        assert result.session_id == "bad-resume"


# ---------------------------------------------------------------------------
# Idempotency cache
# ---------------------------------------------------------------------------


class TestIdempotencyCache:
    def test_make_key_deterministic(self) -> None:
        k1 = make_idempotency_key("sess1", "exploring", 0, 1)
        k2 = make_idempotency_key("sess1", "exploring", 0, 1)
        assert k1 == k2

    def test_different_inputs_produce_different_keys(self) -> None:
        k1 = make_idempotency_key("sess1", "exploring", 0, 1)
        k2 = make_idempotency_key("sess1", "exploring", 0, 2)
        assert k1 != k2

    def test_cache_miss_returns_none(self) -> None:
        cache = IdempotencyCache()
        assert cache.get("nonexistent") is None

    def test_cache_put_and_get(self) -> None:
        cache = IdempotencyCache()
        cache.put("key1", {"result": "success"})
        assert cache.get("key1") == {"result": "success"}

    def test_cache_has(self) -> None:
        cache = IdempotencyCache()
        assert not cache.has("missing")
        cache.put("present", 42)
        assert cache.has("present")

    def test_cache_size(self) -> None:
        cache = IdempotencyCache()
        assert cache.size == 0
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size == 2

    def test_cache_clear(self) -> None:
        cache = IdempotencyCache()
        cache.put("a", 1)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_disk_persistence(self, tmp_path: Path) -> None:
        with patch("agentic_research.orchestrator.idempotency.DEFAULT_CACHE_DIR", tmp_path):
            cache = IdempotencyCache(session_id="persist-test", persist=True)
            cache.put("my_key", {"content": "cached LLM response"})

            cache_file = tmp_path / "persist-test" / "my_key.json"
            assert cache_file.exists()

            cache2 = IdempotencyCache(session_id="persist-test", persist=True)
            result = cache2.get("my_key")
            assert result == {"content": "cached LLM response"}


# ---------------------------------------------------------------------------
# Exponential backoff
# ---------------------------------------------------------------------------


class TestExponentialBackoff:
    def test_success_on_first_try_no_delay(self) -> None:
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        resilient = ResilientRepl(
            repl,
            backoff=ReplBackoffConfig(base_delay=0.01),
        )
        result = resilient.execute_with_backoff("theorem test : True := trivial")
        assert result.status == ToolStatus.SUCCESS

    def test_retries_on_error_with_increasing_delay(self) -> None:
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        resilient = ResilientRepl(
            repl,
            backoff=ReplBackoffConfig(
                base_delay=0.01,
                factor=2.0,
                max_attempts=3,
            ),
        )

        error_result = CompilationResult(
            status=ToolStatus.ERROR,
            error_message="transient error",
        )
        success_result = CompilationResult(
            status=ToolStatus.SUCCESS,
            compilation_status=CompilationStatus.OK,
        )

        call_count = 0

        def mock_execute(code):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return error_result
            return success_result

        repl.execute = mock_execute
        start = time.monotonic()
        result = resilient.execute_with_backoff("test code")
        elapsed = time.monotonic() - start

        assert result.status == ToolStatus.SUCCESS
        assert call_count == 3
        assert elapsed >= 0.02

    def test_all_retries_exhausted(self) -> None:
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        resilient = ResilientRepl(
            repl,
            backoff=ReplBackoffConfig(
                base_delay=0.01,
                max_attempts=2,
            ),
        )

        error_result = CompilationResult(
            status=ToolStatus.ERROR,
            error_message="persistent error",
        )
        repl.execute = lambda code: error_result

        result = resilient.execute_with_backoff("test code")
        assert result.status == ToolStatus.ERROR
        assert result.error_message == "persistent error"

    def test_max_delay_cap(self) -> None:
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        backoff = ReplBackoffConfig(
            base_delay=0.01,
            factor=100.0,
            max_delay=0.02,
            max_attempts=3,
        )
        resilient = ResilientRepl(repl, backoff=backoff)

        error_result = CompilationResult(
            status=ToolStatus.ERROR,
            error_message="err",
        )
        success_result = CompilationResult(
            status=ToolStatus.SUCCESS,
            compilation_status=CompilationStatus.OK,
        )

        call_count = 0

        def mock_execute(code):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return error_result
            return success_result

        repl.execute = mock_execute

        start = time.monotonic()
        resilient.execute_with_backoff("test")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1


# ---------------------------------------------------------------------------
# LLM output validation gates
# ---------------------------------------------------------------------------


class TestValidationGates:
    def test_valid_response_passes(self) -> None:
        response = LLMResponse(
            content="valid output",
            stop_reason="end_turn",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        errors = validate_llm_response(response)
        assert errors == []

    def test_truncated_stop_reason_fails(self) -> None:
        response = LLMResponse(
            content="partial output",
            stop_reason="max_tokens",
            token_usage=TokenUsage(),
        )
        errors = validate_llm_response(response)
        assert len(errors) == 1
        assert "truncated" in errors[0]

    def test_empty_content_fails(self) -> None:
        response = LLMResponse(
            content="   ",
            stop_reason="end_turn",
            token_usage=TokenUsage(),
        )
        errors = validate_llm_response(response)
        assert any("Empty" in e for e in errors)

    def test_schema_validation_passes(self) -> None:
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        response = LLMResponse(
            content='{"name": "test", "value": 42}',
            stop_reason="end_turn",
            token_usage=TokenUsage(),
        )
        errors = validate_llm_response(
            response,
            schema=TestSchema,
            parsed_data={"name": "test", "value": 42},
        )
        assert errors == []

    def test_schema_validation_fails(self) -> None:
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        response = LLMResponse(
            content="bad data",
            stop_reason="end_turn",
            token_usage=TokenUsage(),
        )
        errors = validate_llm_response(
            response,
            schema=TestSchema,
            parsed_data={"name": 123},
        )
        assert any("Schema validation" in e for e in errors)

    def test_tool_use_stop_reason_passes(self) -> None:
        response = LLMResponse(
            content="tool call result",
            stop_reason="tool_use",
            token_usage=TokenUsage(),
        )
        errors = validate_llm_response(response)
        assert errors == []


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_repl_unavailable_after_3_failures(self) -> None:
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        resilient = ResilientRepl(
            repl,
            backoff=ReplBackoffConfig(base_delay=0.001, max_attempts=1),
        )

        error_result = CompilationResult(
            status=ToolStatus.ERROR,
            error_message="crash",
        )
        repl.execute = lambda code: error_result

        resilient.execute_with_backoff("code1")
        resilient.execute_with_backoff("code2")
        result = resilient.execute_with_backoff("code3")

        assert resilient.is_unavailable
        assert result.status == ToolStatus.UNAVAILABLE

    def test_unavailable_repl_returns_early(self) -> None:
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        resilient = ResilientRepl(
            repl,
            backoff=ReplBackoffConfig(base_delay=0.001, max_attempts=1),
        )
        resilient._unavailable = True

        result = resilient.execute_with_backoff("anything")
        assert result.status == ToolStatus.UNAVAILABLE
        assert "unavailable" in (result.error_message or "").lower()

    def test_reset_health_restores_availability(self) -> None:
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        resilient = ResilientRepl(repl)
        resilient._unavailable = True
        resilient._health_failures = 5

        resilient.reset_health()
        assert not resilient.is_unavailable
        assert resilient._health_failures == 0

    def test_success_resets_health_counter(self) -> None:
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        resilient = ResilientRepl(
            repl,
            backoff=ReplBackoffConfig(base_delay=0.001, max_attempts=1),
        )

        error_result = CompilationResult(
            status=ToolStatus.ERROR,
            error_message="process crashed unexpectedly",
        )
        repl.execute = lambda code: error_result
        resilient.execute_with_backoff("fail1")
        resilient.execute_with_backoff("fail2")
        assert resilient._health_failures == 2

        good_result = CompilationResult(
            status=ToolStatus.SUCCESS,
            compilation_status=CompilationStatus.OK,
        )
        repl.execute = lambda code: good_result
        resilient.execute_with_backoff("succeed")
        assert resilient._health_failures == 0
        assert not resilient.is_unavailable
