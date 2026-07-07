"""Tests for Phase 2: Production hardening — circuit breaker, retry, cost tracker, budget, cycles."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agentic_research.agents.llm_client import LLMRetryExhaustedError
from agentic_research.models.agents import TokenUsage
from agentic_research.models.session import OrchestratorConfig, PipelineStage
from agentic_research.orchestrator.circuit_breaker import CircuitBreaker
from agentic_research.orchestrator.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_initially_closed(self) -> None:
        cb = CircuitBreaker()
        assert not cb.is_open()

    def test_opens_after_consecutive_failures(self) -> None:
        cb = CircuitBreaker(consecutive_failure_limit=3, error_rate_threshold=1.0)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open()
        cb.record_failure()
        assert cb.is_open()

    def test_success_resets_consecutive_count(self) -> None:
        cb = CircuitBreaker(consecutive_failure_limit=3, error_rate_threshold=1.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open()

    def test_opens_on_error_rate_threshold(self) -> None:
        cb = CircuitBreaker(
            consecutive_failure_limit=100,
            error_rate_threshold=0.5,
            window_seconds=60.0,
        )
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()

    def test_reset_closes_breaker(self) -> None:
        cb = CircuitBreaker(consecutive_failure_limit=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()
        cb.reset()
        assert not cb.is_open()

    def test_default_consecutive_threshold(self) -> None:
        cb = CircuitBreaker(error_rate_threshold=1.0)
        for _ in range(4):
            cb.record_failure()
        assert not cb.is_open()
        cb.record_failure()
        assert cb.is_open()

    def test_old_events_pruned_from_window(self) -> None:
        cb = CircuitBreaker(
            consecutive_failure_limit=100,
            error_rate_threshold=0.5,
            window_seconds=0.1,
        )
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb._consecutive_failures = 0
        cb.record_success()
        assert not cb.is_open()


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_total_cost_accumulates(self) -> None:
        ct = CostTracker()
        ct.record_usage(input_tokens=1_000_000, output_tokens=0)
        assert ct.total_cost() == pytest.approx(15.0, abs=0.01)
        ct.record_usage(input_tokens=0, output_tokens=1_000_000)
        assert ct.total_cost() == pytest.approx(15.0 + 75.0, abs=0.01)

    def test_velocity_zero_initially(self) -> None:
        ct = CostTracker()
        assert ct.velocity() == 0.0

    def test_velocity_calculation(self) -> None:
        ct = CostTracker(velocity_window=60.0)
        ct.record_usage(input_tokens=1_000_000, output_tokens=0)
        vel = ct.velocity()
        assert vel > 0.0

    def test_velocity_warning_fires(self) -> None:
        ct = CostTracker(velocity_threshold=0.001)
        with patch("agentic_research.orchestrator.cost_tracker.log") as mock_log:
            ct.record_usage(input_tokens=1_000_000, output_tokens=0)
            mock_log.warning.assert_called_once()
            call_kwargs = mock_log.warning.call_args
            assert "cost_velocity_exceeded" in str(call_kwargs)

    def test_velocity_warning_does_not_fire_below_threshold(self) -> None:
        ct = CostTracker(velocity_threshold=99999.0)
        ct.record_usage(input_tokens=100, output_tokens=50)
        time.sleep(0.05)
        with patch("agentic_research.orchestrator.cost_tracker.log") as mock_log:
            ct.record_usage(input_tokens=100, output_tokens=50)
            mock_log.warning.assert_not_called()

    def test_record_returns_call_cost(self) -> None:
        ct = CostTracker()
        cost = ct.record_usage(input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(15.0, abs=0.01)


# ---------------------------------------------------------------------------
# LLM Client Retry
# ---------------------------------------------------------------------------


class TestLLMRetry:
    def test_retry_exhaustion_raises(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from agentic_research.agents.llm_client import LLMClient

                client = LLMClient(api_key="test-key", max_retries=2, backoff_base=0.01)
                client._client.messages.create = MagicMock(
                    side_effect=RuntimeError("API down")
                )

                with pytest.raises(LLMRetryExhaustedError, match="2 retries exhausted"):
                    client.complete(messages=[{"role": "user", "content": "hello"}])

    def test_retry_succeeds_on_second_attempt(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from agentic_research.agents.llm_client import LLMClient

                client = LLMClient(api_key="test-key", max_retries=3, backoff_base=0.01)

                mock_usage = MagicMock()
                mock_usage.input_tokens = 10
                mock_usage.output_tokens = 5
                mock_usage.cache_creation_input_tokens = 0
                mock_usage.cache_read_input_tokens = 0

                mock_response = MagicMock()
                mock_response.content = [MagicMock(type="text", text="ok")]
                mock_response.usage = mock_usage
                mock_response.stop_reason = "end_turn"
                mock_response.model = "test-model"

                client._client.messages.create = MagicMock(
                    side_effect=[RuntimeError("fail"), mock_response]
                )

                result = client.complete(messages=[{"role": "user", "content": "hi"}])
                assert result.content == "ok"
                assert client._client.messages.create.call_count == 2

    def test_retry_counts_match(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                from agentic_research.agents.llm_client import LLMClient

                client = LLMClient(api_key="test-key", max_retries=3, backoff_base=0.01)
                client._client.messages.create = MagicMock(
                    side_effect=RuntimeError("fail")
                )

                with pytest.raises(LLMRetryExhaustedError):
                    client.complete(messages=[{"role": "user", "content": "hi"}])

                assert client._client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# OrchestratorConfig defaults
# ---------------------------------------------------------------------------


class TestOrchestratorConfigDefaults:
    def test_budget_limit_defaults_to_10(self) -> None:
        config = OrchestratorConfig()
        assert config.budget_limit_usd == 10.0

    def test_max_reasoning_cycles_default(self) -> None:
        config = OrchestratorConfig()
        assert config.max_reasoning_cycles == 25


# ---------------------------------------------------------------------------
# Budget enforcement in orchestrator
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_budget_halts_when_exceeded(self) -> None:
        from agentic_research.orchestrator.engine import ResearchOrchestrator
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
        from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

        mock_llm = MagicMock()
        mock_llm.model = "test"
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        config = OrchestratorConfig(budget_limit_usd=0.0001)

        orch = ResearchOrchestrator(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            config=config,
        )
        orch._total_tokens = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)

        result = orch.run("test idea")
        assert result.final_stage == PipelineStage.FAILED
        assert "Budget" in str(
            orch.state_machine.session_state.transitions[-1].reason
        )


# ---------------------------------------------------------------------------
# Reasoning cycle cap
# ---------------------------------------------------------------------------


class TestReasoningCycleCap:
    def test_cycle_cap_enforced(self) -> None:
        from agentic_research.orchestrator.engine import ResearchOrchestrator
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
        from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

        mock_llm = MagicMock()
        mock_llm.model = "test"
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        config = OrchestratorConfig(max_reasoning_cycles=2, budget_limit_usd=1000.0)

        orch = ResearchOrchestrator(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            config=config,
        )

        from agentic_research.models.agents import AgentResult, AgentStatus

        mock_agent_result = AgentResult(
            agent_name="explorer",
            status=AgentStatus.FAILURE,
            result=None,
        )

        with patch(
            "agentic_research.orchestrator.engine.ExplorationAgent"
        ) as MockExplorer:
            instance = MockExplorer.return_value
            instance.run.return_value = mock_agent_result

            result = orch.run("test")

        assert result.final_stage == PipelineStage.FAILED
        assert orch._reasoning_cycles <= 2


# ---------------------------------------------------------------------------
# Circuit breaker integration with orchestrator
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    def test_circuit_breaker_halts_orchestrator(self) -> None:
        from agentic_research.orchestrator.engine import ResearchOrchestrator
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
        from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

        mock_llm = MagicMock()
        mock_llm.model = "test"
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        config = OrchestratorConfig(budget_limit_usd=1000.0, max_reasoning_cycles=100)

        orch = ResearchOrchestrator(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            config=config,
        )
        for _ in range(5):
            orch.circuit_breaker.record_failure()

        result = orch.run("test")
        assert result.final_stage == PipelineStage.FAILED
        transitions = orch.state_machine.session_state.transitions
        assert any("Circuit breaker" in t.reason for t in transitions)
