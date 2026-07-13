"""Tests for eval cost estimation."""

from __future__ import annotations

from agentic_research.eval.cost import estimate_cost
from agentic_research.eval.scorer import compute_aggregate_stats
from agentic_research.models.agents import TokenUsage
from agentic_research.models.eval import EvalMode, ProblemResult, ProofResult


class TestEstimateCost:
    def test_known_token_counts_opus(self) -> None:
        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_read_input_tokens=500_000,
            cache_creation_input_tokens=200_000,
        )
        cost = estimate_cost(usage, "claude-opus-4-6")
        expected = (
            1_000_000 * 5.0 / 1_000_000
            + 100_000 * 25.0 / 1_000_000
            + 500_000 * 0.50 / 1_000_000
            + 200_000 * 6.25 / 1_000_000
        )
        assert cost == expected
        assert cost == 9.0

    def test_known_token_counts_sonnet(self) -> None:
        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_read_input_tokens=500_000,
            cache_creation_input_tokens=200_000,
        )
        cost = estimate_cost(usage, "claude-sonnet-5")
        expected = (
            1_000_000 * 3.0 / 1_000_000
            + 100_000 * 15.0 / 1_000_000
            + 500_000 * 0.30 / 1_000_000
            + 200_000 * 3.75 / 1_000_000
        )
        assert cost == expected

    def test_zero_tokens(self) -> None:
        usage = TokenUsage()
        cost = estimate_cost(usage, "claude-opus-4-6")
        assert cost == 0.0

    def test_unknown_model_falls_back_to_opus(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
        cost_unknown = estimate_cost(usage, "unknown-model-xyz")
        cost_opus = estimate_cost(usage, "claude-opus-4-6")
        assert cost_unknown == cost_opus
        assert cost_unknown == 5.0

    def test_output_only(self) -> None:
        usage = TokenUsage(output_tokens=1_000_000)
        cost = estimate_cost(usage, "claude-opus-4-6")
        assert cost == 25.0

    def test_default_model_is_opus(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000)
        cost = estimate_cost(usage)
        assert cost == 5.0


class TestAggregateCost:
    def _make_result(self, cost: float, result: ProofResult = ProofResult.SUCCESS, input_tok: int = 0, output_tok: int = 0) -> ProblemResult:
        return ProblemResult(
            problem_id="test",
            mode=EvalMode.PROOF_DISCOVERY,
            result=result,
            cost_usd=cost,
            input_tokens=input_tok,
            output_tokens=output_tok,
        )

    def test_aggregate_cost_sums(self) -> None:
        results = [
            self._make_result(1.50, input_tok=100, output_tok=50),
            self._make_result(2.50, input_tok=200, output_tok=100),
            self._make_result(0.00, ProofResult.TIMEOUT, input_tok=0, output_tok=0),
        ]
        agg = compute_aggregate_stats(results)
        assert round(agg.total_cost_usd, 2) == 4.00
        assert round(agg.mean_cost_usd, 6) == round(4.00 / 3, 6)
        assert agg.total_input_tokens == 300
        assert agg.total_output_tokens == 150

    def test_empty_results(self) -> None:
        agg = compute_aggregate_stats([])
        assert agg.total_cost_usd == 0.0
        assert agg.mean_cost_usd == 0.0
        assert agg.total_input_tokens == 0
        assert agg.total_output_tokens == 0

    def test_timeout_zero_cost(self) -> None:
        results = [self._make_result(0.0, ProofResult.TIMEOUT)]
        agg = compute_aggregate_stats(results)
        assert agg.total_cost_usd == 0.0
        assert agg.mean_cost_usd == 0.0
