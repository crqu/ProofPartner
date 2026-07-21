"""Tests for format_run_summary terminal output."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from agentic_research.cli.summary import format_run_summary
from agentic_research.models.agents import TokenUsage
from agentic_research.models.research import Conjecture
from agentic_research.models.session import (
    CostEstimate,
    PipelineStage,
    ResearchSessionResult,
    TriedConjecture,
    ConjectureOutcome,
)


def _make_console() -> tuple[Console, StringIO]:
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120)
    return console, buf


def _make_conjecture(statement: str = "For all primes p > 2, p is odd") -> Conjecture:
    return Conjecture(
        statement=statement,
        natural_language=statement,
        confidence=0.9,
        difficulty=3,
    )


class TestProvedResults:
    def test_proved_output_contains_status(self):
        console, buf = _make_console()
        result = ResearchSessionResult(
            session_id="test-123",
            proved_conjectures=[
                TriedConjecture(
                    conjecture=_make_conjecture(),
                    outcome=ConjectureOutcome.PROVED,
                    lean_statement="theorem odd_prime : True := by trivial",
                    proof_code="by trivial",
                    stage_reached=PipelineStage.PROVING,
                ),
            ],
            cost_estimate=CostEstimate(
                input_cost_usd=0.01,
                output_cost_usd=0.02,
            ),
            total_token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            final_stage=PipelineStage.COMPLETE,
        )
        format_run_summary(result, console)
        output = buf.getvalue()
        assert "PROVED 1 conjecture" in output
        assert "$0.03" in output

    def test_proved_with_elapsed(self):
        console, buf = _make_console()
        result = ResearchSessionResult(
            session_id="t",
            proved_conjectures=[
                TriedConjecture(
                    conjecture=_make_conjecture(),
                    outcome=ConjectureOutcome.PROVED,
                    lean_statement="theorem t : True := by trivial",
                    stage_reached=PipelineStage.PROVING,
                ),
            ],
            cost_estimate=CostEstimate(input_cost_usd=0.10),
            total_token_usage=TokenUsage(input_tokens=500),
            final_stage=PipelineStage.COMPLETE,
        )
        format_run_summary(result, console, elapsed_seconds=12.3)
        output = buf.getvalue()
        assert "12.3s" in output

    def test_multiple_proved(self):
        console, buf = _make_console()
        result = ResearchSessionResult(
            session_id="t",
            proved_conjectures=[
                TriedConjecture(
                    conjecture=_make_conjecture("conj 1"),
                    outcome=ConjectureOutcome.PROVED,
                    lean_statement="theorem t1 : True := by trivial",
                    stage_reached=PipelineStage.PROVING,
                ),
                TriedConjecture(
                    conjecture=_make_conjecture("conj 2"),
                    outcome=ConjectureOutcome.PROVED,
                    lean_statement="theorem t2 : True := by trivial",
                    stage_reached=PipelineStage.PROVING,
                ),
            ],
            cost_estimate=CostEstimate(),
            total_token_usage=TokenUsage(),
            final_stage=PipelineStage.COMPLETE,
        )
        format_run_summary(result, console)
        output = buf.getvalue()
        assert "PROVED 2 conjectures" in output


class TestFailedResults:
    def test_failed_output_contains_stage(self):
        console, buf = _make_console()
        result = ResearchSessionResult(
            session_id="t",
            failed_conjectures=[
                TriedConjecture(
                    conjecture=_make_conjecture(),
                    outcome=ConjectureOutcome.PROOF_FAILED,
                    failure_reason="Proof search exhausted",
                    stage_reached=PipelineStage.PROVING,
                ),
            ],
            cost_estimate=CostEstimate(input_cost_usd=0.05),
            total_token_usage=TokenUsage(input_tokens=200),
            final_stage=PipelineStage.FAILED,
        )
        format_run_summary(result, console)
        output = buf.getvalue()
        assert "FAILED" in output
        assert "FAILED" in output
        assert "Proof search exhausted" in output

    def test_failed_table_truncates_long_strings(self):
        console, buf = _make_console()
        long_stmt = "A" * 100
        long_reason = "B" * 100
        result = ResearchSessionResult(
            session_id="t",
            failed_conjectures=[
                TriedConjecture(
                    conjecture=_make_conjecture(long_stmt),
                    outcome=ConjectureOutcome.FORMALIZATION_FAILED,
                    failure_reason=long_reason,
                    stage_reached=PipelineStage.FORMALIZING,
                ),
            ],
            cost_estimate=CostEstimate(),
            total_token_usage=TokenUsage(),
            final_stage=PipelineStage.FAILED,
        )
        format_run_summary(result, console)
        output = buf.getvalue()
        assert long_stmt not in output


class TestEmptyResults:
    def test_empty_result(self):
        console, buf = _make_console()
        result = ResearchSessionResult(
            session_id="t",
            cost_estimate=CostEstimate(),
            total_token_usage=TokenUsage(),
            final_stage=PipelineStage.FAILED,
        )
        format_run_summary(result, console)
        output = buf.getvalue()
        assert "FAILED" in output
        assert "Cost Breakdown" in output


class TestCostBreakdown:
    def test_cost_table_present(self):
        console, buf = _make_console()
        result = ResearchSessionResult(
            session_id="t",
            cost_estimate=CostEstimate(
                input_cost_usd=0.01,
                output_cost_usd=0.02,
                cache_read_cost_usd=0.001,
            ),
            total_token_usage=TokenUsage(
                input_tokens=1000,
                output_tokens=500,
                cache_read_input_tokens=200,
                cache_creation_input_tokens=100,
            ),
            final_stage=PipelineStage.FAILED,
        )
        format_run_summary(result, console)
        output = buf.getvalue()
        assert "Cost Breakdown" in output
        assert "1,000" in output
        assert "500" in output
