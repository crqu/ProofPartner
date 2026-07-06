"""Composite scoring with per-problem results, aggregate statistics, and Wilson confidence bounds."""

from __future__ import annotations

import math

from agentic_research.logging import get_logger
from agentic_research.models.eval import (
    AggregateStats,
    ConjectureScore,
    EvalMode,
    ProblemResult,
    ProblemSplit,
    ProofResult,
    ScoreReport,
    WilsonInterval,
)

log = get_logger(__name__)


def wilson_confidence_interval(
    successes: int, total: int, z: float = 1.96
) -> WilsonInterval:
    """Compute Wilson score confidence interval for a proportion.

    Uses z=1.96 for 95% confidence by default.
    """
    if total == 0:
        return WilsonInterval(lower=0.0, upper=0.0, center=0.0, n=0, successes=0)

    n = total
    p_hat = successes / n
    z2 = z * z

    denominator = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return WilsonInterval(
        lower=round(lower, 4),
        upper=round(upper, 4),
        center=round(center, 4),
        n=n,
        successes=successes,
    )


def compute_aggregate_stats(results: list[ProblemResult]) -> AggregateStats:
    """Compute aggregate statistics from a list of problem results."""
    if not results:
        return AggregateStats()

    total = len(results)
    successes = sum(1 for r in results if r.result == ProofResult.SUCCESS)
    failures = sum(1 for r in results if r.result == ProofResult.FAILURE)
    timeouts = sum(1 for r in results if r.result == ProofResult.TIMEOUT)
    errors = sum(1 for r in results if r.result == ProofResult.ERROR)

    pass_rate = successes / total if total > 0 else 0.0
    wilson_ci = wilson_confidence_interval(successes, total)

    mean_attempts = sum(r.attempts for r in results) / total
    mean_duration = sum(r.duration_seconds for r in results) / total
    total_tokens = sum(r.token_usage for r in results)

    return AggregateStats(
        total=total,
        successes=successes,
        failures=failures,
        timeouts=timeouts,
        errors=errors,
        pass_rate=round(pass_rate, 4),
        wilson_ci=wilson_ci,
        mean_attempts=round(mean_attempts, 2),
        mean_duration_seconds=round(mean_duration, 2),
        total_tokens=total_tokens,
    )


def compute_conjecture_aggregate(scores: list[ConjectureScore]) -> dict[str, float]:
    """Compute aggregate conjecture quality metrics."""
    if not scores:
        return {
            "mean_formalizability": 0.0,
            "mean_non_triviality": 0.0,
            "mean_relevance": 0.0,
            "mean_composite": 0.0,
        }

    n = len(scores)
    return {
        "mean_formalizability": round(sum(s.formalizability for s in scores) / n, 4),
        "mean_non_triviality": round(sum(s.non_triviality for s in scores) / n, 4),
        "mean_relevance": round(sum(s.relevance for s in scores) / n, 4),
        "mean_composite": round(sum(s.composite for s in scores) / n, 4),
    }


def score_eval_run(
    results: list[ProblemResult],
    mode: EvalMode,
    benchmark: str,
    split: ProblemSplit | None = None,
    conjecture_scores: list[ConjectureScore] | None = None,
) -> ScoreReport:
    """Build a complete score report for an evaluation run."""
    aggregate = compute_aggregate_stats(results)

    report = ScoreReport(
        mode=mode,
        benchmark=benchmark,
        split=split,
        results=results,
        aggregate=aggregate,
        conjecture_scores=conjecture_scores,
    )

    log.info(
        "eval_scored",
        mode=mode.value,
        benchmark=benchmark,
        total=aggregate.total,
        pass_rate=aggregate.pass_rate,
        wilson_lower=aggregate.wilson_ci.lower if aggregate.wilson_ci else None,
        wilson_upper=aggregate.wilson_ci.upper if aggregate.wilson_ci else None,
    )

    return report
