"""Composite scoring with per-problem results, aggregate statistics, and Wilson confidence bounds."""

from __future__ import annotations

import math
from collections import defaultdict

from agentic_research.logging import get_logger
from agentic_research.models.eval import (
    AggregateStats,
    ConjectureScore,
    EvalMode,
    Problem,
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

    total_cost = sum(r.cost_usd for r in results)
    mean_cost = total_cost / total
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)

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
        total_cost_usd=round(total_cost, 6),
        mean_cost_usd=round(mean_cost, 6),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
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


def compute_difficulty_breakdown(
    results: list[ProblemResult],
    problems: list[Problem],
) -> dict[str, AggregateStats]:
    """Group results by problem difficulty and compute per-group stats."""
    difficulty_lookup = {p.id: p.difficulty.value for p in problems}

    grouped: dict[str, list[ProblemResult]] = defaultdict(list)
    for r in results:
        difficulty = difficulty_lookup.get(r.problem_id, "unknown")
        grouped[difficulty].append(r)

    return {
        difficulty: compute_aggregate_stats(group_results)
        for difficulty, group_results in sorted(grouped.items())
    }


def score_eval_run(
    results: list[ProblemResult],
    mode: EvalMode,
    benchmark: str,
    split: ProblemSplit | None = None,
    conjecture_scores: list[ConjectureScore] | None = None,
    problems: list[Problem] | None = None,
) -> ScoreReport:
    """Build a complete score report for an evaluation run."""
    aggregate = compute_aggregate_stats(results)

    by_difficulty: dict[str, AggregateStats] | None = None
    if problems is not None:
        by_difficulty = compute_difficulty_breakdown(results, problems)

    report = ScoreReport(
        mode=mode,
        benchmark=benchmark,
        split=split,
        results=results,
        aggregate=aggregate,
        conjecture_scores=conjecture_scores,
        by_difficulty=by_difficulty,
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

    proved_costs = [r.cost_usd for r in results if r.result == ProofResult.SUCCESS]
    failed_costs = [r.cost_usd for r in results if r.result != ProofResult.SUCCESS]
    log.info(
        "eval_cost_summary",
        total_cost_usd=aggregate.total_cost_usd,
        mean_cost_usd=aggregate.mean_cost_usd,
        mean_cost_proved=round(sum(proved_costs) / len(proved_costs), 6) if proved_costs else 0.0,
        mean_cost_failed=round(sum(failed_costs) / len(failed_costs), 6) if failed_costs else 0.0,
        total_input_tokens=aggregate.total_input_tokens,
        total_output_tokens=aggregate.total_output_tokens,
    )

    if by_difficulty:
        log.info(
            "eval_cost_by_difficulty",
            **{tier: stats.total_cost_usd for tier, stats in by_difficulty.items()},
        )

    return report
