"""Tests for the evaluation scorer."""

from agentic_research.eval.scorer import (
    compute_aggregate_stats,
    compute_conjecture_aggregate,
    compute_difficulty_breakdown,
    score_eval_run,
    wilson_confidence_interval,
)
from agentic_research.models.eval import (
    BenchmarkSource,
    ConjectureScore,
    EvalMode,
    Problem,
    ProblemDifficulty,
    ProblemResult,
    ProblemSplit,
    ProofResult,
)


def test_wilson_confidence_interval_basic():
    ci = wilson_confidence_interval(successes=30, total=100)
    assert ci.n == 100
    assert ci.successes == 30
    assert 0.0 < ci.lower < ci.center < ci.upper < 1.0
    assert abs(ci.center - 0.3) < 0.05


def test_wilson_confidence_interval_zero():
    ci = wilson_confidence_interval(successes=0, total=0)
    assert ci.lower == 0.0
    assert ci.upper == 0.0
    assert ci.center == 0.0


def test_wilson_confidence_interval_all_success():
    ci = wilson_confidence_interval(successes=50, total=50)
    assert ci.upper <= 1.0
    assert ci.lower > 0.9


def test_wilson_confidence_interval_no_success():
    ci = wilson_confidence_interval(successes=0, total=50)
    assert ci.lower == 0.0
    assert ci.upper < 0.1


def test_compute_aggregate_stats():
    results = [
        ProblemResult(problem_id="p1", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.SUCCESS, attempts=3, duration_seconds=10.0, token_usage=500),
        ProblemResult(problem_id="p2", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.FAILURE, attempts=5, duration_seconds=20.0, token_usage=800),
        ProblemResult(problem_id="p3", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.SUCCESS, attempts=1, duration_seconds=5.0, token_usage=200),
        ProblemResult(problem_id="p4", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.TIMEOUT, attempts=5, duration_seconds=600.0, token_usage=2000),
    ]
    stats = compute_aggregate_stats(results)
    assert stats.total == 4
    assert stats.successes == 2
    assert stats.failures == 1
    assert stats.timeouts == 1
    assert stats.pass_rate == 0.5
    assert stats.wilson_ci is not None
    assert stats.wilson_ci.n == 4
    assert stats.total_tokens == 3500


def test_compute_aggregate_stats_empty():
    stats = compute_aggregate_stats([])
    assert stats.total == 0
    assert stats.pass_rate == 0.0


def test_compute_conjecture_aggregate():
    scores = [
        ConjectureScore(formalizability=0.8, non_triviality=0.6, relevance=0.7),
        ConjectureScore(formalizability=0.9, non_triviality=0.5, relevance=0.8),
    ]
    agg = compute_conjecture_aggregate(scores)
    assert abs(agg["mean_formalizability"] - 0.85) < 1e-4
    assert abs(agg["mean_non_triviality"] - 0.55) < 1e-4
    assert abs(agg["mean_relevance"] - 0.75) < 1e-4


def test_score_eval_run():
    results = [
        ProblemResult(problem_id="p1", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.SUCCESS, attempts=1),
        ProblemResult(problem_id="p2", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.FAILURE, attempts=3),
    ]
    report = score_eval_run(
        results=results,
        mode=EvalMode.PROOF_DISCOVERY,
        benchmark="miniF2F",
        split=ProblemSplit.VALIDATION,
    )
    assert report.mode == EvalMode.PROOF_DISCOVERY
    assert report.aggregate.total == 2
    assert report.aggregate.pass_rate == 0.5


def _make_problem(pid: str, difficulty: ProblemDifficulty) -> Problem:
    return Problem(
        id=pid,
        name=pid,
        source=BenchmarkSource.MINIF2F,
        split=ProblemSplit.VALIDATION,
        difficulty=difficulty,
        lean_statement=f"theorem {pid} : True := by trivial",
    )


def test_difficulty_breakdown():
    problems = [
        _make_problem("p1", ProblemDifficulty.AMC),
        _make_problem("p2", ProblemDifficulty.AMC),
        _make_problem("p3", ProblemDifficulty.AIME),
        _make_problem("p4", ProblemDifficulty.IMO),
    ]
    results = [
        ProblemResult(problem_id="p1", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.SUCCESS, attempts=1),
        ProblemResult(problem_id="p2", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.SUCCESS, attempts=1),
        ProblemResult(problem_id="p3", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.FAILURE, attempts=1),
        ProblemResult(problem_id="p4", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.TIMEOUT, attempts=1),
    ]

    breakdown = compute_difficulty_breakdown(results, problems)

    assert "amc" in breakdown
    assert "aime" in breakdown
    assert "imo" in breakdown
    assert breakdown["amc"].total == 2
    assert breakdown["amc"].successes == 2
    assert breakdown["amc"].pass_rate == 1.0
    assert breakdown["aime"].total == 1
    assert breakdown["aime"].failures == 1
    assert breakdown["aime"].pass_rate == 0.0
    assert breakdown["imo"].total == 1
    assert breakdown["imo"].timeouts == 1


def test_difficulty_breakdown_empty():
    breakdown = compute_difficulty_breakdown([], [])
    assert breakdown == {}


def test_score_eval_run_with_difficulty():
    problems = [
        _make_problem("p1", ProblemDifficulty.AMC),
        _make_problem("p2", ProblemDifficulty.AIME),
    ]
    results = [
        ProblemResult(problem_id="p1", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.SUCCESS, attempts=1),
        ProblemResult(problem_id="p2", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.FAILURE, attempts=2),
    ]

    report = score_eval_run(
        results=results,
        mode=EvalMode.PROOF_DISCOVERY,
        benchmark="miniF2F",
        split=ProblemSplit.VALIDATION,
        problems=problems,
    )

    assert report.by_difficulty is not None
    assert "amc" in report.by_difficulty
    assert "aime" in report.by_difficulty
    assert report.by_difficulty["amc"].pass_rate == 1.0
    assert report.by_difficulty["aime"].pass_rate == 0.0


def test_score_eval_run_without_problems():
    results = [
        ProblemResult(problem_id="p1", mode=EvalMode.PROOF_DISCOVERY, result=ProofResult.SUCCESS, attempts=1),
    ]
    report = score_eval_run(
        results=results,
        mode=EvalMode.PROOF_DISCOVERY,
        benchmark="miniF2F",
    )
    assert report.by_difficulty is None
