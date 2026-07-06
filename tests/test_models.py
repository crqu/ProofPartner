"""Tests for the Pydantic eval models."""

from agentic_research.models.eval import (
    AggregateStats,
    BenchmarkSource,
    ConjectureScore,
    EvalConfig,
    EvalMode,
    Problem,
    ProblemDifficulty,
    ProblemSet,
    ProblemSplit,
    WilsonInterval,
)


def test_problem_creation():
    p = Problem(
        id="miniF2F/test_problem",
        name="test_problem",
        source=BenchmarkSource.MINIF2F,
        split=ProblemSplit.TEST,
        lean_statement="theorem test_problem : 1 + 1 = 2 := by sorry",
    )
    assert p.id == "miniF2F/test_problem"
    assert p.source == BenchmarkSource.MINIF2F
    assert p.split == ProblemSplit.TEST
    assert p.difficulty == ProblemDifficulty.UNKNOWN


def test_problem_set_split_filtering():
    problems = [
        Problem(
            id=f"miniF2F/p{i}",
            name=f"p{i}",
            source=BenchmarkSource.MINIF2F,
            split=ProblemSplit.TEST if i < 3 else ProblemSplit.VALIDATION,
            lean_statement=f"theorem p{i} : True := by sorry",
        )
        for i in range(5)
    ]
    ps = ProblemSet(name="test", source=BenchmarkSource.MINIF2F, problems=problems)
    assert len(ps.test_problems) == 3
    assert len(ps.validation_problems) == 2


def test_conjecture_score_composite():
    score = ConjectureScore(formalizability=0.8, non_triviality=0.6, relevance=0.7)
    assert abs(score.composite - 0.7) < 1e-6


def test_eval_config_defaults():
    config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY)
    assert config.benchmark == BenchmarkSource.MINIF2F
    assert config.split == ProblemSplit.VALIDATION
    assert config.pass_k == 1
    assert config.timeout_seconds == 600
    assert config.sample_size is None


def test_wilson_interval_serialization():
    wi = WilsonInterval(lower=0.1, upper=0.5, center=0.3, n=100, successes=30)
    data = wi.model_dump()
    assert data["lower"] == 0.1
    assert data["n"] == 100
    restored = WilsonInterval.model_validate(data)
    assert restored == wi
