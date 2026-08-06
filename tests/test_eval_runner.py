"""Tests for eval runner configuration and ProverConfig threading."""

from agentic_research.models.agents import ProverConfig
from agentic_research.models.eval import (
    BenchmarkSource,
    EvalConfig,
    EvalMode,
    Problem,
    ProblemSplit,
)


def test_eval_config_defaults():
    """EvalConfig new fields have correct defaults."""
    config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY)
    assert config.use_extended_thinking is True
    assert config.thinking_budget == 10000
    assert config.max_critic_retries == 3
    assert config.use_intent_judge is True
    assert config.timeout_seconds == 600
    assert config.problem_filter is None


def test_prover_config_from_eval_config():
    """ProverConfig is constructed with use_extended_thinking from EvalConfig."""
    config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY, use_extended_thinking=True)
    prover_config = ProverConfig(use_extended_thinking=config.use_extended_thinking)
    assert prover_config.use_extended_thinking is True


def test_prover_config_extended_thinking_disabled():
    """ProverConfig respects use_extended_thinking=False from EvalConfig."""
    config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY, use_extended_thinking=False)
    prover_config = ProverConfig(use_extended_thinking=config.use_extended_thinking)
    assert prover_config.use_extended_thinking is False


def test_eval_config_timeout_seconds():
    """EvalConfig accepts custom timeout_seconds."""
    config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY, timeout_seconds=1800)
    assert config.timeout_seconds == 1800


def test_eval_config_problem_filter():
    """EvalConfig accepts problem_filter list."""
    config = EvalConfig(
        mode=EvalMode.PROOF_DISCOVERY,
        problem_filter=["putnam_2024_a1"],
    )
    assert config.problem_filter == ["putnam_2024_a1"]


def test_problem_filter_selects_matching():
    """_select_problems filters by name substring when problem_filter is set."""
    from unittest.mock import patch

    from agentic_research.eval.runner import _select_problems

    problems = [
        Problem(
            id="putnam_bench/putnam_2024_a1",
            name="putnam_2024_a1",
            source=BenchmarkSource.PUTNAM_BENCH,
            split=ProblemSplit.VALIDATION,
            lean_statement="theorem putnam_2024_a1 : True := by sorry",
        ),
        Problem(
            id="putnam_bench/putnam_2024_a2",
            name="putnam_2024_a2",
            source=BenchmarkSource.PUTNAM_BENCH,
            split=ProblemSplit.VALIDATION,
            lean_statement="theorem putnam_2024_a2 : True := by sorry",
        ),
        Problem(
            id="putnam_bench/putnam_2023_b1",
            name="putnam_2023_b1",
            source=BenchmarkSource.PUTNAM_BENCH,
            split=ProblemSplit.VALIDATION,
            lean_statement="theorem putnam_2023_b1 : True := by sorry",
        ),
    ]

    config = EvalConfig(
        mode=EvalMode.PROOF_DISCOVERY,
        benchmark=BenchmarkSource.PUTNAM_BENCH,
        problem_filter=["putnam_2024_a1"],
    )

    with patch("agentic_research.eval.runner.load_putnam_bench") as mock_load:
        from agentic_research.models.eval import ProblemSet
        mock_load.return_value = ProblemSet(
            name="PutnamBench",
            source=BenchmarkSource.PUTNAM_BENCH,
            problems=problems,
        )
        result = _select_problems(config)

    assert len(result) == 1
    assert result[0].name == "putnam_2024_a1"


def test_problem_filter_none_returns_all():
    """_select_problems returns all problems when problem_filter is None."""
    from unittest.mock import patch

    from agentic_research.eval.runner import _select_problems

    problems = [
        Problem(
            id=f"miniF2F/p{i}",
            name=f"p{i}",
            source=BenchmarkSource.MINIF2F,
            split=ProblemSplit.VALIDATION,
            lean_statement=f"theorem p{i} : True := by sorry",
        )
        for i in range(3)
    ]

    config = EvalConfig(
        mode=EvalMode.PROOF_DISCOVERY,
        benchmark=BenchmarkSource.MINIF2F,
    )

    with patch("agentic_research.eval.runner.load_minif2f") as mock_load:
        from agentic_research.models.eval import ProblemSet
        mock_load.return_value = ProblemSet(
            name="miniF2F",
            source=BenchmarkSource.MINIF2F,
            problems=problems,
        )
        result = _select_problems(config)

    assert len(result) == 3
