"""Tests for the eval runner."""

from pathlib import Path

from agentic_research.eval.runner import run_eval
from agentic_research.logging import configure_logging
from agentic_research.models.eval import (
    BenchmarkSource,
    EvalConfig,
    EvalMode,
    ProblemSplit,
    ProofResult,
)


def test_run_eval_stub(tmp_path: Path):
    """Verify the runner executes end-to-end with the stub prover."""
    configure_logging(json_output=False, level="WARNING")

    repo_dir = tmp_path / "miniF2F"
    (repo_dir / ".git").mkdir(parents=True)
    lean_dir = repo_dir / "lean4" / "MiniF2F" / "Test"
    lean_dir.mkdir(parents=True)
    lean_file = lean_dir / "Basic.lean"
    lean_file.write_text(
        "import Mathlib\n\n"
        "theorem test_one (n : Nat) : n = n := by rfl\n\n"
        "theorem test_two : True := by trivial\n"
    )

    config = EvalConfig(
        mode=EvalMode.PROOF_DISCOVERY,
        benchmark=BenchmarkSource.MINIF2F,
        split=ProblemSplit.TEST,
        pass_k=1,
        data_dir=tmp_path,
    )

    report = run_eval(config)
    assert report.mode == EvalMode.PROOF_DISCOVERY
    assert report.aggregate.total == 2
    assert report.aggregate.pass_rate == 0.0
    for r in report.results:
        assert r.result == ProofResult.FAILURE
        assert "not yet implemented" in (r.error_message or "")
