"""Evaluation runner — orchestrates benchmark loading, problem execution, and scoring.

Run via: python -m agentic_research.eval.runner
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from agentic_research.eval.benchmarks import load_minif2f, load_putnam_bench
from agentic_research.eval.scorer import score_eval_run
from agentic_research.logging import configure_logging, get_logger
from agentic_research.models.eval import (
    BenchmarkSource,
    EvalConfig,
    EvalMode,
    Problem,
    ProblemResult,
    ProblemSplit,
    ProofResult,
    ScoreReport,
)

log = get_logger(__name__)


def _select_problems(
    config: EvalConfig, data_dir: Path | None = None
) -> list[Problem]:
    """Load and optionally sample problems based on config."""
    if config.benchmark == BenchmarkSource.MINIF2F:
        problem_set = load_minif2f(data_dir or config.data_dir)
    elif config.benchmark == BenchmarkSource.PUTNAM_BENCH:
        problem_set = load_putnam_bench(data_dir or config.data_dir)
    else:
        raise ValueError(f"Unknown benchmark: {config.benchmark}")

    if config.split == ProblemSplit.TEST:
        problems = problem_set.test_problems
    else:
        problems = problem_set.validation_problems

    if config.sample_size is not None and config.sample_size < len(problems):
        rng = random.Random(config.seed)
        problems = rng.sample(problems, config.sample_size)

    log.info(
        "problems_selected",
        benchmark=config.benchmark.value,
        split=config.split.value,
        count=len(problems),
        sample_size=config.sample_size,
    )

    return problems


def _evaluate_proof_discovery(
    problem: Problem, config: EvalConfig
) -> ProblemResult:
    """Evaluate proof discovery for a single problem.

    Stub: returns FAILURE. Actual proving requires the prover agent (Phase 3).
    """
    start = time.monotonic()

    result = ProblemResult(
        problem_id=problem.id,
        mode=EvalMode.PROOF_DISCOVERY,
        result=ProofResult.FAILURE,
        attempts=0,
        duration_seconds=round(time.monotonic() - start, 3),
        error_message="Prover not yet implemented (Phase 3)",
    )

    log.debug("proof_discovery_result", problem=problem.id, result=result.result.value)
    return result


def _evaluate_conjecture_quality(
    problem: Problem, config: EvalConfig
) -> ProblemResult:
    """Evaluate conjecture quality for a single problem.

    Stub: returns FAILURE. Requires conjecture generator (Phase 4).
    """
    return ProblemResult(
        problem_id=problem.id,
        mode=EvalMode.CONJECTURE_QUALITY,
        result=ProofResult.FAILURE,
        attempts=0,
        error_message="Conjecture generator not yet implemented (Phase 4)",
    )


def _evaluate_end_to_end(
    problem: Problem, config: EvalConfig
) -> ProblemResult:
    """Evaluate end-to-end research for a single problem.

    Stub: returns FAILURE. Requires full pipeline (Phase 9+).
    """
    return ProblemResult(
        problem_id=problem.id,
        mode=EvalMode.END_TO_END,
        result=ProofResult.FAILURE,
        attempts=0,
        error_message="Full pipeline not yet implemented (Phase 9+)",
    )


_EVAL_DISPATCH = {
    EvalMode.PROOF_DISCOVERY: _evaluate_proof_discovery,
    EvalMode.CONJECTURE_QUALITY: _evaluate_conjecture_quality,
    EvalMode.END_TO_END: _evaluate_end_to_end,
}


def run_eval(config: EvalConfig) -> ScoreReport:
    """Run a full evaluation pass."""
    log.info(
        "eval_starting",
        mode=config.mode.value,
        benchmark=config.benchmark.value,
        split=config.split.value,
        pass_k=config.pass_k,
    )

    problems = _select_problems(config)
    evaluate_fn = _EVAL_DISPATCH[config.mode]

    results: list[ProblemResult] = []
    for i, problem in enumerate(problems):
        log.info("eval_problem", index=i + 1, total=len(problems), problem=problem.id)

        best_result: ProblemResult | None = None
        for attempt in range(config.pass_k):
            result = evaluate_fn(problem, config)
            if best_result is None or result.result == ProofResult.SUCCESS:
                best_result = result
            if result.result == ProofResult.SUCCESS:
                break

        assert best_result is not None
        results.append(best_result)

    report = score_eval_run(
        results=results,
        mode=config.mode,
        benchmark=config.benchmark.value,
        split=config.split,
    )

    log.info(
        "eval_complete",
        mode=config.mode.value,
        total=report.aggregate.total,
        pass_rate=report.aggregate.pass_rate,
    )

    return report


def main() -> None:
    """CLI entry point for the eval runner."""
    import json
    import sys

    import click

    @click.command()
    @click.option(
        "--mode",
        type=click.Choice([m.value for m in EvalMode]),
        default=EvalMode.PROOF_DISCOVERY.value,
        help="Evaluation mode",
    )
    @click.option(
        "--benchmark",
        type=click.Choice([b.value for b in BenchmarkSource]),
        default=BenchmarkSource.MINIF2F.value,
        help="Benchmark to evaluate",
    )
    @click.option(
        "--split",
        type=click.Choice([s.value for s in ProblemSplit]),
        default=ProblemSplit.VALIDATION.value,
        help="Problem split",
    )
    @click.option("--pass-k", type=int, default=1, help="Number of attempts per problem")
    @click.option("--sample-size", type=int, default=None, help="Subset size")
    @click.option("--seed", type=int, default=0, help="Random seed")
    @click.option("--data-dir", type=click.Path(), default="data/benchmarks")
    @click.option("--json-logs/--console-logs", default=True, help="Log format")
    @click.option("--output", type=click.Path(), default=None, help="Write JSON report to file")
    def run(
        mode: str,
        benchmark: str,
        split: str,
        pass_k: int,
        sample_size: int | None,
        seed: int,
        data_dir: str,
        json_logs: bool,
        output: str | None,
    ) -> None:
        """Run the evaluation harness."""
        configure_logging(json_output=json_logs)

        config = EvalConfig(
            mode=EvalMode(mode),
            benchmark=BenchmarkSource(benchmark),
            split=ProblemSplit(split),
            pass_k=pass_k,
            sample_size=sample_size,
            seed=seed,
            data_dir=Path(data_dir),
        )

        report = run_eval(config)

        report_json = report.model_dump_json(indent=2)

        if output:
            Path(output).write_text(report_json)
            log.info("report_written", path=output)
        else:
            click.echo(report_json)

        success_rate = report.aggregate.pass_rate
        sys.exit(0 if success_rate >= 0.0 else 1)

    run()


if __name__ == "__main__":
    main()
