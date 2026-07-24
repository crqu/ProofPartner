"""Evaluation runner — orchestrates benchmark loading, problem execution, and scoring.

Run via: python -m agentic_research.eval.runner
"""

from __future__ import annotations

import copy
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.nl_prover import NaturalLanguageProver
from agentic_research.eval.benchmarks import load_minif2f, load_putnam_bench
from agentic_research.eval.cost import estimate_cost
from agentic_research.eval.scorer import score_eval_run
from agentic_research.logging import configure_logging, get_logger
from agentic_research.models.agents import ProverConfig, TokenUsage
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
from agentic_research.models.proof import ProofPipelineResult
from agentic_research.pipelines.proof import ProofPipeline
from agentic_research.tools.lean_repl import LeanRepl, ReplConfig, detect_backend
from agentic_research.tools.lean_search import LeanSearch, SearchConfig, detect_search_backend

log = get_logger(__name__)


@dataclass
class _SharedResources:
    llm_client: LLMClient
    lean_search: LeanSearch


def _sum_token_usage(usage: TokenUsage) -> int:
    return (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )


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
    problem: Problem, config: EvalConfig, shared: _SharedResources
) -> ProblemResult:
    """Evaluate proof discovery for a single problem using the ProofPipeline."""
    start = time.monotonic()

    lean_repl = LeanRepl(ReplConfig(backend=detect_backend()))
    prover_config = ProverConfig(use_extended_thinking=config.use_extended_thinking)
    nl_prover = NaturalLanguageProver(
        llm_client=shared.llm_client,
        prover_config=prover_config,
    )
    pipeline = ProofPipeline(
        llm_client=shared.llm_client,
        lean_repl=lean_repl,
        lean_search=shared.lean_search,
        prover_config=prover_config,
        max_critic_retries=config.max_critic_retries,
        use_intent_judge=config.use_intent_judge,
        nl_prover=nl_prover,
        use_nl_proof_stage=True,
    )

    full_statement = (
        f"{problem.lean_header}\n\n{problem.lean_statement}"
        if problem.lean_header
        else problem.lean_statement
    )

    result_holder: list[ProofPipelineResult] = []
    error_holder: list[Exception] = []

    def _run_pipeline() -> None:
        try:
            r = pipeline.run(
                lean_statement=full_statement,
                statement_nl=problem.natural_language,
            )
            result_holder.append(r)
        except Exception as exc:
            error_holder.append(exc)

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()
    thread.join(timeout=config.timeout_seconds)

    duration = round(time.monotonic() - start, 3)

    if thread.is_alive():
        log.warning("proof_discovery_timeout", problem=problem.id, timeout=config.timeout_seconds)
        return ProblemResult(
            problem_id=problem.id,
            mode=EvalMode.PROOF_DISCOVERY,
            result=ProofResult.TIMEOUT,
            attempts=1,
            duration_seconds=duration,
            error_message=f"Timeout after {config.timeout_seconds}s",
        )

    if error_holder:
        exc = error_holder[0]
        log.error("proof_discovery_error", problem=problem.id, error=str(exc))
        return ProblemResult(
            problem_id=problem.id,
            mode=EvalMode.PROOF_DISCOVERY,
            result=ProofResult.ERROR,
            attempts=1,
            duration_seconds=duration,
            error_message=str(exc),
        )

    if not result_holder:
        return ProblemResult(
            problem_id=problem.id,
            mode=EvalMode.PROOF_DISCOVERY,
            result=ProofResult.ERROR,
            attempts=1,
            duration_seconds=duration,
            error_message="Pipeline returned no result",
        )

    pipeline_result = result_holder[0]
    usage = pipeline_result.total_token_usage
    token_total = _sum_token_usage(usage)
    model = config.model or "claude-opus-4-6"
    cost = estimate_cost(usage, model)

    if pipeline_result.proved:
        log.info("proof_discovery_success", problem=problem.id)
        return ProblemResult(
            problem_id=problem.id,
            mode=EvalMode.PROOF_DISCOVERY,
            result=ProofResult.SUCCESS,
            proof=pipeline_result.final_proof,
            attempts=1,
            duration_seconds=duration,
            token_usage=token_total,
            cost_usd=round(cost, 6),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
        )

    log.debug("proof_discovery_failure", problem=problem.id, stage=pipeline_result.failure_stage)
    return ProblemResult(
        problem_id=problem.id,
        mode=EvalMode.PROOF_DISCOVERY,
        result=ProofResult.FAILURE,
        attempts=1,
        duration_seconds=duration,
        error_message=pipeline_result.failure_reason,
        token_usage=token_total,
        cost_usd=round(cost, 6),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
    )


def _evaluate_conjecture_quality(
    problem: Problem, config: EvalConfig, shared: _SharedResources | None = None
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
    problem: Problem, config: EvalConfig, shared: _SharedResources | None = None
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

    shared: _SharedResources | None = None
    if config.mode == EvalMode.PROOF_DISCOVERY:
        llm_kwargs: dict = {}
        if config.model is not None:
            llm_kwargs["model"] = config.model
        llm_client = LLMClient(**llm_kwargs)
        lean_search = LeanSearch(SearchConfig(backend=detect_search_backend()))
        shared = _SharedResources(llm_client=llm_client, lean_search=lean_search)

    evaluate_fn = _EVAL_DISPATCH[config.mode]

    results: list[ProblemResult] = []
    for i, problem in enumerate(problems):
        log.info("eval_problem", index=i + 1, total=len(problems), problem=problem.id)

        best_result: ProblemResult | None = None
        for attempt in range(config.pass_k):
            attempt_config = copy.copy(config)
            if attempt_config.seed is not None:
                attempt_config.seed = config.seed + attempt
            result = evaluate_fn(problem, attempt_config, shared)
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
        problems=problems,
    )

    log.info(
        "eval_complete",
        mode=config.mode.value,
        total=report.aggregate.total,
        pass_rate=report.aggregate.pass_rate,
        by_difficulty={k: v.pass_rate for k, v in (report.by_difficulty or {}).items()},
    )

    return report


def main() -> None:
    """CLI entry point for the eval runner."""
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
    @click.option("--model", default=None, help="LLM model for proof attempts")
    @click.option("--extended-thinking/--no-extended-thinking", default=True, help="Enable extended thinking for proof search")
    @click.option("--thinking-budget", type=int, default=10000, help="Token budget for extended thinking")
    @click.option("--max-critic-retries", type=int, default=3, help="Max proof critic retry rounds")
    @click.option("--use-intent-judge/--no-use-intent-judge", default=True, help="Enable intent judge for type formalization")
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
        model: str | None,
        extended_thinking: bool,
        thinking_budget: int,
        max_critic_retries: int,
        use_intent_judge: bool,
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
            model=model,
            use_extended_thinking=extended_thinking,
            thinking_budget=thinking_budget,
            max_critic_retries=max_critic_retries,
            use_intent_judge=use_intent_judge,
        )

        report = run_eval(config)

        report_json = report.model_dump_json(indent=2)

        if output:
            Path(output).write_text(report_json)
            log.info("report_written", path=output)
        else:
            click.echo(report_json)

        agg = report.aggregate
        proved_costs = [r.cost_usd for r in report.results if r.result == ProofResult.SUCCESS]
        failed_costs = [r.cost_usd for r in report.results if r.result != ProofResult.SUCCESS]
        mean_proved = sum(proved_costs) / len(proved_costs) if proved_costs else 0.0
        mean_failed = sum(failed_costs) / len(failed_costs) if failed_costs else 0.0

        click.echo("\nCost Summary:")
        click.echo(f"  Total: ${agg.total_cost_usd:.2f} ({agg.total} problems)")
        click.echo(f"  Per problem: ${agg.mean_cost_usd:.2f} avg (${mean_proved:.2f} proved, ${mean_failed:.2f} failed)")
        click.echo(f"  Tokens: {agg.total_input_tokens} input, {agg.total_output_tokens} output, {sum(r.cache_read_input_tokens for r in report.results)} cache_read")

        success_rate = report.aggregate.pass_rate
        sys.exit(0 if success_rate >= 0.0 else 1)

    run()


if __name__ == "__main__":
    main()
