"""CLI entry point for the Agentic Research Partner."""

from __future__ import annotations

from pathlib import Path

import click

from agentic_research import __version__
from agentic_research.logging import configure_logging


@click.group()
@click.version_option(version=__version__)
@click.option("--json-logs/--console-logs", default=False, help="Use JSON log format")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def cli(json_logs: bool, log_level: str) -> None:
    """Agentic Mathematical Research Partner."""
    configure_logging(json_output=json_logs, level=log_level)


@cli.command("eval")
@click.argument("benchmark", type=click.Choice(["miniF2F", "PutnamBench"]))
@click.option("--mode", type=click.Choice(["proof_discovery", "conjecture_quality", "end_to_end"]), default="proof_discovery")
@click.option("--split", type=click.Choice(["test", "valid"]), default="valid")
@click.option("--pass-k", type=int, default=1, help="Number of attempts per problem (pass@k)")
@click.option("--sample-size", type=int, default=None, help="Evaluate a subset of problems")
@click.option("--seed", type=int, default=0, help="Random seed for sampling")
@click.option("--data-dir", type=click.Path(), default="data/benchmarks")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write JSON report to file")
def eval_cmd(
    benchmark: str,
    mode: str,
    split: str,
    pass_k: int,
    sample_size: int | None,
    seed: int,
    data_dir: str,
    output: str | None,
) -> None:
    """Run benchmark evaluation."""
    from agentic_research.eval.runner import run_eval
    from agentic_research.models.eval import BenchmarkSource, EvalConfig, EvalMode, ProblemSplit

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
        click.echo(f"Report written to {output}")
    else:
        click.echo(report_json)


@cli.command("research")
@click.argument("idea")
def research_cmd(idea: str) -> None:
    """Start a research session from a rough mathematical idea."""
    click.echo(f"Research sessions not yet implemented (Phase 4+). Idea: {idea}")


@cli.command("prove")
@click.argument("lean_file", type=click.Path(exists=True))
def prove_cmd(lean_file: str) -> None:
    """Prove a single Lean 4 theorem statement."""
    click.echo(f"Prover not yet implemented (Phase 3). File: {lean_file}")


if __name__ == "__main__":
    cli()
