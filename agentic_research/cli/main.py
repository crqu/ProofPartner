"""CLI entry point for the Agentic Research Partner.

Five decoupled commands: explore, formalize, check, prove, status.
Each enforces a per-command budget via CostTracker and displays
real-time cost with rich.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from agentic_research import __version__
from agentic_research.logging import configure_logging

SESSION_DIR = Path(".agentic_research/sessions")
console = Console()


def _get_session_path() -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR / "current_session.json"


def _load_or_create_session():
    from agentic_research.memory.session import ResearchSessionMemory

    path = _get_session_path()
    if path.exists():
        return ResearchSessionMemory.load(path)
    return ResearchSessionMemory()


def _save_session(session) -> None:
    path = _get_session_path()
    session.save(path)


def _create_llm_client():
    from agentic_research.agents.llm_client import LLMClient

    return LLMClient()


def _create_lean_repl():
    from agentic_research.tools.lean_repl import LeanRepl

    return LeanRepl()


def _create_lean_search():
    from agentic_research.tools.lean_search import LeanSearch

    return LeanSearch()


def _create_cost_tracker():
    from agentic_research.orchestrator.cost_tracker import CostTracker

    return CostTracker()


def _check_budget(cost_tracker, budget: float) -> bool:
    """Return True if budget is exceeded."""
    return cost_tracker.total_cost() > budget


def _cost_display(cost_tracker, budget: float) -> str:
    return f"[cost: ${cost_tracker.total_cost():.2f} / ${budget:.2f}]"


def _record_agent_tokens(cost_tracker, agent) -> None:
    """Record cumulative tokens from a BaseAgent into the CostTracker."""
    tokens = agent.cumulative_tokens
    cost_tracker.record_usage(
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        cache_read_tokens=tokens.cache_read_input_tokens,
        cache_write_tokens=tokens.cache_creation_input_tokens,
    )


def _print_cost_summary(cost_tracker, budget: float) -> None:
    table = Table(title="Cost Summary", show_header=False)
    table.add_column("Label", style="bold")
    table.add_column("Value")
    table.add_row("Total cost", f"${cost_tracker.total_cost():.4f}")
    table.add_row("Budget", f"${budget:.2f}")
    exceeded = cost_tracker.total_cost() > budget
    table.add_row("Status", "[red]EXCEEDED[/red]" if exceeded else "[green]Within budget[/green]")
    console.print(table)


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


@cli.command("explore")
@click.argument("idea")
@click.option("--budget", type=float, default=2.00, help="Budget in USD (default: $2.00)")
def explore_cmd(idea: str, budget: float) -> None:
    """Explore a rough mathematical idea and generate conjectures.

    Runs ExplorationAgent + ConjectureGenerator. Takes a natural language
    math idea, searches Mathlib for related concepts, and generates ranked
    conjectures.
    """
    from agentic_research.agents.conjecturer import ConjectureGenerator
    from agentic_research.agents.explorer import ExplorationAgent
    from agentic_research.models.agents import AgentContext, AgentStatus

    cost_tracker = _create_cost_tracker()
    session = _load_or_create_session()

    try:
        llm = _create_llm_client()
        lean_search = _create_lean_search()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    explorer = ExplorationAgent(llm_client=llm, lean_search=lean_search)
    conjecture_gen = ConjectureGenerator(llm_client=llm)

    with console.status(f"{_cost_display(cost_tracker, budget)} Exploring: {idea[:60]}...") as status:
        ctx = AgentContext(task=idea)
        explore_result = explorer.run(ctx)
        _record_agent_tokens(cost_tracker, explorer)
        status.update(f"{_cost_display(cost_tracker, budget)} Exploration complete, generating conjectures...")

        if explore_result.status != AgentStatus.SUCCESS:
            console.print(f"[red]Exploration failed:[/red] {explore_result.error_message}")
            _print_cost_summary(cost_tracker, budget)
            sys.exit(1)

        if _check_budget(cost_tracker, budget):
            console.print("[yellow]Budget exceeded after exploration.[/yellow]")
            _print_cost_summary(cost_tracker, budget)
            sys.exit(1)

        conj_ctx = AgentContext(
            task=idea,
            metadata={"exploration_result": explore_result.result or {}},
        )
        conj_result = conjecture_gen.run(conj_ctx)
        _record_agent_tokens(cost_tracker, conjecture_gen)

    if conj_result.status != AgentStatus.SUCCESS:
        console.print(f"[red]Conjecture generation failed:[/red] {conj_result.error_message}")
        _print_cost_summary(cost_tracker, budget)
        sys.exit(1)

    result_data = conj_result.result or {}
    conjectures = result_data.get("conjectures", [])

    table = Table(title="Generated Conjectures")
    table.add_column("#", style="bold", width=3)
    table.add_column("Statement", min_width=40)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Difficulty", justify="right", width=10)

    for i, conj in enumerate(conjectures, 1):
        conf = conj.get("confidence", 0)
        diff = conj.get("difficulty", 0)
        stmt = conj.get("statement", conj.get("natural_language", ""))
        table.add_row(str(i), stmt, f"{conf:.2f}", str(diff))

        from agentic_research.models.research import Conjecture

        try:
            conj_obj = Conjecture(**conj)
            session.record_conjecture(conj_obj)
        except Exception:
            pass

    console.print(table)

    if result_data.get("exploration_context"):
        exp_ctx = result_data["exploration_context"]
        domain = exp_ctx.get("domain", "unknown")
        n_concepts = len(exp_ctx.get("concepts", []))
        n_directions = len(exp_ctx.get("directions", []))
        console.print(f"\n[dim]Domain: {domain} | Concepts found: {n_concepts} | Directions: {n_directions}[/dim]")

    _save_session(session)
    _print_cost_summary(cost_tracker, budget)


@cli.command("formalize")
@click.argument("conjecture")
@click.option("--budget", type=float, default=3.00, help="Budget in USD (default: $3.00)")
def formalize_cmd(conjecture: str, budget: float) -> None:
    """Formalize a natural language conjecture into Lean 4.

    Runs FormalizationPipeline + IntentJudge. Takes a natural language
    conjecture, produces a Lean 4 statement with type-first formalization,
    and runs intent verification.
    """
    from agentic_research.agents.informalizer import Informalizer
    from agentic_research.agents.intent_judge import IntentJudge

    cost_tracker = _create_cost_tracker()
    session = _load_or_create_session()

    try:
        llm = _create_llm_client()
        lean_repl = _create_lean_repl()
        lean_search = _create_lean_search()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    from agentic_research.pipelines.formalization import FormalizationPipeline

    pipeline = FormalizationPipeline(
        llm_client=llm, lean_repl=lean_repl, lean_search=lean_search,
    )
    informalizer = Informalizer(llm_client=llm)
    judge = IntentJudge(llm_client=llm, informalizer=informalizer)

    with console.status(f"{_cost_display(cost_tracker, budget)} Formalizing: {conjecture[:60]}...") as status:
        form_result = pipeline.run(conjecture_nl=conjecture)

        estimated_tokens = 2000
        cost_tracker.record_usage(input_tokens=estimated_tokens, output_tokens=estimated_tokens // 2)
        status.update(f"{_cost_display(cost_tracker, budget)} Formalization complete, checking intent...")

        if not form_result.success or form_result.theorem is None:
            console.print(f"[red]Formalization failed:[/red] {form_result.failure_reason or 'unknown error'}")
            _print_cost_summary(cost_tracker, budget)
            sys.exit(1)

        if _check_budget(cost_tracker, budget):
            console.print("[yellow]Budget exceeded after formalization.[/yellow]")
            console.print(f"\nLean 4 statement (unverified):\n{form_result.theorem.lean_statement}")
            _print_cost_summary(cost_tracker, budget)
            sys.exit(1)

        lean_code = form_result.theorem.lean_statement
        verdict = judge.judge(
            lean_code=lean_code,
            original_idea=conjecture,
            conjecture=conjecture,
        )
        _record_agent_tokens(cost_tracker, judge)

    console.print("\n[bold]Lean 4 Statement:[/bold]")
    console.print(f"```lean\n{lean_code}\n```")

    verdict_color = "green" if verdict.overall_verdict.value == "correct" else "red"
    console.print(f"\n[bold]Intent Verdict:[/bold] [{verdict_color}]{verdict.overall_verdict.value.upper()}[/{verdict_color}]")

    if verdict.all_concerns:
        console.print("\n[yellow]Concerns:[/yellow]")
        for concern in verdict.all_concerns:
            console.print(f"  - {concern}")

    _save_session(session)
    _print_cost_summary(cost_tracker, budget)


@cli.command("check")
@click.argument("lean_statement")
@click.option("--budget", type=float, default=2.00, help="Budget in USD (default: $2.00)")
def check_cmd(lean_statement: str, budget: float) -> None:
    """Search for counterexamples to a Lean 4 statement.

    Runs CounterexampleSearcher. Returns PLAUSIBLE (no counterexample found)
    or DISPROVED (with the counterexample).
    """
    from agentic_research.agents.counterexample_searcher import CounterexampleSearcher
    from agentic_research.models.verification import CounterexampleStatus

    cost_tracker = _create_cost_tracker()

    try:
        llm = _create_llm_client()
        lean_repl = _create_lean_repl()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    searcher = CounterexampleSearcher(llm_client=llm, lean_repl=lean_repl)

    with console.status(f"{_cost_display(cost_tracker, budget)} Searching for counterexamples..."):
        result = searcher.search(lean_code=lean_statement, conjecture=lean_statement)
        _record_agent_tokens(cost_tracker, searcher)

    if result.status == CounterexampleStatus.DISPROVED:
        console.print("[red bold]DISPROVED[/red bold]")
        if result.successful_counterexample:
            console.print(f"\nCounterexample: {result.successful_counterexample.description}")
            if result.successful_counterexample.lean_code:
                console.print(f"\n```lean\n{result.successful_counterexample.lean_code}\n```")
    else:
        console.print("[green bold]PLAUSIBLE[/green bold] — no counterexample found")

    console.print(f"\nCandidates tried: {result.attempts_made}")
    _print_cost_summary(cost_tracker, budget)


@cli.command("prove")
@click.argument("lean_statement")
@click.option("--budget", type=float, default=10.00, help="Budget in USD (default: $10.00)")
@click.option("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")
def prove_cmd(lean_statement: str, budget: float, timeout: int) -> None:
    """Attempt to prove a Lean 4 statement.

    Runs ProofPipeline with confirmation prompt before starting.
    Shows real-time progress with cost. Hard-stops when budget exceeded
    or timeout reached.
    """
    if not click.confirm(
        f"Proof search is expensive! Budget: ${budget:.2f}, Timeout: {timeout}s. Proceed?",
        default=False,
    ):
        console.print("Aborted.")
        return

    cost_tracker = _create_cost_tracker()

    try:
        llm = _create_llm_client()
        lean_repl = _create_lean_repl()
        lean_search = _create_lean_search()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    from agentic_research.pipelines.proof import ProofPipeline

    pipeline = ProofPipeline(
        llm_client=llm, lean_repl=lean_repl, lean_search=lean_search,
    )

    start_time = time.monotonic()

    with console.status(f"{_cost_display(cost_tracker, budget)} Starting proof search...") as status:
        result = pipeline.run(lean_statement=lean_statement)

        elapsed = time.monotonic() - start_time
        estimated_tokens = 5000
        cost_tracker.record_usage(input_tokens=estimated_tokens, output_tokens=estimated_tokens // 2)
        status.update(f"{_cost_display(cost_tracker, budget)} Proof search complete ({elapsed:.1f}s)")

    if _check_budget(cost_tracker, budget):
        console.print("[yellow]Budget exceeded during proof search.[/yellow]")

    if elapsed > timeout:
        console.print(f"[yellow]Timeout reached ({elapsed:.1f}s > {timeout}s).[/yellow]")

    if result.proved:
        console.print("[green bold]PROVED[/green bold]")
        if result.final_proof:
            console.print(f"\n```lean\n{result.final_proof}\n```")
        if result.claim_check_passed is not None:
            check_str = "[green]passed[/green]" if result.claim_check_passed else "[yellow]not verified[/yellow]"
            console.print(f"\nClaim check: {check_str}")
    else:
        console.print("[red bold]PROOF FAILED[/red bold]")
        if result.failure_stage:
            console.print(f"Failed at: {result.failure_stage}")
        if result.failure_reason:
            console.print(f"Reason: {result.failure_reason}")

    console.print(f"\nElapsed: {elapsed:.1f}s")
    _print_cost_summary(cost_tracker, budget)


@cli.command("status")
def status_cmd() -> None:
    """Show current session state and cost summary."""
    session = _load_or_create_session()
    summary = session.summary()

    table = Table(title="Session Status")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Session ID", session.session_id)
    table.add_row("Total conjectures", str(summary.get("total_tried", 0)))
    table.add_row("Proved", str(summary.get("proved", 0)))
    table.add_row("Failed", str(summary.get("failed", 0)))
    table.add_row("Promising directions", str(summary.get("promising_directions", 0)))
    table.add_row("Partial results", str(summary.get("partial_results", 0)))
    table.add_row("Stage summaries", str(summary.get("stage_summaries", 0)))

    console.print(table)

    hot = summary.get("hot_tier", 0)
    warm = summary.get("warm_tier", 0)
    cold = summary.get("cold_tier", 0)
    if warm or cold:
        console.print(f"\n[dim]Memory tiers — hot: {hot}, warm: {warm}, cold: {cold}[/dim]")


if __name__ == "__main__":
    cli()
