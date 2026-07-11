"""CLI entry point for the Agentic Research Partner.

Five decoupled commands: explore, formalize, check, prove, status.
Each enforces a per-command budget via CostTracker and displays
real-time cost with rich.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from agentic_research import __version__
from agentic_research.logging import configure_logging

if TYPE_CHECKING:
    from agentic_research.orchestrator.cost_tracker import CostTracker

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


def _create_llm_client(model: str | None = None):
    from agentic_research.agents.llm_client import LLMClient

    if model is not None:
        return LLMClient(model=model)
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


def _check_budget(cost_tracker: CostTracker, budget: float) -> bool:
    """Return True if budget is exceeded."""
    return cost_tracker.total_cost() > budget


def _validate_positive(ctx, param, value):
    if value <= 0:
        raise click.BadParameter(f"must be greater than 0, got {value}")
    return value


def _validate_non_negative(ctx, param, value):
    if value < 0:
        raise click.BadParameter(f"must be non-negative, got {value}")
    return value


def _cost_display(cost_tracker: CostTracker, budget: float) -> str:
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


def _warn_if_lean_missing() -> None:
    if shutil.which("lean") is None:
        click.echo(
            "Warning: Lean 4 not found on PATH. "
            "Proof checking will fall back to mock mode. "
            "Install Lean 4 via https://leanprover.github.io/lean4/doc/setup.html"
        )


def _print_cost_summary(cost_tracker: CostTracker, budget: float) -> None:
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
@click.option("--model", default=None, envvar="AGENTIC_RESEARCH_MODEL", help="LLM model to use (default: claude-opus-4-6-20250616)")
@click.pass_context
def cli(ctx: click.Context, json_logs: bool, log_level: str, model: str | None) -> None:
    """Agentic Mathematical Research Partner."""
    configure_logging(json_output=json_logs, level=log_level)
    ctx.ensure_object(dict)
    ctx.obj["model"] = model


@cli.command("eval")
@click.argument("benchmark", type=click.Choice(["miniF2F", "PutnamBench"]))
@click.option("--mode", type=click.Choice(["proof_discovery", "conjecture_quality", "end_to_end"]), default="proof_discovery")
@click.option("--split", type=click.Choice(["test", "valid"]), default="valid")
@click.option("--pass-k", type=int, default=1, help="Number of attempts per problem (pass@k)")
@click.option("--sample-size", type=int, default=None, help="Evaluate a subset of problems")
@click.option("--seed", type=int, default=0, help="Random seed for sampling")
@click.option("--data-dir", type=click.Path(), default="data/benchmarks")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write JSON report to file")
@click.pass_context
def eval_cmd(
    ctx: click.Context,
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
@click.pass_context
def explore_cmd(ctx: click.Context, idea: str, budget: float) -> None:
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
        llm = _create_llm_client(model=ctx.obj.get("model"))
        lean_search = _create_lean_search()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    explorer = ExplorationAgent(llm_client=llm, lean_search=lean_search)
    conjecture_gen = ConjectureGenerator(llm_client=llm)

    with console.status(f"{_cost_display(cost_tracker, budget)} Exploring: {idea[:60]}...") as status:
        agent_ctx = AgentContext(task=idea)
        explore_result = explorer.run(agent_ctx)
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
@click.option("--artifact-dir", type=click.Path(), default=None, help="Directory to save theorem artifacts")
@click.pass_context
def formalize_cmd(ctx: click.Context, conjecture: str, budget: float, artifact_dir: str | None) -> None:
    """Formalize a natural language conjecture into Lean 4.

    Runs FormalizationPipeline + IntentJudge. Takes a natural language
    conjecture, produces a Lean 4 statement with type-first formalization,
    and runs intent verification.
    """
    _warn_if_lean_missing()

    from agentic_research.agents.informalizer import Informalizer
    from agentic_research.agents.intent_judge import IntentJudge

    cost_tracker = _create_cost_tracker()
    session = _load_or_create_session()

    try:
        llm = _create_llm_client(model=ctx.obj.get("model"))
        lean_repl = _create_lean_repl()
        lean_search = _create_lean_search()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    from agentic_research.pipelines.formalization import FormalizationPipeline

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )
    informalizer = Informalizer(llm_client=llm)
    judge = IntentJudge(llm_client=llm, informalizer=informalizer)

    with progress:
        task_id = progress.add_task(f"Formalizing: {conjecture[:60]}...", total=None)

        def on_formalize_progress(stage: str, message: str) -> None:
            progress.update(task_id, description=f"[bold]{stage}[/bold] — {message}")

        pipeline = FormalizationPipeline(
            llm_client=llm, lean_repl=lean_repl, lean_search=lean_search,
            artifact_dir=Path(artifact_dir) if artifact_dir else None,
            progress_callback=on_formalize_progress,
        )
        form_result = pipeline.run(conjecture_nl=conjecture)

        tokens = form_result.total_token_usage
        cost_tracker.record_usage(
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            cache_read_tokens=tokens.cache_read_input_tokens,
            cache_write_tokens=tokens.cache_creation_input_tokens,
        )
        progress.update(task_id, description=f"{_cost_display(cost_tracker, budget)} Checking intent...")

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
@click.pass_context
def check_cmd(ctx: click.Context, lean_statement: str, budget: float) -> None:
    """Search for counterexamples to a Lean 4 statement.

    Runs CounterexampleSearcher. Returns PLAUSIBLE (no counterexample found)
    or DISPROVED (with the counterexample).
    """
    from agentic_research.agents.counterexample_searcher import CounterexampleSearcher
    from agentic_research.models.verification import CounterexampleStatus

    cost_tracker = _create_cost_tracker()

    try:
        llm = _create_llm_client(model=ctx.obj.get("model"))
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
@click.option(
    "--backend",
    type=click.Choice(["builtin", "leanstral"]),
    default="builtin",
    help="Proof backend to use (default: builtin)",
)
@click.option("--use-critic/--no-critic", default=True, help="Enable ProofCritic for lemma decomposition review (default: enabled)")
@click.option("--use-detailer/--no-detailer", default=True, help="Enable ProofDetailer for proof sketch enrichment (default: enabled)")
@click.pass_context
def prove_cmd(ctx: click.Context, lean_statement: str, budget: float, timeout: int, backend: str, use_critic: bool, use_detailer: bool) -> None:
    """Attempt to prove a Lean 4 statement.

    Runs ProofPipeline with confirmation prompt before starting.
    Shows real-time progress with cost. Hard-stops when budget exceeded
    or timeout reached.
    """
    _warn_if_lean_missing()

    import os

    use_external = backend == "leanstral"
    external_config = None

    if use_external:
        api_url = os.environ.get("LEANSTRAL_API_URL")
        if not api_url:
            console.print("[red]Error:[/red] LEANSTRAL_API_URL environment variable is required for the leanstral backend")
            sys.exit(1)

        from agentic_research.models.external_prover import ExternalProverConfig

        external_config = ExternalProverConfig(
            api_url=api_url,
            api_key=os.environ.get("LEANSTRAL_API_KEY"),
        )

    console.print(f"[dim]Backend: {backend}[/dim]")

    if not click.confirm(
        f"Proof search is expensive! Budget: ${budget:.2f}, Timeout: {timeout}s. Proceed?",
        default=False,
    ):
        console.print("Aborted.")
        return

    cost_tracker = _create_cost_tracker()

    try:
        llm = _create_llm_client(model=ctx.obj.get("model"))
        lean_repl = _create_lean_repl()
        lean_search = _create_lean_search()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    from agentic_research.pipelines.proof import ProofPipeline

    start_time = time.monotonic()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task("Starting proof search...", total=None)

        def on_prove_progress(stage: str, message: str) -> None:
            progress.update(task_id, description=f"[bold]{stage}[/bold] — {message}")

        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=lean_repl,
            lean_search=lean_search,
            use_external_prover=use_external,
            external_prover_config=external_config,
            use_proof_critic=use_critic,
            use_proof_detailer=use_detailer,
            progress_callback=on_prove_progress,
        )
        result = pipeline.run(lean_statement=lean_statement, statement_nl=lean_statement)

        elapsed = time.monotonic() - start_time
        tokens = result.total_token_usage
        cost_tracker.record_usage(
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            cache_read_tokens=tokens.cache_read_input_tokens,
            cache_write_tokens=tokens.cache_creation_input_tokens,
        )
        progress.update(task_id, description=f"{_cost_display(cost_tracker, budget)} Proof search complete ({elapsed:.1f}s)")

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


@cli.command("research")
@click.argument("idea")
@click.option("--budget", type=float, default=20.00, callback=_validate_positive, help="Budget in USD (default: $20.00)")
@click.option("--max-conjectures", type=int, default=5, callback=_validate_positive, help="Max conjectures to evaluate (default: 5)")
@click.option("--max-refinements", type=int, default=3, callback=_validate_non_negative, help="Max refinement attempts per conjecture (default: 3)")
@click.option("--use-critic/--no-critic", default=True, help="Enable ProofCritic for lemma decomposition review (default: enabled)")
@click.option("--use-detailer/--no-detailer", default=True, help="Enable ProofDetailer for proof sketch enrichment (default: enabled)")
@click.pass_context
def research_cmd(ctx: click.Context, idea: str, budget: float, max_conjectures: int, max_refinements: int, use_critic: bool, use_detailer: bool) -> None:
    """Run the full explore-conjecture-prove research loop.

    Automatically explores the idea, generates conjectures, formalizes them
    into Lean 4, checks for counterexamples, attempts proofs, and refines
    on failure. Creates checkpoints at each stage for resumability.
    """
    _warn_if_lean_missing()

    from agentic_research.models.session import (
        OrchestratorConfig,
        PipelineStage,
    )
    from agentic_research.orchestrator.engine import ResearchOrchestrator

    if not click.confirm(
        f"Full research loop. Budget: ${budget:.2f}. Continue?",
        default=False,
    ):
        console.print("Aborted.")
        return

    try:
        llm = _create_llm_client(model=ctx.obj.get("model"))
        lean_repl = _create_lean_repl()
        lean_search = _create_lean_search()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    config = OrchestratorConfig(
        budget_limit_usd=budget,
        max_conjectures=max_conjectures,
        max_refinements=max_refinements,
        use_proof_critic=use_critic,
        use_proof_detailer=use_detailer,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task(f"Researching: {idea[:60]}...", total=None)

        def on_research_progress(stage: str, message: str) -> None:
            progress.update(task_id, description=f"[bold]{stage}[/bold] — {message}")

        orchestrator = ResearchOrchestrator(
            llm_client=llm,
            lean_repl=lean_repl,
            lean_search=lean_search,
            config=config,
            progress_callback=on_research_progress,
        )

        try:
            result = orchestrator.run(idea)
            progress.update(task_id, description="[bold]Research complete.[/bold]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            result = orchestrator._build_result(idea)

    if result.final_stage == PipelineStage.COMPLETE:
        console.print("\n[green bold]RESEARCH COMPLETE[/green bold]")
    else:
        console.print("\n[yellow bold]RESEARCH INCOMPLETE[/yellow bold]")

    stats_table = Table(title="Research Results", show_header=False)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value")
    stats_table.add_row("Stage reached", result.final_stage.value)
    stats_table.add_row("Conjectures tried", str(result.total_conjectures_tried))
    stats_table.add_row("Proofs attempted", str(result.total_refinements + len(result.proved_conjectures) + len(result.failed_conjectures)))
    stats_table.add_row("Proofs succeeded", str(len(result.proved_conjectures)))
    stats_table.add_row("Total cost", f"${result.cost_estimate.total_cost_usd:.4f}")
    console.print(stats_table)

    if result.proved_conjectures:
        proved_table = Table(title="Proved Conjectures")
        proved_table.add_column("Conjecture", style="cyan")
        proved_table.add_column("Lean Statement", style="green")
        for tc in result.proved_conjectures:
            stmt = tc.conjecture.statement
            if len(stmt) > 80:
                stmt = stmt[:77] + "..."
            lean = tc.lean_statement
            if len(lean) > 80:
                lean = lean[:77] + "..."
            proved_table.add_row(stmt, lean)
        console.print(proved_table)

    console.print(f"\n[dim]Session: {result.session_id}[/dim]")
    console.print(f"[dim]Resume with: agentic-research resume {result.session_id}[/dim]")


@cli.command("resume")
@click.argument("session_id", required=False, default=None)
@click.option("--list", "list_sessions", is_flag=True, help="List available sessions with their stages")
@click.option("--budget", type=float, default=20.00, help="Budget in USD (default: $20.00)")
@click.option("--use-critic/--no-critic", default=True, help="Enable ProofCritic for lemma decomposition review (default: enabled)")
@click.option("--use-detailer/--no-detailer", default=True, help="Enable ProofDetailer for proof sketch enrichment (default: enabled)")
@click.pass_context
def resume_cmd(
    ctx: click.Context,
    session_id: str | None,
    list_sessions: bool,
    budget: float,
    use_critic: bool,
    use_detailer: bool,
) -> None:
    """Resume a previously interrupted research session.

    Loads the last checkpoint for a session and continues from the last
    completed stage. Use --list to show available sessions.
    """
    from agentic_research.orchestrator.rollback import CheckpointManager, DEFAULT_CHECKPOINT_DIR

    if list_sessions:
        if not DEFAULT_CHECKPOINT_DIR.exists():
            console.print("[yellow]No sessions found.[/yellow]")
            return

        session_dirs = sorted(
            d for d in DEFAULT_CHECKPOINT_DIR.iterdir() if d.is_dir()
        )
        if not session_dirs:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        table = Table(title="Available Sessions")
        table.add_column("Session ID", style="bold")
        table.add_column("Checkpoints", justify="right")
        table.add_column("Last Stage")

        for session_dir in session_dirs:
            mgr = CheckpointManager(session_id=session_dir.name, persist=False)
            mgr._session_id = session_dir.name
            checkpoint_ids = mgr.list_disk_checkpoints()
            last_stage = "unknown"
            if checkpoint_ids:
                last_ckpt = mgr.latest_disk_checkpoint()
                if last_ckpt:
                    last_stage = last_ckpt.stage.value
            table.add_row(session_dir.name, str(len(checkpoint_ids)), last_stage)

        console.print(table)
        return

    if session_id is None:
        console.print("[red]Error:[/red] Please provide a session ID or use --list to see available sessions.")
        raise SystemExit(1)

    mgr = CheckpointManager(session_id=session_id, persist=True)
    checkpoint = mgr.latest_disk_checkpoint()
    if checkpoint is None:
        console.print(f"[red]Error:[/red] No checkpoints found for session '{session_id}'.")
        raise SystemExit(1)

    console.print(f"[bold]Resuming session:[/bold] {session_id}")
    console.print(f"[dim]Checkpoint: {checkpoint.checkpoint_id} | Stage: {checkpoint.stage.value}[/dim]")

    from agentic_research.models.session import (
        OrchestratorConfig,
        PipelineStage,
    )
    from agentic_research.orchestrator.engine import ResearchOrchestrator

    try:
        llm = _create_llm_client(model=ctx.obj.get("model"))
        lean_repl = _create_lean_repl()
        lean_search = _create_lean_search()
    except Exception as e:
        console.print(f"[red]Setup error:[/red] {e}")
        sys.exit(1)

    config = OrchestratorConfig(
        budget_limit_usd=budget,
        use_proof_critic=use_critic,
        use_proof_detailer=use_detailer,
    )

    orchestrator = ResearchOrchestrator(
        llm_client=llm,
        lean_repl=lean_repl,
        lean_search=lean_search,
        config=config,
        session_id=session_id,
    )

    try:
        with console.status(f"[bold]Resuming from {checkpoint.stage.value}...[/bold]") as status:
            result = orchestrator.resume_from_checkpoint(checkpoint.checkpoint_id)
            status.update("[bold]Research complete.[/bold]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        result = orchestrator._build_result(checkpoint.session_state.raw_idea)

    if result.final_stage == PipelineStage.COMPLETE:
        console.print("\n[green bold]RESEARCH COMPLETE[/green bold]")
    else:
        console.print("\n[yellow bold]RESEARCH INCOMPLETE[/yellow bold]")

    stats_table = Table(title="Research Results", show_header=False)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value")
    stats_table.add_row("Stage reached", result.final_stage.value)
    stats_table.add_row("Conjectures tried", str(result.total_conjectures_tried))
    stats_table.add_row("Proofs succeeded", str(len(result.proved_conjectures)))
    stats_table.add_row("Total cost", f"${result.cost_estimate.total_cost_usd:.4f}")
    console.print(stats_table)

    console.print(f"\n[dim]Session: {result.session_id}[/dim]")


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
