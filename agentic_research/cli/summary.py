"""Terminal run summary for research, prove, and formalize commands."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from agentic_research.models.session import ResearchSessionResult


def format_run_summary(
    result: ResearchSessionResult,
    console: Console,
    *,
    elapsed_seconds: float | None = None,
) -> None:
    """Print a structured end-of-run summary to the terminal."""
    cost = result.cost_estimate.total_cost_usd
    n_proved = len(result.proved_conjectures)

    elapsed_str = f" in {elapsed_seconds:.1f}s" if elapsed_seconds is not None else ""

    if n_proved > 0:
        status = (
            f"[green bold]PROVED {n_proved} conjecture{'s' if n_proved != 1 else ''}"
            f"{elapsed_str} (${cost:.2f})[/green bold]"
        )
    else:
        stage_name = result.final_stage.value.upper()
        status = (
            f"[red bold]FAILED — reached {stage_name} stage"
            f"{elapsed_str} (${cost:.2f})[/red bold]"
        )

    console.print(f"\n{status}")

    for tc in result.proved_conjectures:
        statement = tc.conjecture.statement
        lean_code = tc.proof_code or tc.lean_statement
        if lean_code:
            syntax = Syntax(lean_code, "lean4", theme="monokai")
            console.print(Panel(
                f"[bold]{statement}[/bold]",
                title="Proved",
                border_style="green",
            ))
            console.print(syntax)
        else:
            console.print(Panel(
                f"[bold]{statement}[/bold]",
                title="Proved",
                border_style="green",
            ))

    if result.failed_conjectures:
        fail_table = Table(title="Failed Conjectures")
        fail_table.add_column("Conjecture", min_width=30)
        fail_table.add_column("Stage Reached", width=20)
        fail_table.add_column("Failure Reason", min_width=20)
        for tc in result.failed_conjectures:
            stmt = tc.conjecture.statement
            if len(stmt) > 60:
                stmt = stmt[:57] + "..."
            reason = tc.failure_reason or "unknown"
            if len(reason) > 60:
                reason = reason[:57] + "..."
            fail_table.add_row(stmt, tc.stage_reached.value, reason)
        console.print(fail_table)

    cost_table = Table(title="Cost Breakdown")
    cost_table.add_column("Metric", style="bold")
    cost_table.add_column("Value", justify="right")
    cost_table.add_row("Input tokens", f"{result.total_token_usage.input_tokens:,}")
    cost_table.add_row("Output tokens", f"{result.total_token_usage.output_tokens:,}")
    cost_table.add_row("Cache read tokens", f"{result.total_token_usage.cache_read_input_tokens:,}")
    cost_table.add_row("Cache write tokens", f"{result.total_token_usage.cache_creation_input_tokens:,}")
    cost_table.add_row("Total cost", f"${cost:.4f}")
    console.print(cost_table)
