"""Render trajectory JSON files as GitHub-flavored markdown.

Usage: python -m agentic_research.eval.trajectory_renderer <trajectory-dir> --output-dir <dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agentic_research.models.eval import Trajectory


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _fmt_cost(cost: float) -> str:
    return f"${cost:.2f}"


def _render_stages_table(trajectory: Trajectory) -> str:
    if not trajectory.stage_timings and not trajectory.events:
        return ""
    lines = [
        "<details>",
        "<summary>Pipeline stages</summary>",
        "",
        "| Stage | Duration |",
        "|-------|----------|",
    ]
    for stage, duration in trajectory.stage_timings.items():
        lines.append(f"| {stage} | {_fmt_duration(duration)} |")
    if not trajectory.stage_timings and trajectory.events:
        for event in trajectory.events:
            lines.append(f"| {event.stage} | {_fmt_duration(event.timestamp_s)} |")
    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def _render_nl_sketch(trajectory: Trajectory) -> str:
    pr = trajectory.pipeline_result
    if not pr:
        return ""
    nl_sketch = getattr(pr, "nl_proof_sketch", None)
    if nl_sketch is None and pr.search_result:
        return ""
    if not pr.recursive_result:
        return ""
    return ""


def _render_lemma_tree(trajectory: Trajectory) -> str:
    pr = trajectory.pipeline_result
    if not pr or not pr.recursive_result or not pr.recursive_result.lemma_tree:
        return ""
    tree = pr.recursive_result.lemma_tree
    lines = [
        "<details>",
        "<summary>Lemma tree</summary>",
        "",
    ]

    def _render_node(node_id: str, depth: int = 0) -> None:
        node = tree.get_node(node_id)
        if not node:
            return
        indent = "  " * depth
        status = node.status.value.upper()
        desc = node.statement_nl or node.node_id
        lines.append(f"{indent}- **{node.node_id}** [{status}]: {desc}")
        for child_id in node.children:
            _render_node(child_id, depth + 1)

    _render_node(tree.root_id)
    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def _render_failure_analysis(trajectory: Trajectory) -> str:
    pr = trajectory.pipeline_result
    if not pr or pr.proved:
        return ""
    lines = [
        "<details>",
        "<summary>Failure analysis</summary>",
        "",
    ]
    if pr.failure_stage:
        lines.append(f"- **Failure stage:** {pr.failure_stage}")
    if pr.failure_reason:
        lines.append(f"- **Reason:** {pr.failure_reason}")
    if pr.backtrack_stages:
        lines.append(f"- **Backtrack stages:** {', '.join(pr.backtrack_stages)}")
    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def _render_proof(trajectory: Trajectory) -> str:
    pr = trajectory.pipeline_result
    if not pr or not pr.proved or not pr.final_proof:
        return ""
    lines = [
        "<details>",
        "<summary>Final Lean 4 proof</summary>",
        "",
        "```lean",
        pr.final_proof,
        "```",
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def _render_cost_breakdown(trajectory: Trajectory) -> str:
    r = trajectory.problem_result
    lines = [
        "## Cost breakdown",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total cost | {_fmt_cost(r.cost_usd)} |",
        f"| Input tokens | {r.input_tokens:,} |",
        f"| Output tokens | {r.output_tokens:,} |",
        f"| Cache read tokens | {r.cache_read_input_tokens:,} |",
        f"| Cache creation tokens | {r.cache_creation_input_tokens:,} |",
        "",
    ]
    return "\n".join(lines)


def _render_reproducibility(trajectory: Trajectory) -> str:
    cfg = trajectory.config
    p = trajectory.problem
    lines = [
        "## Reproducibility",
        "",
        f"- **Model:** {trajectory.model}",
    ]
    for k, v in cfg.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    lines.append("```bash")

    benchmark = p.source.value
    cmd_parts = [
        "python -m agentic_research.eval.runner",
        "  --mode proof_discovery",
        f"  --benchmark {benchmark}",
        f"  --problem-filter {p.name}",
    ]
    if cfg.get("extended_thinking"):
        cmd_parts.append("  --extended-thinking")
    if cfg.get("thinking_budget"):
        cmd_parts.append(f"  --thinking-budget {cfg['thinking_budget']}")
    if cfg.get("timeout"):
        cmd_parts.append(f"  --timeout {cfg['timeout']}")
    if cfg.get("seed") is not None:
        cmd_parts.append(f"  --seed {cfg['seed']}")
    if cfg.get("max_critic_retries") is not None:
        cmd_parts.append(f"  --max-critic-retries {cfg['max_critic_retries']}")

    lines.append(" \\\n".join(cmd_parts))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_trajectory(trajectory: Trajectory) -> str:
    """Render a single trajectory as GitHub-flavored markdown."""
    p = trajectory.problem
    r = trajectory.problem_result
    result_label = r.result.value.upper()

    sections = [
        f"# {p.name}",
        "",
        f"**Result:** {result_label} | "
        f"**Duration:** {_fmt_duration(r.duration_seconds)} | "
        f"**Cost:** {_fmt_cost(r.cost_usd)}",
        "",
        "## Problem statement",
        "",
    ]

    if p.natural_language:
        sections.append(p.natural_language)
        sections.append("")

    sections.extend([
        "```lean",
        p.lean_statement,
        "```",
        "",
    ])

    stages = _render_stages_table(trajectory)
    if stages:
        sections.append(stages)

    tree = _render_lemma_tree(trajectory)
    if tree:
        sections.append(tree)

    proof = _render_proof(trajectory)
    if proof:
        sections.append(proof)

    failure = _render_failure_analysis(trajectory)
    if failure:
        sections.append(failure)

    sections.append(_render_cost_breakdown(trajectory))
    sections.append(_render_reproducibility(trajectory))

    return "\n".join(sections)


def render_index(trajectories: list[tuple[str, Trajectory]]) -> str:
    """Render an index.md listing all trajectories."""
    lines = [
        "# Trajectory Index",
        "",
        "| Problem | Result | Duration | Cost | File |",
        "|---------|--------|----------|------|------|",
    ]
    for filename, traj in trajectories:
        r = traj.problem_result
        md_name = filename.replace(".json", ".md")
        lines.append(
            f"| {traj.problem.name} "
            f"| {r.result.value.upper()} "
            f"| {_fmt_duration(r.duration_seconds)} "
            f"| {_fmt_cost(r.cost_usd)} "
            f"| [{md_name}]({md_name}) |"
        )
    lines.append("")
    return "\n".join(lines)


def render_directory(input_dir: Path, output_dir: Path) -> None:
    """Read trajectory JSON files from input_dir, write markdown to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No trajectory JSON files found in {input_dir}", file=sys.stderr)
        return

    trajectories: list[tuple[str, Trajectory]] = []
    for json_file in json_files:
        data = json.loads(json_file.read_text())
        traj = Trajectory.model_validate(data)
        trajectories.append((json_file.name, traj))

        md_content = render_trajectory(traj)
        md_path = output_dir / json_file.with_suffix(".md").name
        md_path.write_text(md_content)
        print(f"  {md_path}")

    index_content = render_index(trajectories)
    index_path = output_dir / "index.md"
    index_path.write_text(index_content)
    print(f"  {index_path}")
    print(f"Rendered {len(trajectories)} trajectories")


def main() -> None:
    """CLI entry point."""
    import click

    @click.command()
    @click.argument("input_dir", type=click.Path(exists=True))
    @click.option("--output-dir", type=click.Path(), required=True, help="Output directory for markdown files")
    def render(input_dir: str, output_dir: str) -> None:
        """Render trajectory JSON files as GitHub-flavored markdown."""
        render_directory(Path(input_dir), Path(output_dir))

    render()


if __name__ == "__main__":
    main()
