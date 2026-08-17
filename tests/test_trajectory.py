"""Tests for trajectory capture infrastructure."""

import json
import textwrap
from pathlib import Path

from agentic_research.eval.benchmarks import _extract_docstring, _parse_lean4_file
from agentic_research.eval.trajectory_renderer import (
    render_index,
    render_trajectory,
)
from agentic_research.models.agents import TokenUsage
from agentic_research.models.eval import (
    BenchmarkSource,
    EvalConfig,
    EvalMode,
    Problem,
    ProblemDifficulty,
    ProblemResult,
    ProblemSplit,
    ProofResult,
    Trajectory,
    TrajectoryEvent,
)
from agentic_research.models.proof import ProofPipelineResult


def _make_problem(**kwargs) -> Problem:
    defaults = dict(
        id="miniF2F/test_thm",
        name="test_thm",
        source=BenchmarkSource.MINIF2F,
        split=ProblemSplit.VALIDATION,
        difficulty=ProblemDifficulty.AMC,
        lean_header="import Mathlib",
        lean_statement="theorem test_thm : 1 + 1 = 2 := by sorry",
        natural_language="Prove that 1 + 1 = 2.",
    )
    defaults.update(kwargs)
    return Problem(**defaults)


def _make_result(**kwargs) -> ProblemResult:
    defaults = dict(
        problem_id="miniF2F/test_thm",
        mode=EvalMode.PROOF_DISCOVERY,
        result=ProofResult.SUCCESS,
        proof="theorem test_thm : 1 + 1 = 2 := by norm_num",
        attempts=1,
        duration_seconds=42.5,
        token_usage=1000,
        cost_usd=0.15,
        input_tokens=800,
        output_tokens=200,
    )
    defaults.update(kwargs)
    return ProblemResult(**defaults)


class TestTrajectoryEventModel:
    def test_create_event(self):
        event = TrajectoryEvent(
            timestamp_s=1.5,
            stage="Proof Search",
            event_type="enter",
            detail="Starting proof search",
        )
        assert event.timestamp_s == 1.5
        assert event.stage == "Proof Search"
        assert event.token_usage is None

    def test_event_with_token_usage(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        event = TrajectoryEvent(
            timestamp_s=10.0,
            stage="complete",
            event_type="exit",
            token_usage=usage,
        )
        assert event.token_usage is not None
        assert event.token_usage.input_tokens == 100

    def test_event_serialization_roundtrip(self):
        event = TrajectoryEvent(
            timestamp_s=5.0,
            stage="NL Proof",
            event_type="enter",
            detail="Generating sketch",
        )
        data = json.loads(event.model_dump_json())
        restored = TrajectoryEvent.model_validate(data)
        assert restored.stage == "NL Proof"


class TestTrajectoryModel:
    def test_create_trajectory(self):
        traj = Trajectory(
            problem=_make_problem(),
            problem_result=_make_result(),
            model="claude-opus-4-6",
            config={"timeout": 600, "extended_thinking": True},
        )
        assert traj.problem.name == "test_thm"
        assert traj.problem_result.result == ProofResult.SUCCESS
        assert traj.model == "claude-opus-4-6"

    def test_trajectory_with_pipeline_result(self):
        pr = ProofPipelineResult(
            statement="theorem test : True := by sorry",
            proved=True,
            final_proof="theorem test : True := by trivial",
            total_token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        traj = Trajectory(
            problem=_make_problem(),
            problem_result=_make_result(),
            pipeline_result=pr,
            model="claude-opus-4-6",
        )
        assert traj.pipeline_result is not None
        assert traj.pipeline_result.proved is True

    def test_trajectory_serialization_roundtrip(self):
        traj = Trajectory(
            problem=_make_problem(),
            problem_result=_make_result(),
            events=[
                TrajectoryEvent(timestamp_s=0.0, stage="start", event_type="enter"),
                TrajectoryEvent(timestamp_s=42.5, stage="complete", event_type="exit"),
            ],
            stage_timings={"Proof Search": 10.0, "Lemma Breakdown": 15.0},
            model="claude-opus-4-6",
            config={"timeout": 600},
        )
        json_str = traj.model_dump_json(indent=2)
        data = json.loads(json_str)
        restored = Trajectory.model_validate(data)
        assert restored.problem.name == "test_thm"
        assert len(restored.events) == 2
        assert restored.stage_timings["Proof Search"] == 10.0

    def test_trajectory_none_pipeline_result(self):
        traj = Trajectory(
            problem=_make_problem(),
            problem_result=_make_result(result=ProofResult.TIMEOUT),
            pipeline_result=None,
            model="claude-opus-4-6",
        )
        assert traj.pipeline_result is None


class TestDocstringParsing:
    def test_extract_single_line_docstring(self):
        content = '/-- Prove that 1 + 1 = 2. -/\ntheorem foo : 1 + 1 = 2 := by sorry'
        result = _extract_docstring(content, content.index("theorem"))
        assert result == "Prove that 1 + 1 = 2."

    def test_extract_multiline_docstring(self):
        content = textwrap.dedent("""\
            /--
            Let $f$ be a function from $\\mathbb{R}$ to $\\mathbb{R}$.
            Prove that $f$ is continuous.
            -/
            theorem foo : True := by sorry
        """)
        result = _extract_docstring(content, content.index("theorem"))
        assert "Let $f$" in result
        assert "continuous" in result

    def test_no_docstring(self):
        content = "import Mathlib\n\ntheorem foo : True := by sorry"
        result = _extract_docstring(content, content.index("theorem"))
        assert result == ""

    def test_docstring_with_leading_whitespace(self):
        content = '/--   Some text with spaces   -/\ntheorem foo : True := by sorry'
        result = _extract_docstring(content, content.index("theorem"))
        assert result == "Some text with spaces"

    def test_parse_lean4_file_extracts_docstrings(self, tmp_path: Path):
        lean_file = tmp_path / "test.lean"
        lean_file.write_text(textwrap.dedent("""\
            import Mathlib

            /-- Prove that True holds. -/
            theorem test_thm : True := by trivial
        """))
        problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        assert len(problems) == 1
        assert problems[0].natural_language == "Prove that True holds."

    def test_parse_lean4_file_no_docstring_gives_empty(self, tmp_path: Path):
        lean_file = tmp_path / "test.lean"
        lean_file.write_text(textwrap.dedent("""\
            import Mathlib

            theorem test_thm : True := by trivial
        """))
        problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        assert len(problems) == 1
        assert problems[0].natural_language == ""

    def test_parse_lean4_file_multiple_docstrings(self, tmp_path: Path):
        lean_file = tmp_path / "test.lean"
        lean_file.write_text(textwrap.dedent("""\
            import Mathlib

            /-- First problem. -/
            theorem thm_one : True := by trivial

            /-- Second problem. -/
            theorem thm_two : 1 = 1 := by rfl
        """))
        problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        assert len(problems) == 2
        assert problems[0].natural_language == "First problem."
        assert problems[1].natural_language == "Second problem."


class TestHelperDefinitionCapture:
    def test_captures_def_in_header(self, tmp_path: Path):
        lean_file = tmp_path / "test.lean"
        lean_file.write_text(textwrap.dedent("""\
            import Mathlib
            open Nat

            def helper (n : Nat) : Nat := n + 1

            theorem test_thm (n : Nat) : helper n > n := by sorry
        """))
        problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        assert len(problems) == 1
        assert "def helper" in problems[0].lean_header

    def test_captures_set_option_in_header(self, tmp_path: Path):
        lean_file = tmp_path / "test.lean"
        lean_file.write_text(textwrap.dedent("""\
            import Mathlib

            set_option maxHeartbeats 400000

            theorem test_thm : True := by trivial
        """))
        problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        assert len(problems) == 1
        assert "set_option" in problems[0].lean_header

    def test_captures_variable_in_header(self, tmp_path: Path):
        lean_file = tmp_path / "test.lean"
        lean_file.write_text(textwrap.dedent("""\
            import Mathlib

            variable (n : Nat)

            theorem test_thm : n = n := by rfl
        """))
        problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        assert len(problems) == 1
        assert "variable" in problems[0].lean_header

    def test_noncomputable_captured_in_header(self, tmp_path: Path):
        lean_file = tmp_path / "test.lean"
        lean_file.write_text(textwrap.dedent("""\
            import Mathlib

            noncomputable def foo : Nat := 42

            theorem test_thm : foo = 42 := by rfl
        """))
        problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        assert len(problems) == 1
        assert "noncomputable" in problems[0].lean_header

    def test_docstring_before_header_def_skipped(self, tmp_path: Path):
        lean_file = tmp_path / "test.lean"
        lean_file.write_text(textwrap.dedent("""\
            import Mathlib

            /-- A helper function -/
            def helper : Nat := 42

            /-- The main theorem. -/
            theorem test_thm : helper = 42 := by rfl
        """))
        problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        assert len(problems) == 1
        assert "def helper" in problems[0].lean_header
        assert "/--" not in problems[0].lean_header


class TestTrajectoryRenderer:
    def test_render_success_trajectory(self):
        traj = Trajectory(
            problem=_make_problem(),
            problem_result=_make_result(),
            pipeline_result=ProofPipelineResult(
                statement="test",
                proved=True,
                final_proof="theorem test_thm : 1 + 1 = 2 := by norm_num",
                total_token_usage=TokenUsage(),
            ),
            stage_timings={"Proof Search": 10.0, "Finalization": 2.0},
            model="claude-opus-4-6",
            config={"timeout": 600, "seed": 42},
        )
        md = render_trajectory(traj)
        assert "# test_thm" in md
        assert "SUCCESS" in md
        assert "1 + 1 = 2" in md
        assert "Proof Search" in md
        assert "norm_num" in md
        assert "python -m agentic_research.eval.runner" in md

    def test_render_failure_trajectory(self):
        traj = Trajectory(
            problem=_make_problem(),
            problem_result=_make_result(result=ProofResult.FAILURE, proof=None, cost_usd=5.0),
            pipeline_result=ProofPipelineResult(
                statement="test",
                proved=False,
                failure_stage="recursive_prover",
                failure_reason="Could not prove all lemmas",
                backtrack_stages=["nl_proof"],
                total_token_usage=TokenUsage(),
            ),
            model="claude-opus-4-6",
        )
        md = render_trajectory(traj)
        assert "FAILURE" in md
        assert "recursive_prover" in md
        assert "nl_proof" in md

    def test_render_index(self):
        traj1 = Trajectory(
            problem=_make_problem(name="thm_a"),
            problem_result=_make_result(result=ProofResult.SUCCESS),
            model="claude-opus-4-6",
        )
        traj2 = Trajectory(
            problem=_make_problem(name="thm_b"),
            problem_result=_make_result(result=ProofResult.FAILURE),
            model="claude-opus-4-6",
        )
        md = render_index([("thm_a.json", traj1), ("thm_b.json", traj2)])
        assert "thm_a" in md
        assert "thm_b" in md
        assert "SUCCESS" in md
        assert "FAILURE" in md


class TestJSONLPersistence:
    def test_append_and_load(self, tmp_path: Path):
        from agentic_research.eval.runner import _append_jsonl, _load_completed_ids

        jsonl_path = tmp_path / "results.jsonl"
        r1 = _make_result(problem_id="p1")
        r2 = _make_result(problem_id="p2")
        _append_jsonl(jsonl_path, r1)
        _append_jsonl(jsonl_path, r2)

        ids = _load_completed_ids(jsonl_path)
        assert ids == {"p1", "p2"}

    def test_load_empty_file(self, tmp_path: Path):
        from agentic_research.eval.runner import _load_completed_ids

        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")
        ids = _load_completed_ids(jsonl_path)
        assert ids == set()

    def test_load_nonexistent_file(self, tmp_path: Path):
        from agentic_research.eval.runner import _load_completed_ids

        jsonl_path = tmp_path / "nonexistent.jsonl"
        ids = _load_completed_ids(jsonl_path)
        assert ids == set()


class TestThinkingBudgetThreading:
    def test_thinking_budget_threaded_to_prover_config(self):
        config = EvalConfig(
            mode=EvalMode.PROOF_DISCOVERY,
            thinking_budget=50000,
            use_extended_thinking=True,
        )
        from agentic_research.models.agents import ProverConfig
        prover_config = ProverConfig(
            use_extended_thinking=config.use_extended_thinking,
            thinking_budget=config.thinking_budget,
        )
        assert prover_config.thinking_budget == 50000
        assert prover_config.use_extended_thinking is True


class TestTrajectoryDirWriting:
    def test_trajectory_json_written(self, tmp_path: Path):
        traj = Trajectory(
            problem=_make_problem(),
            problem_result=_make_result(),
            model="claude-opus-4-6",
            config={"timeout": 600},
        )
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        safe_id = "miniF2F_test_thm"
        traj_path = traj_dir / f"{safe_id}.json"
        traj_path.write_text(traj.model_dump_json(indent=2))

        assert traj_path.exists()
        data = json.loads(traj_path.read_text())
        restored = Trajectory.model_validate(data)
        assert restored.problem.name == "test_thm"


class TestRendererDirectory:
    def test_render_directory(self, tmp_path: Path):
        from agentic_research.eval.trajectory_renderer import render_directory

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        traj = Trajectory(
            problem=_make_problem(),
            problem_result=_make_result(),
            model="claude-opus-4-6",
            config={"timeout": 600, "seed": 42},
        )
        (input_dir / "test_thm.json").write_text(traj.model_dump_json(indent=2))

        render_directory(input_dir, output_dir)

        assert (output_dir / "test_thm.md").exists()
        assert (output_dir / "index.md").exists()
        md = (output_dir / "test_thm.md").read_text()
        assert "test_thm" in md
