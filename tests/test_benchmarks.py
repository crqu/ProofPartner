"""Tests for benchmark loaders."""

import textwrap
from pathlib import Path

from agentic_research.eval.benchmarks import (
    _extract_statement,
    _infer_difficulty,
    _infer_split,
    _parse_lean4_file,
)
from agentic_research.models.eval import BenchmarkSource, ProblemDifficulty, ProblemSplit


def test_infer_difficulty():
    assert _infer_difficulty("amc_12a_2021_p5") == ProblemDifficulty.AMC
    assert _infer_difficulty("aime_2019_p7") == ProblemDifficulty.AIME
    assert _infer_difficulty("imo_2023_p1") == ProblemDifficulty.IMO
    assert _infer_difficulty("mathd_algebra_100") == ProblemDifficulty.MATHD
    assert _infer_difficulty("some_other") == ProblemDifficulty.UNKNOWN


def test_infer_split():
    assert _infer_split("miniF2F/Test/Foo.lean") == ProblemSplit.TEST
    assert _infer_split("miniF2F/Valid/Foo.lean") == ProblemSplit.VALIDATION
    assert _infer_split("test/foo.lean") == ProblemSplit.TEST


def test_extract_statement_by_sorry():
    stmt = _extract_statement("theorem foo (n : Nat) : n = n := by sorry")
    assert stmt == "theorem foo (n : Nat) : n = n := by sorry"


def test_extract_statement_by_proof():
    stmt = _extract_statement("theorem foo (n : Nat) : n = n := by\n  rfl")
    assert stmt == "theorem foo (n : Nat) : n = n := by sorry"


def test_extract_statement_term_proof():
    stmt = _extract_statement("theorem foo : True := trivial")
    assert stmt == "theorem foo : True := sorry"


def test_parse_lean4_file(tmp_path: Path):
    lean_file = tmp_path / "Test.lean"
    lean_file.write_text(textwrap.dedent("""\
        import Mathlib

        open Nat

        theorem amc_2021_p1 (n : Nat) : n + 0 = n := by
          simp

        theorem aime_2022_p3 (x : Int) : x * 1 = x := by
          ring
    """))

    problems = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
    assert len(problems) == 2
    assert problems[0].name == "amc_2021_p1"
    assert problems[0].difficulty == ProblemDifficulty.AMC
    assert "import Mathlib" in problems[0].lean_header
    assert problems[1].name == "aime_2022_p3"
    assert problems[1].difficulty == ProblemDifficulty.AIME
    assert "sorry" in problems[0].lean_statement
