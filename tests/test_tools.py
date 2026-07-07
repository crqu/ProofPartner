"""Tests for the Lean 4 tool layer (Phase 2).

Uses mock backends for unit tests. Integration tests requiring Lean 4
are marked with ``pytest.mark.skipif``.
"""

from __future__ import annotations

import shutil

import pytest

from agentic_research.models.tools import (
    CleanResult,
    CompilationResult,
    CompilationStatus,
    LookupResult,
    SearchResult,
    ToolStatus,
)
from agentic_research.tools.base import BaseTool, Tool, _input_hash
from agentic_research.tools.hint_cleaner import HintCleaner, HintCleanerConfig, _remove_comments
from agentic_research.tools.lean_lookup import LeanLookup, LookupBackend, LookupConfig
from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

LEAN_AVAILABLE = shutil.which("lean") is not None
LEAN_DOJO_AVAILABLE = False
try:
    import lean_dojo  # noqa: F401
    LEAN_DOJO_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------

class _DummyTool(BaseTool):
    _name = "dummy"

    def _run(self, input_data):
        from agentic_research.models.tools import ToolResult
        return ToolResult(tool_name="dummy", status=ToolStatus.SUCCESS)


class _ErrorTool(BaseTool):
    _name = "error_tool"

    def _run(self, input_data):
        raise RuntimeError("boom")


def test_base_tool_satisfies_protocol():
    tool = _DummyTool()
    assert isinstance(tool, Tool)


def test_base_tool_execute_success():
    tool = _DummyTool()
    result = tool.execute("hello")
    assert result.status == ToolStatus.SUCCESS
    assert result.duration_seconds >= 0.0


def test_base_tool_execute_error_handling():
    tool = _ErrorTool()
    result = tool.execute("input")
    assert result.status == ToolStatus.ERROR
    assert "boom" in result.error_message


def test_input_hash_deterministic():
    assert _input_hash("hello") == _input_hash("hello")
    assert _input_hash("hello") != _input_hash("world")


# ---------------------------------------------------------------------------
# lean_repl.py — mock backend
# ---------------------------------------------------------------------------

def test_repl_mock_success():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    result = repl.execute("theorem foo : True := trivial")
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.OK
    assert result.all_goals_closed is True


def test_repl_mock_sorry():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    result = repl.execute("theorem foo : True := by sorry")
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.OK
    assert result.all_goals_closed is False
    assert len(result.goals) > 0
    assert len(result.warnings) > 0


def test_repl_mock_error():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    result = repl.execute("-- MOCK_ERROR\ntheorem foo : True := trivial")
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.ERROR
    assert len(result.errors) > 0


def test_repl_backend_property():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    assert repl.backend == ReplBackend.MOCK


# ---------------------------------------------------------------------------
# lean_repl.py — subprocess backend
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
def test_repl_subprocess_trivial():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    result = repl.execute("theorem foo : True := trivial")
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.OK


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
def test_repl_subprocess_error():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    result = repl.execute("theorem foo : True := 42")
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.ERROR


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
def test_repl_subprocess_unicode():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    result = repl.execute("theorem ℕ_test : ∀ n : Nat, n = n := fun n => rfl")
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.OK


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
def test_repl_subprocess_sorry():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    result = repl.execute("theorem test : True := by sorry")
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.OK


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
def test_repl_subprocess_multiline():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    code = "\n".join([
        "theorem zero_add_test : ∀ (n : Nat), 0 + n = n := by",
        "  intro n",
        "  simp",
    ])
    result = repl.execute(code)
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.OK


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
def test_repl_subprocess_syntax_error():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    result = repl.execute("theorem test : True := garbage#$%")
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.ERROR
    assert len(result.errors) > 0


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
def test_repl_subprocess_multiple_theorems():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    code = "\n".join([
        "theorem t1 : True := trivial",
        "theorem t2 : 1 + 1 = 2 := rfl",
    ])
    result = repl.execute(code)
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.OK


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
def test_repl_subprocess_comments():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    code = "\n".join([
        "-- This is a line comment",
        "/- This is a block comment -/",
        "theorem commented : True := trivial",
    ])
    result = repl.execute(code)
    assert isinstance(result, CompilationResult)
    assert result.compilation_status == CompilationStatus.OK


# ---------------------------------------------------------------------------
# lean_repl.py — LeanDojo backend
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LEAN_DOJO_AVAILABLE, reason="lean-dojo not installed")
def test_repl_leandojo():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.LEAN_DOJO))
    result = repl.execute("trivial")
    assert isinstance(result, CompilationResult)


# ---------------------------------------------------------------------------
# lean_search.py — mock backend
# ---------------------------------------------------------------------------

def test_search_mock_returns_results():
    search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
    result = search.execute("natural number addition")
    assert isinstance(result, SearchResult)
    assert result.status == ToolStatus.SUCCESS
    assert len(result.entries) > 0
    assert result.entries[0].name == "Nat.add_comm"


def test_search_mock_max_results():
    search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK, max_results=1))
    result = search.execute("commutativity")
    assert isinstance(result, SearchResult)
    assert len(result.entries) <= 1


def test_search_backend_property():
    search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
    assert search.search_backend == SearchBackend.MOCK


# ---------------------------------------------------------------------------
# lean_lookup.py — mock backend
# ---------------------------------------------------------------------------

def test_lookup_mock_found():
    lookup = LeanLookup(LookupConfig(backend=LookupBackend.MOCK))
    result = lookup.execute("Nat.add_comm")
    assert isinstance(result, LookupResult)
    assert result.found is True
    assert result.type_signature != ""
    assert result.module == "Mathlib.Data.Nat.Basic"


def test_lookup_mock_not_found():
    lookup = LeanLookup(LookupConfig(backend=LookupBackend.MOCK))
    result = lookup.execute("Nonexistent.theorem")
    assert isinstance(result, LookupResult)
    assert result.found is False


def test_lookup_backend_property():
    lookup = LeanLookup(LookupConfig(backend=LookupBackend.MOCK))
    assert lookup.lookup_backend == LookupBackend.MOCK


# ---------------------------------------------------------------------------
# hint_cleaner.py
# ---------------------------------------------------------------------------

def test_hint_cleaner_removes_line_comments():
    code = "theorem foo : True := by -- this is a hint\n  trivial\n"
    cleaner = HintCleaner()
    result = cleaner.execute(code)
    assert isinstance(result, CleanResult)
    assert result.status == ToolStatus.SUCCESS
    assert "--" not in result.cleaned_code
    assert "trivial" in result.cleaned_code
    assert result.comments_removed >= 1


def test_hint_cleaner_removes_block_comments():
    code = "theorem foo : True := by\n  /- AI generated hint -/\n  trivial\n"
    cleaner = HintCleaner()
    result = cleaner.execute(code)
    assert isinstance(result, CleanResult)
    assert "AI generated" not in result.cleaned_code
    assert "trivial" in result.cleaned_code
    assert result.comments_removed >= 1


def test_hint_cleaner_preserves_string_literals():
    code = 'def msg := "hello -- world"\n'
    cleaner = HintCleaner()
    result = cleaner.execute(code)
    assert isinstance(result, CleanResult)
    assert '"hello -- world"' in result.cleaned_code


def test_hint_cleaner_no_comments():
    code = "theorem foo : True := trivial\n"
    cleaner = HintCleaner()
    result = cleaner.execute(code)
    assert isinstance(result, CleanResult)
    assert result.comments_removed == 0
    assert "trivial" in result.cleaned_code


def test_hint_cleaner_keep_doc_strings():
    code = "/-- Documentation -/\ndef foo := 42\n-- AI hint\n"
    cleaner = HintCleaner(HintCleanerConfig(keep_doc_strings=True))
    result = cleaner.execute(code)
    assert isinstance(result, CleanResult)
    assert "/-- Documentation -/" in result.cleaned_code
    assert "AI hint" not in result.cleaned_code


def test_hint_cleaner_remove_doc_strings_by_default():
    code = "/-- Documentation -/\ndef foo := 42\n"
    cleaner = HintCleaner()
    result = cleaner.execute(code)
    assert isinstance(result, CleanResult)
    assert "Documentation" not in result.cleaned_code
    assert "def foo" in result.cleaned_code


# ---------------------------------------------------------------------------
# _remove_comments standalone
# ---------------------------------------------------------------------------

def test_remove_comments_empty():
    cleaned, count = _remove_comments("", keep_doc_strings=False)
    assert cleaned == ""
    assert count == 0


def test_remove_comments_mixed():
    code = (
        "-- line comment\n"
        "def x := 1\n"
        "/- block -/\n"
        "def y := 2\n"
    )
    cleaned, count = _remove_comments(code, keep_doc_strings=False)
    assert "line comment" not in cleaned
    assert "block" not in cleaned
    assert "def x" in cleaned
    assert "def y" in cleaned
    assert count == 2


# ---------------------------------------------------------------------------
# Model serialization round-trips
# ---------------------------------------------------------------------------

def test_compilation_result_serialization():
    r = CompilationResult(
        status=ToolStatus.SUCCESS,
        compilation_status=CompilationStatus.OK,
        all_goals_closed=True,
    )
    data = r.model_dump()
    restored = CompilationResult.model_validate(data)
    assert restored == r


def test_search_result_serialization():
    r = SearchResult(
        status=ToolStatus.SUCCESS,
        query="test",
        total_results=0,
    )
    data = r.model_dump()
    restored = SearchResult.model_validate(data)
    assert restored == r


def test_lookup_result_serialization():
    r = LookupResult(
        status=ToolStatus.SUCCESS,
        identifier="Nat.add_comm",
        found=True,
        type_signature="∀ (n m : ℕ), n + m = m + n",
    )
    data = r.model_dump()
    restored = LookupResult.model_validate(data)
    assert restored == r


def test_clean_result_serialization():
    r = CleanResult(
        status=ToolStatus.SUCCESS,
        original_code="-- hi\ndef x := 1\n",
        cleaned_code="def x := 1\n",
        comments_removed=1,
    )
    data = r.model_dump()
    restored = CleanResult.model_validate(data)
    assert restored == r


# ---------------------------------------------------------------------------
# Goal parsing
# ---------------------------------------------------------------------------

def test_parse_goals_from_lean_output():
    from agentic_research.tools.lean_repl import _parse_goals

    output = (
        "unsolved goals\n"
        "n m : ℕ\n"
        "⊢ n + m = m + n\n"
    )
    goals = _parse_goals(output)
    assert len(goals) == 1
    assert "n + m = m + n" in goals[0].goal
    assert len(goals[0].hypotheses) > 0


def test_parse_goals_no_goals():
    from agentic_research.tools.lean_repl import _parse_goals

    goals = _parse_goals("no goals")
    assert goals == []


# ---------------------------------------------------------------------------
# Error parsing
# ---------------------------------------------------------------------------

def test_parse_lean_errors():
    from agentic_research.tools.lean_repl import _parse_lean_errors

    output = (
        "file.lean:1:0: error: unknown identifier 'foo'\n"
        "file.lean:2:0: warning: unused variable 'x'\n"
        "all good\n"
    )
    errors, warnings = _parse_lean_errors(output)
    assert len(errors) == 1
    assert len(warnings) == 1
    assert "foo" in errors[0]
    assert "unused" in warnings[0]
