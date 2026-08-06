"""Tests for LemmaLeanifier infrastructure error early termination."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic_research.agents.lemma_leanifier import LemmaLeanifier
from agentic_research.agents.llm_client import LLMClient
from agentic_research.models.agents import LLMResponse, TokenUsage
from agentic_research.models.proof import ProofNode
from agentic_research.models.tools import CompilationResult, CompilationStatus, ToolStatus
from agentic_research.tools.lean_repl import LeanRepl


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )


def _infra_error_result() -> CompilationResult:
    return CompilationResult(
        status=ToolStatus.SUCCESS,
        compilation_status=CompilationStatus.ERROR,
        errors=["Mathlib.olean does not exist"],
        lean_output="error: Mathlib.olean does not exist",
    )


def _compilation_error_result() -> CompilationResult:
    return CompilationResult(
        status=ToolStatus.SUCCESS,
        compilation_status=CompilationStatus.ERROR,
        errors=["unknown identifier 'foo'"],
        lean_output="error: unknown identifier 'foo'",
    )


def _ok_result() -> CompilationResult:
    return CompilationResult(
        status=ToolStatus.SUCCESS,
        compilation_status=CompilationStatus.OK,
        lean_output="",
        all_goals_closed=True,
    )


def test_leanifier_breaks_on_infrastructure_error():
    """Retry loop breaks immediately when compilation returns an infrastructure error."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete.return_value = _mock_llm_response("```lean\ntheorem foo : True := by sorry\n```")

    mock_repl = MagicMock(spec=LeanRepl)
    mock_repl.execute.return_value = _infra_error_result()

    leanifier = LemmaLeanifier(
        llm_client=mock_llm,
        lean_repl=mock_repl,
        max_compile_retries=3,
    )

    node = ProofNode(node_id="lemma_1", statement_nl="Test lemma")
    result, tokens = leanifier.leanify_single_node(node, parent_statement="")

    assert result is None
    # Only 1 LLM call (initial), no retries since infra error was on first compilation
    assert mock_llm.complete.call_count == 1
    # Only 1 compilation (the initial one)
    assert mock_repl.execute.call_count == 1


def test_leanifier_retries_on_compilation_error():
    """Retry loop continues normally for non-infrastructure errors (regression guard)."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete.side_effect = [
        _mock_llm_response("```lean\ntheorem foo : True := by sorry\n```"),
        _mock_llm_response("```lean\ntheorem foo : True := by trivial\n```"),
        _mock_llm_response("```lean\ntheorem foo : True := by exact trivial\n```"),
        _mock_llm_response("```lean\ntheorem foo : True := by simp\n```"),
    ]

    mock_repl = MagicMock(spec=LeanRepl)
    mock_repl.execute.side_effect = [
        _compilation_error_result(),  # initial
        _compilation_error_result(),  # retry 1
        _compilation_error_result(),  # retry 2
        _ok_result(),                 # retry 3 succeeds
    ]

    leanifier = LemmaLeanifier(
        llm_client=mock_llm,
        lean_repl=mock_repl,
        max_compile_retries=3,
    )

    node = ProofNode(node_id="lemma_1", statement_nl="Test lemma")
    result, tokens = leanifier.leanify_single_node(node, parent_statement="")

    assert result is not None
    # 1 initial + 3 retries = 4 LLM calls
    assert mock_llm.complete.call_count == 4
    # 1 initial + 3 retries = 4 compile calls
    assert mock_repl.execute.call_count == 4
