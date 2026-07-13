"""Tests for the automated tactic pre-filter (grind/simp_all before LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agentic_research.models.proof import ProofPipelineResult
from agentic_research.models.tools import (
    CompilationResult,
    CompilationStatus,
    ToolStatus,
)
from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig


STMT = "theorem foo : 1 + 1 = 2 :="


def _ok_result() -> CompilationResult:
    return CompilationResult(
        status=ToolStatus.SUCCESS,
        compilation_status=CompilationStatus.OK,
        all_goals_closed=True,
    )


def _fail_result() -> CompilationResult:
    return CompilationResult(
        status=ToolStatus.SUCCESS,
        compilation_status=CompilationStatus.ERROR,
        errors=["tactic failed"],
        lean_output="error: tactic failed",
    )


def _timeout_result() -> CompilationResult:
    return CompilationResult(
        status=ToolStatus.TIMEOUT,
        compilation_status=CompilationStatus.TIMEOUT,
        error_message="timed out",
    )


# ---------------------------------------------------------------------------
# LeanRepl.try_automated_tactics
# ---------------------------------------------------------------------------


def test_grind_success():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    repl._backend = MagicMock()
    repl._backend.compile.return_value = _ok_result()

    result = repl.try_automated_tactics(STMT)

    assert result == "grind"
    repl._backend.compile.assert_called_once()
    call_code = repl._backend.compile.call_args[0][0]
    assert "by grind" in call_code


def test_grind_fail_simp_success():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    repl._backend = MagicMock()
    repl._backend.compile.side_effect = [_fail_result(), _ok_result()]

    result = repl.try_automated_tactics(STMT)

    assert result == "simp_all"
    assert repl._backend.compile.call_count == 2
    second_call_code = repl._backend.compile.call_args_list[1][0][0]
    assert "by simp_all" in second_call_code


def test_both_fail():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    repl._backend = MagicMock()
    repl._backend.compile.return_value = _fail_result()

    result = repl.try_automated_tactics(STMT)

    assert result is None
    assert repl._backend.compile.call_count == 6


def test_timeout_handling():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    repl._backend = MagicMock()
    repl._backend.compile.return_value = _timeout_result()

    result = repl.try_automated_tactics(STMT, timeout_seconds=1.0)

    assert result is None


def test_imports_included():
    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
    repl._backend = MagicMock()
    repl._backend.compile.return_value = _ok_result()

    repl.try_automated_tactics(STMT, imports=["Mathlib.Tactic"])

    call_code = repl._backend.compile.call_args[0][0]
    assert "import Mathlib.Tactic" in call_code


# ---------------------------------------------------------------------------
# ProofPipeline integration
# ---------------------------------------------------------------------------


def test_proof_pipeline_uses_prefilter():
    from agentic_research.pipelines.proof import ProofPipeline

    mock_llm = MagicMock()
    mock_repl = MagicMock(spec=LeanRepl)
    mock_search = MagicMock()

    mock_repl.try_automated_tactics.return_value = "grind"

    pipeline = ProofPipeline(
        llm_client=mock_llm,
        lean_repl=mock_repl,
        lean_search=mock_search,
    )

    result = pipeline.run(STMT)

    assert isinstance(result, ProofPipelineResult)
    assert result.proved is True
    assert "by grind" in result.final_proof
    mock_repl.try_automated_tactics.assert_called_once_with(STMT)


def test_proof_pipeline_falls_through_when_prefilter_fails():
    from agentic_research.pipelines.proof import ProofPipeline
    from agentic_research.models.proof import ProofSearchResult

    mock_llm = MagicMock()
    mock_repl = MagicMock(spec=LeanRepl)
    mock_search = MagicMock()

    mock_repl.try_automated_tactics.return_value = None

    pipeline = ProofPipeline(
        llm_client=mock_llm,
        lean_repl=mock_repl,
        lean_search=mock_search,
    )

    with patch.object(pipeline, "_run_proof_search") as mock_proof_search:
        mock_proof_search.return_value = ProofSearchResult(
            statement=STMT,
            proved=False,
            needs_decomposition=False,
            failure_reason="no proof found",
        )
        result = pipeline.run(STMT)

    assert result.proved is False
    mock_repl.try_automated_tactics.assert_called_once()
    mock_proof_search.assert_called_once()
