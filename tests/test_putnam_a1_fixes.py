"""Tests for 4 pipeline bugs identified by Putnam 2024 A1 trajectory analysis.

Bug 1: Double preamble — _strip_preamble_lines removes LLM-generated imports
Bug 2: max_critic_retries default changed from 0 to 1
Bug 3: Duplicate input_hash skips recompilation
Bug 4: _extract_compiler_errors filters strategy summaries
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic_research.agents.lean_utils import _strip_preamble_lines
from agentic_research.agents.prover import IterativeProver
from agentic_research.models.agents import (
    LLMResponse,
    ProofAttempt,
    ProofAttemptStatus,
    ProverConfig,
    ProverResult,
    TokenUsage,
)
from agentic_research.models.proof import (
    ProofSearchResult,
    ProofStrategy,
    StrategyType,
)
from agentic_research.pipelines.proof import ProofPipeline


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )


# ---------------------------------------------------------------------------
# Bug 1: _strip_preamble_lines
# ---------------------------------------------------------------------------


class TestStripPreambleLines:
    def test_strips_imports_and_open(self):
        code = (
            "import Mathlib\n"
            "import Aesop\n"
            "set_option maxHeartbeats 400000\n"
            "open BigOperators Real Nat\n"
            "\n"
            "theorem foo : True := trivial"
        )
        result = _strip_preamble_lines(code)
        assert result == "theorem foo : True := trivial"

    def test_strips_section_and_namespace(self):
        code = (
            "section Foo\n"
            "namespace Bar\n"
            "def baz := 42"
        )
        result = _strip_preamble_lines(code)
        assert result == "def baz := 42"

    def test_idempotent_no_preamble(self):
        code = "theorem foo : True := trivial"
        result = _strip_preamble_lines(code)
        assert result == code

    def test_idempotent_with_comment_first(self):
        code = "-- A comment\ntheorem foo : True := trivial"
        result = _strip_preamble_lines(code)
        assert result == code

    def test_preserves_later_imports(self):
        code = (
            "import Mathlib\n"
            "theorem foo : True := by\n"
            "  import_tactic"
        )
        result = _strip_preamble_lines(code)
        assert result.startswith("theorem foo")
        assert "import_tactic" in result

    def test_blank_lines_only_stripped_at_start(self):
        code = "\n\n\ntheorem foo : True := trivial\n\n"
        result = _strip_preamble_lines(code)
        assert result.startswith("theorem foo")

    def test_empty_string(self):
        assert _strip_preamble_lines("") == ""

    def test_all_preamble_lines(self):
        code = "import Mathlib\nset_option maxHeartbeats 400000\n"
        result = _strip_preamble_lines(code)
        assert result == ""


# ---------------------------------------------------------------------------
# Bug 2: max_critic_retries default is 1
# ---------------------------------------------------------------------------


class TestMaxCriticRetriesDefault:
    def test_default_is_one(self):
        llm = MagicMock()
        repl = MagicMock()
        search = MagicMock()
        pipeline = ProofPipeline(llm, repl, search)
        assert pipeline._max_critic_retries == 1


# ---------------------------------------------------------------------------
# Bug 3: Duplicate input_hash skips recompilation
# ---------------------------------------------------------------------------


class TestDuplicateHashDedup:
    def test_duplicate_code_breaks_iteration(self):
        llm = MagicMock()
        repl = MagicMock()

        identical_code = "theorem foo : True := by trivial"
        llm.complete.return_value = _mock_llm_response(
            f"```lean\n{identical_code}\n```"
        )

        compilation_mock = MagicMock()
        compilation_mock.compilation_status.value = "error"
        compilation_mock.compilation_status = MagicMock()
        compilation_mock.compilation_status.__eq__ = lambda self, other: False
        compilation_mock.all_goals_closed = False
        compilation_mock.errors = ["some error"]
        compilation_mock.warnings = []
        compilation_mock.goals = []

        from agentic_research.models.tools import CompilationStatus
        compilation_mock.compilation_status = CompilationStatus.ERROR

        repl.execute.return_value = compilation_mock

        config = ProverConfig(max_iterations=5)
        prover = IterativeProver(llm, repl, config=config)

        prover._prove("theorem foo : True")

        assert repl.execute.call_count == 1, (
            f"Expected 1 REPL call (dedup should skip identical code), got {repl.execute.call_count}"
        )

    def test_different_code_does_not_dedup(self):
        llm = MagicMock()
        repl = MagicMock()

        responses = [
            _mock_llm_response("```lean\ntheorem foo : True := by trivial\n```"),
            _mock_llm_response("```lean\ntheorem foo : True := by exact trivial\n```"),
        ]
        llm.complete.side_effect = responses

        compilation_mock = MagicMock()
        from agentic_research.models.tools import CompilationStatus
        compilation_mock.compilation_status = CompilationStatus.ERROR
        compilation_mock.all_goals_closed = False
        compilation_mock.errors = ["error"]
        compilation_mock.warnings = []
        compilation_mock.goals = []

        repl.execute.return_value = compilation_mock

        config = ProverConfig(max_iterations=2)
        prover = IterativeProver(llm, repl, config=config)
        prover._prove("theorem foo : True")

        assert repl.execute.call_count == 2


# ---------------------------------------------------------------------------
# Bug 4: _extract_compiler_errors filters strategy summaries
# ---------------------------------------------------------------------------


class TestExtractCompilerErrors:
    def test_excludes_strategy_summaries(self):
        prover_result = ProverResult(
            statement="theorem foo : True",
            proved=False,
            attempts=[
                ProofAttempt(
                    iteration=1,
                    proof_code="by sorry",
                    status=ProofAttemptStatus.COMPILATION_ERROR,
                    errors=["unknown identifier 'Real.sqrt'"],
                    token_usage=TokenUsage(),
                ),
            ],
            total_iterations=1,
            total_token_usage=TokenUsage(),
        )

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            needs_decomposition=True,
            failure_reason="All 3 strategies exhausted",
            strategies_tried=[
                ProofStrategy(
                    strategy_type=StrategyType.CONTRADICTION,
                    description="Try proof by contradiction",
                    key_tactics=["by_contra", "push_neg"],
                    prover_result=prover_result,
                ),
            ],
        )

        errors = ProofPipeline._extract_compiler_errors(search_result)

        assert "unknown identifier 'Real.sqrt'" in errors
        for err in errors:
            assert "Strategy" not in err
            assert "strategies exhausted" not in err

    def test_no_prover_result_yields_empty(self):
        search_result = ProofSearchResult(
            statement="theorem foo : True",
            needs_decomposition=True,
            failure_reason="All 3 strategies exhausted",
            strategies_tried=[
                ProofStrategy(
                    strategy_type=StrategyType.DIRECT,
                    description="Direct proof attempt",
                    key_tactics=["simp"],
                ),
            ],
        )

        errors = ProofPipeline._extract_compiler_errors(search_result)
        assert errors == []

    def test_multiple_attempts_collected(self):
        prover_result = ProverResult(
            statement="theorem foo : True",
            proved=False,
            attempts=[
                ProofAttempt(
                    iteration=1,
                    proof_code="by sorry",
                    status=ProofAttemptStatus.COMPILATION_ERROR,
                    errors=["error1.lean:5: unknown tactic"],
                    token_usage=TokenUsage(),
                ),
                ProofAttempt(
                    iteration=2,
                    proof_code="by sorry",
                    status=ProofAttemptStatus.COMPILATION_ERROR,
                    errors=["error2.lean:10: type mismatch"],
                    token_usage=TokenUsage(),
                ),
            ],
            total_iterations=2,
            total_token_usage=TokenUsage(),
        )

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            needs_decomposition=True,
            strategies_tried=[
                ProofStrategy(
                    strategy_type=StrategyType.DIRECT,
                    description="Direct",
                    key_tactics=[],
                    prover_result=prover_result,
                ),
            ],
        )

        errors = ProofPipeline._extract_compiler_errors(search_result)
        assert len(errors) == 2
        assert "error1.lean:5: unknown tactic" in errors
        assert "error2.lean:10: type mismatch" in errors
