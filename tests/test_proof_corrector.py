"""Tests for ProofCorrector agent and pipeline integration.

All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.proof import (
    ErrorCategory,
    ProofCorrection,
    ProofSearchResult,
)


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_proof.py)
# ---------------------------------------------------------------------------


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )


def _extract_json_helper(text: str):
    import re

    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = [_mock_llm_response(text) for text in responses]
    mock.complete.side_effect = side_effects
    mock.extract_json.side_effect = lambda text: _extract_json_helper(text)
    return mock


def _make_mock_repl():
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


def _make_mock_search():
    from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

    return LeanSearch(SearchConfig(backend=SearchBackend.MOCK))


# ---------------------------------------------------------------------------
# models/proof.py — ErrorCategory
# ---------------------------------------------------------------------------


class TestErrorCategory:
    def test_values(self):
        assert ErrorCategory.TYPE_MISMATCH == "type_mismatch"
        assert ErrorCategory.MISSING_IMPORT == "missing_import"
        assert ErrorCategory.TACTIC_FAILURE == "tactic_failure"
        assert ErrorCategory.UNIVERSE_LEVEL == "universe_level"
        assert ErrorCategory.UNKNOWN_IDENTIFIER == "unknown_identifier"
        assert ErrorCategory.TIMEOUT == "timeout"
        assert ErrorCategory.OTHER == "other"

    def test_all_values_count(self):
        assert len(ErrorCategory) == 7


# ---------------------------------------------------------------------------
# models/proof.py — ProofCorrection
# ---------------------------------------------------------------------------


class TestProofCorrection:
    def test_defaults(self):
        c = ProofCorrection(
            error_category=ErrorCategory.OTHER,
            error_message="some error",
        )
        assert c.confidence == 0.5
        assert c.suggested_tactics == []
        assert c.revised_proof_sketch == ""
        assert c.reasoning == ""

    def test_full(self):
        c = ProofCorrection(
            error_category=ErrorCategory.TYPE_MISMATCH,
            error_message="type mismatch, expected Nat got Int",
            suggested_tactics=["exact Int.toNat x", "cast"],
            revised_proof_sketch="theorem foo : Nat := Int.toNat x",
            confidence=0.8,
            reasoning="Need explicit coercion from Int to Nat",
        )
        assert c.error_category == ErrorCategory.TYPE_MISMATCH
        assert c.confidence == 0.8
        assert len(c.suggested_tactics) == 2

    def test_confidence_bounds(self):
        import pytest

        with pytest.raises(Exception):
            ProofCorrection(
                error_category=ErrorCategory.OTHER,
                error_message="err",
                confidence=1.5,
            )
        with pytest.raises(Exception):
            ProofCorrection(
                error_category=ErrorCategory.OTHER,
                error_message="err",
                confidence=-0.1,
            )

    def test_serialization_roundtrip(self):
        c = ProofCorrection(
            error_category=ErrorCategory.TACTIC_FAILURE,
            error_message="simp failed",
            suggested_tactics=["omega", "ring"],
            confidence=0.7,
        )
        restored = ProofCorrection.model_validate(c.model_dump())
        assert restored == c


# ---------------------------------------------------------------------------
# agents/proof_corrector.py
# ---------------------------------------------------------------------------


class TestProofCorrectorAgent:
    def test_type_mismatch_correction(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "type_mismatch",
            "error_message": "type mismatch, expected Nat got Int",
            "suggested_tactics": ["exact Int.toNat x", "norm_cast"],
            "revised_proof_sketch": "by norm_cast",
            "confidence": 0.8,
            "reasoning": "Need explicit coercion from Int to Nat",
        })
        llm = _make_mock_llm([response])

        corrector = ProofCorrector(llm_client=llm)
        correction = corrector.correct(
            failed_proof="theorem foo : Nat := x",
            error_message="type mismatch, expected Nat got Int",
            lean_goal_state="⊢ Nat",
        )

        assert correction.error_category == ErrorCategory.TYPE_MISMATCH
        assert "norm_cast" in correction.suggested_tactics
        assert correction.confidence == 0.8

    def test_missing_import_correction(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "missing_import",
            "error_message": "unknown identifier 'Finset.sum_comm'",
            "suggested_tactics": ["import Mathlib.Algebra.BigOperators.Basic"],
            "revised_proof_sketch": "import Mathlib.Algebra.BigOperators.Basic\nby exact Finset.sum_comm",
            "confidence": 0.9,
            "reasoning": "Finset.sum_comm is in Mathlib.Algebra.BigOperators.Basic",
        })
        llm = _make_mock_llm([response])

        corrector = ProofCorrector(llm_client=llm)
        correction = corrector.correct(
            failed_proof="by exact Finset.sum_comm",
            error_message="unknown identifier 'Finset.sum_comm'",
            lean_goal_state="⊢ Finset.sum f = Finset.sum g",
        )

        assert correction.error_category == ErrorCategory.MISSING_IMPORT
        assert correction.confidence == 0.9

    def test_tactic_failure_correction(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "tactic_failure",
            "error_message": "simp made no progress",
            "suggested_tactics": ["omega", "ring", "norm_num"],
            "revised_proof_sketch": "by omega",
            "confidence": 0.7,
            "reasoning": "simp doesn't handle arithmetic goals well, omega is better",
        })
        llm = _make_mock_llm([response])

        corrector = ProofCorrector(llm_client=llm)
        correction = corrector.correct(
            failed_proof="by simp",
            error_message="simp made no progress",
            lean_goal_state="⊢ n + m = m + n",
        )

        assert correction.error_category == ErrorCategory.TACTIC_FAILURE
        assert "omega" in correction.suggested_tactics
        assert "ring" in correction.suggested_tactics

    def test_with_prior_attempts(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "tactic_failure",
            "error_message": "ring failed",
            "suggested_tactics": ["linarith"],
            "revised_proof_sketch": "by linarith",
            "confidence": 0.6,
            "reasoning": "ring and omega already failed, try linarith",
        })
        llm = _make_mock_llm([response])

        corrector = ProofCorrector(llm_client=llm)
        correction = corrector.correct(
            failed_proof="by ring",
            error_message="ring failed",
            lean_goal_state="⊢ a ≤ b",
            prior_attempts=["by omega", "by simp"],
        )

        assert correction.error_category == ErrorCategory.TACTIC_FAILURE
        assert "linarith" in correction.suggested_tactics

    def test_unparseable_response_fallback(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        llm = _make_mock_llm(["I don't know how to fix this."])

        corrector = ProofCorrector(llm_client=llm)
        correction = corrector.correct(
            failed_proof="by bad_tactic",
            error_message="unknown tactic",
            lean_goal_state="⊢ True",
        )

        assert correction.error_category == ErrorCategory.OTHER
        assert correction.error_message == "unknown tactic"

    def test_execute_via_base_agent(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "unknown_identifier",
            "error_message": "unknown identifier 'foo'",
            "suggested_tactics": ["exact bar"],
            "revised_proof_sketch": "by exact bar",
            "confidence": 0.5,
            "reasoning": "foo might be bar",
        })
        llm = _make_mock_llm([response])

        corrector = ProofCorrector(llm_client=llm)
        ctx = AgentContext(
            task="correct proof",
            metadata={
                "failed_proof": "by exact foo",
                "error_message": "unknown identifier 'foo'",
                "lean_goal_state": "⊢ Nat",
            },
        )
        result = corrector.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        correction = ProofCorrection.model_validate(result.result)
        assert correction.error_category == ErrorCategory.UNKNOWN_IDENTIFIER


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestProofPipelineWithCorrector:
    def test_corrector_invoked_on_search_failure(self):
        from agentic_research.pipelines.proof import ProofPipeline

        strategies_json = json.dumps({
            "strategies": [{
                "strategy_type": "direct",
                "description": "simp",
                "plausibility": 0.9,
                "relevant_lemmas": [],
                "key_tactics": ["simp"],
            }]
        })

        correction_json = json.dumps({
            "error_category": "tactic_failure",
            "error_message": "simp failed",
            "suggested_tactics": ["omega"],
            "revised_proof_sketch": "by omega",
            "confidence": 0.8,
            "reasoning": "use omega instead",
        })

        corrected_strategies_json = json.dumps({
            "strategies": [{
                "strategy_type": "direct",
                "description": "omega",
                "plausibility": 0.95,
                "relevant_lemmas": [],
                "key_tactics": ["omega"],
            }]
        })

        llm = _make_mock_llm([
            strategies_json,
            "```lean\n-- MOCK_ERROR\nbad proof\n```",
            correction_json,
            corrected_strategies_json,
            "```lean\ntheorem foo : True := trivial\n```",
        ])
        repl = _make_mock_repl()
        search = _make_mock_search()

        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            prover_config=ProverConfig(max_iterations=1),
            max_strategies=1,
            use_claim_check=False,
        )

        result = pipeline.run("theorem foo : True")
        assert result.proved
        assert result.final_proof is not None

    def test_corrector_skipped_on_timeout(self):
        from agentic_research.pipelines.proof import ProofPipeline

        strategies_json = json.dumps({
            "strategies": [{
                "strategy_type": "direct",
                "description": "direct",
                "plausibility": 0.5,
                "relevant_lemmas": [],
                "key_tactics": ["simp"],
            }]
        })

        llm = _make_mock_llm([
            strategies_json,
            "```lean\n-- MOCK_ERROR\nbad\n```",
        ])
        repl = _make_mock_repl()
        search = _make_mock_search()

        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            prover_config=ProverConfig(max_iterations=1),
            max_strategies=1,
            use_claim_check=False,
        )

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            needs_decomposition=True,
            failure_reason="timeout during proof search",
        )

        correction = pipeline._try_proof_correction("theorem foo : True", search_result)
        assert correction is None

    def test_corrector_returns_none_no_strategies(self):
        from agentic_research.pipelines.proof import ProofPipeline

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        search = _make_mock_search()

        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            use_claim_check=False,
        )

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            needs_decomposition=True,
            failure_reason="No strategies tried",
            strategies_tried=[],
        )

        correction = pipeline._try_proof_correction("theorem foo : True", search_result)
        assert correction is None
