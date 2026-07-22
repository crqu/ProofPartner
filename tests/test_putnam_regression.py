"""PutnamBench regression test — Putnam 2024 A1 end-to-end.

Uses REAL Vertex AI API and REAL Lean 4 prover.
Validates that best-of-k decomposition helps the pipeline avoid the
previously-stuck decomposition path (lemma_4 stuck on Even(a^n) → Even(a)).
"""

from __future__ import annotations

import os
import shutil

import pytest

LEAN_AVAILABLE = shutil.which("lean") is not None
VERTEX_AVAILABLE = (
    os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1"
    and bool(os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"))
) or bool(os.environ.get("ANTHROPIC_API_KEY"))

lean_required = pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
api_required = pytest.mark.skipif(not VERTEX_AVAILABLE, reason="No API credentials available")

PUTNAM_2024_A1_NL = (
    "Determine all positive integers n such that the equation "
    "2a^n + 3b^n = 4c^n has a solution in positive integers a, b, c. "
    "The answer is n = 1."
)

PUTNAM_2024_A1_LEAN = (
    "theorem putnam_2024_a1 : "
    "{n : ℕ | 0 < n ∧ ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ "
    "2 * a ^ n + 3 * b ^ n = 4 * c ^ n} = {1} := by sorry"
)


@lean_required
@api_required
@pytest.mark.timeout(600)
class TestPutnam2024A1Regression:
    """Real E2E regression test for Putnam 2024 A1 with best-of-k decomposition."""

    def _make_pipeline(self):
        from agentic_research.agents.llm_client import LLMClient
        from agentic_research.pipelines.proof import ProofPipeline
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
        from agentic_research.tools.lean_search import LeanSearch

        llm = LLMClient()
        repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
        search = LeanSearch()

        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            decomposition_k=3,
            use_proof_critic=True,
            use_proof_detailer=True,
            use_claim_check=True,
        )
        return pipeline

    def test_putnam_2024_a1_decomposition_diversity(self):
        """Best-of-k decomposition should avoid the single-path failure mode."""
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown
        from agentic_research.agents.llm_client import LLMClient
        from agentic_research.models.agents import AgentContext
        from agentic_research.models.proof import LemmaTree

        llm = LLMClient()
        breakdown = LemmaBreakdown(llm_client=llm, decomposition_k=3)

        ctx = AgentContext(
            task=PUTNAM_2024_A1_NL,
            metadata={
                "statement_lean": PUTNAM_2024_A1_LEAN,
                "failed_attempts": "None",
            },
        )

        result = breakdown.run(ctx)
        assert result.result is not None

        tree = LemmaTree.model_validate(result.result)
        assert tree.decomposition_score is not None
        assert tree.decomposition_score > 0.0
        assert len(tree.nodes) >= 2

        assert result.token_usage.input_tokens > 0
        assert result.token_usage.output_tokens > 0

    def test_putnam_2024_a1_full_pipeline(self):
        """Full pipeline should get past the previously-stuck decomposition."""
        pipeline = self._make_pipeline()
        result = pipeline.run(
            lean_statement=PUTNAM_2024_A1_LEAN,
            statement_nl=PUTNAM_2024_A1_NL,
        )

        assert result.total_token_usage.input_tokens > 0

        if result.proved:
            assert result.final_proof is not None
            assert "sorry" not in result.final_proof
        else:
            if result.failure_stage:
                assert result.failure_stage != "lemma_breakdown", (
                    "Pipeline should not fail at decomposition with k=3"
                )
