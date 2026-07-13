"""Tests for the NaturalLanguageProver agent and NL proof stage integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentic_research.agents.nl_prover import NaturalLanguageProver
from agentic_research.agents.proof_critic import ProofCritic
from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.proof import (
    CritiqueIssue,
    CritiqueIssueType,
    CritiqueResult,
    NLProofSketch,
    NLProofStep,
)


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


VALID_NL_PROOF_JSON = json.dumps({
    "overall_strategy": "direct proof using properties of natural numbers",
    "assumptions": ["n is a natural number", "n > 0"],
    "key_lemmas": ["Nat.add_comm", "Nat.succ_pos"],
    "proof_steps": [
        {
            "claim": "n + 0 = n by the additive identity",
            "reasoning": "This follows from the definition of addition on natural numbers",
            "sub_claims": [],
        },
        {
            "claim": "Therefore n + 0 = n for all n",
            "reasoning": "Universal generalization from the previous step",
            "sub_claims": ["Base case holds trivially"],
        },
    ],
})

VALID_CRITIQUE_NO_ISSUES = json.dumps({"issues": []})

VALID_CRITIQUE_WITH_ISSUES = json.dumps({
    "issues": [
        {
            "issue_type": "unstated_hypothesis",
            "node_id": "nl_proof",
            "description": "Does this assume n > 0 without stating it?",
            "severity": "blocking",
            "suggested_fix": "Add hypothesis n > 0",
        }
    ]
})


class TestNLProofSketchModel:
    def test_creation_minimal(self) -> None:
        sketch = NLProofSketch()
        assert sketch.proof_steps == []
        assert sketch.assumptions == []
        assert sketch.key_lemmas == []
        assert sketch.overall_strategy == ""

    def test_creation_full(self) -> None:
        step = NLProofStep(
            claim="x = y",
            reasoning="by reflexivity",
            sub_claims=["x = x", "x = y by hypothesis"],
        )
        sketch = NLProofSketch(
            proof_steps=[step],
            assumptions=["x is real"],
            key_lemmas=["eq_refl"],
            overall_strategy="direct",
        )
        assert len(sketch.proof_steps) == 1
        assert sketch.proof_steps[0].claim == "x = y"
        assert sketch.overall_strategy == "direct"
        assert sketch.assumptions == ["x is real"]

    def test_serialization_roundtrip(self) -> None:
        step = NLProofStep(claim="a", reasoning="b", sub_claims=["c"])
        sketch = NLProofSketch(
            proof_steps=[step],
            assumptions=["d"],
            key_lemmas=["e"],
            overall_strategy="induction",
        )
        data = sketch.model_dump()
        restored = NLProofSketch.model_validate(data)
        assert restored == sketch


class TestNaturalLanguageProver:
    def test_generate_proof_valid_json(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(VALID_NL_PROOF_JSON)
        mock_llm.extract_json.return_value = json.loads(VALID_NL_PROOF_JSON)

        prover = NaturalLanguageProver(llm_client=mock_llm)
        sketch, tokens = prover.generate_proof(
            statement="theorem foo : ∀ n : Nat, n + 0 = n",
            statement_nl="For all natural numbers n, n + 0 = n",
        )

        assert isinstance(sketch, NLProofSketch)
        assert len(sketch.proof_steps) == 2
        assert sketch.overall_strategy == "direct proof using properties of natural numbers"
        assert "Nat.add_comm" in sketch.key_lemmas
        assert tokens.input_tokens == 100

    def test_generate_proof_malformed_output(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(
            "This is not JSON at all, just plain text reasoning."
        )
        mock_llm.extract_json.return_value = None

        prover = NaturalLanguageProver(llm_client=mock_llm)
        sketch, tokens = prover.generate_proof(
            statement="theorem foo : True",
        )

        assert isinstance(sketch, NLProofSketch)
        assert sketch.overall_strategy == "unknown"
        assert len(sketch.proof_steps) == 1
        assert "Unparsed proof" in sketch.proof_steps[0].claim

    def test_generate_proof_with_feedback(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(VALID_NL_PROOF_JSON)
        mock_llm.extract_json.return_value = json.loads(VALID_NL_PROOF_JSON)

        critique = CritiqueResult(
            issues=[
                CritiqueIssue(
                    issue_type=CritiqueIssueType.UNSTATED_HYPOTHESIS,
                    node_id="nl_proof",
                    description="Missing n > 0",
                    severity="blocking",
                    suggested_fix="Add n > 0",
                )
            ],
            passed=False,
        )

        prover = NaturalLanguageProver(llm_client=mock_llm)
        sketch, _ = prover.generate_proof(
            statement="theorem foo : True",
            feedback=critique,
        )

        call_args = mock_llm.complete.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "Critic Feedback" in user_msg
        assert "Missing n > 0" in user_msg

    def test_generate_proof_with_extended_thinking(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(VALID_NL_PROOF_JSON)
        mock_llm.extract_json.return_value = json.loads(VALID_NL_PROOF_JSON)

        config = ProverConfig(use_extended_thinking=True)
        prover = NaturalLanguageProver(llm_client=mock_llm, prover_config=config)
        prover.generate_proof(statement="theorem foo : True")

        call_args = mock_llm.complete.call_args
        assert call_args[1]["use_extended_thinking"] is True

    def test_execute_via_run(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(VALID_NL_PROOF_JSON)
        mock_llm.extract_json.return_value = json.loads(VALID_NL_PROOF_JSON)

        prover = NaturalLanguageProver(llm_client=mock_llm)
        ctx = AgentContext(
            task="theorem foo : True",
            metadata={"statement_nl": "True is provable"},
        )
        result = prover.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        assert result.result is not None
        sketch = NLProofSketch.model_validate(result.result)
        assert len(sketch.proof_steps) == 2


class TestProofCriticNLAudit:
    def test_audit_nl_proof_no_issues(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(VALID_CRITIQUE_NO_ISSUES)
        mock_llm.extract_json.return_value = json.loads(VALID_CRITIQUE_NO_ISSUES)

        critic = ProofCritic(llm_client=mock_llm)
        sketch = NLProofSketch(
            proof_steps=[NLProofStep(claim="a", reasoning="b")],
            overall_strategy="direct",
        )
        result = critic.audit_nl_proof(sketch, "theorem foo : True")

        assert isinstance(result, CritiqueResult)
        assert result.passed is True
        assert result.issues == []

    def test_audit_nl_proof_with_issues(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(VALID_CRITIQUE_WITH_ISSUES)
        mock_llm.extract_json.return_value = json.loads(VALID_CRITIQUE_WITH_ISSUES)

        critic = ProofCritic(llm_client=mock_llm)
        sketch = NLProofSketch(
            proof_steps=[NLProofStep(claim="a", reasoning="b")],
            assumptions=["n is a natural number"],
            key_lemmas=["Nat.add_comm"],
            overall_strategy="direct",
        )
        result = critic.audit_nl_proof(sketch, "theorem foo : True")

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == CritiqueIssueType.UNSTATED_HYPOTHESIS


class TestPipelineIntegration:
    def test_pipeline_with_nl_prover(self) -> None:
        """Verify NL stage runs before LemmaBreakdown when configured."""
        from agentic_research.pipelines.proof import ProofPipeline

        mock_llm = MagicMock()
        mock_repl = MagicMock()
        mock_search = MagicMock()

        mock_repl.try_automated_tactics.return_value = None
        mock_llm.complete.return_value = _mock_llm_response(json.dumps({
            "strategies": [],
            "proved": False,
            "needs_decomposition": True,
            "failure_reason": "no direct proof",
        }))
        mock_llm.extract_json.side_effect = lambda text: json.loads(text) if text.strip().startswith("{") else None

        mock_nl_prover = MagicMock()
        mock_nl_prover.generate_proof.return_value = (
            NLProofSketch.model_validate(json.loads(VALID_NL_PROOF_JSON)),
            TokenUsage(input_tokens=100, output_tokens=50),
        )

        pipeline = ProofPipeline(
            llm_client=mock_llm,
            lean_repl=mock_repl,
            lean_search=mock_search,
            nl_prover=mock_nl_prover,
            use_nl_proof_stage=True,
            use_claim_check=False,
            use_proof_critic=False,
            use_proof_detailer=False,
        )

        pipeline.run(
            lean_statement="theorem foo : True := by sorry",
            statement_nl="True is provable",
        )

        assert mock_nl_prover.generate_proof.call_count >= 1

    def test_pipeline_without_nl_prover(self) -> None:
        """Verify pipeline works without NL prover (backward compatible)."""
        from agentic_research.pipelines.proof import ProofPipeline

        mock_llm = MagicMock()
        mock_repl = MagicMock()
        mock_search = MagicMock()

        mock_repl.try_automated_tactics.return_value = "trivial"

        pipeline = ProofPipeline(
            llm_client=mock_llm,
            lean_repl=mock_repl,
            lean_search=mock_search,
            nl_prover=None,
            use_nl_proof_stage=True,
            use_claim_check=False,
        )

        result = pipeline.run(
            lean_statement="theorem foo : True",
            statement_nl="True is provable",
        )

        assert result.proved is True

    def test_pipeline_nl_stage_disabled(self) -> None:
        """Verify NL stage is skipped when use_nl_proof_stage=False."""
        from agentic_research.pipelines.proof import ProofPipeline

        mock_llm = MagicMock()
        mock_repl = MagicMock()
        mock_search = MagicMock()

        mock_repl.try_automated_tactics.return_value = "trivial"

        mock_nl_prover = MagicMock()

        pipeline = ProofPipeline(
            llm_client=mock_llm,
            lean_repl=mock_repl,
            lean_search=mock_search,
            nl_prover=mock_nl_prover,
            use_nl_proof_stage=False,
            use_claim_check=False,
        )

        result = pipeline.run(
            lean_statement="theorem foo : True",
            statement_nl="True is provable",
        )

        mock_nl_prover.generate_proof.assert_not_called()
        assert result.proved is True


class TestLemmaBreakdownNLContext:
    def test_format_nl_proof_context(self) -> None:
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        nl_ctx = {
            "overall_strategy": "induction on n",
            "assumptions": ["n >= 0"],
            "key_lemmas": ["Nat.add_comm"],
            "proof_steps": [
                {
                    "claim": "base case n=0",
                    "reasoning": "trivial",
                    "sub_claims": [],
                },
                {
                    "claim": "inductive step",
                    "reasoning": "assume P(k), show P(k+1)",
                    "sub_claims": ["apply IH"],
                },
            ],
        }

        formatted = LemmaBreakdown._format_nl_proof_context(nl_ctx)
        assert "induction on n" in formatted
        assert "n >= 0" in formatted
        assert "Nat.add_comm" in formatted
        assert "base case n=0" in formatted
        assert "inductive step" in formatted
        assert "apply IH" in formatted
        assert "guide your decomposition" in formatted

    def test_nl_context_passed_to_breakdown(self) -> None:
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(json.dumps({
            "lemmas": [
                {"node_id": "lemma_1", "statement_nl": "base case", "depends_on": []},
            ],
            "topological_order": ["lemma_1"],
        }))
        mock_llm.extract_json.side_effect = lambda text: json.loads(text)

        agent = LemmaBreakdown(llm_client=mock_llm)
        ctx = AgentContext(
            task="For all n, n + 0 = n",
            metadata={
                "statement_lean": "theorem foo : ∀ n, n + 0 = n",
                "failed_attempts": "None",
                "nl_proof_context": {
                    "overall_strategy": "induction",
                    "assumptions": [],
                    "key_lemmas": [],
                    "proof_steps": [
                        {"claim": "base", "reasoning": "trivial", "sub_claims": []},
                    ],
                },
            },
        )
        agent.run(ctx)

        call_args = mock_llm.complete.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "Validated Informal Proof Sketch" in user_msg
        assert "induction" in user_msg
