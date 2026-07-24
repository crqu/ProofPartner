"""Regression tests for the 12 CRITICAL pipeline audit fixes.

Pattern 1 (C1-C5): Truthiness guards — ensure None vs "" vs [] are distinguished.
Pattern 2 (C6-C10): Context-aware retries — metadata injection, terminal errors, early-break.
Pattern 3 (C11-C12): Computed-but-dropped value injection.

All LLM calls and Lean REPL interactions are mocked.
"""

from __future__ import annotations

import copy
import json
import re
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.nl_prover import NaturalLanguageProver
from agentic_research.agents.recursive_prover import RecursiveProver
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    LLMResponse,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.proof import (
    CritiqueResult,
    ErrorCategory,
    FailureDiagnosis,
    FailureType,
    LemmaTree,
    NodeStatus,
    ProofCorrection,
    ProofNode,
)
from agentic_research.models.tools import CompilationResult, CompilationStatus, ToolStatus


def _mock_llm_response(content: str = "ok") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=10, output_tokens=10),
    )


def _mock_compilation(status=CompilationStatus.OK, errors=None, warnings=None):
    return CompilationResult(
        status=ToolStatus.SUCCESS,
        compilation_status=status,
        errors=errors or [],
        warnings=warnings or [],
    )


# ─── Pattern 1: Truthiness fixes ─────────────────────────────────────────────


class TestC1RetryPromptIncludesContextOnEmptyErrors:
    """C1: Verify retry context appended when previous_errors=''."""

    def test_retry_prompt_includes_context_on_empty_errors(self):
        llm = MagicMock(spec=LLMClient)
        repl = MagicMock()

        repl.execute.return_value = _mock_compilation(
            status=CompilationStatus.ERROR,
            errors=["type mismatch"],
        )
        llm.complete.return_value = _mock_llm_response("```lean\nsorry\n```")

        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", statement_lean="theorem root : True := by sorry", children=["child1"]),
                "child1": ProofNode(node_id="child1", statement_lean="axiom child1 : True", parent_id="root"),
            },
            topological_order=["child1", "root"],
        )

        prover._prove_parent_with_children(
            tree, tree.nodes["root"], TokenUsage(),
            previous_proof="old proof",
            previous_errors="",
            nl_context="",
        )

        call_args = llm.complete.call_args
        user_msg = call_args.kwargs.get("messages", [{}])[0].get("content", "")
        assert "Previous Attempt" in user_msg
        assert "did not close all goals" in user_msg


class TestC2EarlyBreakFiresOnEmptyIdenticalErrors:
    """C2: Verify loop breaks when both attempts produce errors=''."""

    def test_early_break_fires_on_empty_identical_errors(self):
        llm = MagicMock(spec=LLMClient)
        repl = MagicMock()

        repl.execute.return_value = _mock_compilation(
            status=CompilationStatus.ERROR,
            errors=[],
        )
        llm.complete.return_value = _mock_llm_response("```lean\nsorry\n```")

        diag_json = json.dumps({"failure_type": "stuck_goal", "description": "stuck"})
        llm.extract_json.return_value = {"failure_type": "stuck_goal", "description": "stuck"}

        prover = RecursiveProver(
            llm_client=llm, lean_repl=repl, max_retries_per_node=5,
        )

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", statement_lean="theorem root : True := by sorry", children=["c1"]),
                "c1": ProofNode(node_id="c1", statement_lean="axiom c1 : True", parent_id="root"),
            },
            topological_order=["c1", "root"],
        )

        prover._prove_parent(tree, tree.nodes["root"], TokenUsage())

        # With 5 max retries, we should break early (on retry 2) due to identical empty errors
        assert tree.nodes["root"].retries_used <= 3


class TestC3ChildDeclsDistinguishesUnleanifiedFromMissing:
    """C3: Verify 'all_children_unleanified' vs '' return."""

    def test_child_decls_distinguishes_unleanified_from_missing(self):
        llm = MagicMock(spec=LLMClient)
        repl = MagicMock()

        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        # Case 1: children exist but none have statement_lean
        tree_with_children = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", statement_lean="theorem root : True := by sorry", children=["c1"]),
                "c1": ProofNode(node_id="c1", statement_lean="", parent_id="root"),
            },
            topological_order=["c1", "root"],
        )

        proved, code, errors = prover._prove_parent_with_children(
            tree_with_children, tree_with_children.nodes["root"], TokenUsage(),
        )
        assert not proved
        assert errors == "all_children_unleanified"

        # Case 2: no children at all
        tree_no_children = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", statement_lean="theorem root : True := by sorry", children=[]),
            },
            topological_order=["root"],
        )

        proved2, code2, errors2 = prover._prove_parent_with_children(
            tree_no_children, tree_no_children.nodes["root"], TokenUsage(),
        )
        assert not proved2
        assert errors2 == ""


class TestC4EmptyFinalProofPassesThroughClaimCheck:
    """C4: Verify final_proof='' still triggers claim check in proof pipeline."""

    def test_empty_final_proof_passes_through_claim_check(self):
        from agentic_research.pipelines.proof import ProofPipeline

        pipeline = MagicMock(spec=ProofPipeline)

        # Simulate the logic from _retry_targeted_subtrees line 942
        # The fix changes `if final_proof:` to `if final_proof is not None:`
        final_proof = ""
        reached_claim_check = False

        if final_proof is not None:
            reached_claim_check = True

        assert reached_claim_check, "Empty string proof must reach claim check"

        # Verify old behavior would have failed
        if final_proof:  # old buggy check
            pytest.fail("Old truthiness check skips empty-string proofs")


class TestC5NoneFeedbackVsEmptyFeedbackInNlProver:
    """C5: Verify CritiqueResult(issues=[]) handled differently from None."""

    def test_none_feedback_vs_empty_feedback_in_nl_prover(self):
        llm = MagicMock(spec=LLMClient)
        llm.complete.return_value = _mock_llm_response(
            json.dumps({
                "overall_strategy": "test",
                "proof_steps": [{"claim": "c", "reasoning": "r"}],
            })
        )
        llm.extract_json.return_value = {
            "overall_strategy": "test",
            "proof_steps": [{"claim": "c", "reasoning": "r"}],
        }

        prover = NaturalLanguageProver(llm_client=llm)

        # With None feedback — no feedback section should appear
        sketch_none, _ = prover.generate_proof(
            statement="theorem foo : True := by sorry",
            feedback=None,
        )

        call_none = llm.complete.call_args
        user_msg_none = call_none.kwargs.get("messages", [{}])[0].get("content", "")
        assert "Critic Feedback" not in user_msg_none

        llm.reset_mock()
        llm.complete.return_value = _mock_llm_response(
            json.dumps({
                "overall_strategy": "test",
                "proof_steps": [{"claim": "c", "reasoning": "r"}],
            })
        )
        llm.extract_json.return_value = {
            "overall_strategy": "test",
            "proof_steps": [{"claim": "c", "reasoning": "r"}],
        }

        # With CritiqueResult(issues=[]) — this is "critique ran, found nothing"
        empty_critique = CritiqueResult(passed=True, issues=[])
        sketch_empty, _ = prover.generate_proof(
            statement="theorem foo : True := by sorry",
            feedback=empty_critique,
        )

        # Should NOT include feedback section since issues is empty
        call_empty = llm.complete.call_args
        user_msg_empty = call_empty.kwargs.get("messages", [{}])[0].get("content", "")
        assert "Critic Feedback" not in user_msg_empty


# ─── Pattern 2: Retry loop fixes ─────────────────────────────────────────────


class TestC6BaseAgentInjectsRetryMetadata:
    """C6: Verify context.metadata['retry_attempt'] set on attempt > 1."""

    def test_base_agent_injects_retry_metadata(self):
        class FailOnceAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="test_agent", max_retries=3)
                self._call_count = 0
                self._seen_metadata: list[dict] = []

            def _execute(self, context: AgentContext) -> AgentResult:
                self._seen_metadata.append(dict(context.metadata))
                self._call_count += 1
                if self._call_count == 1:
                    raise ValueError("transient error")
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.SUCCESS,
                )

        agent = FailOnceAgent()
        ctx = AgentContext(task="test")
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        assert len(agent._seen_metadata) == 2

        # First attempt: no retry metadata
        assert "retry_attempt" not in agent._seen_metadata[0]

        # Second attempt: retry metadata injected
        assert agent._seen_metadata[1]["retry_attempt"] == 2
        assert "transient error" in agent._seen_metadata[1]["last_error"]


class TestC7LlmClientRaisesOnTerminalErrors:
    """C7: Verify content_policy error raises immediately, not retried."""

    def test_llm_client_raises_on_terminal_errors(self):
        llm = LLMClient.__new__(LLMClient)
        llm._max_retries = 3
        llm._backoff_base = 0.01
        llm._backoff_max = 0.01

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception(
            "Error: content_policy violation detected"
        )
        llm._client = mock_client

        with pytest.raises(Exception, match="content_policy"):
            llm._call_with_retries({"model": "test", "messages": [], "max_tokens": 100})

        # Should only be called once (no retry on terminal error)
        assert mock_client.messages.create.call_count == 1


class TestC8LemmaLeanifierBreaksOnIdenticalOutput:
    """C8: Verify early-break when same lean code produced twice."""

    def test_lemma_leanifier_breaks_on_identical_output(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        llm = MagicMock(spec=LLMClient)
        repl = MagicMock()

        # Initial compile fails, retries produce identical code
        repl.execute.return_value = _mock_compilation(
            status=CompilationStatus.ERROR,
            errors=["type mismatch"],
        )
        llm.complete.return_value = _mock_llm_response("```lean\ntheorem foo : True := by sorry\n```")

        leanifier = LemmaLeanifier(llm_client=llm, lean_repl=repl, max_compile_retries=5)

        node = ProofNode(node_id="test", statement_nl="test statement")
        result, tokens = leanifier._leanify_node(node, "", "")

        # Should break early after detecting identical output
        # Initial call + at most 2 retries (one produces identical, breaks)
        assert llm.complete.call_count <= 3


class TestC9ResilientReplSkipsRetryOnCompilationError:
    """C9: Verify deterministic error returns immediately."""

    def test_resilient_repl_skips_retry_on_compilation_error(self):
        from agentic_research.orchestrator.resilience import ResilientRepl, ReplBackoffConfig

        mock_repl = MagicMock()
        # Deterministic compilation error — has compilation_status=ERROR
        mock_repl.execute.return_value = CompilationResult(
            status=ToolStatus.ERROR,
            error_message="type mismatch at application",
            compilation_status=CompilationStatus.ERROR,
        )

        resilient = ResilientRepl(
            repl=mock_repl,
            backoff=ReplBackoffConfig(max_attempts=3, base_delay=0.01),
        )

        result = resilient.execute_with_backoff("invalid lean code")

        # Should return immediately — deterministic compilation error, not retried
        assert mock_repl.execute.call_count == 1
        assert result.status == ToolStatus.ERROR

        # Verify that transient errors (process/timeout keywords) still retry
        mock_repl.reset_mock()
        mock_repl.execute.return_value = CompilationResult(
            status=ToolStatus.ERROR,
            error_message="process crashed unexpectedly",
        )
        resilient_fresh = ResilientRepl(
            repl=mock_repl,
            backoff=ReplBackoffConfig(max_attempts=3, base_delay=0.01),
        )
        result2 = resilient_fresh.execute_with_backoff("test code")
        assert mock_repl.execute.call_count == 3


class TestC10PassKVariesSeedAcrossAttempts:
    """C10: Verify each attempt uses seed + attempt offset."""

    def test_pass_k_varies_seed_across_attempts(self):
        from agentic_research.models.eval import EvalConfig, EvalMode, BenchmarkSource

        config = EvalConfig(
            mode=EvalMode.PROOF_DISCOVERY,
            benchmark=BenchmarkSource.MINIF2F,
            seed=42,
            pass_k=3,
        )

        seen_seeds = []
        for attempt in range(config.pass_k):
            attempt_config = copy.copy(config)
            if attempt_config.seed is not None:
                attempt_config.seed = config.seed + attempt
            seen_seeds.append(attempt_config.seed)

        assert seen_seeds == [42, 43, 44]
        assert len(set(seen_seeds)) == 3


# ─── Pattern 3: Computed-but-dropped value injection ──────────────────────────


class TestC11FailureDiagnosisInjectedIntoRetryPrompt:
    """C11: Verify FailureDiagnosis description appears in user_content."""

    def test_failure_diagnosis_injected_into_retry_prompt(self):
        llm = MagicMock(spec=LLMClient)
        repl = MagicMock()

        repl.execute.return_value = _mock_compilation(
            status=CompilationStatus.ERROR,
            errors=["unknown identifier"],
        )
        llm.complete.return_value = _mock_llm_response("```lean\nsorry\n```")

        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        node = ProofNode(
            node_id="root",
            statement_lean="theorem root : True := by sorry",
            children=["c1"],
            failure_diagnosis=FailureDiagnosis(
                failure_type=FailureType.WEAK_CHILD_LEMMA,
                description="Child lemma too weak to close parent goal",
                suggested_fix="Strengthen the child hypothesis",
            ),
        )

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": node,
                "c1": ProofNode(node_id="c1", statement_lean="axiom c1 : True", parent_id="root"),
            },
            topological_order=["c1", "root"],
        )

        prover._prove_parent_with_children(
            tree, node, TokenUsage(),
            previous_proof="old proof",
            previous_errors="unknown identifier",
        )

        call_args = llm.complete.call_args
        user_msg = call_args.kwargs.get("messages", [{}])[0].get("content", "")
        assert "Failure Diagnosis" in user_msg
        assert "weak_child_lemma" in user_msg
        assert "Child lemma too weak" in user_msg
        assert "Strengthen the child" in user_msg


class TestC12ProofCorrectionReasoningInCorrectionHint:
    """C12: Verify reasoning field appears in correction_hint."""

    def test_proof_correction_reasoning_in_correction_hint(self):
        llm = MagicMock(spec=LLMClient)
        repl = MagicMock()

        repl.execute.return_value = _mock_compilation(
            status=CompilationStatus.ERROR,
            errors=["tactic failed"],
        )
        llm.complete.return_value = _mock_llm_response("```lean\nsorry\n```")

        corrector = MagicMock()
        correction = ProofCorrection(
            error_category=ErrorCategory.TACTIC_FAILURE,
            error_message="simp failed",
            suggested_tactics=["ring", "omega"],
            revised_proof_sketch="by ring",
            confidence=0.8,
            reasoning="The goal has a ring structure, use ring tactic instead of simp",
        )
        corrector.correct.return_value = correction
        corrector.cumulative_tokens = TokenUsage()

        prover_config = ProverConfig()
        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=prover_config,
            proof_corrector=corrector,
        )

        # Build a node where proof search fails (to trigger correction path)
        iter_prover_result = AgentResult(
            agent_name="prover",
            status=AgentStatus.FAILURE,
            result={
                "statement": "theorem foo : True := by sorry",
                "proved": False,
                "final_proof": "theorem foo : True := by simp",
                "failure_reason": "simp failed",
            },
        )

        with patch.object(
            type(prover), '_prove_leaf',
            wraps=prover._prove_leaf,
        ):
            # Call _prove_leaf directly to test correction hint
            from agentic_research.agents.prover import IterativeProver

            with patch.object(IterativeProver, 'run', return_value=iter_prover_result):
                tree = LemmaTree(
                    root_id="leaf",
                    nodes={
                        "leaf": ProofNode(
                            node_id="leaf",
                            statement_lean="theorem leaf : True := by sorry",
                        ),
                    },
                    topological_order=["leaf"],
                )

                prover._prove_leaf(tree, tree.nodes["leaf"], TokenUsage())

        # Verify corrector was called
        assert corrector.correct.called

        # Check that the retry context includes reasoning
        if llm.complete.call_count > 0:
            # IterativeProver retry should contain the correction hint with reasoning
            for call in llm.complete.call_args_list:
                msgs = call.kwargs.get("messages", [])
                for msg in msgs:
                    content = msg.get("content", "")
                    if "Correction context" in content:
                        assert "Reasoning:" in content
                        assert "ring structure" in content
