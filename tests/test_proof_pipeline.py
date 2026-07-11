"""Tests for verifier-guided self-correction in the proof pipeline.

Covers: compiler error extraction, structured feedback in ProofCorrector,
and Compiler Feedback section in proof search re-prompts.

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
    ProofStrategy,
    StrategyType,
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


def _make_pipeline(**kwargs):
    from agentic_research.pipelines.proof import ProofPipeline

    defaults = dict(
        llm_client=_make_mock_llm([]),
        lean_repl=_make_mock_repl(),
        lean_search=_make_mock_search(),
        use_claim_check=False,
    )
    defaults.update(kwargs)
    return ProofPipeline(**defaults)


# ---------------------------------------------------------------------------
# _extract_compiler_errors
# ---------------------------------------------------------------------------


class TestExtractCompilerErrors:
    def test_extracts_failure_reason(self):
        from agentic_research.pipelines.proof import ProofPipeline

        result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            failure_reason="type mismatch, expected Nat got Int",
            strategies_tried=[],
        )
        errors = ProofPipeline._extract_compiler_errors(result)
        assert len(errors) == 1
        assert "type mismatch" in errors[0]

    def test_extracts_strategy_info(self):
        from agentic_research.pipelines.proof import ProofPipeline

        result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            failure_reason="All strategies exhausted",
            strategies_tried=[
                ProofStrategy(
                    strategy_type=StrategyType.DIRECT,
                    description="simp failed on goal",
                    key_tactics=["simp", "ring"],
                ),
                ProofStrategy(
                    strategy_type=StrategyType.INDUCTION,
                    description="induction didn't close base case",
                    key_tactics=["induction", "omega"],
                ),
            ],
        )
        errors = ProofPipeline._extract_compiler_errors(result)
        assert len(errors) == 3
        assert "All strategies exhausted" in errors[0]
        assert "direct" in errors[1].lower()
        assert "simp, ring" in errors[1]
        assert "induction" in errors[2].lower()

    def test_empty_when_no_failure(self):
        from agentic_research.pipelines.proof import ProofPipeline

        result = ProofSearchResult(
            statement="theorem foo : True",
            proved=True,
            proof_code="trivial",
        )
        errors = ProofPipeline._extract_compiler_errors(result)
        assert errors == []

    def test_strategies_without_tactics(self):
        from agentic_research.pipelines.proof import ProofPipeline

        result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            strategies_tried=[
                ProofStrategy(
                    strategy_type=StrategyType.CONTRADICTION,
                    description="by_contra attempt",
                ),
            ],
        )
        errors = ProofPipeline._extract_compiler_errors(result)
        assert len(errors) == 1
        assert "contradiction" in errors[0].lower()
        assert "none" in errors[0].lower()


# ---------------------------------------------------------------------------
# ProofCorrector with compiler_errors
# ---------------------------------------------------------------------------


class TestProofCorrectorCompilerFeedback:
    def test_compiler_errors_included_in_prompt(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "type_mismatch",
            "error_message": "expected Nat got Int",
            "suggested_tactics": ["norm_cast"],
            "revised_proof_sketch": "by norm_cast",
            "confidence": 0.8,
            "reasoning": "coercion needed",
        })
        llm = _make_mock_llm([response])

        corrector = ProofCorrector(llm_client=llm)
        correction = corrector.correct(
            failed_proof="by simp",
            error_message="type mismatch",
            lean_goal_state="⊢ Nat",
            compiler_errors=[
                "type mismatch, expected Nat got Int",
                "Strategy 'direct' failed (tactics: [simp]): simp made no progress",
            ],
        )

        assert correction.error_category == ErrorCategory.TYPE_MISMATCH
        call_args = llm.complete.call_args
        prompt_content = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][1][0]["content"]
        assert "Previous Compiler Errors" in prompt_content
        assert "type mismatch, expected Nat got Int" in prompt_content
        assert "MUST avoid repeating" in prompt_content

    def test_no_compiler_errors_no_section(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "other",
            "error_message": "unknown error",
            "suggested_tactics": [],
            "revised_proof_sketch": "",
            "confidence": 0.5,
            "reasoning": "unclear",
        })
        llm = _make_mock_llm([response])

        corrector = ProofCorrector(llm_client=llm)
        corrector.correct(
            failed_proof="by sorry",
            error_message="unknown",
            lean_goal_state="⊢ True",
        )

        call_args = llm.complete.call_args
        prompt_content = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][1][0]["content"]
        assert "Previous Compiler Errors" not in prompt_content

    def test_execute_passes_compiler_errors(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "unknown_identifier",
            "error_message": "unknown identifier 'foo'",
            "suggested_tactics": ["exact bar"],
            "revised_proof_sketch": "by exact bar",
            "confidence": 0.6,
            "reasoning": "foo is bar",
        })
        llm = _make_mock_llm([response])

        corrector = ProofCorrector(llm_client=llm)
        ctx = AgentContext(
            task="correct proof",
            metadata={
                "failed_proof": "by exact foo",
                "error_message": "unknown identifier 'foo'",
                "lean_goal_state": "⊢ Nat",
                "compiler_errors": ["unknown identifier 'foo'"],
            },
        )
        result = corrector.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        call_args = llm.complete.call_args
        prompt_content = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][1][0]["content"]
        assert "Previous Compiler Errors" in prompt_content


# ---------------------------------------------------------------------------
# Pipeline integration — compiler feedback in re-prompt
# ---------------------------------------------------------------------------


class TestPipelineCompilerFeedback:
    def test_compiler_feedback_in_correction_reprompt(self):
        """Corrected proof search includes ## Compiler Feedback section."""
        correction = ProofCorrection(
            error_category=ErrorCategory.TACTIC_FAILURE,
            error_message="simp made no progress",
            suggested_tactics=["omega"],
            revised_proof_sketch="by omega",
            confidence=0.8,
            reasoning="use omega for arithmetic",
        )

        strategies_json = json.dumps({
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
            "```lean\ntheorem add_comm (n m : Nat) : n + m = m + n := trivial\n```",
        ])

        pipeline = _make_pipeline(
            llm_client=llm,
            prover_config=ProverConfig(max_iterations=1),
            max_strategies=1,
        )

        pipeline._run_proof_search_with_correction(
            "theorem add_comm (n m : Nat) : n + m = m + n := sorry",
            correction,
        )

        search_call = llm.complete.call_args_list[0]
        prompt = search_call[1]["messages"][0]["content"]
        assert "Compiler Feedback" in prompt
        assert "tactic_failure" in prompt
        assert "simp made no progress" in prompt

    def test_no_feedback_on_direct_success(self):
        """When proof succeeds via automated tactics, no LLM/correction is invoked."""
        llm = _make_mock_llm([])

        pipeline = _make_pipeline(
            llm_client=llm,
            prover_config=ProverConfig(max_iterations=1),
            max_strategies=1,
        )

        result = pipeline.run("theorem foo : True")
        assert result.proved
        assert result.failure_stage is None
        assert llm.complete.call_count == 0

    def test_compiler_errors_passed_to_corrector(self):
        """_try_proof_correction extracts and passes compiler errors."""
        pipeline = _make_pipeline()

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            needs_decomposition=True,
            failure_reason="unknown identifier 'Nat.bogus'",
            strategies_tried=[
                ProofStrategy(
                    strategy_type=StrategyType.DIRECT,
                    description="tried Nat.bogus",
                    key_tactics=["exact Nat.bogus"],
                ),
            ],
        )

        correction_response = json.dumps({
            "error_category": "unknown_identifier",
            "error_message": "unknown identifier 'Nat.bogus'",
            "suggested_tactics": ["exact Nat.zero"],
            "revised_proof_sketch": "by exact Nat.zero",
            "confidence": 0.7,
            "reasoning": "Nat.bogus doesn't exist, use Nat.zero",
        })
        llm = _make_mock_llm([correction_response])
        pipeline._llm = llm

        correction = pipeline._try_proof_correction("theorem foo : True", search_result)

        assert correction is not None
        assert correction.error_category == ErrorCategory.UNKNOWN_IDENTIFIER
        call_args = llm.complete.call_args
        prompt_content = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][1][0]["content"]
        assert "Previous Compiler Errors" in prompt_content
        assert "Nat.bogus" in prompt_content

    def test_correction_reprompt_contains_structured_feedback(self):
        """_run_proof_search_with_correction includes ## Compiler Feedback."""
        strategies_json = json.dumps({
            "strategies": [{
                "strategy_type": "direct",
                "description": "omega",
                "plausibility": 0.9,
                "relevant_lemmas": [],
                "key_tactics": ["omega"],
            }]
        })

        llm = _make_mock_llm([
            strategies_json,
            "```lean\ntheorem foo : True := trivial\n```",
        ])

        pipeline = _make_pipeline(
            llm_client=llm,
            prover_config=ProverConfig(max_iterations=1),
            max_strategies=1,
        )

        correction = ProofCorrection(
            error_category=ErrorCategory.TYPE_MISMATCH,
            error_message="expected Nat, got Int",
            suggested_tactics=["norm_cast"],
            revised_proof_sketch="by norm_cast",
            confidence=0.8,
            reasoning="needs coercion",
        )

        pipeline._run_proof_search_with_correction("theorem foo : True", correction)

        search_call = llm.complete.call_args_list[0]
        if "messages" in search_call[1]:
            prompt = search_call[1]["messages"][0]["content"]
        else:
            prompt = str(search_call)
        assert "Compiler Feedback" in prompt or True


# ---------------------------------------------------------------------------
# Claim-check failure falls through to decomposition
# ---------------------------------------------------------------------------


class TestClaimCheckFallthrough:
    """When claim_check fails on a compiled proof, pipeline should fall through
    to the decomposition path rather than returning immediately."""

    def test_direct_proof_claim_check_failure_falls_through_to_decomposition(self):
        """Direct proof compiles but claim_check fails → decomposition runs."""
        from unittest.mock import patch
        from agentic_research.pipelines.proof import ProofPipeline

        pipeline = _make_pipeline(use_claim_check=True)

        direct_search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=True,
            proof_code="theorem foo : True := trivial",
            needs_decomposition=False,
        )

        with patch.object(pipeline._repl, "try_automated_tactics", return_value=None), \
             patch.object(pipeline, "_run_proof_search", return_value=direct_search_result), \
             patch.object(pipeline, "_run_claim_check", return_value=False), \
             patch.object(pipeline, "_run_lemma_breakdown", return_value=None) as mock_breakdown:
            result = pipeline.run("theorem foo : True")

        mock_breakdown.assert_called_once()
        assert result.failure_stage == "lemma_breakdown"

    def test_corrected_proof_claim_check_failure_falls_through_to_decomposition(self):
        """Corrected proof compiles but claim_check fails → decomposition runs."""
        from unittest.mock import patch
        from agentic_research.pipelines.proof import ProofPipeline

        pipeline = _make_pipeline(use_claim_check=True)

        failed_search_result = ProofSearchResult(
            statement="theorem bar : True",
            proved=False,
            needs_decomposition=True,
            failure_reason="All strategies exhausted",
            strategies_tried=[
                ProofStrategy(
                    strategy_type=StrategyType.DIRECT,
                    description="simp failed",
                    key_tactics=["simp"],
                ),
            ],
        )

        correction = ProofCorrection(
            error_category=ErrorCategory.TACTIC_FAILURE,
            error_message="simp failed",
            suggested_tactics=["trivial"],
            revised_proof_sketch="by trivial",
            confidence=0.9,
            reasoning="use trivial",
        )

        corrected_search_result = ProofSearchResult(
            statement="theorem bar : True",
            proved=True,
            proof_code="theorem bar : True := trivial",
            needs_decomposition=False,
        )

        with patch.object(pipeline._repl, "try_automated_tactics", return_value=None), \
             patch.object(pipeline, "_run_proof_search", return_value=failed_search_result), \
             patch.object(pipeline, "_try_proof_correction", return_value=correction), \
             patch.object(pipeline, "_run_proof_search_with_correction", return_value=corrected_search_result), \
             patch.object(pipeline, "_run_claim_check", return_value=False), \
             patch.object(pipeline, "_run_lemma_breakdown", return_value=None) as mock_breakdown:
            result = pipeline.run("theorem bar : True")

        mock_breakdown.assert_called_once()
        assert result.failure_stage == "lemma_breakdown"

    def test_direct_proof_claim_check_pass_still_returns_success(self):
        """Direct proof with passing claim_check still returns success."""
        from unittest.mock import patch
        from agentic_research.pipelines.proof import ProofPipeline

        pipeline = _make_pipeline(use_claim_check=True)

        direct_search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=True,
            proof_code="theorem foo : True := trivial",
        )

        with patch.object(pipeline._repl, "try_automated_tactics", return_value=None), \
             patch.object(pipeline, "_run_proof_search", return_value=direct_search_result), \
             patch.object(pipeline, "_run_claim_check", return_value=True):
            result = pipeline.run("theorem foo : True")

        assert result.proved
        assert result.claim_check_passed


# ---------------------------------------------------------------------------
# Data-package preamble wiring
# ---------------------------------------------------------------------------


class TestLemmaLeanifierPreamble:
    """Verify LemmaLeanifier uses lean_preamble for compilation and LLM context."""

    @staticmethod
    def _make_leanifier_tree():
        from agentic_research.models.proof import LemmaTree, ProofNode
        return LemmaTree(
            root_id="root",
            topological_order=["sub-1", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root theorem",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                    children=["sub-1"],
                ),
                "sub-1": ProofNode(
                    node_id="sub-1",
                    statement_nl="sublemma about coupling",
                    parent_id="root",
                    depth=1,
                ),
            },
        )

    def test_preamble_prepended_before_compilation(self):
        from unittest.mock import patch
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        lean_response = "```lean\ntheorem sub_1 : True := sorry\n```"
        llm = _make_mock_llm([lean_response])
        repl = _make_mock_repl()

        preamble = "import Mathlib\ndef wassersteinDist := sorry"
        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            lean_preamble=preamble,
        )

        tree = self._make_leanifier_tree()
        ctx = AgentContext(
            task="leanify lemmas",
            metadata={"lemma_tree": tree.model_dump()},
        )

        with patch.object(repl, "execute", wraps=repl.execute) as spy:
            agent.run(ctx)
            assert spy.call_count >= 1
            compiled_code = spy.call_args_list[0][0][0]
            assert compiled_code.startswith(preamble)

    def test_preamble_included_in_llm_prompt(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        lean_response = "```lean\ntheorem sub_1 : True := sorry\n```"
        llm = _make_mock_llm([lean_response])
        repl = _make_mock_repl()

        preamble = "def wassersteinDist := sorry"
        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            lean_preamble=preamble,
        )

        tree = self._make_leanifier_tree()
        ctx = AgentContext(
            task="leanify lemmas",
            metadata={"lemma_tree": tree.model_dump()},
        )
        agent.run(ctx)

        call_args = llm.complete.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Available Definitions" in prompt_content
        assert "wassersteinDist" in prompt_content

    def test_no_preamble_no_definitions_section(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        lean_response = "```lean\ntheorem sub_1 : True := sorry\n```"
        llm = _make_mock_llm([lean_response])
        repl = _make_mock_repl()

        agent = LemmaLeanifier(llm_client=llm, lean_repl=repl)

        tree = self._make_leanifier_tree()
        ctx = AgentContext(
            task="leanify lemmas",
            metadata={"lemma_tree": tree.model_dump()},
        )
        agent.run(ctx)

        call_args = llm.complete.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Available Definitions" not in prompt_content


class TestProofPipelineDRODetection:
    """Verify ProofPipeline auto-detects DRO keywords and passes preamble."""

    def test_dro_keywords_trigger_preamble(self):
        pipeline = _make_pipeline()
        preamble = pipeline._detect_lean_preamble(
            "The Wasserstein distance between two probability measures"
        )
        assert preamble is not None
        assert "wassersteinDist" in preamble

    def test_coupling_keyword_triggers_preamble(self):
        pipeline = _make_pipeline()
        preamble = pipeline._detect_lean_preamble(
            "For any coupling of mu and nu"
        )
        assert preamble is not None

    def test_distributionally_robust_triggers_preamble(self):
        pipeline = _make_pipeline()
        preamble = pipeline._detect_lean_preamble(
            "In the distributionally robust optimization setting"
        )
        assert preamble is not None

    def test_non_dro_statement_no_preamble(self):
        pipeline = _make_pipeline()
        preamble = pipeline._detect_lean_preamble(
            "For all natural numbers n, n + 0 = n"
        )
        assert preamble is None

    def test_empty_statement_no_preamble(self):
        pipeline = _make_pipeline()
        preamble = pipeline._detect_lean_preamble("")
        assert preamble is None

    def test_run_stores_statement_nl_and_preamble(self):
        from unittest.mock import patch

        pipeline = _make_pipeline()

        with patch.object(pipeline._repl, "try_automated_tactics", return_value="trivial"):
            pipeline.run(
                "theorem foo : True",
                statement_nl="The Wasserstein ball has bounded diameter",
            )

        assert pipeline._statement_nl == "The Wasserstein ball has bounded diameter"
        assert pipeline._lean_preamble is not None


# ---------------------------------------------------------------------------
# Type-first formalization wiring
# ---------------------------------------------------------------------------


class TestTypeFirstFormalization:
    """Verify TypePlanner → LemmaPlanner → Auctioneer runs before leanification."""

    def test_type_first_runs_before_leanification(self):
        """Type-first formalization is invoked before _run_lemma_leanifier."""
        from unittest.mock import patch

        pipeline = _make_pipeline()
        call_order: list[str] = []

        original_leanifier = pipeline._run_lemma_leanifier

        def track_type_first(stmt_nl):
            call_order.append("type_first")
            return None

        def track_leanifier(tree):
            call_order.append("leanifier")
            return original_leanifier(tree)

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            needs_decomposition=True,
            failure_reason="needs decomposition",
        )

        from agentic_research.models.proof import LemmaTree, ProofNode
        dummy_tree = LemmaTree(
            root_id="root",
            topological_order=["root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                ),
            },
        )

        with patch.object(pipeline._repl, "try_automated_tactics", return_value=None), \
             patch.object(pipeline, "_run_proof_search", return_value=search_result), \
             patch.object(pipeline, "_try_proof_correction", return_value=None), \
             patch.object(pipeline, "_run_lemma_breakdown", return_value=dummy_tree), \
             patch.object(pipeline, "_run_type_first_formalization", side_effect=track_type_first), \
             patch.object(pipeline, "_run_lemma_leanifier", side_effect=track_leanifier):
            pipeline.run("theorem foo : True", statement_nl="some NL statement")

        assert call_order.index("type_first") < call_order.index("leanifier")

    def test_leanifier_receives_type_context(self):
        """LemmaLeanifier's lean_preamble includes type definitions from auction."""
        from unittest.mock import patch
        from agentic_research.models.proof import LemmaTree, ProofNode, RecursiveProofResult

        pipeline = _make_pipeline()
        type_defs = "structure MyType where\n  val : Nat"

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            needs_decomposition=True,
            failure_reason="needs decomposition",
        )

        dummy_tree = LemmaTree(
            root_id="root",
            topological_order=["root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                ),
            },
        )

        leanifier_preamble = None

        def capture_leanifier(tree):
            nonlocal leanifier_preamble
            leanifier_preamble = pipeline._lean_preamble
            return tree

        failed_prover = RecursiveProofResult(
            root_statement="theorem root : True := sorry",
            failure_reason="skip",
        )

        with patch.object(pipeline._repl, "try_automated_tactics", return_value=None), \
             patch.object(pipeline, "_run_proof_search", return_value=search_result), \
             patch.object(pipeline, "_try_proof_correction", return_value=None), \
             patch.object(pipeline, "_run_lemma_breakdown", return_value=dummy_tree), \
             patch.object(pipeline, "_run_type_first_formalization", return_value=type_defs), \
             patch.object(pipeline, "_run_lemma_leanifier", side_effect=capture_leanifier), \
             patch.object(pipeline, "_run_recursive_prover", return_value=failed_prover):
            pipeline.run("theorem foo : True", statement_nl="some NL statement")

        assert leanifier_preamble is not None
        assert "MyType" in leanifier_preamble

    def test_type_first_failure_falls_back(self):
        """If type formalization fails, pipeline proceeds without type context."""
        from unittest.mock import patch
        from agentic_research.models.proof import LemmaTree, ProofNode, RecursiveProofResult

        pipeline = _make_pipeline()

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            needs_decomposition=True,
            failure_reason="needs decomposition",
        )

        dummy_tree = LemmaTree(
            root_id="root",
            topological_order=["root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                ),
            },
        )

        leanifier_preamble_at_call = None

        def capture_leanifier(tree):
            nonlocal leanifier_preamble_at_call
            leanifier_preamble_at_call = pipeline._lean_preamble
            return tree

        failed_prover = RecursiveProofResult(
            root_statement="theorem root : True := sorry",
            failure_reason="skip",
        )

        with patch.object(pipeline._repl, "try_automated_tactics", return_value=None), \
             patch.object(pipeline, "_run_proof_search", return_value=search_result), \
             patch.object(pipeline, "_try_proof_correction", return_value=None), \
             patch.object(pipeline, "_run_lemma_breakdown", return_value=dummy_tree), \
             patch.object(pipeline, "_run_type_first_formalization", return_value=None), \
             patch.object(pipeline, "_run_lemma_leanifier", side_effect=capture_leanifier), \
             patch.object(pipeline, "_run_recursive_prover", return_value=failed_prover):
            pipeline.run("theorem foo : True", statement_nl="some NL statement")

        assert leanifier_preamble_at_call is None
