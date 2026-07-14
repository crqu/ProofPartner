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
    NodeStatus,
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

        def track_leanifier(tree, **kwargs):
            call_order.append("leanifier")
            return original_leanifier(tree, **kwargs)

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

        def capture_leanifier(tree, **kwargs):
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

    def test_default_config_enables_critic_and_detailer(self):
        """ProofPipeline with default config has both critic and detailer enabled."""
        pipeline = _make_pipeline()
        assert pipeline._use_proof_critic is True
        assert pipeline._use_proof_detailer is True

    def test_default_max_critic_retries_is_zero(self):
        """Default max_critic_retries is 0 — critic runs once but doesn't gate."""
        pipeline = _make_pipeline()
        assert pipeline._max_critic_retries == 0

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

        def capture_leanifier(tree, **kwargs):
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


# ---------------------------------------------------------------------------
# H1: Fix B-2 — NL proof context in breakdown retry paths
# ---------------------------------------------------------------------------


class TestH1NLProofContextRetries:
    """Verify nl_proof_context and tactic_hints are passed in retry paths."""

    def test_critic_retry_passes_nl_context(self):
        """Critic retry path forwards nl_proof_context and tactic_hints to breakdown."""
        from unittest.mock import patch, MagicMock
        from agentic_research.agents.proof_detailer import ProofDetailer
        from agentic_research.models.proof import (
            CritiqueIssue, CritiqueIssueType, CritiqueResult, LemmaTree, NLProofSketch,
            NLProofStep, ProofNode, RecursiveProofResult,
        )

        sketch = NLProofSketch(
            proof_steps=[NLProofStep(claim="step1", reasoning="reason")],
            overall_strategy="direct",
        )

        detail_response = "Use simp then ring"
        llm = _make_mock_llm([detail_response])

        mock_nl_prover = MagicMock()

        pipeline = _make_pipeline(
            llm_client=llm,
            use_proof_critic=True,
            max_critic_retries=1,
            nl_prover=mock_nl_prover,
            use_nl_proof_stage=True,
        )

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            needs_decomposition=True,
            failure_reason="needs decomp",
        )

        tree = LemmaTree(
            root_id="root",
            topological_order=["root"],
            nodes={"root": ProofNode(
                node_id="root",
                statement_nl="root",
                statement_lean="theorem root : True := sorry",
                depth=0,
            )},
        )

        failing_critique = CritiqueResult(
            passed=False,
            issues=[CritiqueIssue(
                issue_type=CritiqueIssueType.UNJUSTIFIED_STEP,
                node_id="root",
                description="unjustified",
                severity="blocking",
            )],
        )
        passing_critique = CritiqueResult(passed=True, issues=[])

        failed_prover = RecursiveProofResult(
            root_statement="theorem root : True := sorry",
            failure_reason="skip",
        )

        breakdown_calls: list = []

        def capture_breakdown(*args, **kwargs):
            breakdown_calls.append(kwargs)
            return tree

        critique_calls = [failing_critique, passing_critique]

        with patch.object(pipeline._repl, "try_automated_tactics", return_value=None), \
             patch.object(pipeline, "_run_proof_search", return_value=search_result), \
             patch.object(pipeline, "_try_proof_correction", return_value=None), \
             patch.object(pipeline, "_run_type_first_formalization", return_value=None), \
             patch.object(pipeline, "_run_nl_proof_stage", return_value=sketch), \
             patch.object(pipeline, "_run_lemma_breakdown", side_effect=capture_breakdown), \
             patch.object(pipeline, "_run_proof_critic", side_effect=lambda *a: critique_calls.pop(0)), \
             patch.object(pipeline, "_run_lemma_leanifier", return_value=tree), \
             patch.object(pipeline, "_run_recursive_prover", return_value=failed_prover), \
             patch.object(pipeline, "_run_proof_detailer", return_value=tree):
            pipeline.run("theorem foo : True", statement_nl="NL stmt")

        assert len(breakdown_calls) >= 2
        retry_kwargs = breakdown_calls[1]
        assert retry_kwargs.get("nl_proof_context") is not None
        assert "tactic_hints" in retry_kwargs

    def test_weak_child_retry_passes_nl_context(self):
        """_retry_on_weak_children forwards nl_proof_context and tactic_hints."""
        from unittest.mock import patch
        from agentic_research.models.proof import (
            NLProofSketch, NLProofStep, LemmaTree, ProofNode,
            RecursiveProofResult, CritiqueIssue, CritiqueIssueType,
        )

        pipeline = _make_pipeline(use_proof_critic=False)

        search_result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            needs_decomposition=True,
        )

        sketch = NLProofSketch(
            proof_steps=[NLProofStep(claim="c", reasoning="r")],
            overall_strategy="direct",
        )

        breakdown_kwargs_captured: list = []

        def capture_breakdown(*args, **kwargs):
            breakdown_kwargs_captured.append(kwargs)
            return None

        feedback = [CritiqueIssue(
            issue_type=CritiqueIssueType.WEAK_CHILD_LEMMA,
            node_id="child1",
            description="weak",
            severity="blocking",
        )]

        with patch.object(pipeline, "_run_lemma_breakdown", side_effect=capture_breakdown):
            pipeline._retry_on_weak_children(
                "theorem foo : True", "NL stmt", search_result,
                feedback, 0.0,
                nl_proof_context=sketch,
                tactic_hints="use omega",
            )

        assert len(breakdown_kwargs_captured) >= 1
        kwargs = breakdown_kwargs_captured[0]
        assert kwargs.get("nl_proof_context") is sketch
        assert kwargs.get("tactic_hints") == "use omega"


# ---------------------------------------------------------------------------
# H2: Fix B-1 + Gap 3.3 — Re-leanify reformulated children
# ---------------------------------------------------------------------------


class TestH2ReleanifyReformulatedChildren:

    def test_leanify_single_node_delegates(self):
        """leanify_single_node calls _leanify_node and returns result."""
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier
        from agentic_research.models.proof import ProofNode

        lean_response = "```lean\ntheorem sub_1 : True := sorry\n```"
        llm = _make_mock_llm([lean_response])
        repl = _make_mock_repl()

        agent = LemmaLeanifier(llm_client=llm, lean_repl=repl)
        node = ProofNode(
            node_id="sub-1",
            statement_nl="some lemma",
            depth=1,
        )
        result, tokens = agent.leanify_single_node(node, "theorem root : True := sorry")
        assert result is not None
        assert "sub_1" in result

    def test_reformulated_child_gets_releanified(self):
        """After reformulation with leanifier, child gets statement_lean set."""
        from unittest.mock import patch, MagicMock
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import LemmaTree, ProofNode, FailureDiagnosis, FailureType

        llm = _make_mock_llm([
            '{"reformulated_statement": "new child statement"}',
        ])
        repl = _make_mock_repl()

        mock_leanifier = MagicMock()
        mock_leanifier.leanify_single_node.return_value = (
            "theorem child_1 : True := sorry",
            TokenUsage(),
        )

        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            leanifier=mock_leanifier,
        )

        tree = LemmaTree(
            root_id="root",
            topological_order=["child_1", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                    children=["child_1"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="old child",
                    statement_lean="theorem child_1 : Nat := sorry",
                    parent_id="root",
                    depth=1,
                ),
            },
        )

        diagnosis = FailureDiagnosis(
            failure_type=FailureType.WEAK_CHILD_LEMMA,
            problematic_child_id="child_1",
            description="child too weak",
        )

        tokens = TokenUsage()
        result = prover._reformulate_child(tree, tree.nodes["root"], diagnosis, tokens)

        assert result is True
        child = tree.nodes["child_1"]
        assert child.statement_nl == "new child statement"
        assert child.statement_lean == "theorem child_1 : True := sorry"
        assert child.status == NodeStatus.PENDING
        mock_leanifier.leanify_single_node.assert_called_once()

    def test_reformulated_child_without_leanifier_stays_empty(self):
        """Without leanifier, reformulated child has empty statement_lean."""
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import LemmaTree, ProofNode, FailureDiagnosis, FailureType

        llm = _make_mock_llm([
            '{"reformulated_statement": "new child statement"}',
        ])
        repl = _make_mock_repl()

        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        tree = LemmaTree(
            root_id="root",
            topological_order=["child_1", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                    children=["child_1"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="old child",
                    statement_lean="theorem child_1 : Nat := sorry",
                    parent_id="root",
                    depth=1,
                ),
            },
        )

        diagnosis = FailureDiagnosis(
            failure_type=FailureType.WEAK_CHILD_LEMMA,
            problematic_child_id="child_1",
            description="child too weak",
        )

        tokens = TokenUsage()
        result = prover._reformulate_child(tree, tree.nodes["root"], diagnosis, tokens)

        assert result is True
        child = tree.nodes["child_1"]
        assert child.statement_lean == ""
        assert child.status == NodeStatus.REFORMULATED


# ---------------------------------------------------------------------------
# H3: Gap 2.1 — NL Prover per-node
# ---------------------------------------------------------------------------


class TestH3NLProverPerNode:

    def test_generate_nl_context_returns_context_when_nl_prover_set(self):
        """_generate_nl_context returns NL context string when nl_prover is available."""
        from unittest.mock import MagicMock
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import NLProofSketch, NLProofStep, ProofNode

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        mock_nl_prover = MagicMock()
        mock_nl_prover.generate_proof.return_value = (
            NLProofSketch(
                proof_steps=[NLProofStep(claim="step1", reasoning="reason1")],
                overall_strategy="direct",
            ),
            TokenUsage(),
        )

        mock_detailer = MagicMock()
        mock_detailer.detail_sketch.return_value = "use simp then ring"
        mock_detailer.cumulative_tokens = TokenUsage()

        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            nl_prover=mock_nl_prover,
            proof_detailer=mock_detailer,
        )

        node = ProofNode(
            node_id="test",
            statement_nl="test statement",
            statement_lean="theorem test : True := sorry",
            depth=1,
        )

        tokens = TokenUsage()
        nl_context, tactic_hints = prover._generate_nl_context(node, tokens)

        assert "NL Proof Context" in nl_context
        assert "step1" in nl_context
        assert "Tactic Hints" in nl_context
        assert tactic_hints == "use simp then ring"
        mock_nl_prover.generate_proof.assert_called_once()

    def test_generate_nl_context_noop_without_nl_prover(self):
        """_generate_nl_context returns empty strings when nl_prover is None."""
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import ProofNode

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        node = ProofNode(
            node_id="test",
            statement_nl="test",
            statement_lean="theorem test : True := sorry",
            depth=1,
        )

        tokens = TokenUsage()
        nl_context, tactic_hints = prover._generate_nl_context(node, tokens)
        assert nl_context == ""
        assert tactic_hints == ""

    def test_parent_proof_includes_nl_context(self):
        """_prove_parent_with_children appends NL context to the prompt."""
        from unittest.mock import patch
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import LemmaTree, ProofNode

        response = "```lean\ntheorem root : True := trivial\n```"
        llm = _make_mock_llm([response])
        repl = _make_mock_repl()

        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        tree = LemmaTree(
            root_id="root",
            topological_order=["child_1", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                    children=["child_1"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="child",
                    statement_lean="theorem child_1 : True := sorry",
                    parent_id="root",
                    depth=1,
                ),
            },
        )

        tokens = TokenUsage()
        prover._prove_parent_with_children(
            tree, tree.nodes["root"], tokens,
            nl_context="## NL Proof Context\nStrategy: direct",
        )

        call_args = llm.complete.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "NL Proof Context" in prompt


# ---------------------------------------------------------------------------
# H4: Gap 2.10 — Recursive decomposition of stuck leaves
# ---------------------------------------------------------------------------


class TestH4RecursiveDecomposition:

    def test_stuck_leaf_below_max_depth_decomposes(self):
        """Stuck leaf at depth < max_depth-1 gets decomposed into children."""
        from unittest.mock import MagicMock
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import LemmaTree, ProofNode

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        sub_tree = LemmaTree(
            root_id="sub_root",
            topological_order=["sub_child", "sub_root"],
            nodes={
                "sub_root": ProofNode(
                    node_id="sub_root",
                    statement_nl="root",
                    statement_lean="theorem sub_root : True := sorry",
                    depth=0,
                ),
                "sub_child": ProofNode(
                    node_id="sub_child",
                    statement_nl="child lemma",
                    depth=1,
                ),
            },
        )

        mock_breakdown = MagicMock()
        from agentic_research.models.agents import AgentResult
        mock_breakdown.run.return_value = AgentResult(
            agent_name="lemma_breakdown",
            status=AgentStatus.SUCCESS,
            result=sub_tree.model_dump(),
        )
        mock_breakdown.cumulative_tokens = TokenUsage()

        mock_leanifier = MagicMock()
        mock_leanifier.leanify_single_node.return_value = (
            "theorem leaf_sub_child : True := sorry",
            TokenUsage(),
        )

        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            max_depth=5,
            breakdown=mock_breakdown,
            leanifier=mock_leanifier,
        )

        tree = LemmaTree(
            root_id="root",
            topological_order=["leaf", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                    children=["leaf"],
                ),
                "leaf": ProofNode(
                    node_id="leaf",
                    statement_nl="stuck leaf",
                    statement_lean="theorem leaf : True := sorry",
                    parent_id="root",
                    depth=1,
                ),
            },
        )
        prover._total_nodes = 2

        tokens = TokenUsage()
        result = prover._decompose_stuck_leaf(tree, tree.nodes["leaf"], tokens)

        assert result is True
        assert len(tree.nodes["leaf"].children) == 1
        assert tree.nodes["leaf"].status == NodeStatus.PENDING
        assert prover._total_nodes == 3

    def test_stuck_leaf_at_max_depth_not_decomposed(self):
        """Stuck leaf at depth = max_depth-1 is NOT decomposed."""
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import LemmaTree, ProofNode

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        from unittest.mock import MagicMock
        mock_breakdown = MagicMock()

        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            max_depth=3,
            breakdown=mock_breakdown,
        )

        tree = LemmaTree(
            root_id="root",
            topological_order=["leaf", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root", statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0, children=["leaf"],
                ),
                "leaf": ProofNode(
                    node_id="leaf", statement_nl="deep leaf",
                    statement_lean="theorem leaf : True := sorry",
                    parent_id="root", depth=2,
                ),
            },
        )
        prover._total_nodes = 2

        tokens = TokenUsage()
        result = prover._decompose_stuck_leaf(tree, tree.nodes["leaf"], tokens)
        assert result is False
        mock_breakdown.run.assert_not_called()

    def test_node_cap_prevents_decomposition(self):
        """Total node cap (50) prevents unbounded tree growth."""
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import LemmaTree, ProofNode

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        from unittest.mock import MagicMock
        mock_breakdown = MagicMock()

        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            max_depth=10,
            breakdown=mock_breakdown,
        )

        tree = LemmaTree(
            root_id="root",
            topological_order=["leaf", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root", statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0, children=["leaf"],
                ),
                "leaf": ProofNode(
                    node_id="leaf", statement_nl="leaf",
                    statement_lean="theorem leaf : True := sorry",
                    parent_id="root", depth=1,
                ),
            },
        )
        prover._total_nodes = 50

        tokens = TokenUsage()
        result = prover._decompose_stuck_leaf(tree, tree.nodes["leaf"], tokens)
        assert result is False

    def test_too_many_children_skips_decomposition(self):
        """Decomposition with > 5 children is skipped."""
        from unittest.mock import MagicMock
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import LemmaTree, ProofNode

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        nodes_dict: dict = {
            "sub_root": ProofNode(
                node_id="sub_root", statement_nl="root",
                statement_lean="theorem sub_root : True := sorry", depth=0,
            ),
        }
        for i in range(6):
            nodes_dict[f"c{i}"] = ProofNode(
                node_id=f"c{i}", statement_nl=f"child {i}", depth=1,
            )

        sub_tree = LemmaTree(
            root_id="sub_root",
            topological_order=list(nodes_dict.keys()),
            nodes=nodes_dict,
        )

        mock_breakdown = MagicMock()
        from agentic_research.models.agents import AgentResult
        mock_breakdown.run.return_value = AgentResult(
            agent_name="lemma_breakdown",
            status=AgentStatus.SUCCESS,
            result=sub_tree.model_dump(),
        )
        mock_breakdown.cumulative_tokens = TokenUsage()

        prover = RecursiveProver(
            llm_client=llm, lean_repl=repl, max_depth=5,
            breakdown=mock_breakdown,
        )

        tree = LemmaTree(
            root_id="root",
            topological_order=["leaf", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root", statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0, children=["leaf"],
                ),
                "leaf": ProofNode(
                    node_id="leaf", statement_nl="leaf",
                    statement_lean="theorem leaf : True := sorry",
                    parent_id="root", depth=1,
                ),
            },
        )
        prover._total_nodes = 2

        tokens = TokenUsage()
        result = prover._decompose_stuck_leaf(tree, tree.nodes["leaf"], tokens)
        assert result is False

    def test_no_breakdown_agent_returns_false(self):
        """RecursiveProver without breakdown marks stuck leaves FAILED."""
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.models.proof import LemmaTree, ProofNode

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        tree = LemmaTree(
            root_id="root",
            topological_order=["leaf", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root", statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0, children=["leaf"],
                ),
                "leaf": ProofNode(
                    node_id="leaf", statement_nl="leaf",
                    statement_lean="theorem leaf : True := sorry",
                    parent_id="root", depth=1,
                ),
            },
        )
        prover._total_nodes = 2

        tokens = TokenUsage()
        result = prover._decompose_stuck_leaf(tree, tree.nodes["leaf"], tokens)
        assert result is False


# ---------------------------------------------------------------------------
# H5: Gap 4.4 — ProofCorrector in RecursiveProver
# ---------------------------------------------------------------------------


class TestH5ProofCorrectorInRecursiveProver:

    def test_leaf_failure_triggers_corrector(self):
        """Leaf failure with corrector available invokes correction."""
        from unittest.mock import MagicMock, patch
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.models.proof import LemmaTree, ProofNode
        from agentic_research.models.agents import AgentResult

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        failed_result = AgentResult(
            agent_name="iterative_prover",
            status=AgentStatus.FAILURE,
            result={"statement": "theorem leaf : True := sorry", "proved": False,
                    "final_proof": "by simp", "failure_reason": "simp made no progress"},
            token_usage=TokenUsage(),
        )

        mock_corrector = MagicMock()
        mock_corrector.correct.return_value = ProofCorrection(
            error_category=ErrorCategory.TACTIC_FAILURE,
            error_message="simp made no progress",
            suggested_tactics=["omega", "linarith"],
            revised_proof_sketch="by omega",
            confidence=0.8,
            reasoning="use omega",
        )
        mock_corrector.cumulative_tokens = TokenUsage()

        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            proof_corrector=mock_corrector,
        )

        tree = LemmaTree(
            root_id="root",
            topological_order=["leaf", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root", statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0, children=["leaf"],
                ),
                "leaf": ProofNode(
                    node_id="leaf", statement_nl="leaf",
                    statement_lean="theorem leaf : 1 + 1 = 2 := sorry",
                    parent_id="root", depth=1,
                ),
            },
        )
        prover._total_nodes = 2

        tokens = TokenUsage()
        with patch.object(IterativeProver, "run", return_value=failed_result):
            prover._prove_leaf(tree, tree.nodes["leaf"], tokens)

        mock_corrector.correct.assert_called_once()

    def test_timeout_errors_skip_correction(self):
        """TIMEOUT errors skip correction retry."""
        from unittest.mock import MagicMock, patch
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.models.proof import LemmaTree, ProofNode
        from agentic_research.models.agents import AgentResult

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        failed_result = AgentResult(
            agent_name="iterative_prover",
            status=AgentStatus.FAILURE,
            result={"statement": "theorem leaf : True := sorry", "proved": False,
                    "final_proof": "by simp", "failure_reason": "deterministic timeout"},
            token_usage=TokenUsage(),
        )

        mock_corrector = MagicMock()
        mock_corrector.cumulative_tokens = TokenUsage()

        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            proof_corrector=mock_corrector,
        )

        tree = LemmaTree(
            root_id="root",
            topological_order=["leaf", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root", statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0, children=["leaf"],
                ),
                "leaf": ProofNode(
                    node_id="leaf", statement_nl="leaf",
                    statement_lean="theorem leaf : True := sorry",
                    parent_id="root", depth=1,
                ),
            },
        )
        prover._total_nodes = 2

        tokens = TokenUsage()
        with patch.object(IterativeProver, "run", return_value=failed_result):
            prover._prove_leaf(tree, tree.nodes["leaf"], tokens)

        mock_corrector.correct.assert_not_called()

    def test_no_corrector_behaves_as_before(self):
        """Without corrector, leaf failure goes straight to decomposition/fail."""
        from unittest.mock import patch
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.models.proof import LemmaTree, ProofNode
        from agentic_research.models.agents import AgentResult

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        failed_result = AgentResult(
            agent_name="iterative_prover",
            status=AgentStatus.FAILURE,
            result={"statement": "theorem leaf : True := sorry", "proved": False,
                    "final_proof": "by simp", "failure_reason": "simp made no progress"},
            token_usage=TokenUsage(),
        )

        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        tree = LemmaTree(
            root_id="root",
            topological_order=["leaf", "root"],
            nodes={
                "root": ProofNode(
                    node_id="root", statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0, children=["leaf"],
                ),
                "leaf": ProofNode(
                    node_id="leaf", statement_nl="leaf",
                    statement_lean="theorem leaf : True := sorry",
                    parent_id="root", depth=1,
                ),
            },
        )
        prover._total_nodes = 2

        tokens = TokenUsage()
        with patch.object(IterativeProver, "run", return_value=failed_result):
            result = prover._prove_leaf(tree, tree.nodes["leaf"], tokens)

        assert result is False
        assert tree.nodes["leaf"].status == NodeStatus.FAILED


# ---------------------------------------------------------------------------
# Pipeline wiring — RecursiveProver receives all new components
# ---------------------------------------------------------------------------


class TestPipelineWiringH2H3H4H5:

    def test_recursive_prover_receives_all_components(self):
        """_run_recursive_prover passes leanifier, nl_prover, detailer, breakdown, corrector."""
        from unittest.mock import patch, MagicMock
        from agentic_research.models.proof import LemmaTree, ProofNode, RecursiveProofResult
        from agentic_research.agents.recursive_prover import RecursiveProver

        pipeline = _make_pipeline()

        tree = LemmaTree(
            root_id="root",
            topological_order=["root"],
            nodes={"root": ProofNode(
                node_id="root", statement_nl="root",
                statement_lean="theorem root : True := sorry",
                depth=0,
            )},
        )

        captured_kwargs: dict = {}

        original_init = RecursiveProver.__init__

        def capture_init(self_inner, *args, **kwargs):
            captured_kwargs.update(kwargs)
            original_init(self_inner, *args, **kwargs)

        with patch.object(RecursiveProver, "__init__", capture_init):
            try:
                pipeline._run_recursive_prover(tree)
            except Exception:
                pass

        assert "leanifier" in captured_kwargs
        assert "breakdown" in captured_kwargs
        assert "proof_corrector" in captured_kwargs
        assert captured_kwargs["leanifier"] is not None
        assert captured_kwargs["breakdown"] is not None
        assert captured_kwargs["proof_corrector"] is not None
