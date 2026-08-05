"""Tests for H1: Fix context starvation and blind retries.

Covers preamble propagation (Phase 1) and error context quality (Phase 2).
All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    LLMResponse,
    ProofAttempt,
    ProofAttemptStatus,
    ProverConfig,
    ProverResult,
    TokenUsage,
)
from agentic_research.models.proof import (
    ErrorCategory,
    LemmaTree,
    NodeStatus,
    ProofCorrection,
    ProofNode,
    ProofSearchResult,
    ProofStrategy,
    StrategyType,
)


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
# Phase 1: Preamble Propagation
# ---------------------------------------------------------------------------


class TestDetectLeanPreamble:
    """#120 — Generic Mathlib preamble (no DRO logic)."""

    def test_with_lake_project(self):
        pipeline = _make_pipeline()
        with patch.object(pipeline._repl, "has_lake_project", return_value=True):
            preamble = pipeline._detect_lean_preamble("any math statement")
        assert preamble is not None
        assert "import Mathlib" in preamble
        assert "import Aesop" in preamble
        assert "maxHeartbeats" in preamble

    def test_without_lake_project(self):
        pipeline = _make_pipeline()
        preamble = pipeline._detect_lean_preamble("any math statement")
        assert preamble is None

    def test_no_dro_keywords_in_implementation(self):
        from agentic_research.pipelines.proof import ProofPipeline

        assert not hasattr(ProofPipeline, "_DRO_KEYWORDS")


class TestLeanReplHasLakeProject:
    """Verify LeanRepl exposes has_lake_project()."""

    def test_mock_backend_returns_false(self):
        repl = _make_mock_repl()
        assert repl.has_lake_project() is False

    def test_subprocess_backend_delegates(self):
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        assert hasattr(repl, "has_lake_project")
        result = repl.has_lake_project()
        assert isinstance(result, bool)


class TestIterativeProverCompilesWithPreamble:
    """#121 — IterativeProver prepends preamble to REPL execution."""

    def test_preamble_prepended_to_compile(self):
        from agentic_research.agents.prover import IterativeProver

        preamble = "import Mathlib\nimport Aesop"
        proof_response = "```lean\ntheorem foo : True := trivial\n```"
        llm = _make_mock_llm([proof_response])
        repl = _make_mock_repl()

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=1),
            lean_preamble=preamble,
        )

        with patch.object(repl, "execute", wraps=repl.execute) as spy:
            ctx = AgentContext(task="theorem foo : True := sorry")
            prover.run(ctx)

            assert spy.call_count >= 1
            compiled_code = spy.call_args_list[0][0][0]
            assert compiled_code.startswith(preamble)

    def test_no_preamble_no_prefix(self):
        from agentic_research.agents.prover import IterativeProver

        proof_response = "```lean\ntheorem foo : True := trivial\n```"
        llm = _make_mock_llm([proof_response])
        repl = _make_mock_repl()

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=1),
        )

        with patch.object(repl, "execute", wraps=repl.execute) as spy:
            ctx = AgentContext(task="theorem foo : True := sorry")
            prover.run(ctx)

            assert spy.call_count >= 1
            compiled_code = spy.call_args_list[0][0][0]
            assert not compiled_code.startswith("import")


class TestCorrectionRetryProverHasPreamble:
    """BUG-D — Correction retry prover gets lean_preamble."""

    def test_retry_prover_receives_preamble(self):
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.agents.prover import IterativeProver

        preamble = "import Mathlib\nimport Aesop"
        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        failed_result = AgentResult(
            agent_name="iterative_prover",
            status=AgentStatus.FAILURE,
            result={
                "statement": "theorem leaf : True := sorry",
                "proved": False,
                "final_proof": "by simp",
                "failure_reason": "simp made no progress",
                "attempts": [{"iteration": 1, "proof_code": "by simp", "status": "compilation_error"}],
            },
            token_usage=TokenUsage(),
        )

        mock_corrector = MagicMock()
        mock_corrector.correct.return_value = ProofCorrection(
            error_category=ErrorCategory.TACTIC_FAILURE,
            error_message="simp made no progress",
            suggested_tactics=["omega"],
            revised_proof_sketch="by omega",
            confidence=0.8,
            reasoning="use omega",
        )
        mock_corrector.cumulative_tokens = TokenUsage()

        prover = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            lean_preamble=preamble,
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

        captured_preambles: list[str | None] = []
        original_init = IterativeProver.__init__

        def capture_init(self_inner, *args, **kwargs):
            captured_preambles.append(kwargs.get("lean_preamble"))
            original_init(self_inner, *args, **kwargs)

        tokens = TokenUsage()
        with patch.object(IterativeProver, "run", return_value=failed_result), \
             patch.object(IterativeProver, "__init__", capture_init):
            prover._prove_leaf(tree, tree.nodes["leaf"], tokens)

        assert any(p == preamble for p in captured_preambles)


class TestTryAutomatedTacticsReceivesImports:
    """#122 — Imports from preamble passed to try_automated_tactics."""

    def test_imports_passed_when_preamble_present(self):
        pipeline = _make_pipeline()

        with patch.object(pipeline._repl, "has_lake_project", return_value=True), \
             patch.object(pipeline._repl, "try_automated_tactics", return_value="trivial") as spy:
            pipeline.run("theorem foo : True := sorry")

        spy.assert_called_once()
        call_kwargs = spy.call_args
        imports = call_kwargs[1].get("imports")
        assert imports is not None
        assert "Mathlib" in imports
        assert "Aesop" in imports

    def test_no_imports_when_no_preamble(self):
        pipeline = _make_pipeline()
        pipeline._lean_preamble = None

        with patch.object(pipeline._repl, "try_automated_tactics", return_value="trivial") as spy:
            pipeline.run("theorem foo : True := sorry")

        spy.assert_called_once()
        call_kwargs = spy.call_args
        imports = call_kwargs[1].get("imports")
        assert imports is None


class TestFlattenFinalizePrependsPreamble:
    """BUG-A — FlattenFinalize receives and uses preamble."""

    def test_preamble_in_compilation(self):
        from agentic_research.agents.flatten_finalize import FlattenFinalize

        preamble = "import Mathlib\nimport Aesop"
        assembled = "```lean\ntheorem foo : True := trivial\n```"
        llm = _make_mock_llm([assembled])
        repl = _make_mock_repl()

        agent = FlattenFinalize(llm_client=llm, lean_repl=repl, lean_preamble=preamble)

        tree = LemmaTree(
            root_id="root",
            topological_order=["root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                    status=NodeStatus.PROVED,
                    proof_code="theorem root : True := trivial",
                ),
            },
        )

        ctx = AgentContext(
            task="flatten proof",
            metadata={"lemma_tree": tree.model_dump()},
        )

        with patch.object(repl, "execute", wraps=repl.execute) as spy:
            agent.run(ctx)

            assert spy.call_count >= 1
            compiled_code = spy.call_args_list[0][0][0]
            assert compiled_code.startswith(preamble)

    def test_preamble_in_prompt(self):
        from agentic_research.agents.flatten_finalize import FlattenFinalize

        preamble = "import Mathlib\nimport Aesop"
        assembled = "```lean\ntheorem foo : True := trivial\n```"
        llm = _make_mock_llm([assembled])
        repl = _make_mock_repl()

        agent = FlattenFinalize(llm_client=llm, lean_repl=repl, lean_preamble=preamble)

        tree = LemmaTree(
            root_id="root",
            topological_order=["root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                    status=NodeStatus.PROVED,
                    proof_code="theorem root : True := trivial",
                ),
            },
        )

        ctx = AgentContext(
            task="flatten proof",
            metadata={"lemma_tree": tree.model_dump()},
        )
        agent.run(ctx)

        call_args = llm.complete.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "import Mathlib" in prompt_content

    def test_pipeline_passes_preamble_to_flatten(self):
        from agentic_research.agents.flatten_finalize import FlattenFinalize

        pipeline = _make_pipeline()
        pipeline._lean_preamble = "import Mathlib"

        captured_preambles: list[str | None] = []
        original_init = FlattenFinalize.__init__

        def capture_init(self_inner, *args, **kwargs):
            captured_preambles.append(kwargs.get("lean_preamble"))
            original_init(self_inner, *args, **kwargs)

        tree = LemmaTree(
            root_id="root",
            topological_order=["root"],
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="root",
                    statement_lean="theorem root : True := sorry",
                    depth=0,
                    status=NodeStatus.PROVED,
                    proof_code="theorem root : True := trivial",
                ),
            },
        )

        with patch.object(FlattenFinalize, "__init__", capture_init):
            try:
                pipeline._run_flatten_finalize(tree)
            except Exception:
                pass

        assert "import Mathlib" in captured_preambles


# ---------------------------------------------------------------------------
# Phase 2: Error Context Quality
# ---------------------------------------------------------------------------


class TestExtractCompilerErrorsReturnsReplErrors:
    """BUG-B — _extract_compiler_errors extracts actual REPL errors."""

    def test_extracts_attempt_errors_when_available(self):
        """When strategies have prover_result with attempts, extract REPL errors."""
        from agentic_research.pipelines.proof import ProofPipeline

        mock_prover_result = ProverResult(
            statement="theorem foo : True := sorry",
            proved=False,
            attempts=[
                ProofAttempt(
                    iteration=1,
                    proof_code="by simp",
                    status=ProofAttemptStatus.COMPILATION_ERROR,
                    errors=["unknown identifier 'LE'", "failed to synthesize instance"],
                ),
            ],
        )

        strategy = ProofStrategy(
            strategy_type=StrategyType.DIRECT,
            description="simp failed",
            key_tactics=["simp"],
            prover_result=mock_prover_result,
        )

        result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            failure_reason="All strategies exhausted",
            strategies_tried=[strategy],
        )

        errors = ProofPipeline._extract_compiler_errors(result)
        assert "unknown identifier 'LE'" in errors
        assert "failed to synthesize instance" in errors
        # Strategy summaries should NOT be included
        for err in errors:
            assert "strategies exhausted" not in err.lower()

    def test_no_prover_result_yields_empty(self):
        from agentic_research.pipelines.proof import ProofPipeline

        result = ProofSearchResult(
            statement="theorem foo : True",
            proved=False,
            strategies_tried=[
                ProofStrategy(
                    strategy_type=StrategyType.DIRECT,
                    description="simp failed",
                    key_tactics=["simp"],
                ),
            ],
        )

        errors = ProofPipeline._extract_compiler_errors(result)
        assert errors == []


class TestDiagnoseFailureUsesRealErrors:
    """BUG-F — _diagnose_failure passes real errors to the LLM template."""

    def test_errors_hint_flows_to_template(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        diagnosis_json = json.dumps({
            "failure_type": "stuck_goal",
            "description": "cannot close goal",
        })
        llm = _make_mock_llm([diagnosis_json])
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
                    proof_code="by simp",
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
        real_errors = "type mismatch, expected Nat got Int\nunsolved goals remain"
        prover._diagnose_failure(tree, tree.nodes["root"], tokens, errors_hint=real_errors)

        call_args = llm.complete.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "type mismatch, expected Nat got Int" in prompt
        assert "unsolved goals remain" in prompt

    def test_fallback_message_when_no_errors(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        diagnosis_json = json.dumps({
            "failure_type": "stuck_goal",
            "description": "no info",
        })
        llm = _make_mock_llm([diagnosis_json])
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
        prover._diagnose_failure(tree, tree.nodes["root"], tokens, errors_hint="")

        call_args = llm.complete.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Parent proof did not compile or close all goals" in prompt


class TestProofCorrectorReceivesPriorAttempts:
    """BUG-C — prior_attempts populated from proof search results."""

    def test_prior_attempts_from_recursive_prover(self):
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.agents.prover import IterativeProver

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        failed_result = AgentResult(
            agent_name="iterative_prover",
            status=AgentStatus.FAILURE,
            result={
                "statement": "theorem leaf : True := sorry",
                "proved": False,
                "final_proof": "by simp",
                "failure_reason": "simp made no progress",
                "attempts": [
                    {"iteration": 1, "proof_code": "by simp", "status": "compilation_error"},
                    {"iteration": 2, "proof_code": "by ring", "status": "compilation_error"},
                ],
            },
            token_usage=TokenUsage(),
        )

        mock_corrector = MagicMock()
        mock_corrector.correct.return_value = ProofCorrection(
            error_category=ErrorCategory.TACTIC_FAILURE,
            error_message="simp failed",
            suggested_tactics=["omega"],
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
                    statement_lean="theorem leaf : True := sorry",
                    parent_id="root", depth=1,
                ),
            },
        )
        prover._total_nodes = 2

        tokens = TokenUsage()
        with patch.object(IterativeProver, "run", return_value=failed_result):
            prover._prove_leaf(tree, tree.nodes["leaf"], tokens)

        mock_corrector.correct.assert_called_once()
        call_kwargs = mock_corrector.correct.call_args[1]
        prior = call_kwargs.get("prior_attempts")
        assert prior is not None
        assert "by simp" in prior
        assert "by ring" in prior


class TestImportErrorHeuristicClassification:
    """#123 — Heuristic import error classification."""

    def test_synthesize_instance_classified_as_import(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        llm = _make_mock_llm([])
        corrector = ProofCorrector(llm_client=llm)

        correction = corrector.correct(
            failed_proof="by simp",
            error_message="failed to synthesize instance of type class 'Add'",
            lean_goal_state="⊢ True",
        )

        assert correction.error_category == ErrorCategory.MISSING_IMPORT
        assert correction.confidence == 0.9
        llm.complete.assert_not_called()

    def test_unknown_constant_classified_as_import(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        llm = _make_mock_llm([])
        corrector = ProofCorrector(llm_client=llm)

        correction = corrector.correct(
            failed_proof="by exact Nat.bogus",
            error_message="unknown constant 'Nat.bogus'",
            lean_goal_state="⊢ Nat",
        )

        assert correction.error_category == ErrorCategory.MISSING_IMPORT
        assert correction.confidence == 0.9

    def test_mixed_errors_not_classified_as_import(self):
        from agentic_research.agents.proof_corrector import ProofCorrector

        response = json.dumps({
            "error_category": "tactic_failure",
            "error_message": "simp made no progress",
            "suggested_tactics": ["omega"],
            "revised_proof_sketch": "by omega",
            "confidence": 0.7,
            "reasoning": "try omega",
        })
        llm = _make_mock_llm([response])
        corrector = ProofCorrector(llm_client=llm)

        correction = corrector.correct(
            failed_proof="by simp",
            error_message="simp made no progress",
            lean_goal_state="⊢ Nat",
            compiler_errors=["simp made no progress", "unknown constant 'Foo'"],
        )

        assert correction.error_category != ErrorCategory.MISSING_IMPORT
        llm.complete.assert_called_once()


class TestAssemblyErrorPatternsIncludeImportErrors:
    """#123 — _ASSEMBLY_ERROR_PATTERNS includes import-related patterns."""

    def test_failed_to_synthesize_in_patterns(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        patterns = RecursiveProver._ASSEMBLY_ERROR_PATTERNS
        pattern_strings = [p[0] for p in patterns]
        assert "failed to synthesize instance" in pattern_strings

    def test_unknown_constant_in_patterns(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        patterns = RecursiveProver._ASSEMBLY_ERROR_PATTERNS
        pattern_strings = [p[0] for p in patterns]
        assert "unknown constant" in pattern_strings

    def test_shortcircuit_on_synthesize_instance(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        diagnosis_json = json.dumps({
            "failure_type": "stuck_goal",
            "description": "unneeded",
        })
        llm = _make_mock_llm([diagnosis_json])
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
        diagnosis = prover._diagnose_failure(
            tree, tree.nodes["root"], tokens,
            errors_hint="failed to synthesize instance of type class 'Add'",
        )

        assert diagnosis.failure_type.value == "assembly_error"
        llm.complete.assert_not_called()
