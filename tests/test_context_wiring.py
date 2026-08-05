"""Tests for H1: Wire missing context through the proof pipeline.

Covers all 9 changes from the hypothesis:
1. Preamble in proof prompt (#107)
2. Preamble in _patch_assembly (#105)
3. Set-builder notation in _patch_assembly (#110)
4. Child semantic context (#117)
5. Leaf node context enrichment
6. Strategy metadata wiring
7. max_tokens truncation detection (#111)
8. Multi-line error parsing (#81)
9. Template update for preamble
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agentic_research.models.agents import (
    AgentContext,
    LLMResponse,
    ProofAttemptStatus,
    ProverConfig,
    ProverResult,
    TokenUsage,
)
from agentic_research.models.proof import (
    LemmaTree,
    ProofNode,
)


def _mock_llm_response(content: str, stop_reason: str = "end_turn") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason=stop_reason,
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )


def _make_mock_llm(responses: list[str | tuple[str, str]]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient
    import json
    import re

    mock = MagicMock(spec=LLMClient)
    side_effects = []
    for r in responses:
        if isinstance(r, tuple):
            side_effects.append(_mock_llm_response(r[0], stop_reason=r[1]))
        else:
            side_effects.append(_mock_llm_response(r))
    mock.complete.side_effect = side_effects

    def _extract_json(text):
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass
        for sc, ec in [("{", "}"), ("[", "]")]:
            start = text.find(sc)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == sc:
                    depth += 1
                elif text[i] == ec:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    mock.extract_json.side_effect = _extract_json
    return mock


def _make_mock_repl():
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


# ---------------------------------------------------------------------------
# Change 1: Preamble in proof prompt (#107)
# ---------------------------------------------------------------------------


class TestPreambleInProofPrompt:
    def test_preamble_included_in_parent_proof_prompt(self):
        """When lean_preamble is set, parent proof prompt includes imports."""
        from agentic_research.agents.recursive_prover import RecursiveProver

        preamble = "import Mathlib\nopen MeasureTheory"
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="parent",
                    statement_lean="theorem parent := sorry",
                    children=["child_1"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="helper",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem parent := trivial\n```",
            "```lean\ntheorem child_1 := trivial\n```",
        ])
        repl = _make_mock_repl()

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
            lean_preamble=preamble,
        )
        ctx = AgentContext(task="prove", metadata={"lemma_tree": tree.model_dump()})
        agent.run(ctx)

        parent_call = llm.complete.call_args_list[0]
        user_msg = parent_call[1]["messages"][0]["content"]
        assert "Available Imports & Definitions" in user_msg
        assert "import Mathlib" in user_msg
        assert "open MeasureTheory" in user_msg

    def test_no_preamble_no_section_in_prompt(self):
        """Without preamble, no imports section appears."""
        from agentic_research.agents.recursive_prover import RecursiveProver

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="parent",
                    statement_lean="theorem parent := sorry",
                    children=["child_1"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="helper",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem parent := trivial\n```",
            "```lean\ntheorem child_1 := trivial\n```",
        ])
        repl = _make_mock_repl()

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
        )
        ctx = AgentContext(task="prove", metadata={"lemma_tree": tree.model_dump()})
        agent.run(ctx)

        parent_call = llm.complete.call_args_list[0]
        user_msg = parent_call[1]["messages"][0]["content"]
        assert "Available Imports & Definitions" not in user_msg


# ---------------------------------------------------------------------------
# Change 2: Preamble in _patch_assembly (#105)
# ---------------------------------------------------------------------------


class TestPreambleInPatchAssembly:
    def test_preamble_prepended_to_compilation(self):
        """Preamble is prepended when compiling child declarations."""
        from agentic_research.agents.recursive_prover import RecursiveProver

        preamble = "import Mathlib"
        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        prover = RecursiveProver(
            llm_client=llm, lean_repl=repl, lean_preamble=preamble
        )

        compiled_codes = []
        original_execute = repl.execute

        def capture_execute(code):
            compiled_codes.append(code)
            return original_execute(code)

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_lean="theorem main := sorry",
                    children=["c1"],
                ),
                "c1": ProofNode(
                    node_id="c1",
                    statement_lean="axiom c1 : True",
                    parent_id="root",
                ),
            },
            topological_order=["c1", "root"],
        )

        decls = "axiom c1 : True\n-- Use: have <result> := c1 <args>"
        with patch.object(repl, "execute", side_effect=capture_execute):
            prover._patch_assembly(decls, tree, tree.nodes["root"])

        assert any("import Mathlib" in code for code in compiled_codes)

    def test_no_preamble_no_prefix(self):
        """Without preamble, no prefix is added."""
        from agentic_research.agents.recursive_prover import RecursiveProver

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        compiled_codes = []
        original_execute = repl.execute

        def capture_execute(code):
            compiled_codes.append(code)
            return original_execute(code)

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_lean="theorem main := sorry",
                    children=["c1"],
                ),
                "c1": ProofNode(
                    node_id="c1",
                    statement_lean="axiom c1 : True",
                    parent_id="root",
                ),
            },
            topological_order=["c1", "root"],
        )

        decls = "axiom c1 : True\n-- Use: have <result> := c1 <args>"
        with patch.object(repl, "execute", side_effect=capture_execute):
            prover._patch_assembly(decls, tree, tree.nodes["root"])

        assert all("import Mathlib" not in code for code in compiled_codes)


# ---------------------------------------------------------------------------
# Change 3: Set-builder notation in _patch_assembly (#110)
# ---------------------------------------------------------------------------


class TestSetBuilderBraceBalancing:
    def test_set_builder_braces_not_counted(self):
        from agentic_research.agents.recursive_prover import _count_lean_brace_imbalance

        text = "axiom foo : {x : ℕ | x > 0} → Prop"
        assert _count_lean_brace_imbalance(text) == 0

    def test_set_literal_braces_not_counted(self):
        from agentic_research.agents.recursive_prover import _count_lean_brace_imbalance

        text = "axiom bar : {1, 2, 3} ⊆ Finset.univ"
        assert _count_lean_brace_imbalance(text) == 0

    def test_structural_brace_imbalance_detected(self):
        from agentic_research.agents.recursive_prover import _count_lean_brace_imbalance

        text = "structure Foo where\n  field : Nat\n{"
        assert _count_lean_brace_imbalance(text) == 1

    def test_mixed_set_builder_and_structural(self):
        from agentic_research.agents.recursive_prover import _count_lean_brace_imbalance

        text = "axiom foo : {x : ℕ | x > 0} → Prop\n{"
        assert _count_lean_brace_imbalance(text) == 1

    def test_balanced_structural_braces(self):
        from agentic_research.agents.recursive_prover import _count_lean_brace_imbalance

        text = "def f := { field1 := 1, field2 := 2 }"
        assert _count_lean_brace_imbalance(text) == 0

    def test_patch_assembly_no_spurious_close_on_set_builder(self):
        """Set-builder notation should not cause spurious } to be appended."""
        from agentic_research.agents.recursive_prover import RecursiveProver

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        prover = RecursiveProver(llm_client=llm, lean_repl=repl)

        decls = "axiom c1 : {x : ℕ | x > 0} → Prop\n-- Use: have <result> := c1 <args>"
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_lean="theorem main := sorry",
                    children=["c1"],
                ),
                "c1": ProofNode(
                    node_id="c1",
                    statement_lean="axiom c1 : {x : ℕ | x > 0} → Prop",
                    parent_id="root",
                ),
            },
            topological_order=["c1", "root"],
        )

        result = prover._patch_assembly(decls, tree, tree.nodes["root"])
        assert result.count("}") == decls.count("}")


# ---------------------------------------------------------------------------
# Change 4: Child semantic context (#117)
# ---------------------------------------------------------------------------


class TestChildSemanticContext:
    def test_nl_comment_added(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        child = ProofNode(
            node_id="lemma_1",
            statement_nl="Triangle inequality holds for Wasserstein distance",
            statement_lean="theorem lemma_1 : True := sorry",
        )
        result = RecursiveProver._format_child_declaration(child)
        assert "-- NL: Triangle inequality holds for Wasserstein distance" in result

    def test_no_nl_when_empty(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        child = ProofNode(
            node_id="lemma_1",
            statement_nl="",
            statement_lean="theorem lemma_1 : True := sorry",
        )
        result = RecursiveProver._format_child_declaration(child)
        assert "-- NL:" not in result

    def test_proof_sketch_included(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        child = ProofNode(
            node_id="lemma_1",
            statement_nl="some property",
            statement_lean="theorem lemma_1 : True := sorry",
            proof_sketch_nl="Apply Cauchy-Schwarz then simplify\nUse triangle inequality",
        )
        result = RecursiveProver._format_child_declaration(child)
        assert "-- Sketch: Apply Cauchy-Schwarz then simplify" in result
        assert "Use triangle inequality" not in result

    def test_no_sketch_when_none(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        child = ProofNode(
            node_id="lemma_1",
            statement_nl="some property",
            statement_lean="theorem lemma_1 : True := sorry",
        )
        result = RecursiveProver._format_child_declaration(child)
        assert "-- Sketch:" not in result


# ---------------------------------------------------------------------------
# Change 5: Leaf node context enrichment
# ---------------------------------------------------------------------------


class TestLeafNodeContextEnrichment:
    def test_leaf_prover_receives_preamble(self):
        """IterativeProver gets lean_preamble when proving leaf nodes."""
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.agents.prover import IterativeProver

        preamble = "import Mathlib\ndef WassersteinDist := sorry"
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="trivial",
                    statement_lean="theorem foo : True := sorry",
                ),
            },
            topological_order=["root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem foo : True := trivial\n```",
        ])
        repl = _make_mock_repl()

        captured = []
        orig_init = IterativeProver.__init__

        def spy_init(self_prover, *args, **kwargs):
            captured.append(kwargs.get("lean_preamble"))
            orig_init(self_prover, *args, **kwargs)

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
            lean_preamble=preamble,
        )

        with patch.object(IterativeProver, "__init__", spy_init):
            ctx = AgentContext(task="prove", metadata={"lemma_tree": tree.model_dump()})
            agent.run(ctx)

        assert len(captured) >= 1
        assert captured[0] == preamble

    def test_leaf_task_includes_proof_sketch(self):
        """Leaf proof task includes NL proof sketch when available."""
        from agentic_research.agents.recursive_prover import RecursiveProver
        from agentic_research.agents.prover import IterativeProver

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="lemma with sketch",
                    statement_lean="theorem foo : True := sorry",
                    proof_sketch_nl="Use induction on n, then apply simp",
                ),
            },
            topological_order=["root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem foo : True := trivial\n```",
        ])
        repl = _make_mock_repl()

        captured_tasks = []
        orig_run = IterativeProver.run

        def spy_run(self_prover, ctx):
            captured_tasks.append(ctx.task)
            return orig_run(self_prover, ctx)

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
        )

        with patch.object(IterativeProver, "run", spy_run):
            ctx = AgentContext(task="prove", metadata={"lemma_tree": tree.model_dump()})
            agent.run(ctx)

        assert len(captured_tasks) >= 1
        assert "Use induction on n" in captured_tasks[0]

    def test_leaf_task_includes_sibling_context(self):
        """Leaf proof task includes sibling NL summaries."""
        from agentic_research.agents.recursive_prover import RecursiveProver

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="parent",
                    statement_lean="theorem parent := sorry",
                    children=["child_1", "child_2"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="first sub-goal",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
                "child_2": ProofNode(
                    node_id="child_2",
                    statement_nl="second sub-goal",
                    statement_lean="theorem child_2 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "child_2", "root"],
        )

        ctx_result = RecursiveProver._get_sibling_context(tree, tree.nodes["child_1"])
        assert "child_2" in ctx_result
        assert "second sub-goal" in ctx_result
        assert "child_1" not in ctx_result


# ---------------------------------------------------------------------------
# Change 6: Strategy metadata wiring
# ---------------------------------------------------------------------------


class TestStrategyMetadataWiring:
    def test_strategy_metadata_included_in_prompt(self):
        from agentic_research.agents.prover import IterativeProver

        llm = _make_mock_llm([
            "```lean\ntheorem foo : True := trivial\n```",
        ])
        repl = _make_mock_repl()

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=1),
        )
        ctx = AgentContext(
            task="theorem foo : True",
            metadata={
                "strategy_type": "induction",
                "strategy_description": "Induct on n, then simplify base case",
                "key_tactics": ["induction", "simp", "omega"],
                "relevant_lemmas": ["Nat.add_comm", "Nat.succ_eq_add_one"],
            },
        )
        prover.run(ctx)

        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "## Strategy" in user_content
        assert "induction" in user_content
        assert "Induct on n" in user_content
        assert "omega" in user_content
        assert "Nat.add_comm" in user_content

    def test_no_strategy_metadata_no_section(self):
        from agentic_research.agents.prover import IterativeProver

        llm = _make_mock_llm([
            "```lean\ntheorem foo : True := trivial\n```",
        ])
        repl = _make_mock_repl()

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=1),
        )
        ctx = AgentContext(task="theorem foo : True")
        prover.run(ctx)

        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "## Strategy" not in user_content

    def test_partial_strategy_metadata(self):
        from agentic_research.agents.prover import IterativeProver

        llm = _make_mock_llm([
            "```lean\ntheorem foo : True := trivial\n```",
        ])
        repl = _make_mock_repl()

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=1),
        )
        ctx = AgentContext(
            task="theorem foo : True",
            metadata={
                "strategy_type": "direct",
                "key_tactics": ["simp"],
            },
        )
        prover.run(ctx)

        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "## Strategy" in user_content
        assert "direct" in user_content
        assert "simp" in user_content


# ---------------------------------------------------------------------------
# Change 7: max_tokens truncation detection (#111)
# ---------------------------------------------------------------------------


class TestMaxTokensTruncation:
    def test_truncated_response_retries_with_higher_limit(self):
        from agentic_research.agents.prover import IterativeProver

        llm = _make_mock_llm([
            ("```lean\ntheorem foo : True := by\n  -- truncated", "max_tokens"),
            "```lean\ntheorem foo : True := trivial\n```",
        ])
        repl = _make_mock_repl()

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=3, max_tokens=4096),
        )
        ctx = AgentContext(task="theorem foo : True")
        result = prover.run(ctx)

        prover_result = ProverResult.model_validate(result.result)
        assert prover_result.proved
        assert prover_result.attempts[0].status == ProofAttemptStatus.TRUNCATED

        second_call = llm.complete.call_args_list[1]
        assert second_call[1]["max_tokens"] == int(4096 * 1.5)

    def test_truncated_capped_at_32768(self):
        from agentic_research.agents.prover import IterativeProver

        llm = _make_mock_llm([
            ("truncated", "max_tokens"),
            "```lean\ntheorem foo : True := trivial\n```",
        ])
        repl = _make_mock_repl()

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=3, max_tokens=30000),
        )
        ctx = AgentContext(task="theorem foo : True")
        prover.run(ctx)

        second_call = llm.complete.call_args_list[1]
        assert second_call[1]["max_tokens"] <= 32768

    def test_truncated_status_enum(self):
        assert ProofAttemptStatus.TRUNCATED == "truncated"

    def test_repeated_truncation_at_cap_falls_through(self):
        from agentic_research.agents.prover import IterativeProver

        llm = _make_mock_llm([
            ("truncated1", "max_tokens"),
            ("truncated2", "max_tokens"),
            ("truncated3", "max_tokens"),
        ])
        repl = _make_mock_repl()

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=3, max_tokens=32768),
        )
        ctx = AgentContext(task="theorem foo : True")
        result = prover.run(ctx)

        prover_result = ProverResult.model_validate(result.result)
        assert not prover_result.proved


# ---------------------------------------------------------------------------
# Change 8: Multi-line error parsing (#81)
# ---------------------------------------------------------------------------


class TestMultiLineErrorParsing:
    def test_continuation_lines_included(self):
        from agentic_research.tools.lean_repl import _parse_lean_errors

        output = (
            "file.lean:5:0: error: type mismatch\n"
            "  expected: Nat\n"
            "  actual: Bool\n"
            "\n"
            "other output\n"
        )
        errors, warnings = _parse_lean_errors(output)
        assert len(errors) == 1
        assert "type mismatch" in errors[0]
        assert "expected: Nat" in errors[0]
        assert "actual: Bool" in errors[0]

    def test_multiple_error_blocks(self):
        from agentic_research.tools.lean_repl import _parse_lean_errors

        output = (
            "file.lean:1:0: error: unknown identifier 'foo'\n"
            "  did you mean 'Foo'?\n"
            "\n"
            "file.lean:5:0: error: type mismatch\n"
            "  expected: Nat\n"
            "  actual: String\n"
        )
        errors, warnings = _parse_lean_errors(output)
        assert len(errors) == 2
        assert "foo" in errors[0]
        assert "did you mean" in errors[0]
        assert "type mismatch" in errors[1]
        assert "expected: Nat" in errors[1]

    def test_warning_with_continuation(self):
        from agentic_research.tools.lean_repl import _parse_lean_errors

        output = (
            "file.lean:1:0: warning: unused variable 'x'\n"
            "  consider using _ instead\n"
        )
        errors, warnings = _parse_lean_errors(output)
        assert len(errors) == 0
        assert len(warnings) == 1
        assert "unused variable" in warnings[0]
        assert "consider using" in warnings[0]

    def test_mixed_errors_and_warnings(self):
        from agentic_research.tools.lean_repl import _parse_lean_errors

        output = (
            "file.lean:1:0: error: unknown identifier 'foo'\n"
            "file.lean:2:0: warning: unused variable 'x'\n"
            "non-diagnostic line\n"
        )
        errors, warnings = _parse_lean_errors(output)
        assert len(errors) == 1
        assert len(warnings) == 1

    def test_goal_state_in_error(self):
        from agentic_research.tools.lean_repl import _parse_lean_errors

        output = (
            "file.lean:10:2: error: unsolved goals\n"
            "n m : ℕ\n"
            "⊢ n + m = m + n\n"
        )
        errors, warnings = _parse_lean_errors(output)
        assert len(errors) == 1
        assert "n + m = m + n" in errors[0]

    def test_backward_compatible_simple_errors(self):
        from agentic_research.tools.lean_repl import _parse_lean_errors

        output = (
            "file.lean:1:0: error: unknown identifier 'foo'\n"
            "file.lean:2:0: warning: unused variable 'x'\n"
            "all good\n"
        )
        errors, warnings = _parse_lean_errors(output)
        assert len(errors) == 1
        assert len(warnings) == 1
        assert "foo" in errors[0]
        assert "unused" in warnings[0]

    def test_empty_output(self):
        from agentic_research.tools.lean_repl import _parse_lean_errors

        errors, warnings = _parse_lean_errors("")
        assert errors == []
        assert warnings == []


# ---------------------------------------------------------------------------
# Change 9: Template update for preamble
# ---------------------------------------------------------------------------


class TestTemplateUpdate:
    def test_template_has_preamble_placeholder(self):
        from agentic_research.agents.prompt_templates import PARENT_PROOF_USER_TEMPLATE

        rendered = PARENT_PROOF_USER_TEMPLATE.format(
            parent_statement="theorem p := sorry",
            child_declarations="axiom l1 : True",
            lean_preamble_section="## Preamble\nimport Mathlib",
        )
        assert "## Preamble" in rendered
        assert "import Mathlib" in rendered

    def test_template_empty_preamble(self):
        from agentic_research.agents.prompt_templates import PARENT_PROOF_USER_TEMPLATE

        rendered = PARENT_PROOF_USER_TEMPLATE.format(
            parent_statement="theorem p := sorry",
            child_declarations="axiom l1 : True",
            lean_preamble_section="",
        )
        assert "theorem p" in rendered
        assert "axiom l1" in rendered


# ---------------------------------------------------------------------------
# Integration: preamble in prompt AND assembly (#107 + #105)
# ---------------------------------------------------------------------------


class TestPreamblePromptAndAssemblyTogether:
    def test_preamble_in_both_prompt_and_assembly(self):
        """Both prompt and assembly get the preamble — anti-pattern check."""
        from agentic_research.agents.recursive_prover import RecursiveProver

        preamble = "import Mathlib\nopen Topology"
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="parent",
                    statement_lean="theorem parent := sorry",
                    children=["child_1"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="helper",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        compiled_codes = []
        llm = _make_mock_llm([
            "```lean\ntheorem parent := trivial\n```",
            "```lean\ntheorem child_1 := trivial\n```",
        ])
        repl = _make_mock_repl()
        orig_execute = repl.execute

        def capture_execute(code):
            compiled_codes.append(code)
            return orig_execute(code)

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
            lean_preamble=preamble,
        )

        with patch.object(repl, "execute", side_effect=capture_execute):
            ctx = AgentContext(task="prove", metadata={"lemma_tree": tree.model_dump()})
            agent.run(ctx)

        prompt_call = llm.complete.call_args_list[0]
        user_msg = prompt_call[1]["messages"][0]["content"]
        assert "import Mathlib" in user_msg

        assembly_codes = [c for c in compiled_codes if "-- sentinel" in c]
        assert any("import Mathlib" in code for code in assembly_codes)
