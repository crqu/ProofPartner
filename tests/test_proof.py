"""Tests for Phase 7: Proof Search + Recursive Decomposition.

All LLM calls are mocked — no real API calls are made.
Lean REPL uses mock backend for deterministic testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock


from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.proof import (
    FailureDiagnosis,
    FailureType,
    LemmaTree,
    NodeStatus,
    ProofNode,
    ProofPipelineResult,
    ProofSearchResult,
    ProofStrategy,
    RecursiveProofResult,
    StrategyType,
)


# ---------------------------------------------------------------------------
# models/proof.py
# ---------------------------------------------------------------------------


class TestStrategyType:
    def test_values(self):
        assert StrategyType.DIRECT == "direct"
        assert StrategyType.CONTRADICTION == "contradiction"
        assert StrategyType.INDUCTION == "induction"
        assert StrategyType.CASE_ANALYSIS == "case_analysis"


class TestProofStrategy:
    def test_defaults(self):
        s = ProofStrategy(strategy_type=StrategyType.DIRECT)
        assert s.plausibility == 0.5
        assert s.relevant_lemmas == []
        assert s.key_tactics == []

    def test_full(self):
        s = ProofStrategy(
            strategy_type=StrategyType.INDUCTION,
            description="induction on n",
            relevant_lemmas=["Nat.rec"],
            plausibility=0.8,
            key_tactics=["induction", "simp"],
        )
        assert s.strategy_type == StrategyType.INDUCTION
        assert s.plausibility == 0.8
        assert len(s.key_tactics) == 2

    def test_serialization_roundtrip(self):
        s = ProofStrategy(
            strategy_type=StrategyType.CONTRADICTION,
            description="by_contra",
            plausibility=0.6,
        )
        restored = ProofStrategy.model_validate(s.model_dump())
        assert restored == s


class TestFailureDiagnosis:
    def test_types(self):
        assert FailureType.MISSING_HYPOTHESIS == "missing_hypothesis"
        assert FailureType.WEAK_CHILD_LEMMA == "weak_child_lemma"
        assert FailureType.CONTRADICTORY_CHILD == "contradictory_child"
        assert FailureType.STUCK_GOAL == "stuck_goal"

    def test_with_child_id(self):
        d = FailureDiagnosis(
            failure_type=FailureType.WEAK_CHILD_LEMMA,
            description="lemma too weak",
            problematic_child_id="lemma_1",
            suggested_fix="strengthen hypothesis",
        )
        assert d.problematic_child_id == "lemma_1"


class TestProofNode:
    def test_defaults(self):
        node = ProofNode(node_id="root")
        assert node.depth == 0
        assert node.status == NodeStatus.PENDING
        assert node.children == []
        assert node.proof_code is None
        assert not node.from_prior_work

    def test_full(self):
        node = ProofNode(
            node_id="lemma_1",
            statement_nl="commutativity of addition",
            statement_lean="theorem add_comm : ∀ n m, n + m = m + n := sorry",
            depth=1,
            children=["lemma_1_1"],
            parent_id="root",
            status=NodeStatus.PROVED,
            proof_code="by omega",
            from_prior_work=True,
        )
        assert node.parent_id == "root"
        assert node.from_prior_work


class TestLemmaTree:
    def _make_tree(self) -> LemmaTree:
        root = ProofNode(
            node_id="root",
            statement_nl="main theorem",
            statement_lean="theorem main := sorry",
            children=["lemma_1", "lemma_2"],
        )
        l1 = ProofNode(
            node_id="lemma_1",
            statement_nl="first lemma",
            statement_lean="theorem l1 := sorry",
            depth=1,
            parent_id="root",
            status=NodeStatus.PROVED,
        )
        l2 = ProofNode(
            node_id="lemma_2",
            statement_nl="second lemma",
            statement_lean="theorem l2 := sorry",
            depth=1,
            parent_id="root",
            status=NodeStatus.PENDING,
        )
        return LemmaTree(
            root_id="root",
            nodes={"root": root, "lemma_1": l1, "lemma_2": l2},
            topological_order=["lemma_1", "lemma_2", "root"],
        )

    def test_get_node(self):
        tree = self._make_tree()
        assert tree.get_node("root") is not None
        assert tree.get_node("nonexistent") is None

    def test_get_children(self):
        tree = self._make_tree()
        children = tree.get_children("root")
        assert len(children) == 2
        assert children[0].node_id == "lemma_1"

    def test_all_children_proved(self):
        tree = self._make_tree()
        assert not tree.all_children_proved("root")
        tree.nodes["lemma_2"].status = NodeStatus.PROVED
        assert tree.all_children_proved("root")

    def test_all_proved(self):
        tree = self._make_tree()
        assert not tree.all_proved
        tree.nodes["root"].status = NodeStatus.PROVED
        tree.nodes["lemma_2"].status = NodeStatus.PROVED
        assert tree.all_proved

    def test_serialization_roundtrip(self):
        tree = self._make_tree()
        restored = LemmaTree.model_validate(tree.model_dump())
        assert restored.root_id == tree.root_id
        assert len(restored.nodes) == len(tree.nodes)


class TestProofSearchResult:
    def test_proved(self):
        r = ProofSearchResult(
            statement="theorem foo := sorry",
            proved=True,
            proof_code="theorem foo := trivial",
        )
        assert r.proved
        assert not r.needs_decomposition

    def test_needs_decomposition(self):
        r = ProofSearchResult(
            statement="theorem foo := sorry",
            proved=False,
            needs_decomposition=True,
            failure_reason="all strategies exhausted",
        )
        assert r.needs_decomposition


class TestRecursiveProofResult:
    def test_defaults(self):
        r = RecursiveProofResult(root_statement="theorem foo := sorry")
        assert not r.proved
        assert r.total_nodes == 0

    def test_proved(self):
        r = RecursiveProofResult(
            root_statement="theorem foo := sorry",
            proved=True,
            total_nodes=3,
            proved_nodes=3,
            max_depth_reached=2,
        )
        assert r.proved


class TestProofPipelineResult:
    def test_defaults(self):
        r = ProofPipelineResult(statement="theorem foo := sorry")
        assert not r.proved

    def test_success(self):
        r = ProofPipelineResult(
            statement="theorem foo := sorry",
            proved=True,
            final_proof="theorem foo := trivial",
            claim_check_passed=True,
        )
        assert r.proved
        assert r.claim_check_passed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = [_mock_llm_response(text) for text in responses]
    mock.complete.side_effect = side_effects
    mock.extract_json.side_effect = lambda text: _extract_json_helper(text)
    return mock


def _extract_json_helper(text: str):
    import json
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


def _make_mock_repl(succeed: bool = True):
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


def _make_mock_search():
    from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

    return LeanSearch(SearchConfig(backend=SearchBackend.MOCK))


# ---------------------------------------------------------------------------
# agents/proof_search.py
# ---------------------------------------------------------------------------


class TestProofSearchAgent:
    def test_direct_proof_success(self):
        from agentic_research.agents.proof_search import ProofSearchAgent

        strategies_json = '{"strategies": [{"strategy_type": "direct", "description": "use simp", "plausibility": 0.9, "relevant_lemmas": [], "key_tactics": ["simp"]}]}'
        llm = _make_mock_llm([
            strategies_json,
            "```lean\ntheorem foo : True := trivial\n```",
        ])

        repl = _make_mock_repl()
        search = _make_mock_search()

        agent = ProofSearchAgent(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(task="theorem foo : True")
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        search_result = ProofSearchResult.model_validate(result.result)
        assert search_result.proved
        assert search_result.proof_code is not None

    def test_all_strategies_fail(self):
        from agentic_research.agents.proof_search import ProofSearchAgent

        strategies_json = '{"strategies": [{"strategy_type": "direct", "description": "try simp", "plausibility": 0.5, "relevant_lemmas": [], "key_tactics": ["simp"]}]}'
        llm = _make_mock_llm([
            strategies_json,
            "```lean\n-- MOCK_ERROR\ntheorem foo : True := bad\n```",
        ])

        repl = _make_mock_repl()
        search = _make_mock_search()

        agent = ProofSearchAgent(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(task="theorem foo : True")
        result = agent.run(ctx)

        assert result.status == AgentStatus.FAILURE
        search_result = ProofSearchResult.model_validate(result.result)
        assert not search_result.proved
        assert search_result.needs_decomposition

    def test_second_strategy_succeeds(self):
        from agentic_research.agents.proof_search import ProofSearchAgent

        strategies_json = '{"strategies": [{"strategy_type": "direct", "description": "simp", "plausibility": 0.9, "relevant_lemmas": [], "key_tactics": ["simp"]}, {"strategy_type": "induction", "description": "induction", "plausibility": 0.5, "relevant_lemmas": [], "key_tactics": ["induction"]}]}'
        llm = _make_mock_llm([
            strategies_json,
            "```lean\n-- MOCK_ERROR\nbad\n```",
            "```lean\ntheorem foo : True := trivial\n```",
        ])

        repl = _make_mock_repl()
        search = _make_mock_search()

        agent = ProofSearchAgent(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(task="theorem foo : True")
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        search_result = ProofSearchResult.model_validate(result.result)
        assert search_result.proved


# ---------------------------------------------------------------------------
# agents/lemma_breakdown.py
# ---------------------------------------------------------------------------


class TestLemmaBreakdown:
    def test_decomposition(self):
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        response = '{"lemmas": [{"node_id": "lemma_1", "statement_nl": "base case", "depends_on": [], "from_prior_work": false}, {"node_id": "lemma_2", "statement_nl": "inductive step", "depends_on": ["lemma_1"], "from_prior_work": false}], "topological_order": ["lemma_1", "lemma_2"]}'
        llm = _make_mock_llm([response])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(
            task="prove P(n) for all n",
            metadata={"statement_lean": "theorem p_n : ∀ n, P n := sorry"},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        tree = LemmaTree.model_validate(result.result)
        assert tree.root_id == "root"
        assert "lemma_1" in tree.nodes
        assert "lemma_2" in tree.nodes
        assert tree.nodes["lemma_1"].statement_nl == "base case"

    def test_prior_work_tagging(self):
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        response = '{"lemmas": [{"node_id": "lemma_1", "statement_nl": "commutativity", "depends_on": [], "from_prior_work": true}], "topological_order": ["lemma_1"]}'
        llm = _make_mock_llm([response])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(task="something", metadata={"statement_lean": ""})
        result = agent.run(ctx)

        tree = LemmaTree.model_validate(result.result)
        assert tree.nodes["lemma_1"].from_prior_work


# ---------------------------------------------------------------------------
# agents/lemma_leanifier.py
# ---------------------------------------------------------------------------


class TestLemmaLeanifier:
    def test_leanify_success(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main theorem",
                    statement_lean="theorem main := sorry",
                    children=["lemma_1"],
                ),
                "lemma_1": ProofNode(
                    node_id="lemma_1",
                    statement_nl="sub-lemma",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem l1 : True := sorry\n```",
        ])
        repl = _make_mock_repl()

        agent = LemmaLeanifier(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(
            task="leanify",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        updated_tree = LemmaTree.model_validate(result.result)
        assert updated_tree.nodes["lemma_1"].statement_lean != ""

    def test_leanify_with_retry(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    statement_lean="theorem main := sorry",
                    children=["lemma_1"],
                ),
                "lemma_1": ProofNode(
                    node_id="lemma_1",
                    statement_nl="sub-lemma",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        llm = _make_mock_llm([
            "```lean\n-- MOCK_ERROR\nbad code\n```",
            "```lean\ntheorem l1 : True := sorry\n```",
        ])
        repl = _make_mock_repl()

        agent = LemmaLeanifier(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(
            task="leanify",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS

    def test_missing_tree(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        agent = LemmaLeanifier(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(task="leanify", metadata={})
        result = agent.run(ctx)
        assert result.status == AgentStatus.FAILURE


# ---------------------------------------------------------------------------
# agents/recursive_prover.py
# ---------------------------------------------------------------------------


class TestRecursiveProver:
    def test_prove_leaf_success(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

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

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
        )
        ctx = AgentContext(
            task="prove",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        rr = RecursiveProofResult.model_validate(result.result)
        assert rr.proved
        assert rr.proved_nodes == 1

    def test_depth_limit(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="deep node",
                    statement_lean="theorem foo := sorry",
                    depth=6,
                ),
            },
            topological_order=["root"],
        )

        llm = _make_mock_llm([])
        repl = _make_mock_repl()

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            max_depth=5,
        )
        ctx = AgentContext(
            task="prove",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.FAILURE
        rr = RecursiveProofResult.model_validate(result.result)
        assert not rr.proved
        assert rr.lemma_tree.nodes["root"].status == NodeStatus.FAILED

    def test_parent_with_children_all_proved(self):
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
                    statement_nl="child",
                    statement_lean="theorem child := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem parent := trivial\n```",
            "```lean\ntheorem child := trivial\n```",
        ])
        repl = _make_mock_repl()

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
        )
        ctx = AgentContext(
            task="prove",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        rr = RecursiveProofResult.model_validate(result.result)
        assert rr.proved
        assert rr.proved_nodes == 2

    def test_missing_tree(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        agent = RecursiveProver(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(task="prove", metadata={})
        result = agent.run(ctx)
        assert result.status == AgentStatus.FAILURE


# ---------------------------------------------------------------------------
# agents/flatten_finalize.py
# ---------------------------------------------------------------------------


class TestFlattenFinalize:
    def test_assemble_success(self):
        from agentic_research.agents.flatten_finalize import FlattenFinalize

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    statement_lean="theorem main := sorry",
                    children=["lemma_1"],
                    status=NodeStatus.PROVED,
                    proof_code="theorem main := by exact l1",
                ),
                "lemma_1": ProofNode(
                    node_id="lemma_1",
                    statement_nl="helper",
                    statement_lean="theorem l1 := sorry",
                    depth=1,
                    parent_id="root",
                    status=NodeStatus.PROVED,
                    proof_code="theorem l1 := trivial",
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem l1 := trivial\ntheorem main := by exact l1\n```",
        ])
        repl = _make_mock_repl()

        agent = FlattenFinalize(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(
            task="flatten",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        assert result.result["compiles"]
        assert "l1" in result.result["final_proof"]

    def test_unproved_nodes_rejected(self):
        from agentic_research.agents.flatten_finalize import FlattenFinalize

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    statement_lean="theorem main := sorry",
                    status=NodeStatus.PENDING,
                ),
            },
            topological_order=["root"],
        )

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        agent = FlattenFinalize(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(
            task="flatten",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.FAILURE
        assert "Unproved" in result.error_message

    def test_missing_tree(self):
        from agentic_research.agents.flatten_finalize import FlattenFinalize

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        agent = FlattenFinalize(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(task="flatten", metadata={})
        result = agent.run(ctx)
        assert result.status == AgentStatus.FAILURE


# ---------------------------------------------------------------------------
# pipelines/proof.py
# ---------------------------------------------------------------------------


class TestProofPipeline:
    def test_direct_proof_success(self):
        from agentic_research.pipelines.proof import ProofPipeline

        strategies_json = '{"strategies": [{"strategy_type": "direct", "description": "direct", "plausibility": 0.9, "relevant_lemmas": [], "key_tactics": ["simp"]}]}'

        llm = _make_mock_llm([
            strategies_json,
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
        )

        result = pipeline.run("theorem foo : True")
        assert result.proved
        assert result.final_proof is not None
        assert result.failure_stage is None

    def test_pipeline_result_model(self):
        r = ProofPipelineResult(
            statement="theorem foo : True",
            proved=True,
            final_proof="theorem foo : True := trivial",
            claim_check_passed=True,
        )
        restored = ProofPipelineResult.model_validate(r.model_dump())
        assert restored.proved
        assert restored.claim_check_passed


# ---------------------------------------------------------------------------
# Prompt templates (Phase 7)
# ---------------------------------------------------------------------------


class TestPhase7PromptTemplates:
    def test_strategy_template(self):
        from agentic_research.agents.prompt_templates import PROOF_STRATEGY_USER_TEMPLATE

        rendered = PROOF_STRATEGY_USER_TEMPLATE.format(
            statement="theorem foo : True := sorry",
            mathlib_lemmas="- Nat.add_comm",
        )
        assert "theorem foo" in rendered
        assert "Nat.add_comm" in rendered

    def test_lemma_breakdown_template(self):
        from agentic_research.agents.prompt_templates import LEMMA_BREAKDOWN_USER_TEMPLATE

        rendered = LEMMA_BREAKDOWN_USER_TEMPLATE.format(
            statement_nl="commutativity",
            statement_lean="theorem add_comm := sorry",
            failed_attempts="direct proof failed",
        )
        assert "commutativity" in rendered
        assert "direct proof failed" in rendered

    def test_lemma_leanify_template(self):
        from agentic_research.agents.prompt_templates import LEMMA_LEANIFY_USER_TEMPLATE

        rendered = LEMMA_LEANIFY_USER_TEMPLATE.format(
            node_id="lemma_1",
            statement_nl="base case",
            parent_statement="theorem p := sorry",
            sibling_statements="-- none",
        )
        assert "lemma_1" in rendered
        assert "base case" in rendered

    def test_parent_proof_template(self):
        from agentic_research.agents.prompt_templates import PARENT_PROOF_USER_TEMPLATE

        rendered = PARENT_PROOF_USER_TEMPLATE.format(
            parent_statement="theorem p := sorry",
            child_declarations="theorem l1 := sorry",
        )
        assert "theorem p" in rendered
        assert "theorem l1" in rendered

    def test_flatten_template(self):
        from agentic_research.agents.prompt_templates import FLATTEN_PROOF_TEMPLATE

        rendered = FLATTEN_PROOF_TEMPLATE.format(
            root_statement="theorem main := sorry",
            proved_lemmas="theorem l1 := trivial",
            root_proof="theorem main := by exact l1",
        )
        assert "theorem main" in rendered
        assert "trivial" in rendered

    def test_failure_diagnosis_template(self):
        from agentic_research.agents.prompt_templates import FAILURE_DIAGNOSIS_USER_TEMPLATE

        rendered = FAILURE_DIAGNOSIS_USER_TEMPLATE.format(
            parent_statement="theorem p := sorry",
            child_declarations="theorem l1 := sorry",
            failed_proof="by simp",
            errors="type mismatch",
        )
        assert "type mismatch" in rendered

    def test_child_reformulation_template(self):
        from agentic_research.agents.prompt_templates import CHILD_REFORMULATION_TEMPLATE

        rendered = CHILD_REFORMULATION_TEMPLATE.format(
            parent_statement="theorem p := sorry",
            child_id="lemma_1",
            child_statement_nl="weak claim",
            child_statement_lean="theorem l1 := sorry",
            failure_type="weak_child_lemma",
            failure_description="too weak",
            suggested_fix="strengthen",
        )
        assert "weak_child_lemma" in rendered
        assert "strengthen" in rendered
