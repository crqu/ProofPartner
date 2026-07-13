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
    CritiqueIssue,
    CritiqueIssueType,
    CritiqueResult,
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

    def test_lean_preamble_passed_to_iterative_prover(self):
        from unittest.mock import patch
        from agentic_research.agents.proof_search import ProofSearchAgent
        from agentic_research.agents.prover import IterativeProver

        strategies_json = '{"strategies": [{"strategy_type": "direct", "description": "use simp", "plausibility": 0.9, "relevant_lemmas": [], "key_tactics": ["simp"]}]}'
        llm = _make_mock_llm([
            strategies_json,
            "```lean\ntheorem foo : True := trivial\n```",
        ])

        repl = _make_mock_repl()
        search = _make_mock_search()

        preamble = "import Mathlib\ndef Coupling := sorry"
        agent = ProofSearchAgent(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            prover_config=ProverConfig(max_iterations=1),
            lean_preamble=preamble,
        )

        captured_preamble = []
        original_init = IterativeProver.__init__

        def spy_init(self_prover, *args, **kwargs):
            captured_preamble.append(kwargs.get("lean_preamble"))
            original_init(self_prover, *args, **kwargs)

        with patch.object(IterativeProver, "__init__", spy_init):
            ctx = AgentContext(task="theorem foo : True")
            agent.run(ctx)

        assert len(captured_preamble) >= 1
        assert captured_preamble[0] == preamble


class TestIterativeProverPreamble:
    def test_preamble_included_in_first_attempt(self):
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm(["```lean\ntheorem foo : True := trivial\n```"])

        preamble = "import Mathlib\ndef WassersteinDist := sorry"
        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=1),
            lean_preamble=preamble,
        )

        ctx = AgentContext(task="theorem foo : True")
        prover.run(ctx)

        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "WassersteinDist" in user_content
        assert user_content.index("WassersteinDist") < user_content.index("theorem foo")

    def test_no_preamble_no_prefix(self):
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm(["```lean\ntheorem foo : True := trivial\n```"])

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(task="theorem foo : True")
        prover.run(ctx)

        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "WassersteinDist" not in user_content


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

    def test_circular_axiom_guard(self):
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        root_statement = "strong duality for distributionally robust optimization with T=1"
        child_statement = "Strong duality holds for distributionally robust optimization with T=1"
        response = (
            '{"lemmas": [{"node_id": "lemma_5", "statement_nl": "'
            + child_statement
            + '", "depends_on": [], "from_prior_work": true, '
            '"source_reference": "Blanchet & Murthy 2019"}], '
            '"topological_order": ["lemma_5"]}'
        )
        llm = _make_mock_llm([response])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(task=root_statement, metadata={"statement_lean": ""})
        result = agent.run(ctx)

        tree = LemmaTree.model_validate(result.result)
        assert tree.nodes["lemma_5"].from_prior_work is False
        assert tree.nodes["lemma_5"].source_reference is None

    def test_circular_axiom_guard_allows_genuine_prior_work(self):
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        root_statement = "strong duality for distributionally robust optimization with T=1"
        child_statement = "Kantorovich-Rubinstein duality for Wasserstein distance"
        response = (
            '{"lemmas": [{"node_id": "lemma_1", "statement_nl": "'
            + child_statement
            + '", "depends_on": [], "from_prior_work": true, '
            '"source_reference": "Villani 2009"}], '
            '"topological_order": ["lemma_1"]}'
        )
        llm = _make_mock_llm([response])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(task=root_statement, metadata={"statement_lean": ""})
        result = agent.run(ctx)

        tree = LemmaTree.model_validate(result.result)
        assert tree.nodes["lemma_1"].from_prior_work is True
        assert tree.nodes["lemma_1"].source_reference == "Villani 2009"


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


class TestFormatChildDeclaration:
    def test_axiom_node_gets_axiom_prefix(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        child = ProofNode(
            node_id="lemma_1",
            statement_nl="Kantorovich duality",
            statement_lean="∀ (μ ν : Measure Ω), kantorovich μ ν",
            from_prior_work=True,
        )
        result = RecursiveProver._format_child_declaration(child)
        assert result.startswith("axiom lemma_1 :")
        assert "-- Use: have <result> := lemma_1 <args>" in result

    def test_non_axiom_node_converted_to_axiom(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        child = ProofNode(
            node_id="lemma_2",
            statement_nl="some step",
            statement_lean="theorem lemma_2 : True := sorry",
            from_prior_work=False,
        )
        result = RecursiveProver._format_child_declaration(child)
        assert result.startswith("axiom lemma_2")
        assert "sorry" not in result.split("\n")[0]
        assert "-- Use: have <result> := lemma_2 <args>" in result

    def test_axiom_node_already_prefixed(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        child = ProofNode(
            node_id="lemma_3",
            statement_nl="compactness",
            statement_lean="axiom prokhorov_compact : ∀ K, IsCompact K",
            from_prior_work=True,
        )
        result = RecursiveProver._format_child_declaration(child)
        assert result.startswith("axiom prokhorov_compact")
        assert "axiom lemma_3" not in result


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


class TestVerifyAxiomNodes:
    """Tests for ProofPipeline._verify_axiom_nodes."""

    def _make_pipeline(self, mock_llm):
        from agentic_research.pipelines.proof import ProofPipeline

        repl = _make_mock_repl()
        search = _make_mock_search()
        return ProofPipeline(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
        )

    def test_axiom_node_passes_intent_check(self):
        from unittest.mock import patch

        from agentic_research.models.verification import IntentVerdict, IntentVerdictType

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main theorem",
                    statement_lean="theorem main := sorry",
                    children=["axiom_1"],
                ),
                "axiom_1": ProofNode(
                    node_id="axiom_1",
                    statement_nl="known result",
                    statement_lean="theorem known := sorry",
                    depth=1,
                    parent_id="root",
                    from_prior_work=True,
                    source_reference="Villani 2009",
                ),
            },
            topological_order=["axiom_1", "root"],
        )

        passing_verdict = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
        )

        llm = _make_mock_llm([])
        pipeline = self._make_pipeline(llm)

        with patch.object(
            pipeline, "_verify_axiom_nodes", wraps=pipeline._verify_axiom_nodes
        ):
            from agentic_research.agents.intent_judge import IntentJudge

            with patch.object(IntentJudge, "judge", return_value=passing_verdict):
                pipeline._verify_axiom_nodes(tree, "main theorem")

        node = tree.nodes["axiom_1"]
        assert node.from_prior_work is True
        assert node.status == NodeStatus.PENDING
        assert node.statement_lean == "theorem known := sorry"
        assert node.source_reference == "Villani 2009"

    def test_axiom_node_fails_intent_check(self):
        from unittest.mock import patch

        from agentic_research.models.verification import IntentVerdict, IntentVerdictType

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main theorem",
                    statement_lean="theorem main := sorry",
                    children=["axiom_1"],
                ),
                "axiom_1": ProofNode(
                    node_id="axiom_1",
                    statement_nl="known result",
                    statement_lean="theorem known := sorry",
                    depth=1,
                    parent_id="root",
                    from_prior_work=True,
                    source_reference="Villani 2009",
                ),
            },
            topological_order=["axiom_1", "root"],
        )

        failing_verdict = IntentVerdict(
            overall_verdict=IntentVerdictType.INCORRECT,
            overall_confidence=0.8,
            all_concerns=["statement does not match source"],
        )

        llm = _make_mock_llm([])
        pipeline = self._make_pipeline(llm)

        from agentic_research.agents.intent_judge import IntentJudge

        with patch.object(IntentJudge, "judge", return_value=failing_verdict):
            pipeline._verify_axiom_nodes(tree, "main theorem")

        node = tree.nodes["axiom_1"]
        assert node.status == NodeStatus.FAILED
        assert node.failure_diagnosis is not None
        assert "faithfulness" in node.failure_diagnosis.description.lower()
        assert node.from_prior_work is False
        assert node.source_reference is None
        assert node.statement_lean == "theorem known := sorry"

    def test_axiom_node_low_confidence_incorrect_accepted(self):
        """Low-confidence INCORRECT verdict is accepted with a warning."""
        from unittest.mock import patch

        from agentic_research.models.verification import IntentVerdict, IntentVerdictType

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main theorem",
                    statement_lean="theorem main := sorry",
                    children=["axiom_1"],
                ),
                "axiom_1": ProofNode(
                    node_id="axiom_1",
                    statement_nl="known result",
                    statement_lean="theorem known := sorry",
                    depth=1,
                    parent_id="root",
                    from_prior_work=True,
                    source_reference="Villani 2009",
                ),
            },
            topological_order=["axiom_1", "root"],
        )

        low_confidence_verdict = IntentVerdict(
            overall_verdict=IntentVerdictType.INCORRECT,
            overall_confidence=0.5,
            all_concerns=["extra hypotheses added"],
        )

        llm = _make_mock_llm([])
        pipeline = self._make_pipeline(llm)

        from agentic_research.agents.intent_judge import IntentJudge

        with patch.object(IntentJudge, "judge", return_value=low_confidence_verdict):
            pipeline._verify_axiom_nodes(tree, "main theorem")

        node = tree.nodes["axiom_1"]
        assert node.status == NodeStatus.PENDING
        assert node.from_prior_work is True
        assert node.source_reference == "Villani 2009"

    def test_non_axiom_nodes_skipped(self):
        from unittest.mock import patch

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main theorem",
                    statement_lean="theorem main := sorry",
                ),
            },
            topological_order=["root"],
        )

        llm = _make_mock_llm([])
        pipeline = self._make_pipeline(llm)

        from agentic_research.agents.intent_judge import IntentJudge

        with patch.object(IntentJudge, "judge") as mock_judge:
            pipeline._verify_axiom_nodes(tree, "main theorem")
            mock_judge.assert_not_called()


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

    def test_parent_proof_system_has_have_examples(self):
        from agentic_research.agents.prompt_templates import PARENT_PROOF_SYSTEM

        assert "have" in PARENT_PROOF_SYSTEM
        assert "axiom foo" in PARENT_PROOF_SYSTEM
        assert "axiom bar" in PARENT_PROOF_SYSTEM
        assert "BY NAME" in PARENT_PROOF_SYSTEM

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


# ---------------------------------------------------------------------------
# ProofCritic retry feedback (B-072)
# ---------------------------------------------------------------------------


class TestCriticFeedbackFormatting:
    def test_format_critic_feedback(self):
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        issues = [
            {
                "issue_type": "unstated_hypothesis",
                "node_id": "lemma_1",
                "description": "Assumes x > 0 without stating it",
                "severity": "blocking",
                "suggested_fix": "Add hypothesis x > 0",
            },
            {
                "issue_type": "weak_child_lemma",
                "node_id": "lemma_2",
                "description": "Lemma is too weak to prove parent",
                "severity": "warning",
                "suggested_fix": "",
            },
        ]
        formatted = LemmaBreakdown.format_critic_feedback(issues)
        assert "unstated_hypothesis" in formatted
        assert "lemma_1" in formatted
        assert "Assumes x > 0" in formatted
        assert "Add hypothesis x > 0" in formatted
        assert "weak_child_lemma" in formatted
        assert "lemma_2" in formatted

    def test_format_empty_issues(self):
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        assert LemmaBreakdown.format_critic_feedback([]) == ""

    def test_critic_feedback_appended_to_prompt(self):
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        response = '{"lemmas": [{"node_id": "lemma_1", "statement_nl": "revised step", "depends_on": [], "from_prior_work": false}], "topological_order": ["lemma_1"]}'
        llm = _make_mock_llm([response])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(
            task="prove P(n)",
            metadata={
                "statement_lean": "theorem p_n := sorry",
                "critic_issues": [
                    {
                        "issue_type": "unstated_hypothesis",
                        "node_id": "lemma_1",
                        "description": "Missing boundedness",
                        "severity": "blocking",
                        "suggested_fix": "Add bounded hypothesis",
                    },
                ],
            },
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        call_args = llm.complete.call_args
        user_message = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][1][0]["content"]
        assert "Previous Issues Identified by Proof Critic" in user_message
        assert "Missing boundedness" in user_message

    def test_no_feedback_when_no_issues(self):
        from agentic_research.agents.lemma_breakdown import LemmaBreakdown

        response = '{"lemmas": [{"node_id": "lemma_1", "statement_nl": "step", "depends_on": [], "from_prior_work": false}], "topological_order": ["lemma_1"]}'
        llm = _make_mock_llm([response])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(
            task="prove P(n)",
            metadata={"statement_lean": "theorem p_n := sorry"},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        call_args = llm.complete.call_args
        user_message = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][1][0]["content"]
        assert "Previous Issues" not in user_message


class TestProofCriticRetryFeedback:
    def _make_pipeline(self, mock_llm):
        from agentic_research.pipelines.proof import ProofPipeline

        repl = _make_mock_repl()
        search = _make_mock_search()
        return ProofPipeline(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            use_proof_critic=True,
            max_critic_retries=2,
        )

    def test_proof_critic_feedback_passed_to_retry(self):
        from unittest.mock import patch

        issues = [
            CritiqueIssue(
                issue_type=CritiqueIssueType.UNSTATED_HYPOTHESIS,
                node_id="lemma_1",
                description="Missing measurability",
                severity="blocking",
                suggested_fix="Add Measurable f hypothesis",
            ),
            CritiqueIssue(
                issue_type=CritiqueIssueType.WEAK_CHILD_LEMMA,
                node_id="lemma_2",
                description="Too weak to imply parent",
                severity="blocking",
                suggested_fix="Strengthen conclusion",
            ),
        ]

        failing_critique = CritiqueResult(issues=issues, passed=False)
        passing_critique = CritiqueResult(issues=[], passed=True)

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

        llm = _make_mock_llm([])
        pipeline = self._make_pipeline(llm)

        breakdown_calls = []

        def tracking_run_lemma(*args, **kwargs):
            breakdown_calls.append(kwargs.get("critic_feedback"))
            return tree

        critic_call_count = [0]

        def mock_run_critic(*args, **kwargs):
            critic_call_count[0] += 1
            if critic_call_count[0] == 1:
                return failing_critique
            return passing_critique

        with (
            patch.object(pipeline, "_run_lemma_breakdown", side_effect=tracking_run_lemma),
            patch.object(pipeline, "_run_proof_critic", side_effect=mock_run_critic),
            patch.object(pipeline, "_run_proof_search") as mock_search,
            patch.object(pipeline, "_run_lemma_leanifier", return_value=tree),
            patch.object(pipeline, "_run_recursive_prover") as mock_recursive,
            patch.object(pipeline, "_run_flatten_finalize", return_value="proof code"),
        ):
            mock_search.return_value = ProofSearchResult(
                statement="theorem main := sorry",
                needs_decomposition=True,
            )
            mock_recursive.return_value = RecursiveProofResult(
                root_statement="theorem main := sorry",
                proved=True,
                total_nodes=1,
                proved_nodes=1,
                lemma_tree=tree,
            )

            pipeline.run("theorem main := sorry", "main theorem")

        assert len(breakdown_calls) >= 2
        assert breakdown_calls[0] is None
        assert breakdown_calls[1] is not None
        assert len(breakdown_calls[1]) == 2
        assert breakdown_calls[1][0].issue_type == CritiqueIssueType.UNSTATED_HYPOTHESIS


# ---------------------------------------------------------------------------
# Parent-before-children validation (H3)
# ---------------------------------------------------------------------------


class TestParentBeforeChildrenSorryStubs:
    """Verify parent proof receives child axiom declarations as context."""

    def test_parent_proof_includes_child_axiom_stubs(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="parent",
                    statement_lean="theorem parent : Nat → Nat := sorry",
                    children=["child_1", "child_2"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="first helper",
                    statement_lean="theorem child_1 : Nat → Bool := sorry",
                    depth=1,
                    parent_id="root",
                ),
                "child_2": ProofNode(
                    node_id="child_2",
                    statement_nl="second helper",
                    statement_lean="lemma child_2 : Bool → Nat := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "child_2", "root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem parent : Nat → Nat := trivial\n```",
            "```lean\ntheorem child_1 : Nat → Bool := trivial\n```",
            "```lean\ntheorem child_2 : Bool → Nat := trivial\n```",
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
        agent.run(ctx)

        parent_call = llm.complete.call_args_list[0]
        user_msg = parent_call[1]["messages"][0]["content"]
        assert "axiom child_1" in user_msg
        assert "axiom child_2" in user_msg
        assert "sorry" not in user_msg.split("Child Lemma Declarations")[1]

    def test_child_declarations_formatted_as_axioms(self):
        """All children (not just from_prior_work) become axiom declarations."""
        from agentic_research.agents.recursive_prover import RecursiveProver

        regular_child = ProofNode(
            node_id="lemma_1",
            statement_nl="helper",
            statement_lean="theorem lemma_1 : True := sorry",
            from_prior_work=False,
        )
        result = RecursiveProver._format_child_declaration(regular_child)
        assert result.startswith("axiom lemma_1")
        assert "sorry" not in result.split("\n")[0]

    def test_child_declaration_preserves_axiom_prefix(self):
        from agentic_research.agents.recursive_prover import RecursiveProver

        child = ProofNode(
            node_id="lemma_3",
            statement_nl="existing axiom",
            statement_lean="axiom lemma_3 : True",
            from_prior_work=True,
        )
        result = RecursiveProver._format_child_declaration(child)
        assert result.startswith("axiom lemma_3")
        assert result.count("axiom") == 1


class TestWeakChildLemmaDetection:
    """Verify WEAK_CHILD_LEMMA is detected when parent fails with stubs."""

    def test_weak_child_diagnosed_on_parent_failure(self):
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
                    statement_nl="weak child",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        diagnosis_json = (
            '{"failure_type": "weak_child_lemma", '
            '"description": "child_1 too weak", '
            '"suggested_fix": "strengthen child_1"}'
        )
        llm = _make_mock_llm([
            "```lean\n-- MOCK_ERROR\nfailed proof\n```",
            diagnosis_json,
        ])
        repl = _make_mock_repl()

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
            max_retries_per_node=1,
        )
        ctx = AgentContext(
            task="prove",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.FAILURE
        rr = RecursiveProofResult.model_validate(result.result)
        assert not rr.proved
        root_node = rr.lemma_tree.nodes["root"]
        assert root_node.failure_diagnosis is not None
        assert root_node.failure_diagnosis.failure_type == FailureType.WEAK_CHILD_LEMMA

    def test_successful_parent_then_recurse_children(self):
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
                    statement_nl="first helper",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
                "child_2": ProofNode(
                    node_id="child_2",
                    statement_nl="second helper",
                    statement_lean="theorem child_2 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "child_2", "root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem parent := trivial\n```",
            "```lean\ntheorem child_1 : True := trivial\n```",
            "```lean\ntheorem child_2 : True := trivial\n```",
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
        assert rr.proved_nodes == 3
        assert rr.lemma_tree.nodes["root"].status == NodeStatus.PROVED
        assert rr.lemma_tree.nodes["child_1"].status == NodeStatus.PROVED
        assert rr.lemma_tree.nodes["child_2"].status == NodeStatus.PROVED

    def test_reformulation_on_weak_child(self):
        """When WEAK_CHILD_LEMMA detected, child is reformulated and parent retried."""
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
                    statement_nl="original weak claim",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        diagnosis_json = (
            '{"failure_type": "weak_child_lemma", '
            '"description": "child_1 too weak", '
            '"problematic_child_id": "child_1", '
            '"suggested_fix": "strengthen"}'
        )
        reformulation_json = (
            '{"reformulated_statement": "stronger claim", '
            '"reasoning": "need stronger hypothesis"}'
        )
        llm = _make_mock_llm([
            "```lean\n-- MOCK_ERROR\nfailed\n```",
            diagnosis_json,
            reformulation_json,
            "```lean\ntheorem parent := trivial\n```",
            "```lean\ntheorem child_1 : True := trivial\n```",
        ])
        repl = _make_mock_repl()

        agent = RecursiveProver(
            llm_client=llm,
            lean_repl=repl,
            prover_config=ProverConfig(max_iterations=1),
            max_retries_per_node=2,
        )
        ctx = AgentContext(
            task="prove",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        rr = RecursiveProofResult.model_validate(result.result)
        child_node = rr.lemma_tree.nodes["child_1"]
        assert child_node.statement_nl == "stronger claim"


class TestPipelineWeakChildRetry:
    """Verify ProofPipeline retries LemmaBreakdown on WEAK_CHILD_LEMMA."""

    def _make_pipeline(self, mock_llm):
        from agentic_research.pipelines.proof import ProofPipeline

        repl = _make_mock_repl()
        search = _make_mock_search()
        return ProofPipeline(
            llm_client=mock_llm,
            lean_repl=repl,
            lean_search=search,
            use_proof_critic=False,
            use_proof_detailer=False,
        )

    def test_extract_weak_child_feedback(self):
        from agentic_research.pipelines.proof import ProofPipeline

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="parent",
                    statement_lean="theorem parent := sorry",
                    children=["child_1"],
                    status=NodeStatus.FAILED,
                    failure_diagnosis=FailureDiagnosis(
                        failure_type=FailureType.WEAK_CHILD_LEMMA,
                        description="child_1 is too weak",
                        problematic_child_id="child_1",
                        suggested_fix="strengthen",
                    ),
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="weak helper",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        result = RecursiveProofResult(
            root_statement="theorem parent := sorry",
            proved=False,
            lemma_tree=tree,
            failure_reason="Not all nodes proved",
        )

        feedback = ProofPipeline._extract_weak_child_feedback(result)
        assert len(feedback) == 1
        assert feedback[0].issue_type == CritiqueIssueType.WEAK_CHILD_LEMMA
        assert feedback[0].node_id == "child_1"
        assert "too weak" in feedback[0].description

    def test_no_feedback_on_stuck_goal(self):
        from agentic_research.pipelines.proof import ProofPipeline

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="parent",
                    statement_lean="theorem parent := sorry",
                    status=NodeStatus.FAILED,
                    failure_diagnosis=FailureDiagnosis(
                        failure_type=FailureType.STUCK_GOAL,
                        description="stuck",
                    ),
                ),
            },
            topological_order=["root"],
        )

        result = RecursiveProofResult(
            root_statement="theorem parent := sorry",
            proved=False,
            lemma_tree=tree,
            failure_reason="Not all nodes proved",
        )

        feedback = ProofPipeline._extract_weak_child_feedback(result)
        assert len(feedback) == 0

    def test_pipeline_retries_on_weak_child(self):
        from unittest.mock import patch

        tree_v1 = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    statement_lean="theorem main := sorry",
                    children=["child_1"],
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="weak helper",
                    statement_lean="theorem child_1 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        tree_v2 = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    statement_lean="theorem main := sorry",
                    children=["child_1_v2"],
                ),
                "child_1_v2": ProofNode(
                    node_id="child_1_v2",
                    statement_nl="stronger helper",
                    statement_lean="theorem child_1_v2 : True := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1_v2", "root"],
        )

        failed_tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    statement_lean="theorem main := sorry",
                    children=["child_1"],
                    status=NodeStatus.FAILED,
                    failure_diagnosis=FailureDiagnosis(
                        failure_type=FailureType.WEAK_CHILD_LEMMA,
                        description="child_1 too weak",
                        problematic_child_id="child_1",
                        suggested_fix="strengthen",
                    ),
                ),
                "child_1": ProofNode(
                    node_id="child_1",
                    statement_nl="weak",
                    statement_lean="theorem child_1 := sorry",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["child_1", "root"],
        )

        llm = _make_mock_llm([])
        pipeline = self._make_pipeline(llm)

        breakdown_calls = []

        def tracking_breakdown(*args, critic_feedback=None, **kwargs):
            breakdown_calls.append(critic_feedback)
            if critic_feedback:
                return tree_v2
            return tree_v1

        recursive_call_count = [0]

        def mock_recursive(tree):
            recursive_call_count[0] += 1
            if recursive_call_count[0] == 1:
                return RecursiveProofResult(
                    root_statement="theorem main := sorry",
                    proved=False,
                    lemma_tree=failed_tree,
                    failure_reason="Not all nodes proved",
                )
            return RecursiveProofResult(
                root_statement="theorem main := sorry",
                proved=True,
                total_nodes=2,
                proved_nodes=2,
                lemma_tree=tree_v2,
            )

        with (
            patch.object(pipeline, "_run_lemma_breakdown", side_effect=tracking_breakdown),
            patch.object(pipeline, "_run_proof_search") as mock_search,
            patch.object(pipeline, "_run_lemma_leanifier", side_effect=lambda t, **kw: t),
            patch.object(pipeline, "_run_recursive_prover", side_effect=mock_recursive),
            patch.object(pipeline, "_run_flatten_finalize", return_value="proof code"),
        ):
            mock_search.return_value = ProofSearchResult(
                statement="theorem main := sorry",
                needs_decomposition=True,
            )

            result = pipeline.run("theorem main := sorry", "main theorem")

        assert result.proved
        assert len(breakdown_calls) >= 2
        assert breakdown_calls[0] is None
        assert breakdown_calls[1] is not None
        assert breakdown_calls[1][0].issue_type == CritiqueIssueType.WEAK_CHILD_LEMMA
