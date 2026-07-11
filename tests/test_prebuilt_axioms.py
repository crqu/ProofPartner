"""Tests for pre-built axiom library and axiom matching in LemmaLeanifier.

All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic_research.data_packages import get_package
from agentic_research.data_packages.dro_coupling import LEAN_AXIOMS
from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.proof import LemmaTree, ProofNode


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
    return mock


def _make_mock_repl():
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


# ---------------------------------------------------------------------------
# DROCouplingPackage.provided_axioms()
# ---------------------------------------------------------------------------


class TestProvidedAxioms:
    def test_returns_expected_axiom_names(self):
        pkg = get_package("dro_coupling")
        axioms = pkg.provided_axioms()
        expected = {
            "identity_coupling_exists",
            "wassersteinDist_self",
            "wassersteinDist_nonneg",
            "self_mem_wassersteinBall",
            "wassersteinBall_mono",
            "mem_wassersteinBall_iff",
            "coupling_fst",
            "coupling_snd",
            "wassersteinDist_triangle",
        }
        assert set(axioms.keys()) == expected

    def test_axiom_values_are_complete_declarations(self):
        pkg = get_package("dro_coupling")
        axioms = pkg.provided_axioms()
        for name, decl in axioms.items():
            assert decl.startswith("axiom "), f"{name} doesn't start with 'axiom'"
            assert name in decl, f"{name} not found in its own declaration"

    def test_axioms_present_in_lean_preamble(self):
        pkg = get_package("dro_coupling")
        preamble = pkg.lean_preamble()
        axioms = pkg.provided_axioms()
        for name, decl in axioms.items():
            assert decl in preamble, f"Axiom {name} not found in LEAN_PREAMBLE"

    def test_axiom_keywords_cover_all_axioms(self):
        pkg = get_package("dro_coupling")
        axioms = pkg.provided_axioms()
        keywords = pkg.axiom_keywords()
        assert set(keywords.keys()) == set(axioms.keys())

    def test_returns_copy(self):
        pkg = get_package("dro_coupling")
        a1 = pkg.provided_axioms()
        a2 = pkg.provided_axioms()
        assert a1 == a2
        a1.pop(next(iter(a1)))
        assert len(a2) > len(a1)

    def test_parsed_axioms_match_module_constant(self):
        assert len(LEAN_AXIOMS) == 9


# ---------------------------------------------------------------------------
# LemmaLeanifier._match_prebuilt_axiom
# ---------------------------------------------------------------------------


class TestMatchPrebuiltAxiom:
    def test_matches_wasserstein_self_distance(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        pkg = get_package("dro_coupling")
        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            prebuilt_axioms=pkg.provided_axioms(),
            axiom_keywords=pkg.axiom_keywords(),
        )

        node = ProofNode(
            node_id="lemma_1",
            statement_nl="Wasserstein distance from P to itself is zero",
            from_prior_work=True,
        )
        result = agent._match_prebuilt_axiom(node)
        assert result is not None
        assert "wassersteinDist_self" in result

    def test_matches_triangle_inequality(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        pkg = get_package("dro_coupling")
        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            prebuilt_axioms=pkg.provided_axioms(),
            axiom_keywords=pkg.axiom_keywords(),
        )

        node = ProofNode(
            node_id="lemma_2",
            statement_nl="The Wasserstein distance satisfies the triangle inequality",
        )
        result = agent._match_prebuilt_axiom(node)
        assert result is not None
        assert "wassersteinDist_triangle" in result

    def test_matches_ball_monotonicity(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        pkg = get_package("dro_coupling")
        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            prebuilt_axioms=pkg.provided_axioms(),
            axiom_keywords=pkg.axiom_keywords(),
        )

        node = ProofNode(
            node_id="lemma_3",
            statement_nl="The Wasserstein ball is monotone in its radius",
        )
        result = agent._match_prebuilt_axiom(node)
        assert result is not None
        assert "wassersteinBall_mono" in result

    def test_no_match_unrelated_statement(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        pkg = get_package("dro_coupling")
        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            prebuilt_axioms=pkg.provided_axioms(),
            axiom_keywords=pkg.axiom_keywords(),
        )

        node = ProofNode(
            node_id="lemma_4",
            statement_nl="Every bounded sequence in R^n has a convergent subsequence",
        )
        result = agent._match_prebuilt_axiom(node)
        assert result is None

    def test_no_match_without_axiom_library(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        agent = LemmaLeanifier(llm_client=llm, lean_repl=repl)

        node = ProofNode(
            node_id="lemma_1",
            statement_nl="Wasserstein distance from P to itself is zero",
        )
        result = agent._match_prebuilt_axiom(node)
        assert result is None


# ---------------------------------------------------------------------------
# from_prior_work node with prebuilt axiom skips LLM (zero tokens)
# ---------------------------------------------------------------------------


class TestPrebuiltAxiomSkipsLLM:
    def test_prior_work_node_uses_prebuilt_zero_tokens(self):
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
                    statement_nl="Wasserstein distance from P to itself is zero",
                    depth=1,
                    parent_id="root",
                    from_prior_work=True,
                    source_reference="Villani 2009",
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        pkg = get_package("dro_coupling")

        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            prebuilt_axioms=pkg.provided_axioms(),
            axiom_keywords=pkg.axiom_keywords(),
        )
        ctx = AgentContext(
            task="leanify",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        updated_tree = LemmaTree.model_validate(result.result)
        assert "wassersteinDist_self" in updated_tree.nodes["lemma_1"].statement_lean
        llm.complete.assert_not_called()

    def test_non_prior_work_node_gets_prebuilt_axiom(self):
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
                    statement_nl="The Wasserstein distance satisfies the triangle inequality",
                    depth=1,
                    parent_id="root",
                    from_prior_work=False,
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        pkg = get_package("dro_coupling")

        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            prebuilt_axioms=pkg.provided_axioms(),
            axiom_keywords=pkg.axiom_keywords(),
        )
        ctx = AgentContext(
            task="leanify",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        updated_tree = LemmaTree.model_validate(result.result)
        assert "wassersteinDist_triangle" in updated_tree.nodes["lemma_1"].statement_lean
        llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Pipeline threading
# ---------------------------------------------------------------------------


class TestPipelineAxiomThreading:
    def test_detect_lean_preamble_loads_axioms(self):
        from agentic_research.pipelines.proof import ProofPipeline

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        from agentic_research.tools.lean_search import (
            LeanSearch,
            SearchBackend,
            SearchConfig,
        )

        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            use_claim_check=False,
        )

        result = pipeline._detect_lean_preamble(
            "Wasserstein distance in distributionally robust optimization"
        )
        assert result is not None
        assert pipeline._prebuilt_axioms is not None
        assert "wassersteinDist_self" in pipeline._prebuilt_axioms
        assert pipeline._axiom_keywords is not None
        assert "wassersteinDist_self" in pipeline._axiom_keywords

    def test_detect_lean_preamble_no_match(self):
        from agentic_research.pipelines.proof import ProofPipeline

        llm = _make_mock_llm([])
        repl = _make_mock_repl()
        from agentic_research.tools.lean_search import (
            LeanSearch,
            SearchBackend,
            SearchConfig,
        )

        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            use_claim_check=False,
        )

        result = pipeline._detect_lean_preamble("Every compact set is bounded")
        assert result is None
        assert pipeline._prebuilt_axioms is None
