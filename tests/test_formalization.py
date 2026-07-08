"""Tests for Phase 5: Type-first formalization pipeline.

All LLM calls are mocked — no real API calls are made.
Lean REPL uses mock backend for deterministic testing.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.formalization import (
    AuctionResult,
    AuctionScore,
    AuctionVerdict,
    AuxiliaryLemma,
    ClaimCheckResult,
    ClaimCheckVerdict,
    FormalizationPipelineResult,
    LemmaStatement,
    TheoremFormalization,
    TypeCandidate,
    TypeDependencyGraph,
    TypeFormalizationCandidate,
    TypePlan,
)


# ---------------------------------------------------------------------------
# models/formalization.py
# ---------------------------------------------------------------------------


class TestTypeCandidate:
    def test_basic(self):
        tc = TypeCandidate(name="QuasiRandomGraph", informal_description="A graph with quasi-random properties")
        assert tc.name == "QuasiRandomGraph"
        assert tc.is_in_mathlib is False
        assert tc.depends_on == []

    def test_with_dependencies(self):
        tc = TypeCandidate(
            name="ColoredGraph",
            informal_description="Graph with vertex coloring",
            depends_on=["SimpleGraph"],
            is_in_mathlib=False,
        )
        assert tc.depends_on == ["SimpleGraph"]

    def test_serialization(self):
        tc = TypeCandidate(
            name="T", informal_description="d",
            lean_signature="structure T where", depends_on=["A"],
            mathlib_analog="Nat", is_in_mathlib=False,
        )
        restored = TypeCandidate.model_validate(tc.model_dump())
        assert restored == tc


class TestTypePlan:
    def test_basic(self):
        plan = TypePlan(conjecture_statement="all primes are odd")
        assert plan.candidates == []
        assert plan.mathlib_imports == []

    def test_full(self):
        plan = TypePlan(
            conjecture_statement="test",
            candidates=[TypeCandidate(name="T", informal_description="d")],
            dependency_graph=TypeDependencyGraph(
                edges=[("A", "B")],
                topological_order=["B", "A"],
            ),
            mathlib_imports=["Mathlib.Data.Nat.Basic"],
        )
        assert len(plan.candidates) == 1
        assert plan.dependency_graph.topological_order == ["B", "A"]

    def test_serialization(self):
        plan = TypePlan(
            conjecture_statement="s",
            candidates=[TypeCandidate(name="T", informal_description="d")],
        )
        restored = TypePlan.model_validate(plan.model_dump())
        assert restored == plan


class TestLemmaStatement:
    def test_basic(self):
        ls = LemmaStatement(name="add_comm", statement_nl="addition is commutative", for_type="Nat")
        assert ls.is_well_known is True
        assert ls.statement_lean == ""

    def test_serialization(self):
        ls = LemmaStatement(
            name="l", statement_nl="s", for_type="T", is_well_known=False,
        )
        restored = LemmaStatement.model_validate(ls.model_dump())
        assert restored == ls


class TestTypeFormalizationCandidate:
    def test_proved_ratio_empty(self):
        c = TypeFormalizationCandidate(candidate_id=0, type_name="T")
        assert c.proved_ratio == 0.0
        assert c.proved_count == 0

    def test_proved_ratio(self):
        c = TypeFormalizationCandidate(
            candidate_id=0,
            type_name="T",
            lean_code="structure T where",
            compiles=True,
            auxiliary_lemmas=[
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l1", statement_nl="s1", for_type="T"),
                    proved=True,
                ),
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l2", statement_nl="s2", for_type="T"),
                    proved=False,
                ),
            ],
        )
        assert c.proved_count == 1
        assert c.total_lemma_count == 2
        assert abs(c.proved_ratio - 0.5) < 1e-9

    def test_serialization(self):
        c = TypeFormalizationCandidate(
            candidate_id=1, type_name="T", lean_code="def T := Nat", compiles=True,
        )
        restored = TypeFormalizationCandidate.model_validate(c.model_dump())
        assert restored.candidate_id == c.candidate_id


class TestAuctionScore:
    def test_basic(self):
        score = AuctionScore(
            candidate_id=0,
            lemma_ratio=0.8,
            brevity_score=0.5,
            compilation_score=1.0,
            total_score=0.75,
        )
        assert score.total_score == 0.75


class TestAuctionResult:
    def test_accepted(self):
        ar = AuctionResult(
            type_name="T",
            verdict=AuctionVerdict.ACCEPTED,
            winner_id=0,
            reason="Best candidate",
        )
        assert ar.verdict == AuctionVerdict.ACCEPTED

    def test_retry(self):
        ar = AuctionResult(
            type_name="T",
            verdict=AuctionVerdict.RETRY,
            reason="No candidate met threshold",
        )
        assert ar.verdict == AuctionVerdict.RETRY
        assert ar.winner_id is None


class TestClaimCheckResult:
    def test_pass(self):
        r = ClaimCheckResult(
            verdict=ClaimCheckVerdict.PASS,
            reason="Faithful",
        )
        assert r.verdict == ClaimCheckVerdict.PASS

    def test_fail(self):
        r = ClaimCheckResult(
            verdict=ClaimCheckVerdict.FAIL,
            reason="Weakened statement",
            statement_preserved=False,
        )
        assert r.verdict == ClaimCheckVerdict.FAIL


class TestTheoremFormalization:
    def test_basic(self):
        t = TheoremFormalization(
            conjecture_nl="all primes > 2 are odd",
            lean_statement="theorem foo : True := sorry",
            compiles=True,
            iterations_used=1,
        )
        assert t.compiles
        assert t.failure_reason is None


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = []
    for text in responses:
        side_effects.append(LLMResponse(
            content=text,
            model="claude-opus-4-6-20250616",
            stop_reason="end_turn",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        ))
    mock.complete.side_effect = side_effects

    real_client_cls = LLMClient
    with patch("anthropic.Anthropic"):
        temp_client = real_client_cls.__new__(real_client_cls)
    mock.extract_json = temp_client.__class__.extract_json.__get__(mock, type(mock))

    return mock


MOCK_TYPE_PLAN_RESPONSE = json.dumps({
    "candidates": [
        {
            "name": "QuasiRandomGraph",
            "informal_description": "A graph with quasi-random properties",
            "lean_signature": "structure QuasiRandomGraph where",
            "depends_on": [],
            "mathlib_analog": None,
            "is_in_mathlib": False,
            "composition_alternative": None,
        },
    ],
    "dependency_graph": {
        "edges": [],
        "topological_order": ["QuasiRandomGraph"],
    },
    "mathlib_imports": ["Mathlib.Combinatorics.SimpleGraph.Basic"],
})

MOCK_LEMMA_PLAN_RESPONSE = json.dumps({
    "lemmas": [
        {
            "name": "quasi_random_nonempty",
            "statement_nl": "A quasi-random graph has at least one vertex",
            "for_type": "QuasiRandomGraph",
            "is_well_known": True,
        },
        {
            "name": "quasi_random_symmetric",
            "statement_nl": "The edge relation in a quasi-random graph is symmetric",
            "for_type": "QuasiRandomGraph",
            "is_well_known": True,
        },
    ],
})

MOCK_TYPE_LEAN_CODE = "```lean\nstructure QuasiRandomGraph where\n  vertices : Finset Nat\n  edges : Finset (Nat × Nat)\n```"

MOCK_LEMMA_LEAN_CODE = "```lean\ntheorem quasi_random_nonempty (g : QuasiRandomGraph) : g.vertices.Nonempty := sorry\n```"

MOCK_PROOF_CODE = "```lean\ntheorem quasi_random_nonempty (g : QuasiRandomGraph) : g.vertices.Nonempty := trivial\n```"

MOCK_THEOREM_LEAN = "```lean\ntheorem main_conjecture : True := sorry\n```"

MOCK_CLAIM_CHECK_PASS = json.dumps({
    "verdict": "pass",
    "reason": "Formalization faithfully captures the conjecture",
    "statement_preserved": True,
})

MOCK_CLAIM_CHECK_FAIL = json.dumps({
    "verdict": "fail",
    "reason": "The formalization silently weakened the statement",
    "statement_preserved": False,
})


# ---------------------------------------------------------------------------
# agents/type_planner.py
# ---------------------------------------------------------------------------


class TestTypePlanner:
    def test_basic_planning(self):
        from agentic_research.agents.type_planner import TypePlanner
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm([MOCK_TYPE_PLAN_RESPONSE])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        planner = TypePlanner(llm_client=llm, lean_search=search)
        ctx = AgentContext(task="quasi-random graphs have specific edge distribution")
        result = planner.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        plan = TypePlan.model_validate(result.result)
        assert len(plan.candidates) == 1
        assert plan.candidates[0].name == "QuasiRandomGraph"
        assert not plan.candidates[0].is_in_mathlib

    def test_fallback_on_bad_json(self):
        from agentic_research.agents.type_planner import TypePlanner
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm(["not valid json"])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        planner = TypePlanner(llm_client=llm, lean_search=search)
        ctx = AgentContext(task="test")
        result = planner.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        plan = TypePlan.model_validate(result.result)
        assert plan.candidates == []

    def test_token_tracking(self):
        from agentic_research.agents.type_planner import TypePlanner
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm([MOCK_TYPE_PLAN_RESPONSE])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        planner = TypePlanner(llm_client=llm, lean_search=search)
        ctx = AgentContext(task="test")
        result = planner.run(ctx)

        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 50

    def test_properties(self):
        from agentic_research.agents.type_planner import TypePlanner
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm([])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        planner = TypePlanner(llm_client=llm, lean_search=search)
        assert planner.name == "type_planner"

    def test_composition_alternative_field(self):
        """Test that TypeCandidate accepts composition_alternative field."""
        from agentic_research.agents.type_planner import TypePlanner
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        # Mock response suggesting composition over new type
        response = json.dumps({
            "candidates": [
                {
                    "name": "UniformLipschitz",
                    "informal_description": "f is Lipschitz in x uniformly over y",
                    "lean_signature": "structure UniformLipschitz where",
                    "depends_on": [],
                    "mathlib_analog": "LipschitzWith",
                    "is_in_mathlib": False,
                    "composition_alternative": "∀ y, LipschitzWith K (fun x => f x y)",
                },
            ],
            "dependency_graph": {"edges": [], "topological_order": []},
            "mathlib_imports": ["Mathlib.Topology.MetricSpace.Lipschitz"],
        })

        llm = _make_mock_llm([response])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        planner = TypePlanner(llm_client=llm, lean_search=search)
        ctx = AgentContext(task="f is L-Lipschitz in x uniformly over y")
        result = planner.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        plan = TypePlan.model_validate(result.result)
        assert len(plan.candidates) == 1
        assert plan.candidates[0].composition_alternative == "∀ y, LipschitzWith K (fun x => f x y)"

    def test_genuinely_new_type_null_composition(self):
        """Test that genuinely new types have null composition_alternative."""
        from agentic_research.agents.type_planner import TypePlanner
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        # Mock response for genuinely new type
        response = json.dumps({
            "candidates": [
                {
                    "name": "QuasiRandomGraph",
                    "informal_description": "A graph with quasi-random properties",
                    "lean_signature": "structure QuasiRandomGraph where",
                    "depends_on": [],
                    "mathlib_analog": None,
                    "is_in_mathlib": False,
                    "composition_alternative": None,
                },
            ],
            "dependency_graph": {"edges": [], "topological_order": []},
            "mathlib_imports": [],
        })

        llm = _make_mock_llm([response])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        planner = TypePlanner(llm_client=llm, lean_search=search)
        ctx = AgentContext(task="quasi-random graphs")
        result = planner.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        plan = TypePlan.model_validate(result.result)
        assert len(plan.candidates) == 1
        assert plan.candidates[0].composition_alternative is None


# ---------------------------------------------------------------------------
# agents/lemma_planner.py
# ---------------------------------------------------------------------------


class TestLemmaPlanner:
    def test_basic_planning(self):
        from agentic_research.agents.lemma_planner import LemmaPlanner

        llm = _make_mock_llm([MOCK_LEMMA_PLAN_RESPONSE])

        plan = TypePlan(
            conjecture_statement="test",
            candidates=[TypeCandidate(
                name="QuasiRandomGraph",
                informal_description="A graph with quasi-random properties",
                is_in_mathlib=False,
            )],
        )

        planner = LemmaPlanner(llm_client=llm)
        ctx = AgentContext(
            task="plan lemmas",
            metadata={"type_plan": plan.model_dump()},
        )
        result = planner.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        lemmas = [LemmaStatement.model_validate(lem) for lem in result.result["lemmas"]]
        assert len(lemmas) == 2
        assert lemmas[0].for_type == "QuasiRandomGraph"

    def test_skips_mathlib_types(self):
        from agentic_research.agents.lemma_planner import LemmaPlanner

        llm = _make_mock_llm([])

        plan = TypePlan(
            conjecture_statement="test",
            candidates=[TypeCandidate(
                name="Nat",
                informal_description="Natural numbers",
                is_in_mathlib=True,
            )],
        )

        planner = LemmaPlanner(llm_client=llm)
        ctx = AgentContext(
            task="plan lemmas",
            metadata={"type_plan": plan.model_dump()},
        )
        result = planner.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        assert result.result["lemmas"] == []
        assert llm.complete.call_count == 0

    def test_max_lemmas_per_type(self):
        from agentic_research.agents.lemma_planner import LemmaPlanner

        many_lemmas = json.dumps({
            "lemmas": [
                {"name": f"lemma_{i}", "statement_nl": f"prop {i}", "for_type": "T"}
                for i in range(10)
            ],
        })
        llm = _make_mock_llm([many_lemmas])

        plan = TypePlan(
            conjecture_statement="test",
            candidates=[TypeCandidate(name="T", informal_description="d", is_in_mathlib=False)],
        )

        planner = LemmaPlanner(llm_client=llm, max_lemmas_per_type=3)
        ctx = AgentContext(task="plan", metadata={"type_plan": plan.model_dump()})
        result = planner.run(ctx)

        lemmas = result.result["lemmas"]
        assert len(lemmas) <= 3

    def test_properties(self):
        from agentic_research.agents.lemma_planner import LemmaPlanner

        llm = _make_mock_llm([])
        planner = LemmaPlanner(llm_client=llm)
        assert planner.name == "lemma_planner"


# ---------------------------------------------------------------------------
# agents/type_formalizer.py
# ---------------------------------------------------------------------------


class TestTypeFormalizer:
    def test_successful_formalization(self):
        from agentic_research.agents.type_formalizer import TypeFormalizer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([
            MOCK_TYPE_LEAN_CODE,
            MOCK_LEMMA_LEAN_CODE,
            MOCK_PROOF_CODE,
        ])

        formalizer = TypeFormalizer(
            llm_client=llm,
            lean_repl=repl,
            candidate_id=0,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(
            task="QuasiRandomGraph",
            metadata={
                "type_candidate": TypeCandidate(
                    name="QuasiRandomGraph",
                    informal_description="quasi-random graph",
                ).model_dump(),
                "lemmas": [
                    LemmaStatement(
                        name="qr_nonempty",
                        statement_nl="non-empty",
                        for_type="QuasiRandomGraph",
                    ).model_dump(),
                ],
                "prior_definitions": "",
            },
        )

        result = formalizer.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        candidate = TypeFormalizationCandidate.model_validate(result.result)
        assert candidate.compiles
        assert candidate.candidate_id == 0

    def test_compilation_failure(self):
        from agentic_research.agents.type_formalizer import TypeFormalizer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        error_code = "```lean\n-- MOCK_ERROR\nbad code\n```"
        llm = _make_mock_llm([error_code] * 5)

        formalizer = TypeFormalizer(
            llm_client=llm,
            lean_repl=repl,
            candidate_id=0,
            max_leanify_iterations=2,
        )

        ctx = AgentContext(
            task="BadType",
            metadata={
                "type_candidate": TypeCandidate(
                    name="BadType", informal_description="will fail"
                ).model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = formalizer.run(ctx)
        candidate = TypeFormalizationCandidate.model_validate(result.result)
        assert not candidate.compiles
        assert candidate.auxiliary_lemmas == []

    def test_properties(self):
        from agentic_research.agents.type_formalizer import TypeFormalizer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([])
        formalizer = TypeFormalizer(llm_client=llm, lean_repl=repl)
        assert formalizer.name == "type_formalizer"


# ---------------------------------------------------------------------------
# agents/auctioneer.py — scoring algorithm
# ---------------------------------------------------------------------------


class TestAuctioneerScoring:
    def test_compute_auction_score_compiles(self):
        from agentic_research.agents.auctioneer import compute_auction_score

        candidate = TypeFormalizationCandidate(
            candidate_id=0,
            type_name="T",
            lean_code="structure T where\n  x : Nat",
            compiles=True,
            auxiliary_lemmas=[
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l1", statement_nl="s", for_type="T"),
                    proved=True,
                ),
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l2", statement_nl="s", for_type="T"),
                    proved=True,
                ),
            ],
        )

        score = compute_auction_score(candidate)
        assert score.lemma_ratio == 1.0
        assert score.compilation_score == 1.0
        assert score.total_score > 0

    def test_compute_auction_score_no_compile(self):
        from agentic_research.agents.auctioneer import compute_auction_score

        candidate = TypeFormalizationCandidate(
            candidate_id=1,
            type_name="T",
            lean_code="bad code",
            compiles=False,
        )

        score = compute_auction_score(candidate)
        assert score.compilation_score == 0.0
        assert score.lemma_ratio == 0.0

    def test_brevity_score_short_code(self):
        from agentic_research.agents.auctioneer import compute_auction_score

        short = TypeFormalizationCandidate(
            candidate_id=0, type_name="T", lean_code="x", compiles=True,
        )
        long = TypeFormalizationCandidate(
            candidate_id=1, type_name="T", lean_code="x" * 1000, compiles=True,
        )

        short_score = compute_auction_score(short)
        long_score = compute_auction_score(long)
        assert short_score.brevity_score > long_score.brevity_score

    def test_higher_lemma_ratio_wins(self):
        from agentic_research.agents.auctioneer import compute_auction_score

        high = TypeFormalizationCandidate(
            candidate_id=0, type_name="T", lean_code="code", compiles=True,
            auxiliary_lemmas=[
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l", statement_nl="s", for_type="T"),
                    proved=True,
                ),
            ],
        )
        low = TypeFormalizationCandidate(
            candidate_id=1, type_name="T", lean_code="code", compiles=True,
            auxiliary_lemmas=[
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l", statement_nl="s", for_type="T"),
                    proved=False,
                ),
            ],
        )

        assert compute_auction_score(high).total_score > compute_auction_score(low).total_score


class TestAuctioneer:
    def test_selects_best_candidate(self):
        from agentic_research.agents.auctioneer import Auctioneer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))

        responses = []
        for i in range(3):
            responses.append(MOCK_TYPE_LEAN_CODE)

        llm = _make_mock_llm(responses)

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            quality_threshold=0.0,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(
            task="QuasiRandomGraph",
            metadata={
                "type_candidate": TypeCandidate(
                    name="QuasiRandomGraph",
                    informal_description="test",
                ).model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        auction = AuctionResult.model_validate(result.result)
        assert auction.verdict == AuctionVerdict.ACCEPTED
        assert auction.winner_id is not None

    def test_retry_when_below_threshold(self):
        from agentic_research.agents.auctioneer import Auctioneer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))

        error_response = "```lean\n-- MOCK_ERROR\nbad\n```"
        llm = _make_mock_llm([error_response] * 15)

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            quality_threshold=0.99,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(
            task="BadType",
            metadata={
                "type_candidate": TypeCandidate(
                    name="BadType", informal_description="will fail",
                ).model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        auction = AuctionResult.model_validate(result.result)
        assert auction.verdict == AuctionVerdict.RETRY

    def test_properties(self):
        from agentic_research.agents.auctioneer import Auctioneer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([])
        auctioneer = Auctioneer(llm_client=llm, lean_repl=repl, k=5)
        assert auctioneer.name == "auctioneer"
        assert auctioneer.k == 5


# ---------------------------------------------------------------------------
# agents/theorem_formalizer.py
# ---------------------------------------------------------------------------


class TestTheoremFormalizer:
    def test_successful_formalization(self):
        from agentic_research.agents.theorem_formalizer import TheoremFormalizer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([MOCK_THEOREM_LEAN])

        formalizer = TheoremFormalizer(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(
            task="All primes greater than 2 are odd",
            metadata={"type_definitions": ""},
        )

        result = formalizer.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        theorem = TheoremFormalization.model_validate(result.result)
        assert theorem.compiles
        assert theorem.iterations_used == 1

    def test_compilation_failure(self):
        from agentic_research.agents.theorem_formalizer import TheoremFormalizer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        error_code = "```lean\n-- MOCK_ERROR\nbad theorem\n```"
        llm = _make_mock_llm([error_code] * 5)

        formalizer = TheoremFormalizer(llm_client=llm, lean_repl=repl, max_iterations=3)
        ctx = AgentContext(
            task="impossible conjecture",
            metadata={"type_definitions": ""},
        )

        result = formalizer.run(ctx)
        assert result.status == AgentStatus.FAILURE
        theorem = TheoremFormalization.model_validate(result.result)
        assert not theorem.compiles
        assert theorem.iterations_used == 3
        assert theorem.failure_reason is not None

    def test_succeeds_after_retry(self):
        from agentic_research.agents.theorem_formalizer import TheoremFormalizer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        error_code = "```lean\n-- MOCK_ERROR\nbad\n```"
        llm = _make_mock_llm([error_code, MOCK_THEOREM_LEAN])

        formalizer = TheoremFormalizer(llm_client=llm, lean_repl=repl, max_iterations=5)
        ctx = AgentContext(
            task="test conjecture",
            metadata={"type_definitions": ""},
        )

        result = formalizer.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        theorem = TheoremFormalization.model_validate(result.result)
        assert theorem.compiles
        assert theorem.iterations_used == 2

    def test_properties(self):
        from agentic_research.agents.theorem_formalizer import TheoremFormalizer
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([])
        f = TheoremFormalizer(llm_client=llm, lean_repl=repl)
        assert f.name == "theorem_formalizer"


# ---------------------------------------------------------------------------
# agents/claim_check.py
# ---------------------------------------------------------------------------


class TestClaimCheck:
    def test_pass(self):
        from agentic_research.agents.claim_check import ClaimCheck

        llm = _make_mock_llm([MOCK_CLAIM_CHECK_PASS])

        checker = ClaimCheck(llm_client=llm)
        ctx = AgentContext(
            task="All primes > 2 are odd",
            metadata={
                "lean_code": "theorem primes_odd : True := sorry",
                "type_definitions": "",
            },
        )

        result = checker.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        claim = ClaimCheckResult.model_validate(result.result)
        assert claim.verdict == ClaimCheckVerdict.PASS

    def test_fail(self):
        from agentic_research.agents.claim_check import ClaimCheck

        llm = _make_mock_llm([MOCK_CLAIM_CHECK_FAIL])

        checker = ClaimCheck(llm_client=llm)
        ctx = AgentContext(
            task="Strong conjecture",
            metadata={
                "lean_code": "theorem weakened : True := trivial",
                "type_definitions": "",
            },
        )

        result = checker.run(ctx)
        claim = ClaimCheckResult.model_validate(result.result)
        assert claim.verdict == ClaimCheckVerdict.FAIL

    def test_duplicate_imports_do_not_cause_failure(self):
        from agentic_research.agents.claim_check import ClaimCheck

        llm = _make_mock_llm([MOCK_CLAIM_CHECK_PASS])

        checker = ClaimCheck(llm_client=llm)
        ctx = AgentContext(
            task="test",
            metadata={
                "lean_code": "import Foo\nimport Foo\ntheorem t := sorry",
                "type_definitions": "",
            },
        )

        result = checker.run(ctx)
        claim = ClaimCheckResult.model_validate(result.result)
        assert claim.verdict == ClaimCheckVerdict.PASS

    def test_without_llm_check(self):
        from agentic_research.agents.claim_check import ClaimCheck

        llm = _make_mock_llm([])

        checker = ClaimCheck(llm_client=llm, use_llm_check=False)
        ctx = AgentContext(
            task="test",
            metadata={
                "lean_code": "theorem t := sorry",
                "type_definitions": "",
            },
        )

        result = checker.run(ctx)
        claim = ClaimCheckResult.model_validate(result.result)
        assert claim.verdict == ClaimCheckVerdict.PASS
        assert llm.complete.call_count == 0

    def test_properties(self):
        from agentic_research.agents.claim_check import ClaimCheck

        llm = _make_mock_llm([])
        checker = ClaimCheck(llm_client=llm)
        assert checker.name == "claim_check"


class TestStatementPreserved:
    def test_preserved(self):
        from agentic_research.agents.claim_check import check_statement_preserved

        assert check_statement_preserved(
            "theorem foo : True",
            "import Mathlib\n\ntheorem foo : True := sorry",
        )

    def test_not_preserved(self):
        from agentic_research.agents.claim_check import check_statement_preserved

        assert not check_statement_preserved(
            "theorem foo : False",
            "theorem foo : True := sorry",
        )


# ---------------------------------------------------------------------------
# pipelines/formalization.py
# ---------------------------------------------------------------------------


class TestFormalizationPipeline:
    def test_end_to_end_success(self):
        from agentic_research.pipelines.formalization import FormalizationPipeline
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        responses = [
            MOCK_TYPE_PLAN_RESPONSE,
            MOCK_LEMMA_PLAN_RESPONSE,
        ]
        for _k in range(3):
            responses.append(MOCK_TYPE_LEAN_CODE)
            responses.append(MOCK_LEMMA_LEAN_CODE)
            responses.append(MOCK_PROOF_CODE)
            responses.append(MOCK_LEMMA_LEAN_CODE)
            responses.append(MOCK_PROOF_CODE)
        responses.append(MOCK_THEOREM_LEAN)
        responses.append(MOCK_CLAIM_CHECK_PASS)

        llm = _make_mock_llm(responses)

        pipeline = FormalizationPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            k=3,
            prover_config=ProverConfig(max_iterations=1),
        )

        result = pipeline.run("All quasi-random graphs have specific edge distribution")

        assert isinstance(result, FormalizationPipelineResult)
        assert result.success
        assert result.theorem is not None
        assert result.claim_check is not None

    def test_empty_type_plan_succeeds(self):
        from agentic_research.pipelines.formalization import FormalizationPipeline
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        plan_response = json.dumps({
            "candidates": [],
            "dependency_graph": {"edges": [], "topological_order": []},
            "mathlib_imports": [],
        })
        # Order: type_planner, (lemma_planner gets no LLM calls), theorem_formalizer, claim_check
        llm = _make_mock_llm([plan_response, MOCK_THEOREM_LEAN, MOCK_CLAIM_CHECK_PASS])

        pipeline = FormalizationPipeline(
            llm_client=llm, lean_repl=repl, lean_search=search,
        )

        result = pipeline.run("test conjecture")
        assert result.success

    def test_no_new_types_still_succeeds(self):
        from agentic_research.pipelines.formalization import FormalizationPipeline
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        plan_all_mathlib = json.dumps({
            "candidates": [
                {"name": "Nat", "informal_description": "natural numbers", "is_in_mathlib": True}
            ],
            "dependency_graph": {"edges": [], "topological_order": ["Nat"]},
            "mathlib_imports": ["Mathlib.Data.Nat.Basic"],
        })

        # Lemma planner skips Mathlib types (no LLM call), so sequence is:
        # type_planner, theorem_formalizer, claim_check
        llm = _make_mock_llm([
            plan_all_mathlib,
            MOCK_THEOREM_LEAN,
            MOCK_CLAIM_CHECK_PASS,
        ])

        pipeline = FormalizationPipeline(
            llm_client=llm, lean_repl=repl, lean_search=search,
        )

        result = pipeline.run("Natural numbers have interesting properties")
        assert result.success


# ---------------------------------------------------------------------------
# Prompt templates — Phase 5
# ---------------------------------------------------------------------------


class TestFormalizationPromptTemplates:
    def test_type_planner_template(self):
        from agentic_research.agents.prompt_templates import TYPE_PLANNER_USER_TEMPLATE

        rendered = TYPE_PLANNER_USER_TEMPLATE.format(
            conjecture="test conjecture",
            mathlib_results="- Nat.add_comm",
        )
        assert "test conjecture" in rendered
        assert "Nat.add_comm" in rendered

    def test_lemma_planner_template(self):
        from agentic_research.agents.prompt_templates import LEMMA_PLANNER_USER_TEMPLATE

        rendered = LEMMA_PLANNER_USER_TEMPLATE.format(
            type_name="T",
            type_description="desc",
            lean_signature="structure T",
            dependencies="none",
        )
        assert "T" in rendered

    def test_type_leanifier_template(self):
        from agentic_research.agents.prompt_templates import TYPE_LEANIFIER_USER_TEMPLATE

        rendered = TYPE_LEANIFIER_USER_TEMPLATE.format(
            type_name="Graph",
            type_description="simple graph",
            lean_signature="structure Graph where",
            dependencies="none",
        )
        assert "Graph" in rendered

    def test_theorem_formalizer_template(self):
        from agentic_research.agents.prompt_templates import THEOREM_FORMALIZER_USER_TEMPLATE

        rendered = THEOREM_FORMALIZER_USER_TEMPLATE.format(
            conjecture="all primes are odd",
            type_definitions="-- none",
        )
        assert "all primes are odd" in rendered

    def test_claim_check_template(self):
        from agentic_research.agents.prompt_templates import CLAIM_CHECK_USER_TEMPLATE

        rendered = CLAIM_CHECK_USER_TEMPLATE.format(
            conjecture_nl="conjecture",
            lean_code="theorem t := sorry",
            type_definitions="-- none",
        )
        assert "conjecture" in rendered
