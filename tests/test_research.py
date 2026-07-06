"""Tests for Phase 4: Exploration Agent + Conjecture Generator.

All LLM calls are mocked — no real API calls are made.
LeanSearch uses mock backend for deterministic testing.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.research import (
    Concept,
    Conjecture,
    ConjectureSet,
    ExplorationResult,
    ResearchDirection,
)


# ---------------------------------------------------------------------------
# models/research.py
# ---------------------------------------------------------------------------


class TestConcept:
    def test_basic(self):
        c = Concept(name="graph coloring", description="Assigning colors to vertices")
        assert c.name == "graph coloring"
        assert c.mathlib_ref is None

    def test_with_mathlib_ref(self):
        c = Concept(name="Nat.Prime", domain="number theory", mathlib_ref="Mathlib.Data.Nat.Prime.Basic")
        assert c.mathlib_ref == "Mathlib.Data.Nat.Prime.Basic"

    def test_serialization(self):
        c = Concept(name="test", description="d", domain="algebra", mathlib_ref="ref")
        restored = Concept.model_validate(c.model_dump())
        assert restored == c


class TestResearchDirection:
    def test_basic(self):
        d = ResearchDirection(
            title="Chromatic scheduling",
            description="Model scheduling as graph coloring",
            ambition_level=2,
        )
        assert d.ambition_level == 2
        assert d.estimated_difficulty == 3

    def test_ambition_bounds(self):
        with pytest.raises(Exception):
            ResearchDirection(title="t", description="d", ambition_level=0)
        with pytest.raises(Exception):
            ResearchDirection(title="t", description="d", ambition_level=6)

    def test_serialization(self):
        d = ResearchDirection(
            title="t", description="d", ambition_level=3,
            relevant_concepts=["a", "b"], estimated_difficulty=4,
        )
        restored = ResearchDirection.model_validate(d.model_dump())
        assert restored == d


class TestExplorationResult:
    def test_basic(self):
        r = ExplorationResult(raw_idea="test idea", domain="combinatorics")
        assert r.raw_idea == "test idea"
        assert r.concepts == []
        assert r.directions == []

    def test_full(self):
        r = ExplorationResult(
            raw_idea="idea",
            domain="algebra",
            concepts=[Concept(name="group")],
            known_results=["Lagrange's theorem"],
            directions=[ResearchDirection(title="t", description="d", ambition_level=1)],
        )
        assert len(r.concepts) == 1
        assert len(r.known_results) == 1
        assert len(r.directions) == 1

    def test_serialization(self):
        r = ExplorationResult(
            raw_idea="idea", domain="topology",
            concepts=[Concept(name="manifold", description="smooth")],
            known_results=["Poincaré conjecture (proved)"],
            directions=[ResearchDirection(title="t", description="d", ambition_level=3)],
        )
        restored = ExplorationResult.model_validate(r.model_dump())
        assert restored == r


class TestConjecture:
    def test_basic(self):
        c = Conjecture(
            statement="∀ n, f(n) > 0",
            natural_language="f is always positive",
            confidence=0.8,
            difficulty=2,
        )
        assert c.confidence == 0.8
        assert c.difficulty == 2

    def test_composite_score(self):
        c = Conjecture(
            statement="s", natural_language="nl",
            confidence=0.9, difficulty=1,
            novelty_score=0.3, formalizability_score=0.6,
        )
        expected = (0.3 + 0.9 + 0.6) / 3.0
        assert abs(c.composite_score - expected) < 1e-9

    def test_bounds_enforcement(self):
        with pytest.raises(Exception):
            Conjecture(statement="s", natural_language="n", confidence=1.5, difficulty=1)
        with pytest.raises(Exception):
            Conjecture(statement="s", natural_language="n", confidence=0.5, difficulty=0)

    def test_serialization(self):
        c = Conjecture(
            statement="s", natural_language="n", confidence=0.5, difficulty=3,
            related_results=["r1"], novelty_score=0.7, formalizability_score=0.4,
        )
        restored = Conjecture.model_validate(c.model_dump())
        assert restored == c


class TestConjectureSet:
    def test_basic(self):
        cs = ConjectureSet()
        assert cs.conjectures == []
        assert cs.ranking == []
        assert cs.exploration_context is None

    def test_with_context(self):
        ctx = ExplorationResult(raw_idea="idea", domain="algebra")
        cs = ConjectureSet(
            conjectures=[Conjecture(statement="s", natural_language="n", confidence=0.5, difficulty=3)],
            ranking=[0],
            exploration_context=ctx,
        )
        assert cs.exploration_context.domain == "algebra"

    def test_serialization(self):
        cs = ConjectureSet(
            conjectures=[
                Conjecture(statement="s1", natural_language="n1", confidence=0.8, difficulty=2),
                Conjecture(statement="s2", natural_language="n2", confidence=0.3, difficulty=4),
            ],
            ranking=[0, 1],
        )
        restored = ConjectureSet.model_validate(cs.model_dump())
        assert restored == cs


# ---------------------------------------------------------------------------
# Helpers for mocking LLM responses
# ---------------------------------------------------------------------------


MOCK_EXPLORATION_RESPONSE = json.dumps({
    "domain": "combinatorics",
    "concepts": [
        {
            "name": "graph coloring",
            "description": "Assignment of colors to graph vertices such that no two adjacent vertices share a color",
            "domain": "graph theory",
            "mathlib_ref": "SimpleGraph.Coloring",
        },
        {
            "name": "scheduling",
            "description": "Allocation of resources to tasks over time",
            "domain": "combinatorial optimization",
            "mathlib_ref": None,
        },
    ],
    "known_results": [
        "Every planar graph is 4-colorable (four color theorem)",
        "Graph coloring is equivalent to scheduling with interference constraints",
    ],
    "directions": [
        {
            "title": "Chromatic number bounds for scheduling graphs",
            "description": "Establish upper bounds on the chromatic number of scheduling conflict graphs",
            "ambition_level": 2,
            "relevant_concepts": ["graph coloring", "scheduling"],
            "estimated_difficulty": 3,
        },
        {
            "title": "Optimal coloring algorithms for interval graphs",
            "description": "Prove that greedy coloring is optimal for interval scheduling graphs",
            "ambition_level": 1,
            "relevant_concepts": ["graph coloring"],
            "estimated_difficulty": 2,
        },
        {
            "title": "Novel chromatic polynomial identities",
            "description": "Discover new identities relating chromatic polynomials to scheduling invariants",
            "ambition_level": 4,
            "relevant_concepts": ["graph coloring", "scheduling"],
            "estimated_difficulty": 5,
        },
    ],
})

MOCK_CONJECTURE_RESPONSE = json.dumps({
    "conjectures": [
        {
            "statement": "∀ G : SimpleGraph, G.IsInterval → G.chromaticNumber = G.cliqueNumber",
            "natural_language": "For interval graphs, the chromatic number equals the clique number",
            "confidence": 0.95,
            "difficulty": 2,
            "related_results": ["Perfect graph theorem"],
            "novelty_score": 0.2,
            "formalizability_score": 0.8,
        },
        {
            "statement": "∀ G : SimpleGraph, G.IsChordal → G.chromaticNumber ≤ G.maxDegree + 1",
            "natural_language": "Chordal graphs satisfy a tighter bound than Brooks' theorem",
            "confidence": 0.7,
            "difficulty": 3,
            "related_results": ["Brooks' theorem"],
            "novelty_score": 0.5,
            "formalizability_score": 0.6,
        },
        {
            "statement": "∃ f : ℕ → ℕ, ∀ k, schedulingComplexity(k) = f(chromaticNumber(k))",
            "natural_language": "Scheduling complexity is determined by chromatic number",
            "confidence": 0.3,
            "difficulty": 5,
            "related_results": [],
            "novelty_score": 0.8,
            "formalizability_score": 0.3,
        },
    ]
})

MOCK_RANKING_RESPONSE = json.dumps({"ranking": [0, 1, 2]})


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


# ---------------------------------------------------------------------------
# agents/explorer.py
# ---------------------------------------------------------------------------


class TestExplorationAgent:
    def test_basic_exploration(self):
        from agentic_research.agents.explorer import ExplorationAgent
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm([MOCK_EXPLORATION_RESPONSE])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        agent = ExplorationAgent(llm_client=llm, lean_search=search)
        ctx = AgentContext(task="I think there is a connection between graph coloring and scheduling")
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        exploration = ExplorationResult.model_validate(result.result)
        assert exploration.domain == "combinatorics"
        assert len(exploration.concepts) == 2
        assert any(c.name == "graph coloring" for c in exploration.concepts)
        assert len(exploration.directions) == 3
        assert exploration.raw_idea == "I think there is a connection between graph coloring and scheduling"

    def test_num_directions_limit(self):
        from agentic_research.agents.explorer import ExplorationAgent
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm([MOCK_EXPLORATION_RESPONSE])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        agent = ExplorationAgent(llm_client=llm, lean_search=search, num_directions=2)
        ctx = AgentContext(task="graph coloring and scheduling")
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        exploration = ExplorationResult.model_validate(result.result)
        assert len(exploration.directions) <= 2

    def test_uses_lean_search(self):
        from agentic_research.agents.explorer import ExplorationAgent
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm([MOCK_EXPLORATION_RESPONSE])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        mock_execute = MagicMock(wraps=search.execute)
        search.execute = mock_execute

        agent = ExplorationAgent(llm_client=llm, lean_search=search, max_search_queries=2)
        ctx = AgentContext(task="prime numbers and gaps")
        agent.run(ctx)

        assert mock_execute.call_count >= 1

    def test_fallback_on_bad_json(self):
        from agentic_research.agents.explorer import ExplorationAgent
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm(["This is not JSON at all, just plain text."])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        agent = ExplorationAgent(llm_client=llm, lean_search=search)
        ctx = AgentContext(task="some idea")
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        exploration = ExplorationResult.model_validate(result.result)
        assert exploration.domain == "unknown"
        assert exploration.concepts == []

    def test_token_tracking(self):
        from agentic_research.agents.explorer import ExplorationAgent
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm([MOCK_EXPLORATION_RESPONSE])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        agent = ExplorationAgent(llm_client=llm, lean_search=search)
        ctx = AgentContext(task="test")
        result = agent.run(ctx)

        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 50

    def test_properties(self):
        from agentic_research.agents.explorer import ExplorationAgent
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        llm = _make_mock_llm([])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))

        agent = ExplorationAgent(llm_client=llm, lean_search=search, num_directions=7)
        assert agent.name == "exploration_agent"
        assert agent.num_directions == 7


# ---------------------------------------------------------------------------
# agents/conjecturer.py
# ---------------------------------------------------------------------------


class TestConjectureGenerator:
    def _make_exploration_context(self) -> dict:
        return ExplorationResult(
            raw_idea="graph coloring and scheduling",
            domain="combinatorics",
            concepts=[
                Concept(name="graph coloring", description="vertex coloring", domain="graph theory"),
            ],
            known_results=["Four color theorem"],
            directions=[
                ResearchDirection(
                    title="Chromatic bounds",
                    description="Bounds on chromatic number",
                    ambition_level=2,
                ),
            ],
        ).model_dump()

    def test_basic_generation(self):
        from agentic_research.agents.conjecturer import ConjectureGenerator

        llm = _make_mock_llm([MOCK_CONJECTURE_RESPONSE, MOCK_RANKING_RESPONSE])

        agent = ConjectureGenerator(llm_client=llm)
        ctx = AgentContext(
            task="graph coloring and scheduling",
            metadata={"exploration_result": self._make_exploration_context()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        cs = ConjectureSet.model_validate(result.result)
        assert len(cs.conjectures) == 3
        assert len(cs.ranking) == 3

    def test_num_conjectures_limit(self):
        from agentic_research.agents.conjecturer import ConjectureGenerator

        llm = _make_mock_llm([MOCK_CONJECTURE_RESPONSE, MOCK_RANKING_RESPONSE])

        agent = ConjectureGenerator(llm_client=llm, num_conjectures=2)
        ctx = AgentContext(
            task="graph coloring and scheduling",
            metadata={"exploration_result": self._make_exploration_context()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        cs = ConjectureSet.model_validate(result.result)
        assert len(cs.conjectures) <= 2

    def test_conjecture_ranking_order(self):
        from agentic_research.agents.conjecturer import ConjectureGenerator

        ranking_response = json.dumps({"ranking": [2, 0, 1]})
        llm = _make_mock_llm([MOCK_CONJECTURE_RESPONSE, ranking_response])

        agent = ConjectureGenerator(llm_client=llm)
        ctx = AgentContext(
            task="test",
            metadata={"exploration_result": self._make_exploration_context()},
        )
        result = agent.run(ctx)

        cs = ConjectureSet.model_validate(result.result)
        assert cs.ranking == [2, 0, 1]

    def test_without_exploration_context(self):
        from agentic_research.agents.conjecturer import ConjectureGenerator

        llm = _make_mock_llm([MOCK_CONJECTURE_RESPONSE, MOCK_RANKING_RESPONSE])

        agent = ConjectureGenerator(llm_client=llm)
        ctx = AgentContext(task="some rough idea about primes")
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        cs = ConjectureSet.model_validate(result.result)
        assert cs.exploration_context is not None
        assert cs.exploration_context.domain == "unknown"

    def test_token_tracking(self):
        from agentic_research.agents.conjecturer import ConjectureGenerator

        llm = _make_mock_llm([MOCK_CONJECTURE_RESPONSE, MOCK_RANKING_RESPONSE])

        agent = ConjectureGenerator(llm_client=llm)
        ctx = AgentContext(
            task="test",
            metadata={"exploration_result": self._make_exploration_context()},
        )
        result = agent.run(ctx)

        assert result.token_usage.input_tokens == 200
        assert result.token_usage.output_tokens == 100

    def test_preserves_exploration_context(self):
        from agentic_research.agents.conjecturer import ConjectureGenerator

        llm = _make_mock_llm([MOCK_CONJECTURE_RESPONSE, MOCK_RANKING_RESPONSE])

        agent = ConjectureGenerator(llm_client=llm)
        ctx = AgentContext(
            task="test",
            metadata={"exploration_result": self._make_exploration_context()},
        )
        result = agent.run(ctx)

        cs = ConjectureSet.model_validate(result.result)
        assert cs.exploration_context is not None
        assert cs.exploration_context.domain == "combinatorics"

    def test_properties(self):
        from agentic_research.agents.conjecturer import ConjectureGenerator

        llm = _make_mock_llm([])
        agent = ConjectureGenerator(llm_client=llm, num_conjectures=10)
        assert agent.name == "conjecture_generator"
        assert agent.num_conjectures == 10


# ---------------------------------------------------------------------------
# Pipeline: Exploration → Conjecture
# ---------------------------------------------------------------------------


class TestExplorationToConjecturePipeline:
    def test_full_pipeline(self):
        from agentic_research.agents.conjecturer import ConjectureGenerator
        from agentic_research.agents.explorer import ExplorationAgent
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        explorer_llm = _make_mock_llm([MOCK_EXPLORATION_RESPONSE])
        search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
        explorer = ExplorationAgent(llm_client=explorer_llm, lean_search=search)

        idea = "I think there is a connection between graph coloring and scheduling"
        explore_ctx = AgentContext(task=idea)
        explore_result = explorer.run(explore_ctx)

        assert explore_result.status == AgentStatus.SUCCESS

        conjecturer_llm = _make_mock_llm([MOCK_CONJECTURE_RESPONSE, MOCK_RANKING_RESPONSE])
        conjecturer = ConjectureGenerator(llm_client=conjecturer_llm)

        conjecture_ctx = AgentContext(
            task=idea,
            metadata={"exploration_result": explore_result.result},
        )
        conjecture_result = conjecturer.run(conjecture_ctx)

        assert conjecture_result.status == AgentStatus.SUCCESS
        cs = ConjectureSet.model_validate(conjecture_result.result)
        assert len(cs.conjectures) > 0
        assert len(cs.ranking) == len(cs.conjectures)
        assert cs.exploration_context is not None
        assert cs.exploration_context.domain == "combinatorics"

    def test_pipeline_with_diverse_ideas(self):
        """Verify the pipeline handles varied mathematical domains."""
        from agentic_research.agents.explorer import ExplorationAgent
        from agentic_research.tools.lean_search import LeanSearch, SearchConfig, SearchBackend

        ideas = [
            "prime gaps relate to twin prime density",
            "there might be a topological obstruction to certain embeddings",
            "linear algebra over finite fields has different properties",
        ]

        for idea in ideas:
            llm = _make_mock_llm([MOCK_EXPLORATION_RESPONSE])
            search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
            explorer = ExplorationAgent(llm_client=llm, lean_search=search)

            ctx = AgentContext(task=idea)
            result = explorer.run(ctx)

            assert result.status == AgentStatus.SUCCESS
            exploration = ExplorationResult.model_validate(result.result)
            assert exploration.raw_idea == idea
            assert exploration.domain != ""
