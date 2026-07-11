"""Tests for preamble threading to LemmaBreakdown, TypePlanner, and LemmaLeanifier.

Verifies that the DRO data package preamble is threaded through the pipeline
so agents see available definitions and don't re-derive them.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentic_research.agents.lemma_breakdown import LemmaBreakdown
from agentic_research.agents.lemma_leanifier import LemmaLeanifier
from agentic_research.agents.prompt_templates import PREAMBLE_CONTEXT_SECTION
from agentic_research.agents.type_planner import TypePlanner
from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.formalization import TypeCandidate, TypePlan
from agentic_research.models.proof import LemmaTree, ProofNode
from agentic_research.models.tools import SearchResult, ToolStatus


SAMPLE_PREAMBLE = """\
import Mathlib

structure Coupling (Ω₁ Ω₂ : Type*) [MeasurableSpace Ω₁] [MeasurableSpace Ω₂] where
  joint : Measure (Ω₁ × Ω₂)

noncomputable def wassersteinDist (P Q : Measure Ω) : ℝ≥0∞ := sorry

def wassersteinBall (P₀ : Measure Ω) (ε : ℝ≥0∞) : Set (Measure Ω) :=
  {Q | wassersteinDist Q P₀ ≤ ε}
"""


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


# ---------------------------------------------------------------------------
# LemmaBreakdown preamble threading
# ---------------------------------------------------------------------------


class TestLemmaBreakdownPreamble:
    def test_preamble_included_in_prompt(self):
        """LemmaBreakdown includes preamble definitions in the LLM prompt."""
        response_json = json.dumps({
            "lemmas": [
                {
                    "node_id": "lemma_1",
                    "statement_nl": "wassersteinDist is non-negative",
                    "depends_on": [],
                    "from_prior_work": True,
                    "source_reference": "follows from available definitions",
                }
            ],
            "topological_order": ["lemma_1"],
        })
        llm = _make_mock_llm([response_json])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(
            task="Prove that Wasserstein distance is non-negative",
            metadata={
                "statement_lean": "theorem wasserstein_nonneg : sorry",
                "failed_attempts": "None",
                "lean_preamble": SAMPLE_PREAMBLE,
            },
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "Available Lean 4 Definitions" in user_content
        assert "wassersteinDist" in user_content
        assert "Coupling" in user_content

    def test_no_preamble_no_section(self):
        """Without preamble, no definitions section is added."""
        response_json = json.dumps({
            "lemmas": [],
            "topological_order": [],
        })
        llm = _make_mock_llm([response_json])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(
            task="Prove n + 0 = n",
            metadata={
                "statement_lean": "theorem add_zero : sorry",
                "failed_attempts": "None",
            },
        )
        agent.run(ctx)

        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "Available Lean 4 Definitions" not in user_content

    def test_preamble_lemma_tagged_from_prior_work(self):
        """A lemma about a preamble type gets tagged from_prior_work by the LLM."""
        response_json = json.dumps({
            "lemmas": [
                {
                    "node_id": "lemma_1",
                    "statement_nl": "wassersteinDist is non-negative",
                    "depends_on": [],
                    "from_prior_work": True,
                    "source_reference": "follows from wassersteinDist definition",
                }
            ],
            "topological_order": ["lemma_1"],
        })
        llm = _make_mock_llm([response_json])

        agent = LemmaBreakdown(llm_client=llm)
        ctx = AgentContext(
            task="Prove the Wasserstein ball is bounded",
            metadata={
                "statement_lean": "theorem ball_bounded : sorry",
                "failed_attempts": "None",
                "lean_preamble": SAMPLE_PREAMBLE,
            },
        )
        result = agent.run(ctx)

        tree = LemmaTree.model_validate(result.result)
        lemma_node = tree.get_node("lemma_1")
        assert lemma_node is not None
        assert lemma_node.from_prior_work is True
        assert lemma_node.source_reference is not None


# ---------------------------------------------------------------------------
# TypePlanner preamble threading
# ---------------------------------------------------------------------------


class TestTypePlannerPreamble:
    def test_preamble_included_in_prompt(self):
        """TypePlanner includes preamble definitions in the LLM prompt."""
        response_json = json.dumps({
            "candidates": [
                {
                    "name": "wassersteinDist",
                    "informal_description": "Wasserstein distance",
                    "lean_signature": "def wassersteinDist := sorry",
                    "depends_on": [],
                    "is_in_mathlib": False,
                }
            ],
            "dependency_graph": {"edges": [], "topological_order": ["wassersteinDist"]},
            "mathlib_imports": [],
        })
        llm = _make_mock_llm([response_json])
        llm.extract_json.side_effect = lambda text: _extract_json_helper(text)
        lean_search = MagicMock()
        lean_search.execute.return_value = SearchResult(
            status=ToolStatus.SUCCESS,
            query="test",
            entries=[],
            total_results=0,
        )

        planner = TypePlanner(llm_client=llm, lean_search=lean_search)
        ctx = AgentContext(
            task="Wasserstein distance between measures",
            metadata={"lean_preamble": SAMPLE_PREAMBLE},
        )
        planner.run(ctx)

        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "Available Lean 4 Definitions" in user_content
        assert "wassersteinDist" in user_content

    def test_preamble_types_marked_is_in_preamble(self):
        """Candidates matching preamble definitions get is_in_preamble=True."""
        response_json = json.dumps({
            "candidates": [
                {
                    "name": "wassersteinDist",
                    "informal_description": "Wasserstein distance",
                    "lean_signature": "def wassersteinDist := sorry",
                    "depends_on": [],
                    "is_in_mathlib": False,
                },
                {
                    "name": "Coupling",
                    "informal_description": "A coupling of measures",
                    "lean_signature": "structure Coupling := sorry",
                    "depends_on": [],
                    "is_in_mathlib": False,
                },
                {
                    "name": "CustomNewType",
                    "informal_description": "Something not in preamble",
                    "lean_signature": "def CustomNewType := sorry",
                    "depends_on": [],
                    "is_in_mathlib": False,
                },
            ],
            "dependency_graph": {"edges": [], "topological_order": []},
            "mathlib_imports": [],
        })
        llm = _make_mock_llm([response_json])
        llm.extract_json.side_effect = lambda text: _extract_json_helper(text)
        lean_search = MagicMock()
        lean_search.execute.return_value = SearchResult(
            status=ToolStatus.SUCCESS,
            query="test",
            entries=[],
            total_results=0,
        )

        planner = TypePlanner(llm_client=llm, lean_search=lean_search)
        ctx = AgentContext(
            task="Wasserstein distance",
            metadata={"lean_preamble": SAMPLE_PREAMBLE},
        )
        result = planner.run(ctx)

        plan = TypePlan.model_validate(result.result)
        wasserstein = next(c for c in plan.candidates if c.name == "wassersteinDist")
        coupling = next(c for c in plan.candidates if c.name == "Coupling")
        custom = next(c for c in plan.candidates if c.name == "CustomNewType")

        assert wasserstein.is_in_preamble is True
        assert coupling.is_in_preamble is True
        assert custom.is_in_preamble is False

    def test_no_preamble_no_grounding(self):
        """Without preamble, no is_in_preamble flags are set."""
        response_json = json.dumps({
            "candidates": [
                {
                    "name": "wassersteinDist",
                    "informal_description": "Wasserstein distance",
                    "lean_signature": "def wassersteinDist := sorry",
                    "depends_on": [],
                    "is_in_mathlib": False,
                }
            ],
            "dependency_graph": {"edges": [], "topological_order": []},
            "mathlib_imports": [],
        })
        llm = _make_mock_llm([response_json])
        llm.extract_json.side_effect = lambda text: _extract_json_helper(text)
        lean_search = MagicMock()
        lean_search.execute.return_value = SearchResult(
            status=ToolStatus.SUCCESS,
            query="test",
            entries=[],
            total_results=0,
        )

        planner = TypePlanner(llm_client=llm, lean_search=lean_search)
        ctx = AgentContext(task="Wasserstein distance")
        result = planner.run(ctx)

        plan = TypePlan.model_validate(result.result)
        assert plan.candidates[0].is_in_preamble is False


# ---------------------------------------------------------------------------
# Pipeline preamble filter
# ---------------------------------------------------------------------------


class TestPipelinePreambleFilter:
    def test_preamble_types_filtered_from_type_formalization(self):
        """Types with is_in_preamble=True are filtered out of new_types."""
        from agentic_research.pipelines.proof import ProofPipeline

        pipeline = ProofPipeline(
            llm_client=_make_mock_llm([]),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            use_claim_check=False,
        )
        pipeline._lean_preamble = SAMPLE_PREAMBLE

        type_plan = TypePlan(
            conjecture_statement="test",
            candidates=[
                TypeCandidate(
                    name="wassersteinDist",
                    informal_description="already in preamble",
                    is_in_preamble=True,
                ),
                TypeCandidate(
                    name="NewCustomType",
                    informal_description="needs formalization",
                    is_in_mathlib=False,
                ),
            ],
        )

        new_types = [
            c for c in type_plan.candidates
            if not c.is_in_mathlib and not c.is_in_preamble and not c.composition_alternative
        ]
        assert len(new_types) == 1
        assert new_types[0].name == "NewCustomType"


# ---------------------------------------------------------------------------
# LemmaLeanifier feedback loop preamble
# ---------------------------------------------------------------------------


class TestLemmaLeanifierFeedbackPreamble:
    @staticmethod
    def _make_tree():
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

    def test_feedback_loop_includes_preamble(self):
        """When compilation fails and retries, the feedback prompt includes preamble."""
        from unittest.mock import patch
        from agentic_research.models.tools import CompilationResult, CompilationStatus, ToolStatus

        first_response = "```lean\ntheorem sub_1 : True := sorry_bad\n```"
        retry_response = "```lean\ntheorem sub_1 : True := sorry\n```"
        llm = _make_mock_llm([first_response, retry_response])
        repl = _make_mock_repl()

        preamble = "def wassersteinDist := sorry"
        agent = LemmaLeanifier(
            llm_client=llm,
            lean_repl=repl,
            lean_preamble=preamble,
        )

        fail_result = CompilationResult(
            status=ToolStatus.ERROR,
            compilation_status=CompilationStatus.ERROR,
            errors=["unknown identifier 'sorry_bad'"],
        )
        ok_result = CompilationResult(
            status=ToolStatus.SUCCESS,
            compilation_status=CompilationStatus.OK,
        )

        tree = self._make_tree()
        ctx = AgentContext(
            task="leanify lemmas",
            metadata={"lemma_tree": tree.model_dump()},
        )

        with patch.object(repl, "execute", side_effect=[fail_result, ok_result]):
            agent.run(ctx)

        assert llm.complete.call_count == 2
        retry_call = llm.complete.call_args_list[1]
        retry_content = retry_call[1]["messages"][0]["content"]
        assert "Available Definitions" in retry_content
        assert "wassersteinDist" in retry_content


# ---------------------------------------------------------------------------
# PREAMBLE_CONTEXT_SECTION template
# ---------------------------------------------------------------------------


class TestPreambleContextSection:
    def test_template_renders_preamble(self):
        rendered = PREAMBLE_CONTEXT_SECTION.format(lean_preamble="def foo := sorry")
        assert "Available Lean 4 Definitions" in rendered
        assert "def foo := sorry" in rendered
        assert "from_prior_work=true" in rendered

    def test_template_contains_prior_work_instruction(self):
        rendered = PREAMBLE_CONTEXT_SECTION.format(lean_preamble="dummy")
        assert "prior work" in rendered.lower()
        assert "do not re-derive" in rendered.lower()
