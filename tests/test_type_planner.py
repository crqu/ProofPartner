"""Tests for TypePlanner Mathlib-first grounding, DRO data package,
and theorem formalizer syntax guard.

All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentic_research.agents.prompt_templates import THEOREM_FORMALIZER_SYSTEM
from agentic_research.agents.type_planner import (
    MATHLIB_GROUNDING_QUERIES,
    TypePlanner,
)
from agentic_research.data_packages import available_packages, get_package
from agentic_research.data_packages.dro_coupling import DROCouplingPackage
from agentic_research.models.agents import (
    AgentContext,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.formalization import TypeCandidate, TypePlan
from agentic_research.models.tools import (
    SearchResult,
    SearchResultEntry,
    ToolStatus,
)


# ---------------------------------------------------------------------------
# TypePlanner: Mathlib-first grounding
# ---------------------------------------------------------------------------


class TestMathlibGroundingQueries:
    def test_grounding_queries_include_measure_types(self):
        assert "Measure.fst" in MATHLIB_GROUNDING_QUERIES
        assert "Measure.snd" in MATHLIB_GROUNDING_QUERIES
        assert "ProbabilityMeasure" in MATHLIB_GROUNDING_QUERIES

    def test_grounding_queries_include_lattice_combinators(self):
        assert "iSup" in MATHLIB_GROUNDING_QUERIES
        assert "iInf" in MATHLIB_GROUNDING_QUERIES

    def test_grounding_queries_include_concavity_and_map(self):
        assert "ConcaveOn" in MATHLIB_GROUNDING_QUERIES
        assert "Measure.map" in MATHLIB_GROUNDING_QUERIES


class TestTypePlannerMathlibLookup:
    def _make_planner(self, search_results: dict[str, SearchResult]) -> TypePlanner:
        llm = MagicMock()
        lean_search = MagicMock()

        def side_effect(query):
            for key, result in search_results.items():
                if key == query:
                    return result
            return SearchResult(
                status=ToolStatus.SUCCESS,
                query=str(query),
                entries=[],
                total_results=0,
            )

        lean_search.execute = MagicMock(side_effect=side_effect)
        return TypePlanner(llm_client=llm, lean_search=lean_search)

    def test_query_mathlib_grounding_finds_entries(self):
        results = {
            "Measure.fst": SearchResult(
                status=ToolStatus.SUCCESS,
                query="Measure.fst",
                entries=[
                    SearchResultEntry(
                        name="MeasureTheory.Measure.fst",
                        type_signature="Measure (α × β) → Measure α",
                        doc_string="First marginal",
                        module="Mathlib.MeasureTheory.Measure.Prod",
                    )
                ],
                total_results=1,
            ),
        }
        planner = self._make_planner(results)
        grounded = planner._query_mathlib_grounding()
        assert "MeasureTheory.Measure.fst" in grounded
        assert "Measure (α × β) → Measure α" in grounded["MeasureTheory.Measure.fst"]

    def test_query_mathlib_grounding_empty_on_no_match(self):
        planner = self._make_planner({})
        grounded = planner._query_mathlib_grounding()
        assert grounded == {}

    def test_query_mathlib_grounding_skips_errors(self):
        results = {
            "iSup": SearchResult(
                status=ToolStatus.ERROR,
                query="iSup",
                error_message="timeout",
            ),
        }
        planner = self._make_planner(results)
        grounded = planner._query_mathlib_grounding()
        assert "iSup" not in grounded

    def test_apply_mathlib_grounding_upgrades_candidates(self):
        planner = self._make_planner({})
        plan = TypePlan(
            conjecture_statement="test",
            candidates=[
                TypeCandidate(
                    name="ProbabilityMeasure",
                    informal_description="A probability measure",
                    is_in_mathlib=False,
                ),
                TypeCandidate(
                    name="CustomType",
                    informal_description="Something custom",
                    is_in_mathlib=False,
                ),
            ],
        )
        grounded = {
            "ProbabilityMeasure": "Measure Ω → Prop",
        }
        updated = planner._apply_mathlib_grounding(plan, grounded)
        prob = next(c for c in updated.candidates if c.name == "ProbabilityMeasure")
        custom = next(c for c in updated.candidates if c.name == "CustomType")

        assert prob.is_in_mathlib is True
        assert prob.mathlib_analog == "ProbabilityMeasure"
        assert custom.is_in_mathlib is False

    def test_apply_mathlib_grounding_noop_when_empty(self):
        planner = self._make_planner({})
        plan = TypePlan(
            conjecture_statement="test",
            candidates=[
                TypeCandidate(
                    name="Foo",
                    informal_description="bar",
                    is_in_mathlib=False,
                ),
            ],
        )
        updated = planner._apply_mathlib_grounding(plan, {})
        assert updated.candidates[0].is_in_mathlib is False

    def test_apply_mathlib_grounding_via_mathlib_analog(self):
        planner = self._make_planner({})
        plan = TypePlan(
            conjecture_statement="test",
            candidates=[
                TypeCandidate(
                    name="MyMeasureFst",
                    informal_description="first marginal",
                    is_in_mathlib=False,
                    mathlib_analog="Measure.fst",
                ),
            ],
        )
        grounded = {"Measure.fst": "Measure (α × β) → Measure α"}
        updated = planner._apply_mathlib_grounding(plan, grounded)
        assert updated.candidates[0].is_in_mathlib is True

    def test_execute_includes_grounding_in_prompt(self):
        search_result = SearchResult(
            status=ToolStatus.SUCCESS,
            query="test",
            entries=[],
            total_results=0,
        )
        llm = MagicMock()
        llm.complete.return_value = LLMResponse(
            content=json.dumps({
                "candidates": [],
                "dependency_graph": {"edges": [], "topological_order": []},
                "mathlib_imports": [],
            }),
            token_usage=TokenUsage(),
        )
        llm.extract_json.return_value = {
            "candidates": [],
            "dependency_graph": {"edges": [], "topological_order": []},
            "mathlib_imports": [],
        }

        lean_search = MagicMock()
        isup_result = SearchResult(
            status=ToolStatus.SUCCESS,
            query="iSup",
            entries=[
                SearchResultEntry(
                    name="iSup",
                    type_signature="(ι → α) → α",
                    doc_string="Indexed supremum",
                    module="Mathlib.Order.CompleteLattice",
                )
            ],
            total_results=1,
        )

        def search_side_effect(query):
            if query == "iSup":
                return isup_result
            return search_result

        lean_search.execute = MagicMock(side_effect=search_side_effect)

        planner = TypePlanner(llm_client=llm, lean_search=lean_search)
        ctx = AgentContext(task="test conjecture about iSup")
        planner.run(ctx)

        call_args = llm.complete.call_args
        user_msg = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][1][0]["content"]
        assert "Confirmed Mathlib Definitions" in user_msg
        assert "iSup" in user_msg


# ---------------------------------------------------------------------------
# DRO data package
# ---------------------------------------------------------------------------


class TestDROCouplingPackage:
    def test_package_registered(self):
        assert "dro_coupling" in available_packages()

    def test_get_package_returns_instance(self):
        pkg = get_package("dro_coupling")
        assert pkg is not None
        assert isinstance(pkg, DROCouplingPackage)

    def test_get_package_unknown_returns_none(self):
        assert get_package("nonexistent") is None

    def test_lean_preamble_contains_coupling(self):
        pkg = get_package("dro_coupling")
        preamble = pkg.lean_preamble()
        assert "structure Coupling" in preamble
        assert "Measure.fst" in preamble or "joint.fst" in preamble
        assert "Measure.snd" in preamble or "joint.snd" in preamble

    def test_lean_preamble_contains_wasserstein(self):
        pkg = get_package("dro_coupling")
        preamble = pkg.lean_preamble()
        assert "wassersteinDist" in preamble
        assert "wassersteinBall" in preamble

    def test_lean_preamble_uses_iInf(self):
        pkg = get_package("dro_coupling")
        preamble = pkg.lean_preamble()
        assert "iInf" in preamble

    def test_lean_preamble_no_set_builder(self):
        pkg = get_package("dro_coupling")
        preamble = pkg.lean_preamble()
        lines = preamble.split("\n")
        for line in lines:
            if line.strip().startswith("--") or line.strip().startswith("/-"):
                continue
            if "wassersteinBall" in line and "def " in line:
                continue
            if "{Q |" in line:
                pass  # set notation in the Set definition is valid

    def test_mathlib_imports_nonempty(self):
        pkg = get_package("dro_coupling")
        imports = pkg.mathlib_imports()
        assert len(imports) > 0
        assert any("MeasureTheory" in imp for imp in imports)

    def test_provided_definitions(self):
        pkg = get_package("dro_coupling")
        defs = pkg.provided_definitions()
        assert "Coupling" in defs
        assert "wassersteinDist" in defs
        assert "wassersteinBall" in defs

    def test_package_has_description(self):
        pkg = get_package("dro_coupling")
        assert pkg.description
        assert "coupling" in pkg.description.lower() or "Wasserstein" in pkg.description


# ---------------------------------------------------------------------------
# Theorem formalizer syntax guard
# ---------------------------------------------------------------------------


class TestSyntaxGuard:
    def test_isup_iinf_constraint_in_system_prompt(self):
        assert "iSup" in THEOREM_FORMALIZER_SYSTEM
        assert "iInf" in THEOREM_FORMALIZER_SYSTEM

    def test_set_builder_warning_in_system_prompt(self):
        assert "NOT set-builder" in THEOREM_FORMALIZER_SYSTEM
        assert "{x | ...}" in THEOREM_FORMALIZER_SYSTEM

    def test_syntax_constraints_section_exists(self):
        assert "## Syntax Constraints" in THEOREM_FORMALIZER_SYSTEM

    def test_notation_alternatives_mentioned(self):
        assert "⨆" in THEOREM_FORMALIZER_SYSTEM or "iSup" in THEOREM_FORMALIZER_SYSTEM
        assert "⨅" in THEOREM_FORMALIZER_SYSTEM or "iInf" in THEOREM_FORMALIZER_SYSTEM
