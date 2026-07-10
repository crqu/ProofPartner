"""Type Planner agent — analyzes a conjecture and determines which types
beyond Lean 4 + Mathlib need to be defined for formalization.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    DATA_PACKAGE_SYSTEM,
    DATA_PACKAGE_USER_TEMPLATE,
    TYPE_PLANNER_SYSTEM,
    TYPE_PLANNER_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.formalization import (
    DataPackageCandidate,
    TypeCandidate,
    TypeDependencyGraph,
    TypePlan,
)
from agentic_research.models.tools import ToolStatus
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)

MATHLIB_GROUNDING_QUERIES: list[str] = [
    "Measure.fst",
    "Measure.snd",
    "ProbabilityMeasure",
    "iSup",
    "iInf",
    "ConcaveOn",
    "Measure.map",
]


class TypePlanner(BaseAgent):
    """Determines which custom types are needed to formalize a conjecture."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_search: LeanSearch,
    ) -> None:
        super().__init__(name="type_planner", max_retries=2)
        self._llm = llm_client
        self._search = lean_search

    def _execute(self, context: AgentContext) -> AgentResult:
        conjecture = context.task
        log.info("type_planner_start", conjecture_len=len(conjecture))

        search_result = self._search.execute(conjecture[:120])
        mathlib_summary = self._format_search(search_result)

        mathlib_grounded = self._query_mathlib_grounding()
        if mathlib_grounded:
            grounded_lines = [
                f"- **{name}**: `{sig}` [from Mathlib]"
                for name, sig in mathlib_grounded.items()
            ]
            mathlib_summary += "\n\n## Confirmed Mathlib Definitions\n" + "\n".join(
                grounded_lines
            )

        user_content = TYPE_PLANNER_USER_TEMPLATE.format(
            conjecture=conjecture,
            mathlib_results=mathlib_summary,
        )

        response = self._llm.complete(
            system=TYPE_PLANNER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
            use_cache=True,
        )

        plan = self._parse_response(response.content, conjecture)

        plan = self._apply_mathlib_grounding(plan, mathlib_grounded)

        log.info(
            "type_planner_done",
            num_candidates=len(plan.candidates),
            num_new_types=sum(1 for c in plan.candidates if not c.is_in_mathlib),
            num_from_mathlib=sum(1 for c in plan.candidates if c.is_in_mathlib),
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=plan.model_dump(),
            token_usage=response.token_usage,
        )

    def _query_mathlib_grounding(self) -> dict[str, str]:
        """Query Loogle/LeanSearch for known measure-theoretic identifiers.

        Returns a mapping of identifier name to type signature for those
        confirmed present in Mathlib.
        """
        found: dict[str, str] = {}
        for query in MATHLIB_GROUNDING_QUERIES:
            result = self._search.execute(query)
            if result.status != ToolStatus.SUCCESS:
                continue
            entries = getattr(result, "entries", [])
            for entry in entries:
                if entry.name and query.lower() in entry.name.lower():
                    found[entry.name] = entry.type_signature
                    break
        log.info(
            "mathlib_grounding_done",
            queries=len(MATHLIB_GROUNDING_QUERIES),
            found=len(found),
        )
        return found

    def _apply_mathlib_grounding(
        self,
        plan: TypePlan,
        grounded: dict[str, str],
    ) -> TypePlan:
        """Upgrade candidates whose names match confirmed Mathlib identifiers
        to from_mathlib status, preventing unnecessary axiomatization."""
        if not grounded:
            return plan
        grounded_lower = {k.lower(): k for k in grounded}
        for candidate in plan.candidates:
            key = candidate.name.lower()
            if key in grounded_lower:
                original = grounded_lower[key]
                candidate.is_in_mathlib = True
                candidate.mathlib_analog = original
                candidate.lean_signature = grounded[original]
            elif candidate.mathlib_analog:
                analog_lower = candidate.mathlib_analog.lower()
                if analog_lower in grounded_lower:
                    candidate.is_in_mathlib = True
        return plan

    def _format_search(self, search_result) -> str:
        entries = getattr(search_result, "entries", [])
        if not entries:
            return "No Mathlib results found."
        lines = []
        for entry in entries:
            line = f"- **{entry.name}**: `{entry.type_signature}`"
            if entry.doc_string:
                line += f" — {entry.doc_string}"
            lines.append(line)
        return "\n".join(lines)

    def _parse_response(self, content: str, conjecture: str) -> TypePlan:
        parsed = self._llm.extract_json(content)
        if not isinstance(parsed, dict):
            log.warning("type_planner_parse_fallback", reason="no valid JSON")
            return TypePlan(conjecture_statement=conjecture)

        candidates = []
        for c in parsed.get("candidates", []):
            if isinstance(c, dict):
                candidates.append(TypeCandidate(
                    name=c.get("name", ""),
                    informal_description=c.get("informal_description", ""),
                    lean_signature=c.get("lean_signature", ""),
                    depends_on=c.get("depends_on", []),
                    mathlib_analog=c.get("mathlib_analog"),
                    is_in_mathlib=c.get("is_in_mathlib", False),
                    composition_alternative=c.get("composition_alternative"),
                ))

        dep_graph_raw = parsed.get("dependency_graph", {})
        dep_graph = TypeDependencyGraph(
            edges=[tuple(e) for e in dep_graph_raw.get("edges", []) if len(e) == 2],
            topological_order=dep_graph_raw.get("topological_order", []),
        )

        return TypePlan(
            conjecture_statement=conjecture,
            candidates=candidates,
            dependency_graph=dep_graph,
            mathlib_imports=parsed.get("mathlib_imports", []),
        )

    def suggest_data_package(
        self,
        type_name: str,
        type_description: str,
        search_summary: str = "No Mathlib results found.",
    ) -> DataPackageCandidate | None:
        """Suggest a data package parameterization for a type not found in Mathlib.

        Only call this when Loogle search returns 0 results for a candidate type.
        Returns a DataPackageCandidate with a bundled structure declaration.
        """
        log.info(
            "data_package_suggest_start",
            type_name=type_name,
        )

        user_content = DATA_PACKAGE_USER_TEMPLATE.format(
            type_name=type_name,
            type_description=type_description,
            search_results=search_summary,
        )

        response = self._llm.complete(
            system=DATA_PACKAGE_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
            use_cache=True,
        )
        self._accumulate_tokens(response.token_usage)

        parsed = self._llm.extract_json(response.content)
        if not isinstance(parsed, dict):
            log.warning("data_package_parse_failed", type_name=type_name)
            return None

        candidate = DataPackageCandidate(
            package_name=parsed.get("package_name", f"{type_name}Data"),
            description=parsed.get("description", type_description),
            bundled_fields=parsed.get("bundled_fields", []),
            assumed_properties=parsed.get("assumed_properties", []),
            mathlib_foundation=parsed.get("mathlib_foundation", ""),
            lean_structure=parsed.get("lean_structure", ""),
        )

        log.info(
            "data_package_suggest_done",
            type_name=type_name,
            package_name=candidate.package_name,
            num_fields=len(candidate.bundled_fields),
        )

        return candidate
