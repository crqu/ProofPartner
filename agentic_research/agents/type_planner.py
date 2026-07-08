"""Type Planner agent — analyzes a conjecture and determines which types
beyond Lean 4 + Mathlib need to be defined for formalization.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
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
    TypeCandidate,
    TypeDependencyGraph,
    TypePlan,
)
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)


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

        log.info(
            "type_planner_done",
            num_candidates=len(plan.candidates),
            num_new_types=sum(1 for c in plan.candidates if not c.is_in_mathlib),
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=plan.model_dump(),
            token_usage=response.token_usage,
        )

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
