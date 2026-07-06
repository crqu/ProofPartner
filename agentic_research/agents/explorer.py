"""Exploration Agent — takes a rough mathematical idea and identifies
domains, concepts, known results, and candidate research directions.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    EXPLORATION_SYSTEM_PROMPT,
    EXPLORATION_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.research import (
    Concept,
    ExplorationResult,
    ResearchDirection,
)
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)


class ExplorationAgent(BaseAgent):
    """Explores a rough mathematical idea and proposes research directions."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_search: LeanSearch,
        *,
        num_directions: int = 5,
        max_search_queries: int = 3,
    ) -> None:
        super().__init__(name="exploration_agent", max_retries=2)
        self._llm = llm_client
        self._search = lean_search
        self._num_directions = num_directions
        self._max_search_queries = max_search_queries

    @property
    def num_directions(self) -> int:
        return self._num_directions

    def _execute(self, context: AgentContext) -> AgentResult:
        idea = context.task
        log.info("explorer_start", idea_len=len(idea))

        search_results = self._search_mathlib(idea)

        search_summary = self._format_search_results(search_results)

        user_content = EXPLORATION_USER_TEMPLATE.format(
            idea=idea,
            search_results=search_summary,
        )

        response = self._llm.complete(
            system=EXPLORATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
            use_cache=True,
        )

        exploration = self._parse_response(response.content, idea)

        if len(exploration.directions) > self._num_directions:
            exploration.directions = exploration.directions[: self._num_directions]

        log.info(
            "explorer_done",
            domain=exploration.domain,
            num_concepts=len(exploration.concepts),
            num_directions=len(exploration.directions),
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=exploration.model_dump(),
            token_usage=response.token_usage,
        )

    def _search_mathlib(self, idea: str) -> list[dict]:
        keywords = idea.split()
        queries = []
        queries.append(idea[:120])
        chunk_size = max(1, len(keywords) // self._max_search_queries)
        for i in range(0, len(keywords), chunk_size):
            q = " ".join(keywords[i : i + chunk_size])
            if q and q != queries[0]:
                queries.append(q)
            if len(queries) >= self._max_search_queries:
                break

        all_results: list[dict] = []
        seen_names: set[str] = set()
        for query in queries:
            result = self._search.execute(query)
            for entry in getattr(result, "entries", []):
                if entry.name not in seen_names:
                    seen_names.add(entry.name)
                    all_results.append({
                        "name": entry.name,
                        "type_signature": entry.type_signature,
                        "doc_string": entry.doc_string,
                        "module": entry.module,
                    })

        return all_results

    def _format_search_results(self, results: list[dict]) -> str:
        if not results:
            return "No Mathlib results found."
        lines = []
        for r in results:
            line = f"- **{r['name']}**: `{r['type_signature']}`"
            if r.get("doc_string"):
                line += f" — {r['doc_string']}"
            if r.get("module"):
                line += f" (from {r['module']})"
            lines.append(line)
        return "\n".join(lines)

    def _parse_response(self, content: str, idea: str) -> ExplorationResult:
        parsed = self._llm.extract_json(content)
        if not isinstance(parsed, dict):
            log.warning("explorer_parse_fallback", reason="no valid JSON")
            return ExplorationResult(
                raw_idea=idea,
                domain="unknown",
            )

        concepts = []
        for c in parsed.get("concepts", []):
            if isinstance(c, dict):
                concepts.append(Concept(
                    name=c.get("name", ""),
                    description=c.get("description", ""),
                    domain=c.get("domain", ""),
                    mathlib_ref=c.get("mathlib_ref"),
                ))

        directions = []
        for d in parsed.get("directions", []):
            if isinstance(d, dict):
                directions.append(ResearchDirection(
                    title=d.get("title", ""),
                    description=d.get("description", ""),
                    ambition_level=max(1, min(5, int(d.get("ambition_level", 3)))),
                    relevant_concepts=d.get("relevant_concepts", []),
                    estimated_difficulty=max(1, min(5, int(d.get("estimated_difficulty", 3)))),
                ))

        return ExplorationResult(
            raw_idea=idea,
            domain=parsed.get("domain", "unknown"),
            concepts=concepts,
            known_results=parsed.get("known_results", []),
            directions=directions,
        )
