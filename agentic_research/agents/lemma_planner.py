"""Lemma Planner agent — generates well-known auxiliary lemmas for each
type candidate, following the 'unit testing for types' methodology.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    LEMMA_PLANNER_SYSTEM,
    LEMMA_PLANNER_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.formalization import (
    LemmaStatement,
    TypeCandidate,
    TypePlan,
)

log = get_logger(__name__)


class LemmaPlanner(BaseAgent):
    """Generates auxiliary lemmas to validate type definitions."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        max_lemmas_per_type: int = 5,
    ) -> None:
        super().__init__(name="lemma_planner", max_retries=2)
        self._llm = llm_client
        self._max_lemmas_per_type = max_lemmas_per_type

    def _execute(self, context: AgentContext) -> AgentResult:
        plan = TypePlan.model_validate(context.metadata.get("type_plan", {}))
        log.info("lemma_planner_start", num_types=len(plan.candidates))

        all_lemmas: list[LemmaStatement] = []
        total_tokens = TokenUsage()

        new_types = [c for c in plan.candidates if not c.is_in_mathlib]
        for candidate in new_types:
            lemmas, tokens = self._plan_lemmas(candidate)
            all_lemmas.extend(lemmas[:self._max_lemmas_per_type])
            total_tokens.input_tokens += tokens.input_tokens
            total_tokens.output_tokens += tokens.output_tokens
            total_tokens.cache_creation_input_tokens += tokens.cache_creation_input_tokens
            total_tokens.cache_read_input_tokens += tokens.cache_read_input_tokens

        log.info(
            "lemma_planner_done",
            total_lemmas=len(all_lemmas),
            types_covered=len(new_types),
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result={
                "lemmas": [l.model_dump() for l in all_lemmas],
                "type_plan": plan.model_dump(),
            },
            token_usage=total_tokens,
        )

    def _plan_lemmas(
        self, candidate: TypeCandidate
    ) -> tuple[list[LemmaStatement], TokenUsage]:
        user_content = LEMMA_PLANNER_USER_TEMPLATE.format(
            type_name=candidate.name,
            type_description=candidate.informal_description,
            lean_signature=candidate.lean_signature or "not specified",
            dependencies=", ".join(candidate.depends_on) if candidate.depends_on else "none",
        )

        response = self._llm.complete(
            system=LEMMA_PLANNER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
            use_cache=True,
        )

        lemmas = self._parse_lemmas(response.content, candidate.name)
        return lemmas, response.token_usage

    def _parse_lemmas(self, content: str, type_name: str) -> list[LemmaStatement]:
        parsed = self._llm.extract_json(content)
        if not isinstance(parsed, dict):
            log.warning("lemma_planner_parse_fallback", type_name=type_name)
            return []

        lemmas: list[LemmaStatement] = []
        for item in parsed.get("lemmas", []):
            if isinstance(item, dict):
                lemmas.append(LemmaStatement(
                    name=item.get("name", f"{type_name}_lemma_{len(lemmas)}"),
                    statement_nl=item.get("statement_nl", ""),
                    for_type=item.get("for_type", type_name),
                    is_well_known=item.get("is_well_known", True),
                ))

        return lemmas
