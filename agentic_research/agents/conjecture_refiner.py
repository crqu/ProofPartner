"""Conjecture Refiner: produces refined variants of failed conjectures.

Takes a failed conjecture (disproved or unprovable) and generates 2-4
refined variants using one of four strategies:
  1. Weakening — add hypotheses, restrict to special cases
  2. Strengthening — if too weak, strengthen and check provability
  3. Reformulation — express in a different mathematical framework
  4. Specialization — try specific instances (n=2, finite case, etc.)
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    CONJECTURE_REFINEMENT_SYSTEM,
    CONJECTURE_REFINEMENT_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.refinement import RefinementType
from agentic_research.models.research import Conjecture

log = get_logger(__name__)

_STRATEGY_DESCRIPTIONS: dict[RefinementType, str] = {
    RefinementType.WEAKENING: (
        "WEAKENING: Add additional hypotheses, restrict to special cases, "
        "reduce quantifier strength (∀→∃, 'for all n' → 'for sufficiently large n'). "
        "The goal is to make the conjecture more likely to be true by narrowing its scope."
    ),
    RefinementType.STRENGTHENING: (
        "STRENGTHENING: The conjecture may be too weak to be interesting. "
        "Strengthen the statement — add stronger conclusions, tighten bounds, "
        "or remove unnecessary hypotheses — and check if it's still provable."
    ),
    RefinementType.REFORMULATION: (
        "REFORMULATION: Express the same mathematical idea in a different framework. "
        "For example, translate an algebraic statement into a combinatorial one, "
        "or rephrase in terms of different mathematical objects that may be easier to reason about."
    ),
    RefinementType.SPECIALIZATION: (
        "SPECIALIZATION: Try specific instances of the general conjecture. "
        "For example: n=2, finite case, commutative case, abelian groups, "
        "compact spaces, or other concrete restrictions."
    ),
}


class ConjectureRefiner(BaseAgent):
    """Refine a failed conjecture into provable variants."""

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="conjecture_refiner", max_retries=1)
        self._llm = llm_client

    def _execute(self, context: AgentContext) -> AgentResult:
        conjecture_data = context.metadata.get("conjecture")
        if not conjecture_data:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                error_message="No conjecture provided in metadata",
            )

        conjecture = Conjecture.model_validate(conjecture_data)
        failure_reason = context.metadata.get("failure_reason", "unknown")
        failure_outcome = context.metadata.get("failure_outcome", "unknown")
        original_idea = context.metadata.get("original_idea", conjecture.natural_language)
        strategy_str = context.metadata.get("strategy", RefinementType.WEAKENING.value)
        strategy = RefinementType(strategy_str)

        refined = self.refine(
            conjecture=conjecture,
            failure_reason=failure_reason,
            failure_outcome=failure_outcome,
            original_idea=original_idea,
            strategy=strategy,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result={
                "refined_conjectures": [c.model_dump() for c in refined],
                "strategy": strategy.value,
            },
        )

    def refine(
        self,
        *,
        conjecture: Conjecture,
        failure_reason: str,
        failure_outcome: str,
        original_idea: str,
        strategy: RefinementType,
    ) -> list[Conjecture]:
        prompt = CONJECTURE_REFINEMENT_USER_TEMPLATE.format(
            original_statement=conjecture.statement,
            original_nl=conjecture.natural_language,
            failure_outcome=failure_outcome,
            failure_reason=failure_reason,
            strategy=_STRATEGY_DESCRIPTIONS[strategy],
            original_idea=original_idea,
        )

        response = self._llm.complete(
            system=CONJECTURE_REFINEMENT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        parsed = self._llm.extract_json(response.content)
        if not isinstance(parsed, dict):
            log.warning("conjecture_refiner_parse_failed")
            return []

        raw_conjectures = parsed.get("refined_conjectures", [])
        if not isinstance(raw_conjectures, list):
            return []

        refined: list[Conjecture] = []
        for raw in raw_conjectures[:4]:
            if not isinstance(raw, dict):
                continue
            try:
                c = Conjecture(
                    statement=str(raw.get("statement", "")),
                    natural_language=str(raw.get("natural_language", "")),
                    confidence=float(raw.get("confidence", 0.5)),
                    difficulty=int(raw.get("difficulty", 3)),
                    related_results=raw.get("related_results", []),
                    novelty_score=float(raw.get("novelty_score", 0.5)),
                    formalizability_score=float(raw.get("formalizability_score", 0.5)),
                )
                refined.append(c)
            except (ValueError, TypeError):
                continue

        return refined
