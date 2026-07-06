"""Conjecture Generator — takes an ExplorationResult and generates
ranked conjecture candidates ranging from conservative to ambitious.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    CONJECTURE_RANKING_PROMPT,
    CONJECTURE_SYSTEM_PROMPT,
    CONJECTURE_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.research import (
    Conjecture,
    ConjectureSet,
    ExplorationResult,
)

log = get_logger(__name__)


class ConjectureGenerator(BaseAgent):
    """Generates and ranks conjecture candidates from an exploration result."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        num_conjectures: int = 5,
    ) -> None:
        super().__init__(name="conjecture_generator", max_retries=2)
        self._llm = llm_client
        self._num_conjectures = num_conjectures

    @property
    def num_conjectures(self) -> int:
        return self._num_conjectures

    def _execute(self, context: AgentContext) -> AgentResult:
        exploration = self._load_exploration(context)
        log.info("conjecturer_start", domain=exploration.domain)

        conjectures, gen_tokens = self._generate_conjectures(exploration)

        if len(conjectures) > self._num_conjectures:
            conjectures = conjectures[: self._num_conjectures]

        ranking, rank_tokens = self._rank_conjectures(conjectures)

        total_tokens = TokenUsage(
            input_tokens=gen_tokens.input_tokens + rank_tokens.input_tokens,
            output_tokens=gen_tokens.output_tokens + rank_tokens.output_tokens,
            cache_creation_input_tokens=(
                gen_tokens.cache_creation_input_tokens + rank_tokens.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                gen_tokens.cache_read_input_tokens + rank_tokens.cache_read_input_tokens
            ),
        )

        result = ConjectureSet(
            conjectures=conjectures,
            ranking=ranking,
            exploration_context=exploration,
        )

        log.info(
            "conjecturer_done",
            num_conjectures=len(conjectures),
            top_confidence=conjectures[ranking[0]].confidence if ranking and conjectures else 0,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=result.model_dump(),
            token_usage=total_tokens,
        )

    def _load_exploration(self, context: AgentContext) -> ExplorationResult:
        if "exploration_result" in context.metadata:
            return ExplorationResult.model_validate(context.metadata["exploration_result"])
        return ExplorationResult(
            raw_idea=context.task,
            domain="unknown",
        )

    def _generate_conjectures(
        self, exploration: ExplorationResult
    ) -> tuple[list[Conjecture], TokenUsage]:
        concepts_text = "\n".join(
            f"- {c.name}: {c.description}" for c in exploration.concepts
        ) or "None identified"

        known_text = "\n".join(
            f"- {r}" for r in exploration.known_results
        ) or "None identified"

        directions_text = "\n".join(
            f"- [{d.ambition_level}/5] {d.title}: {d.description}"
            for d in exploration.directions
        ) or "None identified"

        user_content = CONJECTURE_USER_TEMPLATE.format(
            idea=exploration.raw_idea,
            domain=exploration.domain,
            concepts=concepts_text,
            known_results=known_text,
            directions=directions_text,
            num_conjectures=self._num_conjectures,
        )

        response = self._llm.complete(
            system=CONJECTURE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.3,
            use_cache=True,
        )

        conjectures = self._parse_conjectures(response.content)
        return conjectures, response.token_usage

    def _rank_conjectures(
        self, conjectures: list[Conjecture]
    ) -> tuple[list[int], TokenUsage]:
        if len(conjectures) <= 1:
            return list(range(len(conjectures))), TokenUsage()

        conjectures_text = "\n".join(
            f"[{i}] {c.statement} (confidence={c.confidence}, "
            f"novelty={c.novelty_score}, formalizability={c.formalizability_score})"
            for i, c in enumerate(conjectures)
        )

        response = self._llm.complete(
            system="You are a mathematical research advisor ranking conjectures.",
            messages=[{
                "role": "user",
                "content": CONJECTURE_RANKING_PROMPT.format(conjectures=conjectures_text),
            }],
            temperature=0.0,
        )

        parsed = self._llm.extract_json(response.content)
        if isinstance(parsed, dict) and "ranking" in parsed:
            raw_ranking = parsed["ranking"]
            valid_indices = set(range(len(conjectures)))
            ranking = [i for i in raw_ranking if isinstance(i, int) and i in valid_indices]
            for i in range(len(conjectures)):
                if i not in ranking:
                    ranking.append(i)
            return ranking, response.token_usage

        ranking = sorted(
            range(len(conjectures)),
            key=lambda i: conjectures[i].composite_score,
            reverse=True,
        )
        return ranking, response.token_usage

    def _parse_conjectures(self, content: str) -> list[Conjecture]:
        parsed = self._llm.extract_json(content)

        raw_list: list[dict] = []
        if isinstance(parsed, dict) and "conjectures" in parsed:
            raw_list = parsed["conjectures"]
        elif isinstance(parsed, list):
            raw_list = parsed

        conjectures: list[Conjecture] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            conjectures.append(Conjecture(
                statement=item.get("statement", ""),
                natural_language=item.get("natural_language", ""),
                confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                difficulty=max(1, min(5, int(item.get("difficulty", 3)))),
                related_results=item.get("related_results", []),
                novelty_score=max(0.0, min(1.0, float(item.get("novelty_score", 0.5)))),
                formalizability_score=max(
                    0.0, min(1.0, float(item.get("formalizability_score", 0.5)))
                ),
            ))

        return conjectures
