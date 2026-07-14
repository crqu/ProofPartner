"""Extractor agent — extracts formalization targets from research papers.

Reads a mathematical paper (LaTeX source or extracted PDF text) and
produces an ExtractionResult with theorems, definitions, lemmas, and
prior work suitable for downstream formalization.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    EXTRACTOR_SYSTEM,
    EXTRACTOR_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.extraction import (
    ExtractionResult,
    ExtractedDefinition,
    ExtractedLemma,
    ExtractedPriorWork,
    ExtractedTheorem,
)

log = get_logger(__name__)

MAX_INPUT_CHARS = 100_000


class Extractor(BaseAgent):
    """Extracts formalization targets from mathematical research papers."""

    def __init__(
        self,
        llm_client: LLMClient,
        use_extended_thinking: bool = False,
    ) -> None:
        super().__init__(name="extractor", max_retries=2)
        self._llm = llm_client
        self._use_extended_thinking = use_extended_thinking

    def extract(self, paper_text: str) -> ExtractionResult:
        """Extract theorems, definitions, lemmas, and prior work from paper text."""
        truncated = paper_text[:MAX_INPUT_CHARS]

        user_content = EXTRACTOR_USER_TEMPLATE.format(paper_text=truncated)

        response = self._llm.complete(
            system=EXTRACTOR_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_extended_thinking=self._use_extended_thinking,
            use_cache=True,
        )

        self._accumulate_tokens(response.token_usage)

        return self._parse_result(response.content)

    def _execute(self, context: AgentContext) -> AgentResult:
        paper_text = context.task
        result = self.extract(paper_text)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=result.model_dump(),
            token_usage=self.cumulative_tokens,
        )

    def _parse_result(self, response_text: str) -> ExtractionResult:
        parsed = self._llm.extract_json(response_text)
        if not isinstance(parsed, dict):
            log.warning("extractor_parse_fallback")
            return ExtractionResult()

        theorems = [
            ExtractedTheorem(**t)
            for t in parsed.get("theorems", [])
            if isinstance(t, dict)
        ]
        definitions = [
            ExtractedDefinition(**d)
            for d in parsed.get("definitions", [])
            if isinstance(d, dict)
        ]
        lemmas = [
            ExtractedLemma(**lem)
            for lem in parsed.get("lemmas", [])
            if isinstance(lem, dict)
        ]
        prior_work = [
            ExtractedPriorWork(**p)
            for p in parsed.get("prior_work", [])
            if isinstance(p, dict)
        ]

        return ExtractionResult(
            theorems=theorems,
            definitions=definitions,
            lemmas=lemmas,
            prior_work=prior_work,
            paper_title=parsed.get("paper_title", ""),
            paper_domain=parsed.get("paper_domain", ""),
        )
