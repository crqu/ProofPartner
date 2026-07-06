"""Back-translation agent: Lean 4 code → natural language.

Used by the blind verification path of the Intent Judge.
Strips AI-generated comments via hint_cleaner before translating.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import INFORMALIZE_PROMPT
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.verification import InformalizationResult
from agentic_research.tools.hint_cleaner import HintCleaner

log = get_logger(__name__)


class Informalizer(BaseAgent):
    """Convert Lean 4 code to natural language via LLM back-translation."""

    def __init__(
        self,
        llm_client: LLMClient,
        hint_cleaner: HintCleaner | None = None,
    ) -> None:
        super().__init__(name="informalizer", max_retries=2)
        self._llm = llm_client
        self._cleaner = hint_cleaner or HintCleaner()

    def _execute(self, context: AgentContext) -> AgentResult:
        lean_code = context.task
        result = self.informalize(lean_code)
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=result.model_dump(),
        )

    def informalize(self, lean_code: str) -> InformalizationResult:
        clean_result = self._cleaner.execute(lean_code)
        cleaned = clean_result.cleaned_code

        prompt = INFORMALIZE_PROMPT.format(lean_code=cleaned)
        response = self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        return InformalizationResult(
            lean_input=lean_code,
            natural_language_output=response.content.strip(),
        )
