"""Counterexample Searcher: tries to disprove a conjecture before proving.

Strategy:
  1. Generate candidate counterexamples via LLM (edge/small/degenerate cases)
  2. Formalize each candidate in Lean 4
  3. If a counterexample compiles and proves the negation → DISPROVED
  4. If none found after N attempts → PLAUSIBLE
"""

from __future__ import annotations

import re

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    COUNTEREXAMPLE_FORMALIZATION_PROMPT,
    COUNTEREXAMPLE_GENERATION_PROMPT,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.tools import CompilationStatus
from agentic_research.models.verification import (
    CounterexampleCandidate,
    CounterexampleResult,
    CounterexampleStatus,
)
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)

_DEFAULT_MAX_CANDIDATES = 5


def _extract_lean_code(text: str) -> str:
    match = re.search(r"```lean\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class CounterexampleSearcher(BaseAgent):
    """Try to disprove a conjecture by finding counterexamples."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    ) -> None:
        super().__init__(name="counterexample_searcher", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._max_candidates = max_candidates

    def _execute(self, context: AgentContext) -> AgentResult:
        lean_code = context.task
        conjecture = context.metadata.get("conjecture", lean_code)

        result = self.search(lean_code=lean_code, conjecture=conjecture)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=result.model_dump(),
        )

    def search(
        self,
        *,
        lean_code: str,
        conjecture: str,
    ) -> CounterexampleResult:
        raw_candidates = self._generate_candidates(lean_code, conjecture)

        tried: list[CounterexampleCandidate] = []
        for raw in raw_candidates:
            candidate = self._formalize_and_check(raw, lean_code)
            tried.append(candidate)

            if candidate.proves_negation:
                log.info("counterexample_found", description=candidate.description)
                return CounterexampleResult(
                    status=CounterexampleStatus.DISPROVED,
                    candidates_tried=tried,
                    successful_counterexample=candidate,
                    attempts_made=len(tried),
                )

        return CounterexampleResult(
            status=CounterexampleStatus.PLAUSIBLE,
            candidates_tried=tried,
            attempts_made=len(tried),
        )

    def _generate_candidates(
        self, lean_code: str, conjecture: str
    ) -> list[dict[str, str]]:
        prompt = COUNTEREXAMPLE_GENERATION_PROMPT.format(
            conjecture=conjecture,
            lean_code=lean_code,
            num_candidates=self._max_candidates,
        )

        response = self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        parsed = self._llm.extract_json(response.content)
        if not isinstance(parsed, dict):
            return []

        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list):
            return []

        return candidates[: self._max_candidates]

    def _formalize_and_check(
        self, raw_candidate: dict[str, str], lean_code: str
    ) -> CounterexampleCandidate:
        description = str(raw_candidate.get("description", "unknown"))
        values = str(raw_candidate.get("values", ""))

        prompt = COUNTEREXAMPLE_FORMALIZATION_PROMPT.format(
            lean_code=lean_code,
            counterexample_description=description,
            counterexample_values=values,
        )

        response = self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        formalized = _extract_lean_code(response.content)
        compilation = self._repl.execute(formalized)

        compiles_ok = compilation.compilation_status == CompilationStatus.OK
        proves_negation = compiles_ok and compilation.all_goals_closed

        return CounterexampleCandidate(
            description=description,
            lean_code=formalized,
            compilation_status=compilation.compilation_status.value,
            proves_negation=proves_negation,
        )
