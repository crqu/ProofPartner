"""Refinement Reporter: generates human-readable reports of the refinement journey.

Shows: original conjecture -> refinement 1 (failed: counterexample) ->
       refinement 2 (proved!) with explanations of each step.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    REFINEMENT_REPORT_SYSTEM,
    REFINEMENT_REPORT_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.refinement import (
    RefinementHistory,
    RefinementReport,
)

log = get_logger(__name__)


class RefinementReporter(BaseAgent):
    """Generate a human-readable report of the refinement journey."""

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="refinement_reporter", max_retries=1)
        self._llm = llm_client

    def _execute(self, context: AgentContext) -> AgentResult:
        history_data = context.metadata.get("history")
        if not history_data:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                error_message="No refinement history provided",
            )

        history = RefinementHistory.model_validate(history_data)
        report = self.generate_report(history)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=report.model_dump(),
        )

    def generate_report(self, history: RefinementHistory) -> RefinementReport:
        attempts_text = self._format_attempts(history)

        original_conjecture_text = "Not available"
        if history.original_conjecture:
            original_conjecture_text = (
                f"Statement: {history.original_conjecture.statement}\n"
                f"Description: {history.original_conjecture.natural_language}"
            )

        final_status = history.final_result.value if history.final_result else "in_progress"

        prompt = REFINEMENT_REPORT_USER_TEMPLATE.format(
            original_idea=history.original_idea,
            original_conjecture=original_conjecture_text,
            attempts=attempts_text,
            final_status=final_status,
        )

        response = self._llm.complete(
            system=REFINEMENT_REPORT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return RefinementReport(
            markdown_report=response.content,
            structured_history=history,
        )

    def _format_attempts(self, history: RefinementHistory) -> str:
        if not history.attempts:
            return "No refinement attempts were made."

        parts: list[str] = []
        for i, attempt in enumerate(history.attempts, 1):
            part = (
                f"### Attempt {i} (depth={attempt.depth})\n"
                f"- Strategy: {attempt.refinement_type.value}\n"
                f"- Original: {attempt.original.natural_language}\n"
                f"- Refined: {attempt.refined.natural_language}\n"
                f"- Outcome: {attempt.outcome.value}\n"
            )
            if attempt.failure_reason:
                part += f"- Failure reason: {attempt.failure_reason}\n"
            if attempt.proof_code:
                part += "- Proof found: yes\n"
            parts.append(part)

        return "\n".join(parts)
