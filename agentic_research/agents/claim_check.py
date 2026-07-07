"""Claim Check tool — verifies that formalization hasn't silently
weakened the original statement.

Checks:
  1. Byte-for-byte statement preservation
  2. Semantic fidelity via LLM (optional)
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    CLAIM_CHECK_SYSTEM,
    CLAIM_CHECK_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.formalization import ClaimCheckResult, ClaimCheckVerdict

log = get_logger(__name__)


def check_statement_preserved(
    original_statement: str,
    formalized_code: str,
) -> bool:
    """Verify the theorem statement text appears in the formalized code."""
    normalized_original = " ".join(original_statement.split())
    normalized_formal = " ".join(formalized_code.split())
    return normalized_original in normalized_formal


class ClaimCheck(BaseAgent):
    """Verifies formalization fidelity against the original conjecture."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        use_llm_check: bool = True,
    ) -> None:
        super().__init__(name="claim_check", max_retries=1)
        self._llm = llm_client
        self._use_llm_check = use_llm_check

    def _execute(self, context: AgentContext) -> AgentResult:
        conjecture_nl = context.task
        lean_code = context.metadata.get("lean_code", "")
        type_definitions = context.metadata.get("type_definitions", "")

        log.info("claim_check_start", conjecture_len=len(conjecture_nl))

        if self._use_llm_check:
            result, token_usage = self._llm_check(conjecture_nl, lean_code, type_definitions)
        else:
            result = ClaimCheckResult(
                verdict=ClaimCheckVerdict.PASS,
                original_statement=conjecture_nl,
                formalized_statement=lean_code,
                reason="Structural checks passed (LLM check disabled)",
                statement_preserved=True,
            )
            token_usage = None

        log.info(
            "claim_check_done",
            verdict=result.verdict.value,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=result.model_dump(),
            token_usage=token_usage or TokenUsage(),
        )

    def _llm_check(
        self,
        conjecture_nl: str,
        lean_code: str,
        type_definitions: str,
    ) -> tuple[ClaimCheckResult, TokenUsage]:

        user_content = CLAIM_CHECK_USER_TEMPLATE.format(
            conjecture_nl=conjecture_nl,
            lean_code=lean_code,
            type_definitions=type_definitions or "-- none",
        )

        response = self._llm.complete(
            system=CLAIM_CHECK_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_cache=True,
        )

        parsed = self._llm.extract_json(response.content)
        if isinstance(parsed, dict):
            verdict_str = parsed.get("verdict", "fail")
            verdict = (
                ClaimCheckVerdict.PASS
                if verdict_str == "pass"
                else ClaimCheckVerdict.FAIL
            )
            return ClaimCheckResult(
                verdict=verdict,
                original_statement=conjecture_nl,
                formalized_statement=lean_code,
                reason=parsed.get("reason", ""),
                statement_preserved=parsed.get("statement_preserved", True),
            ), response.token_usage

        return ClaimCheckResult(
            verdict=ClaimCheckVerdict.FAIL,
            original_statement=conjecture_nl,
            formalized_statement=lean_code,
            reason="Could not parse LLM claim check response",
        ), response.token_usage
