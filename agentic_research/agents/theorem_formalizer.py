"""Theorem Formalizer — takes a NL conjecture + accepted type
formalizations and produces a compilable Lean 4 theorem statement.
"""

from __future__ import annotations

import re

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    THEOREM_FORMALIZER_FEEDBACK_TEMPLATE,
    THEOREM_FORMALIZER_SYSTEM,
    THEOREM_FORMALIZER_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.formalization import TheoremFormalization
from agentic_research.models.tools import CompilationStatus
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)


def _extract_lean_code(text: str) -> str:
    match = re.search(r"```lean\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class TheoremFormalizer(BaseAgent):
    """Produces a Lean 4 theorem statement from NL + type definitions."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        *,
        max_iterations: int = 5,
    ) -> None:
        super().__init__(name="theorem_formalizer", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._max_iterations = max_iterations

    def _execute(self, context: AgentContext) -> AgentResult:
        conjecture = context.task
        type_definitions = context.metadata.get("type_definitions", "")

        log.info("theorem_formalizer_start", conjecture_len=len(conjecture))

        result, tokens = self._formalize(conjecture, type_definitions)

        status = AgentStatus.SUCCESS if result.compiles else AgentStatus.FAILURE

        log.info(
            "theorem_formalizer_done",
            compiles=result.compiles,
            iterations=result.iterations_used,
        )

        return AgentResult(
            agent_name=self.name,
            status=status,
            result=result.model_dump(),
            token_usage=tokens,
            error_message=result.failure_reason,
        )

    def _formalize(
        self,
        conjecture: str,
        type_definitions: str,
    ) -> tuple[TheoremFormalization, TokenUsage]:
        total_tokens = TokenUsage()
        previous_code: str | None = None
        previous_errors: str = ""

        for iteration in range(1, self._max_iterations + 1):
            if previous_code is None:
                user_content = THEOREM_FORMALIZER_USER_TEMPLATE.format(
                    conjecture=conjecture,
                    type_definitions=type_definitions or "-- no custom types",
                )
            else:
                user_content = THEOREM_FORMALIZER_FEEDBACK_TEMPLATE.format(
                    conjecture=conjecture,
                    type_definitions=type_definitions or "-- no custom types",
                    previous_attempt=previous_code,
                    errors=previous_errors,
                )

            response = self._llm.complete(
                system=THEOREM_FORMALIZER_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.1,
                use_cache=True,
            )
            total_tokens.input_tokens += response.token_usage.input_tokens
            total_tokens.output_tokens += response.token_usage.output_tokens

            lean_code = _extract_lean_code(response.content)
            full_code = (
                f"{type_definitions}\n\n{lean_code}" if type_definitions else lean_code
            )
            compilation = self._repl.execute(full_code)

            if compilation.compilation_status == CompilationStatus.OK:
                return TheoremFormalization(
                    conjecture_nl=conjecture,
                    lean_statement=lean_code,
                    compiles=True,
                    iterations_used=iteration,
                    type_imports=[type_definitions] if type_definitions else [],
                ), total_tokens

            previous_code = lean_code
            previous_errors = (
                "\n".join(compilation.errors)
                if compilation.errors
                else "Unknown compilation error"
            )

            log.info(
                "theorem_formalizer_retry",
                iteration=iteration,
                error_count=len(compilation.errors),
            )

        failure = (
            f"Failed to compile theorem after {self._max_iterations} iterations. "
            f"Last errors: {previous_errors}"
        )
        return TheoremFormalization(
            conjecture_nl=conjecture,
            lean_statement=previous_code or "",
            compiles=False,
            iterations_used=self._max_iterations,
            failure_reason=failure,
        ), total_tokens
