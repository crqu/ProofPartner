"""ProofCorrector agent — error-driven proof refinement.

Parses Lean 4 compiler errors, classifies them, and suggests targeted
fixes (type coercions, missing imports, alternative tactics) to feed
back into the next prover iteration.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.proof import ErrorCategory, ProofCorrection

log = get_logger(__name__)

PROOF_CORRECTOR_SYSTEM = """\
You are a Lean 4 proof correction specialist. Given a failed proof attempt,
the compiler error message, and the current goal state, you must:

1. Parse the Lean 4 compiler error message carefully.
2. Classify the error into one of these categories:
   - type_mismatch: Type errors, wrong types, failed unification
   - missing_import: Missing Mathlib or other module imports
   - tactic_failure: A tactic failed (simp, omega, ring, etc.)
   - universe_level: Universe level issues (Sort, Type, Prop mismatch)
   - unknown_identifier: Unknown name, undeclared identifier
   - timeout: Deterministic timeout or heartbeat exceeded
   - other: Anything else
3. Suggest specific tactics or code fixes based on the error type:
   - For type_mismatch: suggest explicit type coercions, casts, or the correct types
   - For missing_import: suggest the specific Mathlib module to import
   - For tactic_failure: suggest alternative tactics (omega, simp, ring, norm_num, linarith, etc.)
   - For unknown_identifier: suggest the correct identifier name or required import
   - For universe_level: suggest universe annotations or Prop/Type adjustments
   - For timeout: suggest simplifying the proof or breaking it into steps
4. Provide a revised proof sketch incorporating the fix.

Respond with a JSON object:
```json
{
  "error_category": "<category>",
  "error_message": "<original error>",
  "suggested_tactics": ["tactic1", "tactic2"],
  "revised_proof_sketch": "<corrected lean 4 code>",
  "confidence": 0.7,
  "reasoning": "<why this fix should work>"
}
```
"""

PROOF_CORRECTOR_USER_TEMPLATE = """\
## Failed Proof
```lean
{failed_proof}
```

## Compiler Error
{error_message}

## Goal State
{lean_goal_state}

## Prior Attempts
{prior_attempts}

{compiler_feedback}Analyze the error and suggest a correction.
"""


class ProofCorrector(BaseAgent):
    """Analyzes Lean 4 compilation errors and suggests targeted fixes."""

    SYSTEM_PROMPT = PROOF_CORRECTOR_SYSTEM

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="proof_corrector", max_retries=1)
        self._llm = llm_client

    def correct(
        self,
        failed_proof: str,
        error_message: str,
        lean_goal_state: str,
        prior_attempts: list[str] | None = None,
        compiler_errors: list[str] | None = None,
    ) -> ProofCorrection:
        """Analyze a failed proof and return a structured correction."""
        log.info(
            "proof_corrector_start",
            error_len=len(error_message),
            prior_count=len(prior_attempts) if prior_attempts else 0,
            compiler_error_count=len(compiler_errors) if compiler_errors else 0,
        )

        prior_text = "\n".join(
            f"Attempt {i + 1}:\n```lean\n{a}\n```" for i, a in enumerate(prior_attempts)
        ) if prior_attempts else "None"

        if compiler_errors:
            feedback_lines = ["## Previous Compiler Errors"]
            for i, err in enumerate(compiler_errors, 1):
                feedback_lines.append(f"{i}. {err}")
            feedback_lines.append("")
            feedback_lines.append(
                "IMPORTANT: Your correction MUST avoid repeating these errors. "
                "Each error above represents a failed approach — do not retry it.\n\n"
            )
            compiler_feedback = "\n".join(feedback_lines)
        else:
            compiler_feedback = ""

        user_content = PROOF_CORRECTOR_USER_TEMPLATE.format(
            failed_proof=failed_proof,
            error_message=error_message,
            lean_goal_state=lean_goal_state,
            prior_attempts=prior_text,
            compiler_feedback=compiler_feedback,
        )

        response = self._llm.complete(
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_cache=True,
        )
        self._accumulate_tokens(response.token_usage)

        parsed = self._llm.extract_json(response.content)
        if isinstance(parsed, dict):
            try:
                category = ErrorCategory(parsed.get("error_category", "other"))
            except ValueError:
                category = ErrorCategory.OTHER

            correction = ProofCorrection(
                error_category=category,
                error_message=parsed.get("error_message", error_message),
                suggested_tactics=parsed.get("suggested_tactics", []),
                revised_proof_sketch=parsed.get("revised_proof_sketch", ""),
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", ""),
            )
        else:
            correction = ProofCorrection(
                error_category=ErrorCategory.OTHER,
                error_message=error_message,
                reasoning="Could not parse LLM correction response",
            )

        log.info(
            "proof_corrector_done",
            category=correction.error_category.value,
            confidence=correction.confidence,
            tactic_count=len(correction.suggested_tactics),
        )
        return correction

    def _execute(self, context: AgentContext) -> AgentResult:
        """BaseAgent protocol — delegates to correct()."""
        failed_proof = context.metadata.get("failed_proof", "")
        error_message = context.metadata.get("error_message", "")
        lean_goal_state = context.metadata.get("lean_goal_state", "")
        prior_attempts = context.metadata.get("prior_attempts", [])
        compiler_errors = context.metadata.get("compiler_errors", []) or None

        correction = self.correct(
            failed_proof=failed_proof,
            error_message=error_message,
            lean_goal_state=lean_goal_state,
            prior_attempts=prior_attempts,
            compiler_errors=compiler_errors,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=correction.model_dump(),
            token_usage=self.cumulative_tokens,
        )
