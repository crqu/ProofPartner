"""Natural Language Prover — generates structured informal proof sketches.

Produces an NLProofSketch (strategy, assumptions, key lemmas, proof steps)
BEFORE Lean formalization. Optionally uses extended thinking for deep reasoning.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    NL_PROVER_SYSTEM,
    NL_PROVER_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.proof import (
    CritiqueResult,
    NLProofSketch,
    NLProofStep,
)

log = get_logger(__name__)


class NaturalLanguageProver(BaseAgent):
    """Generates structured informal proof sketches before Lean formalization."""

    def __init__(
        self,
        llm_client: LLMClient,
        prover_config: ProverConfig | None = None,
    ) -> None:
        super().__init__(name="nl_prover", max_retries=2)
        self._llm = llm_client
        self._config = prover_config or ProverConfig()

    def _execute(self, context: AgentContext) -> AgentResult:
        statement = context.task
        statement_nl = context.metadata.get("statement_nl")
        feedback_str = context.metadata.get("critique_feedback", "")

        sketch, token_usage = self._generate(
            statement=statement,
            statement_nl=statement_nl,
            feedback=feedback_str,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=sketch.model_dump(),
            token_usage=token_usage,
        )

    def generate_proof(
        self,
        statement: str,
        statement_nl: str | None = None,
        feedback: CritiqueResult | None = None,
    ) -> tuple[NLProofSketch, TokenUsage]:
        """Generate an informal proof sketch for the given statement."""
        feedback_str = ""
        if feedback is not None and feedback.issues:
            lines = []
            for issue in feedback.issues:
                lines.append(
                    f"- [{issue.severity}] {issue.issue_type.value}: "
                    f"{issue.description}"
                )
                if issue.suggested_fix:
                    lines.append(f"  Fix: {issue.suggested_fix}")
            feedback_str = "\n".join(lines)

        return self._generate(
            statement=statement,
            statement_nl=statement_nl,
            feedback=feedback_str,
        )

    def _generate(
        self,
        statement: str,
        statement_nl: str | None = None,
        feedback: str = "",
    ) -> tuple[NLProofSketch, TokenUsage]:
        nl_section = ""
        if statement_nl:
            nl_section = (
                f"## Natural Language Description\n{statement_nl}"
            )

        feedback_section = ""
        if feedback is not None and feedback != "":
            feedback_section = (
                "## Critic Feedback (address these issues)\n"
                f"{feedback}"
            )

        user_content = NL_PROVER_USER_TEMPLATE.format(
            statement=statement,
            nl_description_section=nl_section,
            feedback_section=feedback_section,
        )

        response = self._llm.complete(
            system=NL_PROVER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.3,
            use_extended_thinking=self._config.use_extended_thinking,
            thinking_budget=getattr(self._config, "thinking_budget", 10000),
            use_cache=True,
        )

        sketch = self._parse_sketch(response.content)
        log.info(
            "nl_prover_done",
            steps=len(sketch.proof_steps),
            strategy=sketch.overall_strategy,
        )
        return sketch, response.token_usage

    def _parse_sketch(self, response_text: str) -> NLProofSketch:
        parsed = self._llm.extract_json(response_text)
        if not isinstance(parsed, dict):
            log.warning("nl_prover_parse_fallback")
            return NLProofSketch(
                overall_strategy="unknown",
                proof_steps=[
                    NLProofStep(
                        claim="Unparsed proof",
                        reasoning=response_text[:500],
                    )
                ],
            )

        steps = []
        for step_data in parsed.get("proof_steps", []):
            steps.append(
                NLProofStep(
                    claim=step_data.get("claim", ""),
                    reasoning=step_data.get("reasoning", ""),
                    sub_claims=step_data.get("sub_claims", []),
                )
            )

        return NLProofSketch(
            proof_steps=steps,
            assumptions=parsed.get("assumptions", []),
            key_lemmas=parsed.get("key_lemmas", []),
            overall_strategy=parsed.get("overall_strategy", ""),
        )
