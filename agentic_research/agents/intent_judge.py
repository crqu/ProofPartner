"""Intent Judge: verifies a Lean formalization captures the user's idea.

Uses three varied prompting strategies for diversity:
  1. Blind path — back-translate via Informalizer, then compare
  2. Direct path — directly compare Lean code to idea + conjecture
  3. Adversarial path — devil's advocate seeking semantic mismatches

Adjudication: any VALID concern from any path → overall INCORRECT.

Feature flag: OPENAI_ENABLED=true adds a 4th GPT verification path.
"""

from __future__ import annotations

import os

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.informalizer import Informalizer
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    INTENT_ADVERSARIAL_SYSTEM_PROMPT,
    INTENT_ADVERSARIAL_USER_TEMPLATE,
    INTENT_BLIND_SYSTEM_PROMPT,
    INTENT_BLIND_USER_TEMPLATE,
    INTENT_DIRECT_SYSTEM_PROMPT,
    INTENT_DIRECT_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.verification import (
    IntentVerdict,
    IntentVerdictType,
    PathVerdict,
    VerificationPath,
)

log = get_logger(__name__)


class IntentJudge(BaseAgent):
    """Verify a Lean formalization captures the user's original idea."""

    def __init__(
        self,
        llm_client: LLMClient,
        informalizer: Informalizer,
    ) -> None:
        super().__init__(name="intent_judge", max_retries=1)
        self._llm = llm_client
        self._informalizer = informalizer

    def _execute(self, context: AgentContext) -> AgentResult:
        lean_code = context.task
        original_idea = context.metadata.get("original_idea", "")
        conjecture = context.metadata.get("conjecture", "")

        verdict = self.judge(
            lean_code=lean_code,
            original_idea=original_idea,
            conjecture=conjecture,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=verdict.model_dump(),
        )

    def judge(
        self,
        *,
        lean_code: str,
        original_idea: str,
        conjecture: str,
    ) -> IntentVerdict:
        path_verdicts: list[PathVerdict] = []

        blind = self._run_blind_path(lean_code, original_idea)
        path_verdicts.append(blind)

        direct = self._run_direct_path(lean_code, original_idea, conjecture)
        path_verdicts.append(direct)

        adversarial = self._run_adversarial_path(lean_code, original_idea, conjecture)
        path_verdicts.append(adversarial)

        if _openai_enabled():
            openai_verdict = self._run_openai_path(lean_code, original_idea, conjecture)
            path_verdicts.append(openai_verdict)

        return _adjudicate(path_verdicts)

    def _run_blind_path(self, lean_code: str, original_idea: str) -> PathVerdict:
        informal = self._informalizer.informalize(lean_code)

        prompt = INTENT_BLIND_USER_TEMPLATE.format(
            back_translation=informal.natural_language_output,
            original_idea=original_idea,
        )

        response = self._llm.complete(
            system=INTENT_BLIND_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        return _parse_path_verdict(VerificationPath.BLIND, response.content, self._llm)

    def _run_direct_path(
        self, lean_code: str, original_idea: str, conjecture: str
    ) -> PathVerdict:
        prompt = INTENT_DIRECT_USER_TEMPLATE.format(
            lean_code=lean_code,
            original_idea=original_idea,
            conjecture=conjecture,
        )

        response = self._llm.complete(
            system=INTENT_DIRECT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return _parse_path_verdict(VerificationPath.DIRECT, response.content, self._llm)

    def _run_adversarial_path(
        self, lean_code: str, original_idea: str, conjecture: str
    ) -> PathVerdict:
        prompt = INTENT_ADVERSARIAL_USER_TEMPLATE.format(
            lean_code=lean_code,
            original_idea=original_idea,
            conjecture=conjecture,
        )

        response = self._llm.complete(
            system=INTENT_ADVERSARIAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return _parse_path_verdict(
            VerificationPath.ADVERSARIAL, response.content, self._llm
        )

    def _run_openai_path(
        self, lean_code: str, original_idea: str, conjecture: str
    ) -> PathVerdict:
        try:
            import openai

            client = openai.OpenAI()
            prompt = INTENT_DIRECT_USER_TEMPLATE.format(
                lean_code=lean_code,
                original_idea=original_idea,
                conjecture=conjecture,
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": INTENT_DIRECT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            return _parse_path_verdict(VerificationPath.OPENAI, content, self._llm)
        except Exception as exc:
            log.warning("openai_path_failed", error=str(exc))
            return PathVerdict(
                path=VerificationPath.OPENAI,
                verdict=IntentVerdictType.CORRECT,
                concerns=[],
                confidence=0.0,
                reasoning=f"OpenAI path failed: {exc}",
            )


def _openai_enabled() -> bool:
    return os.environ.get("OPENAI_ENABLED", "false").lower() in ("true", "1", "yes")


def _parse_path_verdict(
    path: VerificationPath, content: str, llm: LLMClient
) -> PathVerdict:
    parsed = llm.extract_json(content)
    if not isinstance(parsed, dict):
        return PathVerdict(
            path=path,
            verdict=IntentVerdictType.INCORRECT,
            concerns=["Failed to parse verification response"],
            confidence=0.0,
            reasoning=content[:500],
        )

    raw_verdict = str(parsed.get("verdict", "incorrect")).lower()
    verdict = (
        IntentVerdictType.CORRECT
        if raw_verdict == "correct"
        else IntentVerdictType.INCORRECT
    )
    concerns = parsed.get("concerns", [])
    if not isinstance(concerns, list):
        concerns = [str(concerns)] if concerns else []

    return PathVerdict(
        path=path,
        verdict=verdict,
        concerns=[str(c) for c in concerns],
        confidence=float(parsed.get("confidence", 0.5)),
        reasoning=str(parsed.get("reasoning", "")),
    )


def _adjudicate(path_verdicts: list[PathVerdict]) -> IntentVerdict:
    all_concerns: list[str] = []
    notes_parts: list[str] = []

    for pv in path_verdicts:
        if pv.concerns:
            all_concerns.extend(pv.concerns)
            notes_parts.append(
                f"{pv.path.value} path raised {len(pv.concerns)} concern(s)"
            )

    if all_concerns:
        overall = IntentVerdictType.INCORRECT
        notes_parts.append("Any valid concern requires refinement")
    else:
        overall = IntentVerdictType.CORRECT
        notes_parts.append("All paths agree: formalization is correct")

    return IntentVerdict(
        overall_verdict=overall,
        path_verdicts=path_verdicts,
        adjudication_notes="; ".join(notes_parts),
        all_concerns=all_concerns,
    )
