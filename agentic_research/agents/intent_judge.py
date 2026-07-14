"""Intent Judge: verifies a Lean formalization captures the user's idea.

Uses three varied prompting strategies for diversity:
  1. Blind path — back-translate via Informalizer, then compare
  2. Direct path — directly compare Lean code to idea + conjecture
  3. Adversarial path — devil's advocate seeking semantic mismatches

Adjudication: concerns go through a second-pass LLM review that classifies
each as false_positive or genuine_error.  Only genuine errors trigger
INCORRECT.  If the second-pass fails to parse, all concerns are treated
as genuine (safe fallback).

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
    INTENT_FP_REVIEW_SYSTEM,
    INTENT_FP_REVIEW_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.verification import (
    ConcernClassification,
    IntentVerdict,
    IntentVerdictType,
    PathVerdict,
    VerificationPath,
)
from agentic_research.tools.hint_cleaner import HintCleaner

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
        self._hint_cleaner = HintCleaner()

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

        cleaned_code = self._hint_cleaner.execute(lean_code).cleaned_code

        blind = self._run_blind_path(lean_code, original_idea)
        path_verdicts.append(blind)

        direct = self._run_direct_path(cleaned_code, original_idea, conjecture)
        path_verdicts.append(direct)

        adversarial = self._run_adversarial_path(cleaned_code, original_idea, conjecture)
        path_verdicts.append(adversarial)

        if _openai_enabled():
            openai_verdict = self._run_openai_path(lean_code, original_idea, conjecture)
            path_verdicts.append(openai_verdict)

        return self._adjudicate(
            path_verdicts,
            lean_code=lean_code,
            original_idea=original_idea,
            conjecture=conjecture,
        )

    def _adjudicate(
        self,
        path_verdicts: list[PathVerdict],
        *,
        lean_code: str,
        original_idea: str,
        conjecture: str,
    ) -> IntentVerdict:
        return _adjudicate(
            path_verdicts,
            llm_client=self._llm,
            lean_code=lean_code,
            original_idea=original_idea,
            conjecture=conjecture,
        )

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
        type_fidelity=float(parsed.get("type_fidelity", 0.5)),
        quantifier_accuracy=float(parsed.get("quantifier_accuracy", 0.5)),
        constraint_preservation=float(parsed.get("constraint_preservation", 0.5)),
    )


def _adjudicate(
    path_verdicts: list[PathVerdict],
    *,
    llm_client: LLMClient | None = None,
    lean_code: str = "",
    original_idea: str = "",
    conjecture: str = "",
) -> IntentVerdict:
    all_concerns: list[str] = []
    notes_parts: list[str] = []

    for pv in path_verdicts:
        if pv.concerns:
            all_concerns.extend(pv.concerns)
            notes_parts.append(
                f"{pv.path.value} path raised {len(pv.concerns)} concern(s)"
            )

    dismissed: list[str] = []

    if all_concerns and llm_client is not None:
        classifications = _run_fp_review(
            llm_client,
            lean_code=lean_code,
            original_idea=original_idea,
            conjecture=conjecture,
            concerns=all_concerns,
        )
        if classifications is not None:
            genuine = [
                c.concern for c in classifications
                if c.classification == "genuine_error"
            ]
            dismissed = [
                c.concern for c in classifications
                if c.classification == "false_positive"
            ]
            if genuine:
                overall = IntentVerdictType.INCORRECT
                notes_parts.append(
                    f"{len(genuine)} genuine error(s) after FP review"
                )
            else:
                overall = IntentVerdictType.CORRECT
                notes_parts.append(
                    f"All {len(dismissed)} concern(s) dismissed as false positives"
                )
        else:
            overall = IntentVerdictType.INCORRECT
            notes_parts.append(
                "FP review parse failed; treating all concerns as genuine"
            )
    elif all_concerns:
        overall = IntentVerdictType.INCORRECT
        notes_parts.append("Any valid concern requires refinement")
    else:
        overall = IntentVerdictType.CORRECT
        notes_parts.append("All paths agree: formalization is correct")

    dims = _aggregate_dimensions(path_verdicts)

    log.info(
        "intent_judge_verdict",
        type_fidelity=dims["type_fidelity"],
        quantifier_accuracy=dims["quantifier_accuracy"],
        constraint_preservation=dims["constraint_preservation"],
        overall_confidence=dims["overall_confidence"],
        passes=dims["passes"],
    )

    return IntentVerdict(
        overall_verdict=overall,
        path_verdicts=path_verdicts,
        adjudication_notes="; ".join(notes_parts),
        all_concerns=all_concerns,
        dismissed_concerns=dismissed,
        type_fidelity=dims["type_fidelity"],
        quantifier_accuracy=dims["quantifier_accuracy"],
        constraint_preservation=dims["constraint_preservation"],
        overall_confidence=dims["overall_confidence"],
    )


def _run_fp_review(
    llm_client: LLMClient,
    *,
    lean_code: str,
    original_idea: str,
    conjecture: str,
    concerns: list[str],
) -> list[ConcernClassification] | None:
    """Run the second-pass false-positive review. Returns None on parse failure."""
    concerns_text = "\n".join(f"- {c}" for c in concerns)
    prompt = INTENT_FP_REVIEW_USER_TEMPLATE.format(
        lean_code=lean_code,
        original_idea=original_idea,
        conjecture=conjecture,
        concerns=concerns_text,
    )
    try:
        response = llm_client.complete(
            system=INTENT_FP_REVIEW_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        parsed = llm_client.extract_json(response.content)
        if not isinstance(parsed, list):
            log.warning("fp_review_parse_failed", reason="expected JSON array")
            return None

        return [
            ConcernClassification(
                concern=str(item.get("concern", "")),
                classification=item.get("classification", "genuine_error"),
                reasoning=str(item.get("reasoning", "")),
            )
            for item in parsed
            if isinstance(item, dict)
            and item.get("classification") in ("false_positive", "genuine_error")
        ]
    except Exception as exc:
        log.warning("fp_review_failed", error=str(exc))
        return None


def _aggregate_dimensions(
    path_verdicts: list[PathVerdict],
) -> dict[str, float | bool]:
    total_weight = 0.0
    tf_sum = 0.0
    qa_sum = 0.0
    cp_sum = 0.0

    for pv in path_verdicts:
        w = 1.5 if pv.path == VerificationPath.ADVERSARIAL else 1.0
        total_weight += w
        tf_sum += pv.type_fidelity * w
        qa_sum += pv.quantifier_accuracy * w
        cp_sum += pv.constraint_preservation * w

    if total_weight == 0.0:
        return {
            "type_fidelity": 0.5,
            "quantifier_accuracy": 0.5,
            "constraint_preservation": 0.5,
            "overall_confidence": 0.5,
            "passes": False,
        }

    tf = tf_sum / total_weight
    qa = qa_sum / total_weight
    cp = cp_sum / total_weight
    oc = (tf + qa + cp) / 3.0

    return {
        "type_fidelity": round(tf, 4),
        "quantifier_accuracy": round(qa, 4),
        "constraint_preservation": round(cp, 4),
        "overall_confidence": round(oc, 4),
        "passes": oc >= 0.6 and tf >= 0.4 and qa >= 0.4 and cp >= 0.4,
    }
