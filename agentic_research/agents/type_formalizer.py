"""Type Formalizer agent — translates informal types to Lean 4 and
validates via auxiliary lemma proving.

Sub-components:
  1. Type Leanifier: informal type → Lean 4 definition (iterative)
  2. Lemma Formalizer: NL lemma → Lean 4 theorem statement
  3. Lemma Prover: proves auxiliary lemmas via IterativeProver
"""

from __future__ import annotations

import re

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    LEMMA_FORMALIZER_SYSTEM,
    LEMMA_FORMALIZER_USER_TEMPLATE,
    TYPE_LEANIFIER_FEEDBACK_TEMPLATE,
    TYPE_LEANIFIER_SYSTEM,
    TYPE_LEANIFIER_USER_TEMPLATE,
)
from agentic_research.agents.prover import IterativeProver
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    ProverConfig,
    ProverResult,
    TokenUsage,
)
from agentic_research.models.formalization import (
    AuxiliaryLemma,
    LemmaStatement,
    TypeCandidate,
    TypeFormalizationCandidate,
)
from agentic_research.models.tools import CompilationStatus
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)


def _extract_lean_code(text: str) -> str:
    match = re.search(r"```lean\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class TypeFormalizer(BaseAgent):
    """Formalizes a single type candidate into compilable Lean 4 code
    and validates it via auxiliary lemma proving.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        *,
        candidate_id: int = 0,
        max_leanify_iterations: int = 5,
        prover_config: ProverConfig | None = None,
        intent_judge: object | None = None,
    ) -> None:
        super().__init__(name="type_formalizer", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._candidate_id = candidate_id
        self._max_leanify_iterations = max_leanify_iterations
        self._prover_config = prover_config or ProverConfig(max_iterations=3)
        self._intent_judge = intent_judge

    def _execute(self, context: AgentContext) -> AgentResult:
        type_candidate = TypeCandidate.model_validate(
            context.metadata.get("type_candidate", {})
        )
        lemmas = [
            LemmaStatement.model_validate(lem)
            for lem in context.metadata.get("lemmas", [])
        ]
        prior_definitions = context.metadata.get("prior_definitions", "")

        log.info(
            "type_formalizer_start",
            type_name=type_candidate.name,
            candidate_id=self._candidate_id,
            num_lemmas=len(lemmas),
        )

        total_tokens = TokenUsage()

        lean_code, leanify_tokens, compiles = self._leanify_type(
            type_candidate, prior_definitions
        )
        total_tokens.input_tokens += leanify_tokens.input_tokens
        total_tokens.output_tokens += leanify_tokens.output_tokens

        aux_lemmas: list[AuxiliaryLemma] = []
        if compiles and lemmas:
            full_context = f"{prior_definitions}\n\n{lean_code}" if prior_definitions else lean_code
            for lemma_stmt in lemmas:
                aux, lemma_tokens = self._formalize_and_prove_lemma(
                    lemma_stmt, full_context
                )
                aux_lemmas.append(aux)
                total_tokens.input_tokens += lemma_tokens.input_tokens
                total_tokens.output_tokens += lemma_tokens.output_tokens

        result = TypeFormalizationCandidate(
            candidate_id=self._candidate_id,
            type_name=type_candidate.name,
            lean_code=lean_code,
            compiles=compiles,
            auxiliary_lemmas=aux_lemmas,
        )

        log.info(
            "type_formalizer_done",
            type_name=type_candidate.name,
            candidate_id=self._candidate_id,
            compiles=compiles,
            proved_lemmas=result.proved_count,
            total_lemmas=result.total_lemma_count,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=result.model_dump(),
            token_usage=total_tokens,
        )

    def _leanify_type(
        self,
        candidate: TypeCandidate,
        prior_definitions: str,
    ) -> tuple[str, TokenUsage, bool]:
        """Translate informal type to Lean 4, iterating with compiler feedback."""
        total_tokens = TokenUsage()
        previous_code: str | None = None
        previous_errors: str = ""

        for iteration in range(1, self._max_leanify_iterations + 1):
            if previous_code is None:
                user_content = TYPE_LEANIFIER_USER_TEMPLATE.format(
                    type_name=candidate.name,
                    type_description=candidate.informal_description,
                    lean_signature=candidate.lean_signature or "not specified",
                    dependencies=prior_definitions or "none",
                )
            else:
                user_content = TYPE_LEANIFIER_FEEDBACK_TEMPLATE.format(
                    type_name=candidate.name,
                    type_description=candidate.informal_description,
                    previous_attempt=previous_code,
                    errors=previous_errors,
                )

            response = self._llm.complete(
                system=TYPE_LEANIFIER_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.1,
                use_cache=True,
            )
            total_tokens.input_tokens += response.token_usage.input_tokens
            total_tokens.output_tokens += response.token_usage.output_tokens

            lean_code = _extract_lean_code(response.content)
            full_code = f"{prior_definitions}\n\n{lean_code}" if prior_definitions else lean_code
            compilation = self._repl.execute(full_code)

            if compilation.compilation_status == CompilationStatus.OK:
                if self._intent_judge is not None:
                    try:
                        from agentic_research.agents.intent_judge import IntentJudge as IJType
                        from agentic_research.models.verification import IntentVerdictType
                        judge: IJType = self._intent_judge  # type: ignore[assignment]
                        verdict = judge.judge(
                            lean_code=lean_code,
                            original_idea=candidate.informal_description,
                            conjecture=candidate.informal_description,
                        )
                        if (
                            verdict.overall_verdict == IntentVerdictType.INCORRECT
                            and verdict.overall_confidence >= 0.7
                        ):
                            previous_code = lean_code
                            previous_errors = (
                                f"Faithfulness check failed: {verdict.all_concerns}"
                            )
                            log.info(
                                "type_leanify_intent_reject",
                                type_name=candidate.name,
                                iteration=iteration,
                                confidence=verdict.overall_confidence,
                            )
                            continue
                    except Exception as exc:
                        log.warning("type_leanify_intent_judge_error", error=str(exc))
                log.info("type_leanify_success", type_name=candidate.name, iteration=iteration)
                return lean_code, total_tokens, True

            previous_code = lean_code
            previous_errors = "\n".join(compilation.errors) if compilation.errors else "Unknown error"
            log.info(
                "type_leanify_retry",
                type_name=candidate.name,
                iteration=iteration,
                error_count=len(compilation.errors),
            )

        return previous_code or "", total_tokens, False

    def _formalize_and_prove_lemma(
        self,
        lemma: LemmaStatement,
        type_definitions: str,
    ) -> tuple[AuxiliaryLemma, TokenUsage]:
        """Formalize a single lemma statement and attempt to prove it."""
        total_tokens = TokenUsage()

        user_content = LEMMA_FORMALIZER_USER_TEMPLATE.format(
            lemma_name=lemma.name,
            statement_nl=lemma.statement_nl,
            for_type=lemma.for_type,
            type_definitions=type_definitions,
        )

        response = self._llm.complete(
            system=LEMMA_FORMALIZER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.1,
            use_cache=True,
        )
        total_tokens.input_tokens += response.token_usage.input_tokens
        total_tokens.output_tokens += response.token_usage.output_tokens

        lean_stmt = _extract_lean_code(response.content)
        full_code = f"{type_definitions}\n\n{lean_stmt}"

        stmt_compilation = self._repl.execute(full_code)
        if stmt_compilation.compilation_status == CompilationStatus.ERROR:
            return AuxiliaryLemma(
                lemma=lemma,
                lean_code=lean_stmt,
                proved=False,
                error_message="Lemma statement failed to compile: "
                + "; ".join(stmt_compilation.errors),
            ), total_tokens

        prover = IterativeProver(
            llm_client=self._llm,
            lean_repl=self._repl,
            config=self._prover_config,
        )

        prove_ctx = AgentContext(task=full_code)
        prove_result = prover.run(prove_ctx)

        prover_data = ProverResult.model_validate(prove_result.result or {})
        total_tokens.input_tokens += prove_result.token_usage.input_tokens
        total_tokens.output_tokens += prove_result.token_usage.output_tokens

        return AuxiliaryLemma(
            lemma=lemma,
            lean_code=lean_stmt,
            proved=prover_data.proved,
            proof_code=prover_data.final_proof or "",
            error_message=prover_data.failure_reason,
        ), total_tokens
