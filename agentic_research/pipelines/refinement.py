"""Refinement pipeline coordinator.

Coordinates the conjecture refinement loop:
  1. Take failed conjecture + failure reason
  2. Generate refined variants via ConjectureRefiner
  3. Each variant: Formalize -> Intent Judge -> Counterexample Search -> Proof Search
  4. Track refinement history
  5. Limit refinement depth to max_depth (default 3)
"""

from __future__ import annotations

from agentic_research.agents.conjecture_refiner import ConjectureRefiner
from agentic_research.agents.counterexample_searcher import CounterexampleSearcher
from agentic_research.agents.informalizer import Informalizer
from agentic_research.agents.intent_judge import IntentJudge
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.refinement_reporter import RefinementReporter
from agentic_research.logging import get_logger
from agentic_research.models.agents import TokenUsage
from agentic_research.models.refinement import RefinementReport
from agentic_research.models.refinement import (
    RefinementAttempt,
    RefinementHistory,
    RefinementOutcome,
    RefinementResult,
    RefinementStatus,
    RefinementType,
)
from agentic_research.models.research import Conjecture
from agentic_research.models.verification import (
    CounterexampleStatus,
    IntentVerdictType,
)
from agentic_research.pipelines.formalization import FormalizationPipeline
from agentic_research.pipelines.proof import ProofPipeline
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)

_DEFAULT_MAX_DEPTH = 3
_STRATEGY_ORDER = [
    RefinementType.WEAKENING,
    RefinementType.SPECIALIZATION,
    RefinementType.REFORMULATION,
    RefinementType.STRENGTHENING,
]


class RefinementPipeline:
    """End-to-end refinement loop: failed conjecture -> proved variant or exhausted."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        lean_search: LeanSearch,
        *,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        generate_report: bool = True,
    ) -> None:
        self._llm = llm_client
        self._repl = lean_repl
        self._search = lean_search
        self._max_depth = max_depth
        self._generate_report = generate_report
        self._total_tokens = TokenUsage()

    def run(
        self,
        conjecture: Conjecture,
        failure_reason: str,
        failure_outcome: str,
        original_idea: str,
    ) -> RefinementResult:
        log.info(
            "refinement_pipeline_start",
            conjecture=conjecture.statement[:80],
            failure_outcome=failure_outcome,
        )

        history = RefinementHistory(
            original_idea=original_idea,
            original_conjecture=conjecture,
        )

        candidates = [(conjecture, failure_reason, failure_outcome, 0)]
        max_depth_reached = 0

        while candidates:
            current, fail_reason, fail_outcome, depth = candidates.pop(0)

            if depth >= self._max_depth:
                log.info("refinement_depth_limit", depth=depth)
                continue

            max_depth_reached = max(max_depth_reached, depth + 1)

            strategy = _STRATEGY_ORDER[depth % len(_STRATEGY_ORDER)]

            refined_conjectures = self._generate_refinements(
                current, fail_reason, fail_outcome, original_idea, strategy
            )

            for refined in refined_conjectures:
                attempt = RefinementAttempt(
                    original=current,
                    refined=refined,
                    refinement_type=strategy,
                    depth=depth + 1,
                )

                outcome, proof_code, new_fail_reason = self._evaluate_variant(
                    refined, original_idea
                )

                attempt.outcome = outcome
                attempt.failure_reason = new_fail_reason
                attempt.proof_code = proof_code
                history.attempts.append(attempt)

                if outcome == RefinementOutcome.PROVED:
                    history.final_result = RefinementStatus.PROVED
                    log.info(
                        "refinement_proved",
                        variant=refined.natural_language[:80],
                        depth=depth + 1,
                    )
                    report = self._make_report(history) if self._generate_report else None
                    return RefinementResult(
                        status=RefinementStatus.PROVED,
                        proved_variant=refined,
                        proof_code=proof_code,
                        history=history,
                        report=report,
                        total_token_usage=self._total_tokens,
                        max_depth_reached=max_depth_reached,
                    )

                if outcome in (
                    RefinementOutcome.DISPROVED,
                    RefinementOutcome.PROOF_FAILED,
                ):
                    candidates.append(
                        (refined, new_fail_reason, outcome.value, depth + 1)
                    )

        history.final_result = RefinementStatus.EXHAUSTED
        log.info("refinement_exhausted", total_attempts=history.total_attempts)

        report = self._make_report(history) if self._generate_report else None
        return RefinementResult(
            status=RefinementStatus.EXHAUSTED,
            history=history,
            report=report,
            total_token_usage=self._total_tokens,
            max_depth_reached=max_depth_reached,
        )

    def _generate_refinements(
        self,
        conjecture: Conjecture,
        failure_reason: str,
        failure_outcome: str,
        original_idea: str,
        strategy: RefinementType,
    ) -> list[Conjecture]:
        refiner = ConjectureRefiner(llm_client=self._llm)
        refined = refiner.refine(
            conjecture=conjecture,
            failure_reason=failure_reason,
            failure_outcome=failure_outcome,
            original_idea=original_idea,
            strategy=strategy,
        )
        self._accumulate_tokens(refiner.cumulative_tokens)
        return refined

    def _evaluate_variant(
        self,
        conjecture: Conjecture,
        original_idea: str,
    ) -> tuple[RefinementOutcome, str | None, str]:
        """Run a variant through Formalize -> Intent Judge -> Counterexample -> Proof."""

        # Stage 1: Formalize
        formalization = FormalizationPipeline(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_search=self._search,
        )
        form_result = formalization.run(conjecture.natural_language)

        if not form_result.success or not form_result.theorem:
            return (
                RefinementOutcome.FORMALIZATION_FAILED,
                None,
                form_result.failure_reason or "Formalization failed",
            )

        lean_statement = form_result.theorem.lean_statement

        # Stage 2: Intent Judge
        informalizer = Informalizer(llm_client=self._llm)
        judge = IntentJudge(llm_client=self._llm, informalizer=informalizer)
        verdict = judge.judge(
            lean_code=lean_statement,
            original_idea=original_idea,
            conjecture=conjecture.natural_language,
        )
        self._accumulate_tokens(judge.cumulative_tokens)

        if verdict.overall_verdict == IntentVerdictType.INCORRECT:
            return (
                RefinementOutcome.INTENT_MISMATCH,
                None,
                f"Intent mismatch: {'; '.join(verdict.all_concerns[:3])}",
            )

        # Stage 3: Counterexample Search
        searcher = CounterexampleSearcher(
            llm_client=self._llm, lean_repl=self._repl
        )
        cx_result = searcher.search(
            lean_code=lean_statement,
            conjecture=conjecture.natural_language,
        )
        self._accumulate_tokens(searcher.cumulative_tokens)

        if cx_result.status == CounterexampleStatus.DISPROVED:
            desc = ""
            if cx_result.successful_counterexample:
                desc = cx_result.successful_counterexample.description
            return (
                RefinementOutcome.DISPROVED,
                None,
                f"Disproved by counterexample: {desc}",
            )

        # Stage 4: Proof Search
        proof_pipeline = ProofPipeline(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_search=self._search,
        )
        proof_result = proof_pipeline.run(lean_statement, conjecture.natural_language)

        if proof_result.proved and proof_result.final_proof:
            return (
                RefinementOutcome.PROVED,
                proof_result.final_proof,
                "",
            )

        return (
            RefinementOutcome.PROOF_FAILED,
            None,
            proof_result.failure_reason or "Proof search exhausted",
        )

    def _make_report(self, history: RefinementHistory) -> RefinementReport | None:

        reporter = RefinementReporter(llm_client=self._llm)
        report = reporter.generate_report(history)
        self._accumulate_tokens(reporter.cumulative_tokens)
        return report

    def _accumulate_tokens(self, usage: TokenUsage) -> None:
        self._total_tokens.input_tokens += usage.input_tokens
        self._total_tokens.output_tokens += usage.output_tokens
        self._total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self._total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens
