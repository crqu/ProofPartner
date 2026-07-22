"""Formalization pipeline coordinator.

Wires: Conjecture → Type Planner → Lemma Planner → [k × Type Formalizer]
       → Auctioneer → Theorem Formalizer → Claim Check

Handles retry logic when the Auctioneer returns RETRY.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from agentic_research.agents.auctioneer import Auctioneer
from agentic_research.agents.claim_check import ClaimCheck
from agentic_research.agents.informalizer import Informalizer
from agentic_research.agents.intent_judge import IntentJudge
from agentic_research.agents.lemma_planner import LemmaPlanner
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.theorem_formalizer import TheoremFormalizer
from agentic_research.agents.type_planner import TypePlanner
from agentic_research.logging import get_logger
from agentic_research.models.agents import AgentContext, AgentStatus, TokenUsage
from agentic_research.models.formalization import (
    AuctionResult,
    AuctionVerdict,
    ClaimCheckResult,
    ClaimCheckVerdict,
    DataPackageCandidate,
    FormalizationPipelineResult,
    LemmaStatement,
    TheoremFormalization,
    TypeCandidate,
    TypeFormalizationCandidate,
    TypeFormalizationResult,
    TypePlan,
)
from agentic_research.models.interaction import InteractionOption, InteractionRequest, InteractionResponse
from agentic_research.models.agents import ProverConfig
from agentic_research.cache.formalization_cache import CachedFormalization, FormalizationCache
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_K = 3
DEFAULT_MAX_REFINEMENT_ITERATIONS = 0
REFINEMENT_PROOF_RATE_THRESHOLD = 0.8
REFINEMENT_STALL_THRESHOLD = 0.05


class FormalizationPipeline:
    """End-to-end formalization pipeline from NL conjecture to Lean 4 statement."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        lean_search: LeanSearch,
        *,
        k: int = DEFAULT_K,
        max_retries: int = DEFAULT_MAX_RETRIES,
        prover_config: ProverConfig | None = None,
        artifact_dir: Path | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
        interaction_callback: Callable[[InteractionRequest], InteractionResponse] | None = None,
        use_intent_judge: bool = False,
        formalization_cache: FormalizationCache | None = None,
        lean_toolchain: str = "",
        max_refinement_iterations: int = DEFAULT_MAX_REFINEMENT_ITERATIONS,
    ) -> None:
        self._llm = llm_client
        self._repl = lean_repl
        self._search = lean_search
        self._k = k
        self._max_retries = max_retries
        self._prover_config = prover_config
        self._artifact_dir = artifact_dir
        self._progress_callback = progress_callback
        self._interaction_callback = interaction_callback
        self._use_intent_judge = use_intent_judge
        self._cache = formalization_cache
        self._lean_toolchain = lean_toolchain
        self._max_refinement_iterations = max_refinement_iterations
        self._total_tokens = TokenUsage()
        self._conjecture_nl: str = ""

    def _notify_progress(self, stage: str, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(stage, message)

    def _accumulate_tokens(self, usage: TokenUsage) -> None:
        self._total_tokens.input_tokens += usage.input_tokens
        self._total_tokens.output_tokens += usage.output_tokens
        self._total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self._total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens

    def run(self, conjecture_nl: str) -> FormalizationPipelineResult:
        """Execute the full formalization pipeline."""
        self._conjecture_nl = conjecture_nl
        log.info("formalization_pipeline_start", conjecture_len=len(conjecture_nl))

        # Stage 1: Type Planning
        self._notify_progress("Type Planning", "Planning types for formalization")
        type_plan = self._run_type_planner(conjecture_nl)
        if type_plan is None:
            return FormalizationPipelineResult(
                conjecture_nl=conjecture_nl,
                failure_stage="type_planning",
                failure_reason="Type planner failed",
                total_token_usage=self._total_tokens,
            )

        # Stage 2: Lemma Planning
        self._notify_progress("Lemma Planning", "Planning auxiliary lemmas")
        lemmas_by_type = self._run_lemma_planner(type_plan)

        # Stage 3 + 4: Type Formalization + Auction (with retries)
        self._notify_progress("Type Formalization", "Formalizing types with auction")
        type_result = self._run_type_formalization(type_plan, lemmas_by_type)
        if not type_result.all_types_accepted:
            return FormalizationPipelineResult(
                conjecture_nl=conjecture_nl,
                type_formalization=type_result,
                failure_stage="type_formalization",
                failure_reason="Not all types could be formalized",
                total_token_usage=self._total_tokens,
            )

        # Stage 5: Theorem Formalization
        self._notify_progress("Theorem Formalization", "Formalizing theorem statement")
        type_defs = "\n\n".join(
            t.lean_code for t in type_result.accepted_types
        )
        theorem = self._run_theorem_formalizer(conjecture_nl, type_defs)
        if theorem is None or not theorem.compiles:
            return FormalizationPipelineResult(
                conjecture_nl=conjecture_nl,
                type_formalization=type_result,
                theorem=theorem,
                failure_stage="theorem_formalization",
                failure_reason=theorem.failure_reason if theorem else "Theorem formalizer failed",
                total_token_usage=self._total_tokens,
            )

        if self._artifact_dir is not None:
            self._artifact_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            theorem_path = self._artifact_dir / f"{timestamp}-theorem.lean"
            full_theorem = f"{type_defs}\n\n{theorem.lean_statement}"
            theorem_path.write_text(full_theorem)
            metadata_path = self._artifact_dir / f"{timestamp}-metadata.json"
            metadata = {
                "conjecture_nl": conjecture_nl,
                "compiles": theorem.compiles,
                "iterations_used": theorem.iterations_used,
                "type_imports": theorem.type_imports,
                "timestamp": timestamp,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))
            log.info("theorem_artifact_saved", path=str(theorem_path))

        # Stage 6: Claim Check
        self._notify_progress("Claim Check", "Verifying formalization preserves intent")
        claim_result = self._run_claim_check(conjecture_nl, theorem.lean_statement, type_defs)
        if claim_result and claim_result.verdict == ClaimCheckVerdict.FAIL:
            return FormalizationPipelineResult(
                conjecture_nl=conjecture_nl,
                type_formalization=type_result,
                theorem=theorem,
                claim_check=claim_result,
                failure_stage="claim_check",
                failure_reason=claim_result.reason,
                total_token_usage=self._total_tokens,
            )

        log.info("formalization_pipeline_success")
        return FormalizationPipelineResult(
            conjecture_nl=conjecture_nl,
            type_formalization=type_result,
            theorem=theorem,
            claim_check=claim_result,
            success=True,
            total_token_usage=self._total_tokens,
        )

    def _run_type_planner(self, conjecture_nl: str) -> TypePlan | None:
        planner = TypePlanner(llm_client=self._llm, lean_search=self._search)
        ctx = AgentContext(task=conjecture_nl)
        result = planner.run(ctx)
        self._accumulate_tokens(planner.cumulative_tokens)
        if result.status != AgentStatus.SUCCESS or not result.result:
            return None
        return TypePlan.model_validate(result.result)

    def _run_lemma_planner(self, type_plan: TypePlan) -> dict[str, list[LemmaStatement]]:
        planner = LemmaPlanner(llm_client=self._llm)
        ctx = AgentContext(
            task="plan lemmas",
            metadata={"type_plan": type_plan.model_dump()},
        )
        result = planner.run(ctx)
        self._accumulate_tokens(planner.cumulative_tokens)
        if result.status != AgentStatus.SUCCESS or not result.result:
            return {}

        all_lemmas = [
            LemmaStatement.model_validate(lem)
            for lem in result.result.get("lemmas", [])
        ]

        by_type: dict[str, list[LemmaStatement]] = {}
        for lemma in all_lemmas:
            by_type.setdefault(lemma.for_type, []).append(lemma)
        return by_type

    def _run_type_formalization(
        self,
        type_plan: TypePlan,
        lemmas_by_type: dict[str, list[LemmaStatement]],
    ) -> TypeFormalizationResult:
        accepted: list[TypeFormalizationCandidate] = []
        auction_results: list[AuctionResult] = []
        prior_definitions = ""
        total_proved = 0
        total_failed = 0

        new_types: list[TypeCandidate] = []
        for c in type_plan.candidates:
            if c.is_in_mathlib:
                continue
            if c.composition_alternative:
                log.info(
                    "type_composition_alternative_used",
                    type_name=c.name,
                    composition=c.composition_alternative,
                )
                prior_definitions = (
                    f"{prior_definitions}\n\n-- {c.name}: {c.composition_alternative}"
                    if prior_definitions
                    else f"-- {c.name}: {c.composition_alternative}"
                )
                continue
            new_types.append(c)
        topo_order = type_plan.dependency_graph.topological_order
        if topo_order:
            ordered = sorted(
                new_types,
                key=lambda c: topo_order.index(c.name) if c.name in topo_order else len(topo_order),
            )
        else:
            ordered = new_types

        data_packages: list[DataPackageCandidate] = []

        for candidate in ordered:
            if self._cache:
                cache_hit = self._cache.get(candidate.name, self._lean_toolchain)
                if cache_hit:
                    prior_definitions = (
                        f"{prior_definitions}\n\n{cache_hit.lean_code}"
                        if prior_definitions
                        else cache_hit.lean_code
                    )
                    accepted.append(TypeFormalizationCandidate(
                        candidate_id=0,
                        type_name=candidate.name,
                        lean_code=cache_hit.lean_code,
                        compiles=True,
                        auxiliary_lemmas=[],
                    ))
                    continue

            search_result = self._search.execute(candidate.name)
            search_entries = getattr(search_result, "entries", [])

            if not search_entries:
                planner = TypePlanner(llm_client=self._llm, lean_search=self._search)
                data_pkg = planner.suggest_data_package(
                    type_name=candidate.name,
                    type_description=candidate.informal_description,
                )
                if data_pkg and data_pkg.lean_structure:
                    log.info(
                        "data_package_used",
                        type_name=candidate.name,
                        package_name=data_pkg.package_name,
                    )
                    data_packages.append(data_pkg)
                    prior_definitions = (
                        f"{prior_definitions}\n\n{data_pkg.lean_structure}"
                        if prior_definitions
                        else data_pkg.lean_structure
                    )
                    continue

            lemmas = lemmas_by_type.get(candidate.name, [])

            auction_result = self._auction_type(
                candidate, lemmas, prior_definitions
            )

            if (
                self._interaction_callback is not None
                and auction_result.verdict == AuctionVerdict.ACCEPTED
                and auction_result.all_candidates
            ):
                auction_result = self._apply_interactive_selection(auction_result)

            auction_results.append(auction_result)

            if auction_result.verdict == AuctionVerdict.ACCEPTED and auction_result.winning_candidate:
                winner = auction_result.winning_candidate
                accepted.append(winner)
                prior_definitions = (
                    f"{prior_definitions}\n\n{winner.lean_code}"
                    if prior_definitions
                    else winner.lean_code
                )
                total_proved += winner.proved_count
                total_failed += winner.total_lemma_count - winner.proved_count
                if self._cache and winner:
                    self._cache.put(CachedFormalization(
                        type_name=candidate.name,
                        type_signature=winner.lean_code.split("\n")[0] if winner.lean_code else "",
                        lean_code=winner.lean_code,
                        lean_toolchain=self._lean_toolchain,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        proved_lemmas=[],
                    ))
            else:
                total_failed += len(lemmas)

        return TypeFormalizationResult(
            type_plan=type_plan,
            auction_results=auction_results,
            accepted_types=accepted,
            all_types_accepted=(len(accepted) + len(data_packages)) == len(new_types),
            total_proved_lemmas=total_proved,
            total_failed_lemmas=total_failed,
        )

    def _apply_interactive_selection(self, auction_result: AuctionResult) -> AuctionResult:
        """Let the user override the auctioneer's pick via interaction_callback."""
        assert self._interaction_callback is not None

        scores_by_id = {s.candidate_id: s for s in auction_result.scores}
        options: list[InteractionOption] = []
        for c in auction_result.all_candidates:
            score = scores_by_id.get(c.candidate_id)
            sig = c.lean_code.split("\n")[0] if c.lean_code else "(empty)"
            label = (
                f"#{c.candidate_id + 1}: {sig}  "
                f"[score={score.total_score:.3f}, lemmas={c.total_lemma_count}]"
                if score
                else f"#{c.candidate_id + 1}: {sig}"
            )
            options.append(InteractionOption(
                label=label,
                value=c.candidate_id,
                score=score.total_score if score else 0.0,
            ))

        options.sort(key=lambda o: o.score, reverse=True)

        default_id = auction_result.winner_id
        request = InteractionRequest(
            type="select",
            prompt=f"Select type formalization for '{auction_result.type_name}'",
            options=options,
            default_value=default_id,
        )

        response = self._interaction_callback(request)

        if response.aborted or response.selected_value is None:
            log.info("interactive_selection_aborted", type_name=auction_result.type_name)
            return auction_result

        selected_id = response.selected_value
        if selected_id == auction_result.winner_id:
            return auction_result

        selected = next(
            (c for c in auction_result.all_candidates if c.candidate_id == selected_id),
            None,
        )
        if selected is None:
            log.warning("interactive_selection_invalid", selected_id=selected_id)
            return auction_result

        log.info(
            "interactive_selection_override",
            type_name=auction_result.type_name,
            original_winner=auction_result.winner_id,
            user_pick=selected_id,
        )
        return AuctionResult(
            type_name=auction_result.type_name,
            verdict=AuctionVerdict.ACCEPTED,
            winner_id=selected_id,
            scores=auction_result.scores,
            winning_candidate=selected,
            all_candidates=auction_result.all_candidates,
            reason=f"User selected candidate {selected_id} (interactive override)",
        )

    def _make_intent_judge(self) -> IntentJudge | None:
        try:
            informalizer = Informalizer(llm_client=self._llm)
            return IntentJudge(llm_client=self._llm, informalizer=informalizer)
        except Exception:
            return None

    def _auction_type(
        self,
        candidate: TypeCandidate,
        lemmas: list[LemmaStatement],
        prior_definitions: str,
    ) -> AuctionResult:
        intent_judge = self._make_intent_judge() if self._use_intent_judge else None
        iteration_context: str | None = None
        proof_rate_history: list[float] = []

        for attempt in range(1, self._max_retries + 1):
            auctioneer = Auctioneer(
                llm_client=self._llm,
                lean_repl=self._repl,
                k=self._k,
                prover_config=self._prover_config,
                intent_judge=intent_judge,
                original_idea=self._conjecture_nl,
                conjecture=self._conjecture_nl,
            )

            ctx = AgentContext(
                task=candidate.name,
                metadata={
                    "type_candidate": candidate.model_dump(),
                    "lemmas": [lem.model_dump() for lem in lemmas],
                    "prior_definitions": prior_definitions,
                    **({"iteration_context": iteration_context} if iteration_context else {}),
                },
            )

            result = auctioneer.run(ctx)
            self._accumulate_tokens(auctioneer.cumulative_tokens)
            if not result.result:
                continue

            auction_result = AuctionResult.model_validate(result.result)

            if auction_result.verdict == AuctionVerdict.ACCEPTED:
                current_rate = (
                    auction_result.winning_candidate.proved_ratio
                    if auction_result.winning_candidate
                    else 0.0
                )
                proof_rate_history.append(current_rate)

                if current_rate >= REFINEMENT_PROOF_RATE_THRESHOLD:
                    auction_result.refinement_iterations = len(proof_rate_history) - 1
                    auction_result.proof_rate_history = proof_rate_history
                    return auction_result

                if (
                    len(proof_rate_history) >= 2
                    and (proof_rate_history[-1] - proof_rate_history[-2]) < REFINEMENT_STALL_THRESHOLD
                ):
                    log.info(
                        "refinement_stall_detected",
                        type_name=candidate.name,
                        proof_rate_history=proof_rate_history,
                    )
                    auction_result.refinement_iterations = len(proof_rate_history) - 1
                    auction_result.proof_rate_history = proof_rate_history
                    return auction_result

                if len(proof_rate_history) > self._max_refinement_iterations:
                    auction_result.refinement_iterations = len(proof_rate_history) - 1
                    auction_result.proof_rate_history = proof_rate_history
                    return auction_result

                feedback = Auctioneer.build_failure_feedback(auction_result)
                if not feedback:
                    auction_result.refinement_iterations = len(proof_rate_history) - 1
                    auction_result.proof_rate_history = proof_rate_history
                    return auction_result

                iteration_context = feedback
                log.info(
                    "refinement_iteration",
                    type_name=candidate.name,
                    iteration=len(proof_rate_history),
                    proof_rate=current_rate,
                )
                continue

            log.info(
                "formalization_auction_retry",
                type_name=candidate.name,
                attempt=attempt,
                reason=auction_result.reason,
            )

        final_result = AuctionResult(
            type_name=candidate.name,
            verdict=AuctionVerdict.RETRY,
            reason=f"All {self._max_retries} retry attempts exhausted",
            refinement_iterations=len(proof_rate_history),
            proof_rate_history=proof_rate_history,
        )
        return final_result

    def _run_theorem_formalizer(
        self, conjecture_nl: str, type_definitions: str
    ) -> TheoremFormalization | None:
        formalizer = TheoremFormalizer(
            llm_client=self._llm, lean_repl=self._repl
        )
        ctx = AgentContext(
            task=conjecture_nl,
            metadata={"type_definitions": type_definitions},
        )
        result = formalizer.run(ctx)
        self._accumulate_tokens(formalizer.cumulative_tokens)
        if not result.result:
            return None
        return TheoremFormalization.model_validate(result.result)

    def _run_claim_check(
        self, conjecture_nl: str, lean_code: str, type_definitions: str
    ) -> ClaimCheckResult | None:
        checker = ClaimCheck(llm_client=self._llm)
        ctx = AgentContext(
            task=conjecture_nl,
            metadata={
                "lean_code": lean_code,
                "type_definitions": type_definitions,
            },
        )
        result = checker.run(ctx)
        self._accumulate_tokens(checker.cumulative_tokens)
        if not result.result:
            return None
        return ClaimCheckResult.model_validate(result.result)
