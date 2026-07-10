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
from agentic_research.models.agents import ProverConfig
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_K = 3


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
    ) -> None:
        self._llm = llm_client
        self._repl = lean_repl
        self._search = lean_search
        self._k = k
        self._max_retries = max_retries
        self._prover_config = prover_config
        self._artifact_dir = artifact_dir
        self._progress_callback = progress_callback
        self._total_tokens = TokenUsage()

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
            # Check if this type has 0 Loogle results — if so, suggest a data package
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

    def _auction_type(
        self,
        candidate: TypeCandidate,
        lemmas: list[LemmaStatement],
        prior_definitions: str,
    ) -> AuctionResult:
        for attempt in range(1, self._max_retries + 1):
            auctioneer = Auctioneer(
                llm_client=self._llm,
                lean_repl=self._repl,
                k=self._k,
                prover_config=self._prover_config,
            )

            ctx = AgentContext(
                task=candidate.name,
                metadata={
                    "type_candidate": candidate.model_dump(),
                    "lemmas": [lem.model_dump() for lem in lemmas],
                    "prior_definitions": prior_definitions,
                },
            )

            result = auctioneer.run(ctx)
            self._accumulate_tokens(auctioneer.cumulative_tokens)
            if not result.result:
                continue

            auction_result = AuctionResult.model_validate(result.result)
            if auction_result.verdict == AuctionVerdict.ACCEPTED:
                return auction_result

            log.info(
                "formalization_auction_retry",
                type_name=candidate.name,
                attempt=attempt,
                reason=auction_result.reason,
            )

        return AuctionResult(
            type_name=candidate.name,
            verdict=AuctionVerdict.RETRY,
            reason=f"All {self._max_retries} retry attempts exhausted",
        )

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
