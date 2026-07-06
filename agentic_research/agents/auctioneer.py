"""Best-of-k Auctioneer — evaluates parallel Type Formalizer candidates
and selects the best one based on proved lemma ratio, brevity, and
compilation success.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.type_formalizer import TypeFormalizer
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.formalization import (
    AuctionResult,
    AuctionScore,
    AuctionVerdict,
    LemmaStatement,
    TypeCandidate,
    TypeFormalizationCandidate,
)
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)

WEIGHT_LEMMA_RATIO = 0.5
WEIGHT_BREVITY = 0.2
WEIGHT_COMPILATION = 0.3
QUALITY_THRESHOLD = 0.3


def compute_auction_score(candidate: TypeFormalizationCandidate) -> AuctionScore:
    """Score a single candidate formalization."""
    lemma_ratio = candidate.proved_ratio
    compilation = 1.0 if candidate.compiles else 0.0
    code_len = max(len(candidate.lean_code), 1)
    brevity = 1.0 / (1.0 + code_len / 500.0)

    total = (
        WEIGHT_LEMMA_RATIO * lemma_ratio
        + WEIGHT_BREVITY * brevity
        + WEIGHT_COMPILATION * compilation
    )

    return AuctionScore(
        candidate_id=candidate.candidate_id,
        lemma_ratio=round(lemma_ratio, 4),
        brevity_score=round(brevity, 4),
        compilation_score=compilation,
        total_score=round(total, 4),
    )


class Auctioneer(BaseAgent):
    """Evaluates k parallel type formalization candidates."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        *,
        k: int = 3,
        quality_threshold: float = QUALITY_THRESHOLD,
        prover_config: ProverConfig | None = None,
    ) -> None:
        super().__init__(name="auctioneer", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._k = k
        self._quality_threshold = quality_threshold
        self._prover_config = prover_config

    @property
    def k(self) -> int:
        return self._k

    def _execute(self, context: AgentContext) -> AgentResult:
        type_candidate = TypeCandidate.model_validate(
            context.metadata.get("type_candidate", {})
        )
        lemmas = [
            LemmaStatement.model_validate(l)
            for l in context.metadata.get("lemmas", [])
        ]
        prior_definitions = context.metadata.get("prior_definitions", "")

        log.info(
            "auctioneer_start",
            type_name=type_candidate.name,
            k=self._k,
            num_lemmas=len(lemmas),
        )

        candidates = self._run_parallel_formalizers(
            type_candidate, lemmas, prior_definitions
        )

        auction_result = self._evaluate(type_candidate.name, candidates)

        log.info(
            "auctioneer_done",
            type_name=type_candidate.name,
            verdict=auction_result.verdict.value,
            winner_id=auction_result.winner_id,
        )

        total_tokens = TokenUsage()
        for c in candidates:
            total_tokens.input_tokens += sum(
                l.lemma.name.__len__() for l in c.auxiliary_lemmas
            ) * 0  # tokens tracked inside formalizers

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=auction_result.model_dump(),
            token_usage=total_tokens,
        )

    def _run_parallel_formalizers(
        self,
        type_candidate: TypeCandidate,
        lemmas: list[LemmaStatement],
        prior_definitions: str,
    ) -> list[TypeFormalizationCandidate]:
        """Run k type formalizers in parallel using asyncio."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return self._run_sequential(type_candidate, lemmas, prior_definitions)

        return asyncio.run(
            self._run_async(type_candidate, lemmas, prior_definitions)
        )

    async def _run_async(
        self,
        type_candidate: TypeCandidate,
        lemmas: list[LemmaStatement],
        prior_definitions: str,
    ) -> list[TypeFormalizationCandidate]:
        loop = asyncio.get_event_loop()
        tasks = []
        for i in range(self._k):
            tasks.append(
                loop.run_in_executor(
                    None,
                    self._run_single_formalizer,
                    type_candidate,
                    lemmas,
                    prior_definitions,
                    i,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[TypeFormalizationCandidate] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning("auctioneer_candidate_error", candidate_id=i, error=str(result))
                candidates.append(TypeFormalizationCandidate(
                    candidate_id=i,
                    type_name=type_candidate.name,
                    compiles=False,
                ))
            else:
                candidates.append(result)
        return candidates

    def _run_sequential(
        self,
        type_candidate: TypeCandidate,
        lemmas: list[LemmaStatement],
        prior_definitions: str,
    ) -> list[TypeFormalizationCandidate]:
        """Fallback when already inside an event loop."""
        candidates: list[TypeFormalizationCandidate] = []
        for i in range(self._k):
            try:
                candidate = self._run_single_formalizer(
                    type_candidate, lemmas, prior_definitions, i
                )
                candidates.append(candidate)
            except Exception as exc:
                log.warning("auctioneer_candidate_error", candidate_id=i, error=str(exc))
                candidates.append(TypeFormalizationCandidate(
                    candidate_id=i,
                    type_name=type_candidate.name,
                    compiles=False,
                ))
        return candidates

    def _run_single_formalizer(
        self,
        type_candidate: TypeCandidate,
        lemmas: list[LemmaStatement],
        prior_definitions: str,
        candidate_id: int,
    ) -> TypeFormalizationCandidate:
        formalizer = TypeFormalizer(
            llm_client=self._llm,
            lean_repl=self._repl,
            candidate_id=candidate_id,
            prover_config=self._prover_config,
        )

        ctx = AgentContext(
            task=type_candidate.name,
            metadata={
                "type_candidate": type_candidate.model_dump(),
                "lemmas": [l.model_dump() for l in lemmas],
                "prior_definitions": prior_definitions,
            },
        )

        result = formalizer.run(ctx)
        if result.result:
            return TypeFormalizationCandidate.model_validate(result.result)

        return TypeFormalizationCandidate(
            candidate_id=candidate_id,
            type_name=type_candidate.name,
            compiles=False,
        )

    def _evaluate(
        self,
        type_name: str,
        candidates: list[TypeFormalizationCandidate],
    ) -> AuctionResult:
        scores = [compute_auction_score(c) for c in candidates]
        scores.sort(key=lambda s: s.total_score, reverse=True)

        best = scores[0]
        if best.total_score >= self._quality_threshold:
            winner = next(c for c in candidates if c.candidate_id == best.candidate_id)
            return AuctionResult(
                type_name=type_name,
                verdict=AuctionVerdict.ACCEPTED,
                winner_id=best.candidate_id,
                scores=scores,
                winning_candidate=winner,
                reason=f"Candidate {best.candidate_id} scored {best.total_score:.3f} "
                f"(threshold {self._quality_threshold})",
            )

        return AuctionResult(
            type_name=type_name,
            verdict=AuctionVerdict.RETRY,
            scores=scores,
            reason=f"Best score {best.total_score:.3f} below threshold "
            f"{self._quality_threshold}",
        )
