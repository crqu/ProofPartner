"""Proof Search Agent — discovers proofs from scratch.

Three-phase approach:
  1. Exploration: search Mathlib for useful lemmas, identify proof strategies
  2. Strategy selection: LLM proposes 2-3 strategies ranked by plausibility
  3. Attempt: try each strategy via IterativeProver with Lean REPL feedback

When direct proof fails after all strategies, signals that decomposition is needed.
"""

from __future__ import annotations

import re

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    PROOF_STRATEGY_SYSTEM,
    PROOF_STRATEGY_USER_TEMPLATE,
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
from agentic_research.models.proof import ProofSearchResult, ProofStrategy, StrategyType
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)


def _extract_lean_code(text: str) -> str:
    match = re.search(r"```lean\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class ProofSearchAgent(BaseAgent):
    """Discovers proofs from scratch using multi-strategy search."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        lean_search: LeanSearch,
        *,
        prover_config: ProverConfig | None = None,
        max_strategies: int = 3,
    ) -> None:
        super().__init__(name="proof_search", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._search = lean_search
        self._prover_config = prover_config or ProverConfig()
        self._max_strategies = max_strategies

    def _execute(self, context: AgentContext) -> AgentResult:
        statement = context.task
        total_tokens = TokenUsage()

        mathlib_lemmas = self._explore(statement)

        strategies, tokens = self._select_strategies(statement, mathlib_lemmas)
        total_tokens.input_tokens += tokens.input_tokens
        total_tokens.output_tokens += tokens.output_tokens

        for strategy in strategies[: self._max_strategies]:
            log.info(
                "proof_search_trying_strategy",
                strategy_type=strategy.strategy_type.value,
                plausibility=strategy.plausibility,
            )

            prover_result = self._attempt_strategy(statement, strategy)
            total_tokens.input_tokens += prover_result.total_token_usage.input_tokens
            total_tokens.output_tokens += prover_result.total_token_usage.output_tokens

            if prover_result.proved:
                search_result = ProofSearchResult(
                    statement=statement,
                    proved=True,
                    proof_code=prover_result.final_proof,
                    strategies_tried=strategies[: strategies.index(strategy) + 1],
                    mathlib_lemmas_found=[lem for lem in mathlib_lemmas],
                    iterations_used=prover_result.total_iterations,
                )
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.SUCCESS,
                    result=search_result.model_dump(),
                    token_usage=total_tokens,
                )

        search_result = ProofSearchResult(
            statement=statement,
            proved=False,
            strategies_tried=strategies,
            needs_decomposition=True,
            mathlib_lemmas_found=[lem for lem in mathlib_lemmas],
            failure_reason=f"All {len(strategies)} strategies exhausted",
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILURE,
            result=search_result.model_dump(),
            token_usage=total_tokens,
        )

    def _explore(self, statement: str) -> list[str]:
        """Search Mathlib for potentially useful lemmas."""
        log.info("proof_search_explore", statement_len=len(statement))
        search_result = self._search.execute(statement)
        if not hasattr(search_result, "entries"):
            return []
        return [
            f"{e.name}: {e.type_signature}" for e in search_result.entries if e.name
        ]

    def _select_strategies(
        self, statement: str, mathlib_lemmas: list[str]
    ) -> tuple[list[ProofStrategy], TokenUsage]:
        """Ask the LLM to propose proof strategies."""
        log.info("proof_search_select_strategies")

        lemmas_text = "\n".join(f"- {lem}" for lem in mathlib_lemmas) if mathlib_lemmas else "No relevant lemmas found."
        user_content = PROOF_STRATEGY_USER_TEMPLATE.format(
            statement=statement,
            mathlib_lemmas=lemmas_text,
        )

        response = self._llm.complete(
            system=PROOF_STRATEGY_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.3,
            use_cache=True,
        )

        strategies = self._parse_strategies(response.content)
        strategies.sort(key=lambda s: s.plausibility, reverse=True)

        log.info("proof_search_strategies_selected", count=len(strategies))
        return strategies, response.token_usage

    def _parse_strategies(self, response_text: str) -> list[ProofStrategy]:
        parsed = self._llm.extract_json(response_text)
        if not isinstance(parsed, dict):
            return [ProofStrategy(strategy_type=StrategyType.DIRECT, description="fallback direct proof", plausibility=0.5)]

        strategies = []
        for item in parsed.get("strategies", []):
            try:
                st = StrategyType(item.get("strategy_type", "direct"))
            except ValueError:
                st = StrategyType.DIRECT
            strategies.append(
                ProofStrategy(
                    strategy_type=st,
                    description=item.get("description", ""),
                    relevant_lemmas=item.get("relevant_lemmas", []),
                    plausibility=float(item.get("plausibility", 0.5)),
                    key_tactics=item.get("key_tactics", []),
                )
            )

        if not strategies:
            strategies.append(ProofStrategy(strategy_type=StrategyType.DIRECT, description="fallback", plausibility=0.5))

        return strategies

    def _attempt_strategy(self, statement: str, strategy: ProofStrategy) -> ProverResult:
        """Try to prove the statement using a specific strategy via IterativeProver."""
        prover = IterativeProver(
            llm_client=self._llm,
            lean_repl=self._repl,
            config=self._prover_config,
        )

        ctx = AgentContext(
            task=statement,
            metadata={
                "strategy_type": strategy.strategy_type.value,
                "strategy_description": strategy.description,
                "key_tactics": ", ".join(strategy.key_tactics),
                "relevant_lemmas": ", ".join(strategy.relevant_lemmas),
            },
        )

        result = prover.run(ctx)
        if result.result:
            return ProverResult.model_validate(result.result)

        return ProverResult(
            statement=statement,
            proved=False,
            failure_reason="Prover returned no result",
        )
