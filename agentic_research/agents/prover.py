"""Iterative prover agent.

Proposes Lean proofs via LLM, checks with the Lean REPL, feeds
compilation errors back, and refines iteratively up to N iterations.
This is the 'proposer + reviewer + memory' minimal architecture.
"""

from __future__ import annotations

import hashlib
import re

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.lean_utils import _strip_preamble_lines
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    ERROR_FEEDBACK_TEMPLATE,
    LEAN4_PROVER_SYSTEM,
    PROOF_ATTEMPT_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    LLMResponse,
    ProofAttempt,
    ProofAttemptStatus,
    ProverConfig,
    ProverResult,
    TokenUsage,
)
from agentic_research.models.tools import CompilationStatus
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)


def _extract_lean_code(text: str) -> str:
    """Extract Lean code from a ```lean code block, or return raw text."""
    match = re.search(r"```lean\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class IterativeProver(BaseAgent):
    """Iteratively refine a Lean 4 proof using LLM + compiler feedback."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        config: ProverConfig | None = None,
        lean_preamble: str | None = None,
    ) -> None:
        self._config = config or ProverConfig()
        super().__init__(name="iterative_prover", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._lean_preamble = lean_preamble

    @property
    def config(self) -> ProverConfig:
        return self._config

    def _execute(self, context: AgentContext) -> AgentResult:
        statement = context.task

        strategy_meta = {}
        for key in ("strategy_type", "strategy_description", "key_tactics", "relevant_lemmas"):
            if key in context.metadata:
                strategy_meta[key] = context.metadata[key]

        prover_result = self._prove(statement, strategy_metadata=strategy_meta or None)

        status = AgentStatus.SUCCESS if prover_result.proved else AgentStatus.FAILURE
        return AgentResult(
            agent_name=self.name,
            status=status,
            result=prover_result.model_dump(),
            token_usage=prover_result.total_token_usage,
            error_message=prover_result.failure_reason,
        )

    _MAX_TOKENS_CAP = 32768
    _TRUNCATION_MULTIPLIER = 1.5

    def _prove(
        self,
        statement: str,
        strategy_metadata: dict | None = None,
    ) -> ProverResult:
        attempts: list[ProofAttempt] = []
        total_tokens = TokenUsage()
        previous_code: str | None = None
        previous_errors: str = ""
        previous_goals: str = ""
        current_max_tokens: int = self._config.max_tokens
        seen_hashes: set[str] = set()

        for iteration in range(1, self._config.max_iterations + 1):
            log.info("prover_iteration", iteration=iteration, max=self._config.max_iterations)

            llm_response = self._request_proof(
                statement=statement,
                previous_attempt=previous_code,
                errors=previous_errors,
                goals=previous_goals,
                strategy_metadata=strategy_metadata,
                max_tokens_override=current_max_tokens if current_max_tokens != self._config.max_tokens else None,
            )

            total_tokens.input_tokens += llm_response.token_usage.input_tokens
            total_tokens.output_tokens += llm_response.token_usage.output_tokens
            total_tokens.cache_creation_input_tokens += llm_response.token_usage.cache_creation_input_tokens
            total_tokens.cache_read_input_tokens += llm_response.token_usage.cache_read_input_tokens

            if llm_response.stop_reason == "max_tokens":
                new_limit = min(
                    int(current_max_tokens * self._TRUNCATION_MULTIPLIER),
                    self._MAX_TOKENS_CAP,
                )
                log.warning(
                    "prover_truncated",
                    iteration=iteration,
                    old_max_tokens=current_max_tokens,
                    new_max_tokens=new_limit,
                )
                attempt = ProofAttempt(
                    iteration=iteration,
                    proof_code=_extract_lean_code(llm_response.content),
                    status=ProofAttemptStatus.TRUNCATED,
                    token_usage=llm_response.token_usage,
                )
                attempts.append(attempt)
                if new_limit > current_max_tokens:
                    current_max_tokens = new_limit
                    previous_code = None
                    previous_errors = ""
                    previous_goals = ""
                    continue
                previous_code = attempt.proof_code
                previous_errors = "Output was truncated due to max_tokens limit."
                previous_goals = "None"
                continue

            proof_code = _extract_lean_code(llm_response.content)
            if self._lean_preamble:
                stripped = _strip_preamble_lines(proof_code)
                compile_code = self._lean_preamble + "\n\n" + stripped
            else:
                compile_code = proof_code

            input_hash = hashlib.md5(compile_code.encode()).hexdigest()[:12]
            if input_hash in seen_hashes:
                log.warning("prover_duplicate_code", iteration=iteration, input_hash=input_hash)
                break
            seen_hashes.add(input_hash)

            compilation = self._repl.execute(compile_code)

            uses_sorry = any('sorry' in w for w in (compilation.warnings or []))
            if compilation.compilation_status == CompilationStatus.OK and compilation.all_goals_closed and not uses_sorry:
                attempt = ProofAttempt(
                    iteration=iteration,
                    proof_code=proof_code,
                    status=ProofAttemptStatus.SUCCESS,
                    warnings=compilation.warnings,
                    token_usage=llm_response.token_usage,
                )
                attempts.append(attempt)
                log.info("prover_success", iteration=iteration)
                return ProverResult(
                    statement=statement,
                    proved=True,
                    final_proof=proof_code,
                    attempts=attempts,
                    total_iterations=iteration,
                    total_token_usage=total_tokens,
                )

            if compilation.compilation_status == CompilationStatus.ERROR:
                attempt_status = ProofAttemptStatus.COMPILATION_ERROR
            elif compilation.compilation_status == CompilationStatus.TIMEOUT:
                attempt_status = ProofAttemptStatus.TIMEOUT
            else:
                attempt_status = ProofAttemptStatus.INCOMPLETE

            attempt = ProofAttempt(
                iteration=iteration,
                proof_code=proof_code,
                status=attempt_status,
                errors=compilation.errors,
                warnings=compilation.warnings,
                goals_remaining=[g.goal for g in compilation.goals],
                token_usage=llm_response.token_usage,
            )
            attempts.append(attempt)

            previous_code = proof_code
            previous_errors = "\n".join(compilation.errors) if compilation.errors else "No explicit errors."
            previous_goals = "\n".join(g.goal for g in compilation.goals) if compilation.goals else "None"

            log.info(
                "prover_iteration_failed",
                iteration=iteration,
                status=attempt_status.value,
                error_count=len(compilation.errors),
            )

        return ProverResult(
            statement=statement,
            proved=False,
            attempts=attempts,
            total_iterations=self._config.max_iterations,
            total_token_usage=total_tokens,
            failure_reason=f"Failed to prove after {self._config.max_iterations} iterations",
        )

    def _request_proof(
        self,
        *,
        statement: str,
        previous_attempt: str | None,
        errors: str,
        goals: str,
        strategy_metadata: dict | None = None,
        max_tokens_override: int | None = None,
    ) -> "LLMResponse":
        full_statement = (
            f"{self._lean_preamble}\n\n{statement}"
            if self._lean_preamble
            else statement
        )

        if previous_attempt is None:
            user_content = PROOF_ATTEMPT_TEMPLATE.format(statement=full_statement)
        else:
            user_content = ERROR_FEEDBACK_TEMPLATE.format(
                statement=full_statement,
                previous_attempt=previous_attempt,
                errors=errors,
                goals=goals,
            )

        if strategy_metadata:
            strategy_section = "\n## Strategy\n"
            if strategy_metadata.get("strategy_type"):
                strategy_section += f"Type: {strategy_metadata['strategy_type']}\n"
            if strategy_metadata.get("strategy_description"):
                strategy_section += f"Description: {strategy_metadata['strategy_description']}\n"
            if strategy_metadata.get("key_tactics"):
                tactics = strategy_metadata["key_tactics"]
                if isinstance(tactics, list):
                    tactics = ", ".join(tactics)
                strategy_section += f"Key tactics: {tactics}\n"
            if strategy_metadata.get("relevant_lemmas"):
                lemmas = strategy_metadata["relevant_lemmas"]
                if isinstance(lemmas, list):
                    lemmas = ", ".join(lemmas)
                strategy_section += f"Relevant lemmas: {lemmas}\n"
            user_content += strategy_section

        effective_max_tokens = max_tokens_override or self._config.max_tokens

        return self._llm.complete(
            system=LEAN4_PROVER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=effective_max_tokens,
            temperature=self._config.temperature,
            use_extended_thinking=self._config.use_extended_thinking,
            use_cache=True,
        )
