"""Proof pipeline coordinator.

Wires: Lean statement -> Proof Search -> [Lemma Breakdown -> Lemma Leanifier
       -> Recursive Prover] -> Flatten & Finalize

Integrates ClaimCheck at each node to verify statement preservation.
"""

from __future__ import annotations

from agentic_research.agents.claim_check import ClaimCheck
from agentic_research.agents.flatten_finalize import FlattenFinalize
from agentic_research.agents.lemma_breakdown import LemmaBreakdown
from agentic_research.agents.lemma_leanifier import LemmaLeanifier
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.proof_search import ProofSearchAgent
from agentic_research.agents.recursive_prover import RecursiveProver
from agentic_research.logging import get_logger
from agentic_research.models.agents import AgentContext, AgentStatus, ProverConfig, TokenUsage
from agentic_research.models.external_prover import ExternalProverConfig
from agentic_research.models.formalization import ClaimCheckVerdict
from agentic_research.models.proof import (
    LemmaTree,
    ProofPipelineResult,
    ProofSearchResult,
    RecursiveProofResult,
)
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)


class ProofPipeline:
    """End-to-end proof pipeline from Lean statement to verified proof."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        lean_search: LeanSearch,
        *,
        prover_config: ProverConfig | None = None,
        max_strategies: int = 3,
        max_depth: int = 5,
        max_retries_per_node: int = 3,
        use_claim_check: bool = True,
        use_external_prover: bool = False,
        external_prover_config: ExternalProverConfig | None = None,
    ) -> None:
        self._llm = llm_client
        self._repl = lean_repl
        self._search = lean_search
        self._prover_config = prover_config or ProverConfig()
        self._max_strategies = max_strategies
        self._max_depth = max_depth
        self._max_retries_per_node = max_retries_per_node
        self._use_claim_check = use_claim_check
        self._use_external_prover = use_external_prover
        self._external_prover_config = external_prover_config
        self._total_tokens = TokenUsage()

    def _accumulate_tokens(self, usage: TokenUsage) -> None:
        self._total_tokens.input_tokens += usage.input_tokens
        self._total_tokens.output_tokens += usage.output_tokens
        self._total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self._total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens

    def run(self, lean_statement: str, statement_nl: str = "") -> ProofPipelineResult:
        """Execute the full proof pipeline."""
        log.info("proof_pipeline_start", statement_len=len(lean_statement))

        if self._use_external_prover and self._external_prover_config is not None:
            ext_result = self._run_external_prover(lean_statement)
            if ext_result is not None:
                return ext_result
            log.info("external_prover_fallback_to_builtin")

        search_result = self._run_proof_search(lean_statement)

        if search_result.proved and search_result.proof_code:
            if self._use_claim_check:
                passed = self._run_claim_check(lean_statement, search_result.proof_code)
                if not passed:
                    log.warning("proof_pipeline_claim_check_failed_direct")
                    return ProofPipelineResult(
                        statement=lean_statement,
                        search_result=search_result,
                        claim_check_passed=False,
                        failure_stage="claim_check",
                        failure_reason="Claim check failed on direct proof",
                        total_token_usage=self._total_tokens,
                    )

            log.info("proof_pipeline_direct_success")
            return ProofPipelineResult(
                statement=lean_statement,
                proved=True,
                final_proof=search_result.proof_code,
                search_result=search_result,
                claim_check_passed=True,
                total_token_usage=self._total_tokens,
            )

        if not search_result.needs_decomposition:
            return ProofPipelineResult(
                statement=lean_statement,
                search_result=search_result,
                failure_stage="proof_search",
                failure_reason=search_result.failure_reason,
                total_token_usage=self._total_tokens,
            )

        log.info("proof_pipeline_decomposing")

        tree = self._run_lemma_breakdown(lean_statement, statement_nl, search_result)
        if tree is None:
            return ProofPipelineResult(
                statement=lean_statement,
                search_result=search_result,
                failure_stage="lemma_breakdown",
                failure_reason="Lemma breakdown failed",
                total_token_usage=self._total_tokens,
            )

        tree = self._run_lemma_leanifier(tree)
        if tree is None:
            return ProofPipelineResult(
                statement=lean_statement,
                search_result=search_result,
                failure_stage="lemma_leanifier",
                failure_reason="Lemma leanification failed",
                total_token_usage=self._total_tokens,
            )

        recursive_result = self._run_recursive_prover(tree)

        if not recursive_result.proved or not recursive_result.lemma_tree:
            return ProofPipelineResult(
                statement=lean_statement,
                search_result=search_result,
                recursive_result=recursive_result,
                failure_stage="recursive_prover",
                failure_reason=recursive_result.failure_reason,
                total_token_usage=self._total_tokens,
            )

        final_proof = self._run_flatten_finalize(recursive_result.lemma_tree)
        if not final_proof:
            return ProofPipelineResult(
                statement=lean_statement,
                search_result=search_result,
                recursive_result=recursive_result,
                failure_stage="flatten_finalize",
                failure_reason="Proof assembly failed",
                total_token_usage=self._total_tokens,
            )

        if self._use_claim_check:
            passed = self._run_claim_check(lean_statement, final_proof)
            if not passed:
                return ProofPipelineResult(
                    statement=lean_statement,
                    search_result=search_result,
                    recursive_result=recursive_result,
                    claim_check_passed=False,
                    failure_stage="claim_check",
                    failure_reason="Claim check failed on assembled proof",
                    total_token_usage=self._total_tokens,
                )

        log.info("proof_pipeline_recursive_success")
        return ProofPipelineResult(
            statement=lean_statement,
            proved=True,
            final_proof=final_proof,
            search_result=search_result,
            recursive_result=recursive_result,
            claim_check_passed=True,
            total_token_usage=self._total_tokens,
        )

    def _run_external_prover(self, lean_statement: str) -> ProofPipelineResult | None:
        """Try the external prover. Returns a result on success, None on failure."""
        from agentic_research.tools.external_prover import ExternalProverClient

        assert self._external_prover_config is not None
        client = ExternalProverClient(self._external_prover_config)
        log.info("external_prover_attempt", model=self._external_prover_config.model_name)

        ext_result = client.prove(lean_statement)
        self._accumulate_tokens(ext_result.tokens_used)

        if not ext_result.success or not ext_result.proof_code:
            log.warning("external_prover_failed", error=ext_result.error)
            return None

        if self._use_claim_check:
            passed = self._run_claim_check(lean_statement, ext_result.proof_code)
            if not passed:
                log.warning("external_prover_claim_check_failed")
                return None

        log.info("external_prover_success")
        return ProofPipelineResult(
            statement=lean_statement,
            proved=True,
            final_proof=ext_result.proof_code,
            claim_check_passed=True,
            total_token_usage=self._total_tokens,
        )

    def _run_proof_search(self, statement: str) -> ProofSearchResult:
        agent = ProofSearchAgent(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_search=self._search,
            prover_config=self._prover_config,
            max_strategies=self._max_strategies,
        )
        ctx = AgentContext(task=statement)
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.result:
            return ProofSearchResult.model_validate(result.result)
        return ProofSearchResult(
            statement=statement,
            needs_decomposition=True,
            failure_reason="Proof search agent returned no result",
        )

    def _run_lemma_breakdown(
        self,
        lean_statement: str,
        statement_nl: str,
        search_result: ProofSearchResult,
    ) -> LemmaTree | None:
        failed_strategies = "\n".join(
            f"- {s.strategy_type.value}: {s.description}"
            for s in search_result.strategies_tried
        )

        agent = LemmaBreakdown(llm_client=self._llm)
        ctx = AgentContext(
            task=statement_nl or lean_statement,
            metadata={
                "statement_lean": lean_statement,
                "failed_attempts": failed_strategies or "None",
            },
        )
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.status == AgentStatus.SUCCESS and result.result:
            return LemmaTree.model_validate(result.result)
        return None

    def _run_lemma_leanifier(self, tree: LemmaTree) -> LemmaTree | None:
        agent = LemmaLeanifier(llm_client=self._llm, lean_repl=self._repl)
        ctx = AgentContext(
            task="leanify lemmas",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.result:
            return LemmaTree.model_validate(result.result)
        return None

    def _run_recursive_prover(self, tree: LemmaTree) -> RecursiveProofResult:
        agent = RecursiveProver(
            llm_client=self._llm,
            lean_repl=self._repl,
            prover_config=self._prover_config,
            max_depth=self._max_depth,
            max_retries_per_node=self._max_retries_per_node,
        )
        ctx = AgentContext(
            task="prove recursively",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.result:
            return RecursiveProofResult.model_validate(result.result)
        return RecursiveProofResult(
            root_statement=tree.nodes[tree.root_id].statement_lean,
            failure_reason="Recursive prover returned no result",
        )

    def _run_flatten_finalize(self, tree: LemmaTree) -> str | None:
        agent = FlattenFinalize(llm_client=self._llm, lean_repl=self._repl)
        ctx = AgentContext(
            task="flatten proof",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.status == AgentStatus.SUCCESS and result.result:
            return result.result.get("final_proof")
        return None

    def _run_claim_check(self, statement: str, proof_code: str) -> bool:
        checker = ClaimCheck(llm_client=self._llm, use_llm_check=False)
        ctx = AgentContext(
            task=statement,
            metadata={"lean_code": proof_code},
        )
        result = checker.run(ctx)
        self._accumulate_tokens(checker.cumulative_tokens)
        if result.result:
            verdict = result.result.get("verdict", "fail")
            return verdict == ClaimCheckVerdict.PASS.value
        return True
