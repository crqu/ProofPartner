"""Proof pipeline coordinator.

Wires: Lean statement -> Proof Search -> [Lemma Breakdown -> Lemma Leanifier
       -> Recursive Prover] -> Flatten & Finalize

Integrates ClaimCheck at each node to verify statement preservation.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from agentic_research.agents.claim_check import ClaimCheck
from agentic_research.agents.flatten_finalize import FlattenFinalize
from agentic_research.agents.informalizer import Informalizer
from agentic_research.agents.intent_judge import IntentJudge
from agentic_research.agents.lemma_breakdown import LemmaBreakdown
from agentic_research.agents.proof_critic import ProofCritic
from agentic_research.agents.proof_detailer import ProofDetailer
from agentic_research.agents.lemma_leanifier import LemmaLeanifier
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.proof_corrector import ProofCorrector
from agentic_research.agents.proof_search import ProofSearchAgent
from agentic_research.agents.recursive_prover import RecursiveProver
from agentic_research.logging import get_logger
from agentic_research.models.agents import AgentContext, AgentStatus, ProverConfig, TokenUsage
from agentic_research.models.external_prover import ExternalProverConfig
from agentic_research.models.formalization import ClaimCheckVerdict
from agentic_research.models.proof import (
    CritiqueIssue,
    CritiqueResult,
    ErrorCategory,
    FailureDiagnosis,
    FailureType,
    LemmaTree,
    NodeStatus,
    ProofCorrection,
    ProofPipelineResult,
    ProofSearchResult,
    RecursiveProofResult,
)
from agentic_research.models.verification import IntentVerdictType
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
        use_proof_critic: bool = False,
        max_critic_retries: int = 2,
        use_proof_detailer: bool = False,
        progress_callback: Callable[[str, str], None] | None = None,
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
        self._use_proof_critic = use_proof_critic
        self._max_critic_retries = max_critic_retries
        self._use_proof_detailer = use_proof_detailer
        self._progress_callback = progress_callback
        self._total_tokens = TokenUsage()
        self._statement_nl: str = ""
        self._lean_preamble: str | None = None

    def _notify_progress(self, stage: str, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(stage, message)

    def _accumulate_tokens(self, usage: TokenUsage) -> None:
        self._total_tokens.input_tokens += usage.input_tokens
        self._total_tokens.output_tokens += usage.output_tokens
        self._total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self._total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens

    _DRO_KEYWORDS = frozenset([
        "wasserstein", "coupling", "distributionally robust",
        "probability measure", "transport cost",
    ])

    def _detect_lean_preamble(self, statement_nl: str) -> str | None:
        """Return the DRO data-package preamble if NL statement matches keywords."""
        lower = statement_nl.lower()
        if any(kw in lower for kw in self._DRO_KEYWORDS):
            from agentic_research.data_packages import get_package
            pkg = get_package("dro_coupling")
            if pkg is not None:
                return pkg.lean_preamble()
        return None

    def run(self, lean_statement: str, statement_nl: str = "") -> ProofPipelineResult:
        """Execute the full proof pipeline."""
        pipeline_start = time.monotonic()
        self._statement_nl = statement_nl
        self._lean_preamble = self._detect_lean_preamble(statement_nl)
        log.info("proof_pipeline_start", statement_len=len(lean_statement))

        self._notify_progress("Proof Search", "Starting proof search")

        if self._use_external_prover and self._external_prover_config is not None:
            ext_result = self._run_external_prover(lean_statement)
            if ext_result is not None:
                return ext_result
            log.info("external_prover_fallback_to_builtin")

        tactic_start = time.monotonic()
        tactic = self._repl.try_automated_tactics(lean_statement)
        if tactic is not None:
            tactic_elapsed = time.monotonic() - tactic_start
            proof_code = f"{lean_statement} by {tactic}"
            log.info(
                "automated_tactic_success",
                tactic=tactic,
                elapsed_seconds=round(tactic_elapsed, 3),
            )
            return ProofPipelineResult(
                statement=lean_statement,
                proved=True,
                final_proof=proof_code,
                total_token_usage=self._total_tokens,
            )

        search_result = self._run_proof_search(lean_statement)
        force_decomposition = False

        if search_result.proved and search_result.proof_code:
            direct_ok = True
            if self._use_claim_check:
                passed = self._run_claim_check(lean_statement, search_result.proof_code)
                if not passed:
                    log.warning("proof_pipeline_claim_check_failed_direct")
                    direct_ok = False
                    force_decomposition = True

            if direct_ok:
                log.info("proof_pipeline_direct_success", elapsed_seconds=round(time.monotonic() - pipeline_start, 3))
                return ProofPipelineResult(
                    statement=lean_statement,
                    proved=True,
                    final_proof=search_result.proof_code,
                    search_result=search_result,
                    claim_check_passed=True,
                    total_token_usage=self._total_tokens,
                )

        if not force_decomposition:
            correction = self._try_proof_correction(lean_statement, search_result)
            if correction is not None:
                corrected_result = self._run_proof_search_with_correction(
                    lean_statement, correction
                )
                if corrected_result.proved and corrected_result.proof_code:
                    corrected_ok = True
                    if self._use_claim_check:
                        passed = self._run_claim_check(
                            lean_statement, corrected_result.proof_code
                        )
                        if not passed:
                            log.warning("proof_pipeline_claim_check_failed_corrected")
                            corrected_ok = False
                            force_decomposition = True

                    if corrected_ok:
                        log.info("proof_pipeline_corrected_success", elapsed_seconds=round(time.monotonic() - pipeline_start, 3))
                        return ProofPipelineResult(
                            statement=lean_statement,
                            proved=True,
                            final_proof=corrected_result.proof_code,
                            search_result=corrected_result,
                            claim_check_passed=True,
                            total_token_usage=self._total_tokens,
                        )

        if not search_result.needs_decomposition and not force_decomposition:
            return ProofPipelineResult(
                statement=lean_statement,
                search_result=search_result,
                failure_stage="proof_search",
                failure_reason=search_result.failure_reason,
                total_token_usage=self._total_tokens,
            )

        log.info("proof_pipeline_decomposing")
        self._notify_progress("Lemma Breakdown", "Decomposing into lemmas")

        tree = self._run_lemma_breakdown(lean_statement, statement_nl, search_result)
        if tree is None:
            return ProofPipelineResult(
                statement=lean_statement,
                search_result=search_result,
                failure_stage="lemma_breakdown",
                failure_reason="Lemma breakdown failed",
                total_token_usage=self._total_tokens,
            )

        if self._use_proof_critic:
            critique = self._run_proof_critic(tree, statement_nl, lean_statement)
            if critique and not critique.passed:
                critic_issues = critique.issues
                for _retry in range(self._max_critic_retries):
                    tree = self._run_lemma_breakdown(
                        lean_statement, statement_nl, search_result,
                        critic_feedback=critic_issues,
                    )
                    if tree is None:
                        break
                    critique = self._run_proof_critic(tree, statement_nl, lean_statement)
                    if critique is None or critique.passed:
                        break
                    critic_issues = critique.issues

            if tree is None:
                return ProofPipelineResult(
                    statement=lean_statement,
                    search_result=search_result,
                    failure_stage="proof_critic",
                    failure_reason="Lemma breakdown failed after critic retries",
                    total_token_usage=self._total_tokens,
                )

            if critique and not critique.passed:
                blocking = [i for i in critique.issues if i.severity == "blocking"]
                if blocking:
                    log.warning(
                        "proof_critic_exhausted",
                        blocking_issues=len(blocking),
                        retries=self._max_critic_retries,
                    )

        if self._use_proof_detailer:
            tree = self._run_proof_detailer(tree)

        self._notify_progress("Leanification", "Converting lemmas to Lean 4")
        tree = self._run_lemma_leanifier(tree)
        if tree is None:
            return ProofPipelineResult(
                statement=lean_statement,
                search_result=search_result,
                failure_stage="lemma_leanifier",
                failure_reason="Lemma leanification failed",
                total_token_usage=self._total_tokens,
            )

        has_axiom_nodes = any(n.from_prior_work for n in tree.nodes.values())
        if has_axiom_nodes:
            self._verify_axiom_nodes(tree, statement_nl)

        self._notify_progress("Recursive Prover", "Proving lemmas recursively")
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

        self._notify_progress("Finalization", "Assembling final proof")
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

        log.info("proof_pipeline_recursive_success", elapsed_seconds=round(time.monotonic() - pipeline_start, 3))
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
        critic_feedback: list[CritiqueIssue] | None = None,
    ) -> LemmaTree | None:
        failed_strategies = "\n".join(
            f"- {s.strategy_type.value}: {s.description}"
            for s in search_result.strategies_tried
        )

        metadata: dict = {
            "statement_lean": lean_statement,
            "failed_attempts": failed_strategies or "None",
        }
        if critic_feedback:
            metadata["critic_issues"] = [
                issue.model_dump() for issue in critic_feedback
            ]

        agent = LemmaBreakdown(llm_client=self._llm)
        ctx = AgentContext(
            task=statement_nl or lean_statement,
            metadata=metadata,
        )
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.status == AgentStatus.SUCCESS and result.result:
            return LemmaTree.model_validate(result.result)
        return None

    def _run_lemma_leanifier(self, tree: LemmaTree) -> LemmaTree | None:
        agent = LemmaLeanifier(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_preamble=self._lean_preamble,
        )
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
            lean_preamble=self._lean_preamble,
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

    @staticmethod
    def _extract_compiler_errors(search_result: ProofSearchResult) -> list[str]:
        """Extract structured compiler error strings from a failed proof search."""
        errors: list[str] = []
        if search_result.failure_reason:
            errors.append(search_result.failure_reason)
        for strategy in search_result.strategies_tried:
            tactic_desc = ", ".join(strategy.key_tactics) if strategy.key_tactics else "none"
            errors.append(
                f"Strategy '{strategy.strategy_type.value}' failed "
                f"(tactics: [{tactic_desc}]): {strategy.description}"
            )
        return errors

    def _try_proof_correction(
        self, statement: str, search_result: ProofSearchResult
    ) -> ProofCorrection | None:
        """Invoke ProofCorrector if the search failed with compilation errors."""
        if not search_result.strategies_tried:
            return None

        if search_result.failure_reason and "timeout" in search_result.failure_reason.lower():
            return None

        last_proof = search_result.proof_code
        if not last_proof and search_result.strategies_tried:
            last_strategy = search_result.strategies_tried[-1]
            tactics = ", ".join(last_strategy.key_tactics) if last_strategy.key_tactics else "sorry"
            last_proof = f"{statement} by {tactics}"
        if not last_proof:
            last_proof = statement
        error_msg = search_result.failure_reason or "Proof search exhausted all strategies"

        compiler_errors = self._extract_compiler_errors(search_result)
        log.info(
            "proof_correction_compiler_errors",
            error_count=len(compiler_errors),
        )

        corrector = ProofCorrector(llm_client=self._llm)
        correction = corrector.correct(
            failed_proof=last_proof,
            error_message=error_msg,
            lean_goal_state=statement,
            compiler_errors=compiler_errors if compiler_errors else None,
        )
        self._accumulate_tokens(corrector.cumulative_tokens)

        log.info(
            "proof_corrector_category",
            category=correction.error_category.value,
        )

        if correction.error_category == ErrorCategory.TIMEOUT:
            log.info("proof_corrector_skip_timeout")
            return None

        return correction

    def _run_proof_search_with_correction(
        self, statement: str, correction: ProofCorrection
    ) -> ProofSearchResult:
        """Re-run proof search with correction context as additional hints."""
        correction_hint = (
            f"Previous attempt failed with {correction.error_category.value}: "
            f"{correction.reasoning}\n"
            f"Suggested tactics: {', '.join(correction.suggested_tactics)}\n"
            f"Revised sketch:\n{correction.revised_proof_sketch}"
        )

        compiler_feedback = (
            f"## Compiler Feedback\n"
            f"Error category: {correction.error_category.value}\n"
            f"Error message: {correction.error_message}\n"
            f"Constraint: Do NOT repeat this error — "
            f"avoid the {correction.error_category.value} pattern described above."
        )

        augmented_task = (
            f"{statement}\n\n"
            f"[Correction context from previous failed attempt]\n"
            f"{correction_hint}\n\n"
            f"{compiler_feedback}"
        )

        agent = ProofSearchAgent(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_search=self._search,
            prover_config=self._prover_config,
            max_strategies=self._max_strategies,
        )
        ctx = AgentContext(task=augmented_task)
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.result:
            return ProofSearchResult.model_validate(result.result)
        return ProofSearchResult(
            statement=statement,
            needs_decomposition=True,
            failure_reason="Corrected proof search returned no result",
        )

    def _run_proof_detailer(self, tree: LemmaTree) -> LemmaTree:
        detailer = ProofDetailer(llm_client=self._llm)
        ctx = AgentContext(
            task="detail proof nodes",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = detailer.run(ctx)
        self._accumulate_tokens(detailer.cumulative_tokens)
        if result.status == AgentStatus.SUCCESS and result.result:
            return LemmaTree.model_validate(result.result)
        return tree

    def _run_proof_critic(
        self, tree: LemmaTree, statement_nl: str, statement_lean: str
    ) -> CritiqueResult | None:
        critic = ProofCritic(llm_client=self._llm)
        ctx = AgentContext(
            task=statement_nl,
            metadata={
                "lemma_tree": tree.model_dump(),
                "statement_lean": statement_lean,
            },
        )
        result = critic.run(ctx)
        self._accumulate_tokens(critic.cumulative_tokens)
        if result.status == AgentStatus.SUCCESS and result.result:
            return CritiqueResult.model_validate(result.result)
        return None

    def _verify_axiom_nodes(self, tree: LemmaTree, statement_nl: str) -> None:
        """Run IntentJudge on axiom nodes to verify faithfulness.

        For axiom nodes, a more-complete axiom (with extra assumptions like
        Polish space requirements) will legitimately differ from the brief NL
        description. We accept low-confidence INCORRECT verdicts with a warning
        since the LLM isn't confident the axiom is actually wrong — it's likely
        just flagging the additional hypotheses.
        """
        informalizer = Informalizer(llm_client=self._llm)
        judge = IntentJudge(llm_client=self._llm, informalizer=informalizer)

        for node_id, node in tree.nodes.items():
            if not node.from_prior_work or not node.statement_lean:
                continue

            verdict = judge.judge(
                lean_code=node.statement_lean,
                original_idea=node.source_reference or node.statement_nl,
                conjecture=node.statement_nl,
            )
            self._accumulate_tokens(judge.cumulative_tokens)

            if verdict.overall_verdict == IntentVerdictType.INCORRECT:
                if verdict.overall_confidence < 0.7:
                    log.warning(
                        "axiom_intent_check_low_confidence_accept",
                        node_id=node_id,
                        confidence=verdict.overall_confidence,
                        concerns=verdict.all_concerns,
                    )
                else:
                    log.warning(
                        "axiom_intent_check_failed",
                        node_id=node_id,
                        confidence=verdict.overall_confidence,
                        concerns=verdict.all_concerns,
                    )
                    node.status = NodeStatus.FAILED
                    node.failure_diagnosis = FailureDiagnosis(
                        failure_type=FailureType.MISSING_HYPOTHESIS,
                        description=f"Axiom faithfulness check failed: {verdict.all_concerns}",
                    )
                    node.from_prior_work = False
                    node.source_reference = None

    def _run_claim_check(self, statement: str, proof_code: str) -> bool:
        checker = ClaimCheck(llm_client=self._llm, use_llm_check=True)
        ctx = AgentContext(
            task=statement,
            metadata={"lean_code": proof_code},
        )
        result = checker.run(ctx)
        self._accumulate_tokens(checker.cumulative_tokens)
        if result.result:
            verdict: str = result.result.get("verdict", "fail")
            return verdict == ClaimCheckVerdict.PASS.value
        return True
