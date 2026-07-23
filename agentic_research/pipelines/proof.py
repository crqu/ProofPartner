"""Proof pipeline coordinator.

Wires: Lean statement -> Proof Search -> [Lemma Breakdown -> Lemma Leanifier
       -> Recursive Prover] -> Flatten & Finalize

Integrates ClaimCheck at each node to verify statement preservation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_research.agents.nl_prover import NaturalLanguageProver

from agentic_research.agents.auctioneer import Auctioneer
from agentic_research.agents.claim_check import ClaimCheck
from agentic_research.agents.flatten_finalize import FlattenFinalize
from agentic_research.agents.informalizer import Informalizer
from agentic_research.agents.intent_judge import IntentJudge
from agentic_research.agents.lemma_breakdown import LemmaBreakdown
from agentic_research.agents.lemma_planner import LemmaPlanner
from agentic_research.agents.proof_critic import ProofCritic
from agentic_research.agents.proof_detailer import ProofDetailer
from agentic_research.agents.lemma_leanifier import LemmaLeanifier
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.proof_corrector import ProofCorrector
from agentic_research.agents.proof_search import ProofSearchAgent
from agentic_research.agents.recursive_prover import RecursiveProver
from agentic_research.agents.type_planner import TypePlanner
from agentic_research.logging import get_logger
from agentic_research.models.agents import AgentContext, AgentStatus, ProverConfig, TokenUsage
from agentic_research.models.external_prover import ExternalProverConfig
from agentic_research.models.formalization import (
    AuctionResult,
    AuctionVerdict,
    ClaimCheckVerdict,
    LemmaStatement,
    TypePlan,
)
from agentic_research.models.proof import (
    CritiqueIssue,
    CritiqueIssueType,
    CritiqueResult,
    ErrorCategory,
    FailureDiagnosis,
    FailureType,
    LemmaTree,
    NLProofSketch,
    NodeStatus,
    ProofCorrection,
    ProofPipelineResult,
    ProofSearchResult,
    RecursiveProofResult,
)
from agentic_research.models.verification import IntentVerdictType
from agentic_research.tools.lean_repl import LeanRepl, ReplBackend
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
        max_strategies: int = 2,
        max_depth: int = 5,
        max_retries_per_node: int = 3,
        use_claim_check: bool = True,
        use_external_prover: bool = False,
        external_prover_config: ExternalProverConfig | None = None,
        use_proof_critic: bool = True,
        max_critic_retries: int = 0,
        use_proof_detailer: bool = True,
        use_intent_judge: bool = False,
        nl_prover: NaturalLanguageProver | None = None,
        use_nl_proof_stage: bool = True,
        decomposition_k: int = 1,
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
        self._use_intent_judge = use_intent_judge
        self._nl_prover = nl_prover
        self._use_nl_proof_stage = use_nl_proof_stage
        self._decomposition_k = decomposition_k
        self._progress_callback = progress_callback
        self._total_tokens = TokenUsage()
        self._statement_nl: str = ""
        self._lean_preamble: str | None = None
        self._prebuilt_axioms: dict[str, str] | None = None
        self._axiom_keywords: dict[str, list[str]] | None = None

    def _make_result(self, **kwargs) -> ProofPipelineResult:
        """Construct a ProofPipelineResult with backend info pre-filled."""
        backend = getattr(self._repl, "backend", None)
        if isinstance(backend, ReplBackend):
            kwargs.setdefault("backend", backend.value)
            kwargs.setdefault("verified", backend != ReplBackend.MOCK)
        return ProofPipelineResult(**kwargs)

    def _notify_progress(self, stage: str, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(stage, message)

    def _accumulate_tokens(self, usage: TokenUsage) -> None:
        self._total_tokens.input_tokens += usage.input_tokens
        self._total_tokens.output_tokens += usage.output_tokens
        self._total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self._total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens

    _MAX_BACKTRACK_ATTEMPTS = 2

    _DRO_KEYWORDS = frozenset([
        "wasserstein", "coupling", "distributionally robust",
        "probability measure", "transport cost",
    ])

    def _detect_lean_preamble(self, statement_nl: str) -> str | None:
        """Return the DRO data-package preamble if NL statement matches keywords.

        Also loads pre-built axioms and keywords if available.
        """
        lower = statement_nl.lower()
        if any(kw in lower for kw in self._DRO_KEYWORDS):
            from agentic_research.data_packages import get_package
            pkg = get_package("dro_coupling")
            if pkg is not None:
                if hasattr(pkg, "provided_axioms"):
                    self._prebuilt_axioms = pkg.provided_axioms()
                if hasattr(pkg, "axiom_keywords"):
                    self._axiom_keywords = pkg.axiom_keywords()
                return pkg.lean_preamble()
        return None

    def run(self, lean_statement: str, statement_nl: str = "") -> ProofPipelineResult:
        """Execute the full proof pipeline."""
        pipeline_start = time.monotonic()
        self._statement_nl = statement_nl
        self._lean_preamble = self._detect_lean_preamble(statement_nl)
        self._backtrack_counts: dict[str, int] = {}
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
            return self._make_result(
                statement=lean_statement,
                proved=True,
                final_proof=proof_code,
                total_token_usage=self._total_tokens,
            )

        type_defs = self._run_type_first_formalization(self._statement_nl)
        if type_defs:
            self._lean_preamble = (
                f"{self._lean_preamble}\n\n{type_defs}"
                if self._lean_preamble
                else type_defs
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
                return self._make_result(
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
                        return self._make_result(
                            statement=lean_statement,
                            proved=True,
                            final_proof=corrected_result.proof_code,
                            search_result=corrected_result,
                            claim_check_passed=True,
                            total_token_usage=self._total_tokens,
                        )

        if not search_result.needs_decomposition and not force_decomposition:
            log.info("proof_pipeline_forcing_decomposition",
                     reason="proof_search_exhausted")
            force_decomposition = True

        log.info("proof_pipeline_decomposing")

        nl_sketch: NLProofSketch | None = None
        if self._nl_prover and self._use_nl_proof_stage:
            nl_sketch = self._run_nl_proof_stage(lean_statement, statement_nl)

        tactic_hints = ""
        if nl_sketch and nl_sketch.proof_steps:
            detailer = ProofDetailer(llm_client=self._llm)
            tactic_hints = detailer.detail_sketch(nl_sketch)
            self._accumulate_tokens(detailer.cumulative_tokens)
            log.info("proof_pipeline_sketch_detailed", hint_len=len(tactic_hints))

        self._notify_progress("Lemma Breakdown", "Decomposing into lemmas")

        tree = self._run_lemma_breakdown(
            lean_statement, statement_nl, search_result,
            nl_proof_context=nl_sketch,
            tactic_hints=tactic_hints,
        )
        if tree is None:
            return self._make_result(
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
                        nl_proof_context=nl_sketch,
                        tactic_hints=tactic_hints,
                    )
                    if tree is None:
                        break
                    critique = self._run_proof_critic(tree, statement_nl, lean_statement)
                    if critique is None or critique.passed:
                        break
                    critic_issues = critique.issues

            if tree is None:
                return self._make_result(
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
        tree = self._run_lemma_leanifier(tree, tactic_hints=tactic_hints)
        if tree is None:
            return self._make_result(
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
            weak_feedback = self._extract_weak_child_feedback(recursive_result)
            if weak_feedback:
                retried = self._retry_on_weak_children(
                    lean_statement, statement_nl, search_result,
                    weak_feedback, pipeline_start,
                    nl_proof_context=nl_sketch,
                    tactic_hints=tactic_hints,
                    existing_tree=recursive_result.lemma_tree,
                )
                if retried is not None:
                    return retried

            backtrack_target = self._classify_backtrack_target(recursive_result)
            if (
                backtrack_target
                and self._backtrack_counts.get(backtrack_target, 0)
                < self._MAX_BACKTRACK_ATTEMPTS
            ):
                self._backtrack_counts[backtrack_target] = (
                    self._backtrack_counts.get(backtrack_target, 0) + 1
                )
                log.info(
                    "proof_pipeline_backtracking",
                    target=backtrack_target,
                    attempt=self._backtrack_counts[backtrack_target],
                )

                if backtrack_target == "type_formalization":
                    backtrack_result = self._backtrack_to_type_formalization(
                        lean_statement, statement_nl, search_result,
                        recursive_result, nl_sketch, tactic_hints,
                        pipeline_start,
                    )
                    if backtrack_result is not None:
                        backtrack_result.backtrack_stages.append("type_formalization")
                        return backtrack_result

                elif backtrack_target == "nl_proof":
                    backtrack_result = self._backtrack_to_nl_proof(
                        lean_statement, statement_nl, search_result,
                        recursive_result, pipeline_start,
                    )
                    if backtrack_result is not None:
                        backtrack_result.backtrack_stages.append("nl_proof")
                        return backtrack_result

            return self._make_result(
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
            return self._make_result(
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
                return self._make_result(
                    statement=lean_statement,
                    search_result=search_result,
                    recursive_result=recursive_result,
                    claim_check_passed=False,
                    failure_stage="claim_check",
                    failure_reason="Claim check failed on assembled proof",
                    total_token_usage=self._total_tokens,
                )

        log.info("proof_pipeline_recursive_success", elapsed_seconds=round(time.monotonic() - pipeline_start, 3))
        return self._make_result(
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
        return self._make_result(
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
            lean_preamble=self._lean_preamble,
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

    _MAX_NL_CRITIC_SAFETY_CAP = 8
    _NL_CRITIC_STALL_WINDOW = 2

    def _run_nl_proof_stage(
        self,
        lean_statement: str,
        statement_nl: str,
    ) -> NLProofSketch | None:
        """Generate and critique an NL proof sketch before decomposition."""
        assert self._nl_prover is not None
        self._notify_progress("NL Proof", "Generating informal proof sketch")

        sketch, gen_tokens = self._nl_prover.generate_proof(
            statement=lean_statement,
            statement_nl=statement_nl or None,
        )
        self._accumulate_tokens(gen_tokens)

        if not sketch.proof_steps:
            log.warning("nl_proof_stage_empty_sketch")
            return None

        critic = ProofCritic(llm_client=self._llm)
        stall_count = 0
        prev_issue_count: int | None = None
        for iteration in range(self._MAX_NL_CRITIC_SAFETY_CAP):
            critique = critic.audit_nl_proof(sketch, lean_statement)
            self._accumulate_tokens(critique.token_usage)

            if not critique.issues:
                log.info("nl_proof_stage_passed", iterations=iteration + 1)
                break

            current_count = len(critique.issues)
            if prev_issue_count is not None and current_count >= prev_issue_count:
                stall_count += 1
            else:
                stall_count = 0
            prev_issue_count = current_count

            if stall_count >= self._NL_CRITIC_STALL_WINDOW:
                log.info(
                    "nl_proof_critic_stalled",
                    iteration=iteration + 1,
                    issues=current_count,
                )
                break

            log.info(
                "nl_proof_stage_critique",
                iteration=iteration + 1,
                issues=current_count,
            )

            sketch, regen_tokens = self._nl_prover.generate_proof(
                statement=lean_statement,
                statement_nl=statement_nl or None,
                feedback=critique,
            )
            self._accumulate_tokens(regen_tokens)

        log.info(
            "nl_proof_stage_done",
            steps=len(sketch.proof_steps),
            strategy=sketch.overall_strategy,
        )
        return sketch

    def _run_lemma_breakdown(
        self,
        lean_statement: str,
        statement_nl: str,
        search_result: ProofSearchResult,
        critic_feedback: list[CritiqueIssue] | None = None,
        nl_proof_context: NLProofSketch | None = None,
        tactic_hints: str = "",
    ) -> LemmaTree | None:
        failed_strategies = "\n".join(
            f"- {s.strategy_type.value}: {s.description}"
            for s in search_result.strategies_tried
        )

        metadata: dict = {
            "statement_lean": lean_statement,
            "failed_attempts": failed_strategies or "None",
        }
        if nl_proof_context:
            metadata["nl_proof_context"] = nl_proof_context.model_dump()
        if tactic_hints:
            metadata["tactic_hints"] = tactic_hints
        if self._lean_preamble:
            metadata["lean_preamble"] = self._lean_preamble
        if critic_feedback:
            metadata["critic_issues"] = [
                issue.model_dump() for issue in critic_feedback
            ]

        agent = LemmaBreakdown(
            llm_client=self._llm,
            decomposition_k=self._decomposition_k,
        )
        ctx = AgentContext(
            task=statement_nl or lean_statement,
            metadata=metadata,
        )
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.status == AgentStatus.SUCCESS and result.result:
            return LemmaTree.model_validate(result.result)
        return None

    def _make_intent_judge(self) -> IntentJudge | None:
        try:
            informalizer = Informalizer(llm_client=self._llm)
            return IntentJudge(llm_client=self._llm, informalizer=informalizer)
        except Exception:
            return None

    def _run_type_first_formalization(self, statement_nl: str) -> str | None:
        """Run TypePlanner → LemmaPlanner → Auctioneer to get validated type defs.

        Returns accepted type definitions as Lean 4 code, or None on failure.
        """
        if not statement_nl:
            return None

        self._notify_progress("Type Planning", "Identifying domain types")
        planner = TypePlanner(llm_client=self._llm, lean_search=self._search)
        metadata: dict = {}
        if self._lean_preamble:
            metadata["lean_preamble"] = self._lean_preamble
        ctx = AgentContext(task=statement_nl, metadata=metadata)
        result = planner.run(ctx)
        self._accumulate_tokens(planner.cumulative_tokens)
        if result.status != AgentStatus.SUCCESS or not result.result:
            log.warning("type_first_planning_failed")
            return None
        type_plan = TypePlan.model_validate(result.result)

        new_types = [
            c for c in type_plan.candidates
            if not c.is_in_mathlib and not c.is_in_preamble and not c.composition_alternative
        ]
        if not new_types:
            log.info("type_first_no_new_types")
            return None

        lemma_planner = LemmaPlanner(llm_client=self._llm)
        lemma_ctx = AgentContext(
            task="plan lemmas",
            metadata={"type_plan": type_plan.model_dump()},
        )
        lemma_result = lemma_planner.run(lemma_ctx)
        self._accumulate_tokens(lemma_planner.cumulative_tokens)

        lemmas_by_type: dict[str, list[LemmaStatement]] = {}
        if lemma_result.status == AgentStatus.SUCCESS and lemma_result.result:
            for lem in lemma_result.result.get("lemmas", []):
                ls = LemmaStatement.model_validate(lem)
                lemmas_by_type.setdefault(ls.for_type, []).append(ls)

        topo_order = type_plan.dependency_graph.topological_order
        if topo_order:
            ordered = sorted(
                new_types,
                key=lambda c: topo_order.index(c.name) if c.name in topo_order else len(topo_order),
            )
        else:
            ordered = new_types

        self._notify_progress("Type Formalization", "Formalizing types with auction")
        intent_judge = self._make_intent_judge() if self._use_intent_judge else None
        auctioneer = Auctioneer(
            llm_client=self._llm,
            lean_repl=self._repl,
            k=3,
            prover_config=self._prover_config,
            intent_judge=intent_judge,
            original_idea=self._statement_nl,
            conjecture=self._statement_nl,
        )

        accepted_defs: list[str] = []
        prior_definitions = ""
        for candidate in ordered:
            lemmas = lemmas_by_type.get(candidate.name, [])
            auction_ctx = AgentContext(
                task=candidate.name,
                metadata={
                    "type_candidate": candidate.model_dump(),
                    "lemmas": [lem.model_dump() for lem in lemmas],
                    "prior_definitions": prior_definitions,
                },
            )
            auction_result_raw = auctioneer.run(auction_ctx)
            if auction_result_raw.result:
                ar = AuctionResult.model_validate(auction_result_raw.result)
                if ar.verdict == AuctionVerdict.ACCEPTED and ar.winning_candidate:
                    code = ar.winning_candidate.lean_code
                    accepted_defs.append(code)
                    prior_definitions = (
                        f"{prior_definitions}\n\n{code}" if prior_definitions else code
                    )

        self._accumulate_tokens(auctioneer.cumulative_tokens)

        if not accepted_defs:
            log.info("type_first_no_types_accepted")
            return None

        type_context = "\n\n".join(accepted_defs)
        log.info("type_first_formalization_done", num_types=len(accepted_defs))
        return type_context

    def _run_lemma_leanifier(
        self, tree: LemmaTree, tactic_hints: str = ""
    ) -> LemmaTree | None:
        agent = LemmaLeanifier(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_preamble=self._lean_preamble,
            prebuilt_axioms=self._prebuilt_axioms,
            axiom_keywords=self._axiom_keywords,
            lean_search=self._search,
        )
        metadata: dict = {"lemma_tree": tree.model_dump()}
        if tactic_hints:
            metadata["tactic_hints"] = tactic_hints
        ctx = AgentContext(
            task="leanify lemmas",
            metadata=metadata,
        )
        result = agent.run(ctx)
        self._accumulate_tokens(agent.cumulative_tokens)
        if result.result:
            return LemmaTree.model_validate(result.result)
        return None

    def _run_recursive_prover(self, tree: LemmaTree) -> RecursiveProofResult:
        leanifier = LemmaLeanifier(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_preamble=self._lean_preamble,
            prebuilt_axioms=self._prebuilt_axioms,
            axiom_keywords=self._axiom_keywords,
            lean_search=self._search,
        )
        nl_prover = self._nl_prover if self._use_nl_proof_stage else None
        detailer = ProofDetailer(llm_client=self._llm) if nl_prover else None
        breakdown_agent = LemmaBreakdown(llm_client=self._llm)
        corrector = ProofCorrector(llm_client=self._llm)
        agent = RecursiveProver(
            llm_client=self._llm,
            lean_repl=self._repl,
            prover_config=self._prover_config,
            max_depth=self._max_depth,
            max_retries_per_node=self._max_retries_per_node,
            lean_preamble=self._lean_preamble,
            leanifier=leanifier,
            nl_prover=nl_prover,
            proof_detailer=detailer,
            breakdown=breakdown_agent,
            proof_corrector=corrector,
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

    _MAX_WEAK_CHILD_RETRIES = 2

    @staticmethod
    def _extract_weak_child_feedback(
        recursive_result: RecursiveProofResult,
    ) -> list[CritiqueIssue]:
        """Extract WEAK_CHILD_LEMMA diagnoses from a failed recursive proof."""
        if not recursive_result.lemma_tree:
            return []
        issues: list[CritiqueIssue] = []
        for node in recursive_result.lemma_tree.nodes.values():
            if (
                node.failure_diagnosis
                and node.failure_diagnosis.failure_type == FailureType.WEAK_CHILD_LEMMA
            ):
                issues.append(
                    CritiqueIssue(
                        issue_type=CritiqueIssueType.WEAK_CHILD_LEMMA,
                        node_id=node.failure_diagnosis.problematic_child_id or node.node_id,
                        description=node.failure_diagnosis.description,
                        severity="blocking",
                        suggested_fix=node.failure_diagnosis.suggested_fix,
                    )
                )
        return issues

    def _graft_subtree(
        self,
        parent_tree: LemmaTree,
        target_node_id: str,
        sub_tree: LemmaTree,
    ) -> bool:
        """Graft a sub-tree onto a target node, remapping IDs to avoid collision."""
        target_node = parent_tree.get_node(target_node_id)
        if not target_node:
            return False

        child_nodes = [
            n for nid, n in sub_tree.nodes.items() if nid != sub_tree.root_id
        ]
        if not child_nodes:
            return False

        from agentic_research.models.proof import ProofNode

        new_children: list[str] = []
        for child in child_nodes:
            child_id = f"{target_node_id}_{child.node_id}"
            new_node = ProofNode(
                node_id=child_id,
                statement_nl=child.statement_nl,
                statement_lean=child.statement_lean,
                parent_id=target_node_id,
                depth=target_node.depth + 1,
            )
            parent_tree.nodes[child_id] = new_node
            new_children.append(child_id)

        target_node.children = new_children
        target_node.status = NodeStatus.PENDING
        target_node.failure_diagnosis = None
        target_node.proof_code = None

        parent_tree.topological_order = new_children + parent_tree.topological_order

        log.info(
            "graft_subtree_done",
            target=target_node_id,
            new_children=len(new_children),
        )
        return True

    def _run_lemma_leanifier_selective(
        self,
        tree: LemmaTree,
        node_ids: list[str],
        tactic_hints: str = "",
    ) -> LemmaTree | None:
        """Leanify only a subset of nodes, leaving existing nodes untouched."""
        agent = LemmaLeanifier(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_preamble=self._lean_preamble,
            prebuilt_axioms=self._prebuilt_axioms,
            axiom_keywords=self._axiom_keywords,
            lean_search=self._search,
        )

        for node_id in node_ids:
            node = tree.get_node(node_id)
            if not node or node.statement_lean:
                continue
            parent = tree.get_node(node.parent_id) if node.parent_id else None
            parent_stmt = parent.statement_lean if parent else ""
            lean_code, usage = agent.leanify_single_node(node, parent_stmt)
            self._accumulate_tokens(usage)
            if lean_code:
                node.statement_lean = lean_code

        return tree

    def _retry_on_weak_children(
        self,
        lean_statement: str,
        statement_nl: str,
        search_result: ProofSearchResult,
        weak_feedback: list[CritiqueIssue],
        pipeline_start: float,
        nl_proof_context: NLProofSketch | None = None,
        tactic_hints: str = "",
        existing_tree: LemmaTree | None = None,
    ) -> ProofPipelineResult | None:
        """Retry via targeted subtree re-breakdown, falling back to full re-breakdown."""
        if existing_tree is not None:
            targeted_result = self._retry_targeted_subtrees(
                lean_statement, statement_nl, search_result,
                weak_feedback, pipeline_start, existing_tree,
                nl_proof_context=nl_proof_context,
                tactic_hints=tactic_hints,
            )
            if targeted_result is not None:
                return targeted_result

        return self._retry_full_rebreakdown(
            lean_statement, statement_nl, search_result,
            weak_feedback, pipeline_start,
            nl_proof_context=nl_proof_context,
            tactic_hints=tactic_hints,
        )

    def _retry_targeted_subtrees(
        self,
        lean_statement: str,
        statement_nl: str,
        search_result: ProofSearchResult,
        weak_feedback: list[CritiqueIssue],
        pipeline_start: float,
        existing_tree: LemmaTree,
        nl_proof_context: NLProofSketch | None = None,
        tactic_hints: str = "",
    ) -> ProofPipelineResult | None:
        """Re-break down only the weak children, grafting results back."""
        weak_node_ids = [
            issue.node_id for issue in weak_feedback
            if existing_tree.get_node(issue.node_id)
        ]
        if not weak_node_ids:
            return None

        any_grafted = False
        new_node_ids: list[str] = []
        for weak_id in weak_node_ids:
            weak_node = existing_tree.get_node(weak_id)
            if not weak_node:
                continue

            sub_tree = self._run_lemma_breakdown(
                weak_node.statement_lean or lean_statement,
                weak_node.statement_nl or statement_nl,
                search_result,
                nl_proof_context=nl_proof_context,
                tactic_hints=tactic_hints,
            )
            if sub_tree is None:
                continue

            if self._graft_subtree(existing_tree, weak_id, sub_tree):
                any_grafted = True
                new_node_ids.extend(existing_tree.get_node(weak_id).children)

        if not any_grafted:
            return None

        if new_node_ids:
            self._run_lemma_leanifier_selective(
                existing_tree, new_node_ids, tactic_hints=tactic_hints,
            )

        self._notify_progress("Recursive Prover", "Re-proving after subtree graft")
        recursive_result = self._run_recursive_prover(existing_tree)

        if recursive_result.proved and recursive_result.lemma_tree:
            self._notify_progress("Finalization", "Assembling final proof")
            final_proof = self._run_flatten_finalize(recursive_result.lemma_tree)
            if final_proof:
                if self._use_claim_check:
                    passed = self._run_claim_check(lean_statement, final_proof)
                    if not passed:
                        return None

                log.info(
                    "proof_pipeline_subtree_graft_success",
                    elapsed_seconds=round(time.monotonic() - pipeline_start, 3),
                )
                return self._make_result(
                    statement=lean_statement,
                    proved=True,
                    final_proof=final_proof,
                    search_result=search_result,
                    recursive_result=recursive_result,
                    claim_check_passed=True,
                    total_token_usage=self._total_tokens,
                )

        return None

    def _retry_full_rebreakdown(
        self,
        lean_statement: str,
        statement_nl: str,
        search_result: ProofSearchResult,
        weak_feedback: list[CritiqueIssue],
        pipeline_start: float,
        nl_proof_context: NLProofSketch | None = None,
        tactic_hints: str = "",
    ) -> ProofPipelineResult | None:
        """Full-root re-breakdown fallback."""
        for attempt in range(self._MAX_WEAK_CHILD_RETRIES):
            log.info(
                "proof_pipeline_weak_child_retry",
                attempt=attempt + 1,
                weak_children=len(weak_feedback),
            )
            self._notify_progress(
                "Lemma Breakdown",
                f"Retrying decomposition (weak child feedback, attempt {attempt + 1})",
            )

            tree = self._run_lemma_breakdown(
                lean_statement, statement_nl, search_result,
                critic_feedback=weak_feedback,
                nl_proof_context=nl_proof_context,
                tactic_hints=tactic_hints,
            )
            if tree is None:
                return None

            if self._use_proof_detailer:
                tree = self._run_proof_detailer(tree)

            self._notify_progress("Leanification", "Re-converting lemmas to Lean 4")
            tree = self._run_lemma_leanifier(tree)
            if tree is None:
                return None

            has_axiom_nodes = any(n.from_prior_work for n in tree.nodes.values())
            if has_axiom_nodes:
                self._verify_axiom_nodes(tree, statement_nl)

            self._notify_progress("Recursive Prover", "Re-proving with revised decomposition")
            recursive_result = self._run_recursive_prover(tree)

            if recursive_result.proved and recursive_result.lemma_tree:
                self._notify_progress("Finalization", "Assembling final proof")
                final_proof = self._run_flatten_finalize(recursive_result.lemma_tree)
                if final_proof:
                    if self._use_claim_check:
                        passed = self._run_claim_check(lean_statement, final_proof)
                        if not passed:
                            return self._make_result(
                                statement=lean_statement,
                                search_result=search_result,
                                recursive_result=recursive_result,
                                claim_check_passed=False,
                                failure_stage="claim_check",
                                failure_reason="Claim check failed after weak-child retry",
                                total_token_usage=self._total_tokens,
                            )

                    log.info(
                        "proof_pipeline_weak_child_retry_success",
                        attempt=attempt + 1,
                        elapsed_seconds=round(time.monotonic() - pipeline_start, 3),
                    )
                    return self._make_result(
                        statement=lean_statement,
                        proved=True,
                        final_proof=final_proof,
                        search_result=search_result,
                        recursive_result=recursive_result,
                        claim_check_passed=True,
                        total_token_usage=self._total_tokens,
                    )

            new_feedback = self._extract_weak_child_feedback(recursive_result)
            if not new_feedback:
                break
            weak_feedback = new_feedback

        return None

    _BACKTRACK_TYPE_KEYWORDS = frozenset([
        "undefined structure", "undefined type", "unknown identifier",
        "type mismatch", "expected type", "has type", "definition",
        "structure", "inductive", "class", "instance",
    ])
    _BACKTRACK_NL_KEYWORDS = frozenset([
        "stuck_goal", "logical gap", "missing step",
    ])

    def _classify_backtrack_target(
        self, recursive_result: RecursiveProofResult,
    ) -> str | None:
        """Analyze failure diagnoses to decide which stage to backtrack to."""
        if not recursive_result.lemma_tree:
            return None

        failed_nodes = [
            n for n in recursive_result.lemma_tree.nodes.values()
            if n.failure_diagnosis is not None
        ]
        if not failed_nodes:
            return None

        type_signals = 0
        stuck_signals = 0
        for node in failed_nodes:
            diag = node.failure_diagnosis
            assert diag is not None
            combined = f"{diag.description} {diag.suggested_fix}".lower()
            if any(kw in combined for kw in self._BACKTRACK_TYPE_KEYWORDS):
                type_signals += 1
            if diag.failure_type == FailureType.STUCK_GOAL:
                stuck_signals += 1

        if type_signals > 0:
            return "type_formalization"
        if stuck_signals > len(failed_nodes) // 2 and self._use_nl_proof_stage:
            return "nl_proof"
        return None

    def _backtrack_to_type_formalization(
        self,
        lean_statement: str,
        statement_nl: str,
        search_result: ProofSearchResult,
        recursive_result: RecursiveProofResult,
        nl_sketch: NLProofSketch | None,
        tactic_hints: str,
        pipeline_start: float,
    ) -> ProofPipelineResult | None:
        """Re-run type formalization with failure context, then re-enter decomposition."""
        failure_context = []
        if recursive_result.lemma_tree:
            for n in recursive_result.lemma_tree.nodes.values():
                if n.failure_diagnosis:
                    failure_context.append(n.failure_diagnosis.description)

        augmented_nl = (
            f"{statement_nl}\n\n[Prior type formalization failed]\n"
            + "\n".join(failure_context[:5])
        )

        self._notify_progress("Backtrack", "Re-running type formalization")
        type_defs = self._run_type_first_formalization(augmented_nl)
        if not type_defs:
            return None

        self._lean_preamble = (
            f"{self._lean_preamble}\n\n{type_defs}"
            if self._lean_preamble
            else type_defs
        )

        if nl_sketch is None and self._nl_prover and self._use_nl_proof_stage:
            nl_sketch = self._run_nl_proof_stage(lean_statement, statement_nl)

        bt_tactic_hints = tactic_hints
        if nl_sketch and nl_sketch.proof_steps:
            detailer = ProofDetailer(llm_client=self._llm)
            bt_tactic_hints = detailer.detail_sketch(nl_sketch)
            self._accumulate_tokens(detailer.cumulative_tokens)

        tree = self._run_lemma_breakdown(
            lean_statement, statement_nl, search_result,
            nl_proof_context=nl_sketch,
            tactic_hints=bt_tactic_hints,
        )
        if tree is None:
            return None

        tree = self._run_lemma_leanifier(tree, tactic_hints=bt_tactic_hints)
        if tree is None:
            return None

        bt_result = self._run_recursive_prover(tree)
        if not bt_result.proved or not bt_result.lemma_tree:
            return None

        final_proof = self._run_flatten_finalize(bt_result.lemma_tree)
        if not final_proof:
            return None

        if self._use_claim_check and not self._run_claim_check(lean_statement, final_proof):
            return None

        log.info(
            "proof_pipeline_backtrack_type_success",
            elapsed_seconds=round(time.monotonic() - pipeline_start, 3),
        )
        return self._make_result(
            statement=lean_statement,
            proved=True,
            final_proof=final_proof,
            search_result=search_result,
            recursive_result=bt_result,
            claim_check_passed=True,
            total_token_usage=self._total_tokens,
        )

    def _backtrack_to_nl_proof(
        self,
        lean_statement: str,
        statement_nl: str,
        search_result: ProofSearchResult,
        recursive_result: RecursiveProofResult,
        pipeline_start: float,
    ) -> ProofPipelineResult | None:
        """Re-run NL proof stage with failure context, then re-enter decomposition."""
        if not self._nl_prover:
            return None

        failure_feedback = []
        if recursive_result.lemma_tree:
            for n in recursive_result.lemma_tree.nodes.values():
                if n.failure_diagnosis:
                    failure_feedback.append(n.failure_diagnosis.description)

        augmented_nl = (
            f"{statement_nl}\n\n[Prior proof attempt feedback]\n"
            + "\n".join(failure_feedback[:5])
        )

        self._notify_progress("Backtrack", "Re-running NL proof stage")
        nl_sketch = self._run_nl_proof_stage(lean_statement, augmented_nl)
        if nl_sketch is None or not nl_sketch.proof_steps:
            return None

        detailer = ProofDetailer(llm_client=self._llm)
        tactic_hints = detailer.detail_sketch(nl_sketch)
        self._accumulate_tokens(detailer.cumulative_tokens)

        tree = self._run_lemma_breakdown(
            lean_statement, statement_nl, search_result,
            nl_proof_context=nl_sketch,
            tactic_hints=tactic_hints,
        )
        if tree is None:
            return None

        tree = self._run_lemma_leanifier(tree, tactic_hints=tactic_hints)
        if tree is None:
            return None

        bt_result = self._run_recursive_prover(tree)
        if not bt_result.proved or not bt_result.lemma_tree:
            return None

        final_proof = self._run_flatten_finalize(bt_result.lemma_tree)
        if not final_proof:
            return None

        if self._use_claim_check and not self._run_claim_check(lean_statement, final_proof):
            return None

        log.info(
            "proof_pipeline_backtrack_nl_success",
            elapsed_seconds=round(time.monotonic() - pipeline_start, 3),
        )
        return self._make_result(
            statement=lean_statement,
            proved=True,
            final_proof=final_proof,
            search_result=search_result,
            recursive_result=bt_result,
            claim_check_passed=True,
            total_token_usage=self._total_tokens,
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
            lean_preamble=self._lean_preamble,
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
