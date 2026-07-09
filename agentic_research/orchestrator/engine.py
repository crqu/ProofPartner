"""Central Orchestrator — manages the explore-conjecture-prove loop.

Takes a rough idea as input and produces a ResearchSessionResult by
routing between pipeline stages based on outcomes:

  Start: Explorer -> Conjecturer
  For each conjecture: Formalize -> IntentJudge -> CounterexampleSearch
  If plausible: ProofPipeline -> success or failure
  If disproved or proof failed: RefinementPipeline
  If refinement exhausted: back to Explorer for new directions
"""

from __future__ import annotations

import uuid

from agentic_research.agents.conjecturer import ConjectureGenerator
from agentic_research.agents.counterexample_searcher import CounterexampleSearcher
from agentic_research.agents.explorer import ExplorationAgent
from agentic_research.agents.informalizer import Informalizer
from agentic_research.agents.intent_judge import IntentJudge
from agentic_research.agents.llm_client import LLMClient
from agentic_research.logging import get_logger
from agentic_research.memory.session import ResearchSessionMemory
from agentic_research.models.agents import AgentContext, AgentStatus, TokenUsage
from agentic_research.models.research import ConjectureSet, ExplorationResult
from agentic_research.models.session import (
    ConjectureOutcome,
    OrchestratorConfig,
    PipelineStage,
    ResearchSessionResult,
    StageTokenUsage,
    TriedConjecture,
    compute_cost,
)
from agentic_research.models.verification import CounterexampleStatus, IntentVerdictType
from agentic_research.orchestrator.circuit_breaker import CircuitBreaker
from agentic_research.orchestrator.cost_tracker import CostTracker
from agentic_research.orchestrator.rollback import CheckpointManager
from agentic_research.orchestrator.state import PipelineStateMachine
from agentic_research.pipelines.formalization import FormalizationPipeline
from agentic_research.pipelines.proof import ProofPipeline
from agentic_research.pipelines.refinement import RefinementPipeline
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

log = get_logger(__name__)


class ResearchOrchestrator:
    """Central orchestrator for the explore-conjecture-prove loop."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        lean_search: LeanSearch,
        config: OrchestratorConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self._llm = llm_client
        self._repl = lean_repl
        self._search = lean_search
        self._config = config or OrchestratorConfig()
        self._session_id = session_id or uuid.uuid4().hex[:12]

        self._state_machine = PipelineStateMachine()
        self._memory = ResearchSessionMemory(self._session_id)
        self._checkpoint_mgr = CheckpointManager(
            session_id=self._session_id, persist=True
        )
        self._total_tokens = TokenUsage()
        self._stage_usages: list[StageTokenUsage] = []
        self._exploration_rounds = 0
        self._circuit_breaker = CircuitBreaker()
        self._cost_tracker = CostTracker()
        self._reasoning_cycles = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state_machine(self) -> PipelineStateMachine:
        return self._state_machine

    @property
    def memory(self) -> ResearchSessionMemory:
        return self._memory

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._checkpoint_mgr

    @property
    def total_tokens(self) -> TokenUsage:
        return self._total_tokens

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    def run(self, raw_idea: str) -> ResearchSessionResult:
        """Execute the full explore-conjecture-prove loop."""
        log.info("orchestrator_start", session_id=self._session_id, idea_len=len(raw_idea))
        self._state_machine.session_state.raw_idea = raw_idea

        try:
            self._run_loop(raw_idea)
        except Exception as exc:
            log.error("orchestrator_error", error=str(exc))
            if not self._state_machine.is_terminal:
                self._state_machine.transition(
                    PipelineStage.FAILED, reason=f"Unhandled error: {exc}"
                )

        return self._build_result(raw_idea)

    def resume_from_checkpoint(self, checkpoint_id: str) -> ResearchSessionResult:
        """Resume a session from a previously saved checkpoint.

        Loads the checkpoint (from in-memory list or disk), restores state
        machine position and session memory, then continues from the last
        completed stage.
        """
        checkpoint = self._checkpoint_mgr.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            checkpoint = CheckpointManager.load_checkpoint_from_disk(
                self._session_id, checkpoint_id
            )
        if checkpoint is None:
            log.error("resume_checkpoint_not_found", checkpoint_id=checkpoint_id)
            return self._build_result("")

        self._state_machine.restore(checkpoint.session_state.model_copy(deep=True))
        self._memory._data = checkpoint.memory.model_copy(deep=True)
        self._stage_usages = list(checkpoint.stage_token_usages)

        raw_idea = checkpoint.session_state.raw_idea
        log.info(
            "orchestrator_resumed",
            checkpoint_id=checkpoint_id,
            stage=checkpoint.stage.value,
            raw_idea_len=len(raw_idea),
        )

        try:
            self._run_loop(raw_idea)
        except Exception as exc:
            log.error("orchestrator_error_after_resume", error=str(exc))
            if not self._state_machine.is_terminal:
                self._state_machine.transition(
                    PipelineStage.FAILED, reason=f"Unhandled error after resume: {exc}"
                )

        return self._build_result(raw_idea)

    def _checkpoint_current_stage(self) -> None:
        self._checkpoint_mgr.create_checkpoint(
            self._state_machine, self._memory, self._stage_usages
        )

    def _run_loop(self, raw_idea: str) -> None:
        while not self._state_machine.is_terminal:
            if self._budget_exceeded():
                log.info("orchestrator_budget_exceeded")
                self._state_machine.transition(
                    PipelineStage.FAILED, reason="Budget limit exceeded"
                )
                self._checkpoint_current_stage()
                break

            if self._reasoning_cycles >= self._config.max_reasoning_cycles:
                log.info(
                    "orchestrator_max_reasoning_cycles",
                    cycles=self._reasoning_cycles,
                    limit=self._config.max_reasoning_cycles,
                )
                self._state_machine.transition(
                    PipelineStage.FAILED,
                    reason=f"Max reasoning cycles ({self._config.max_reasoning_cycles}) reached",
                )
                self._checkpoint_current_stage()
                break

            if self._circuit_breaker.is_open():
                log.info("orchestrator_circuit_breaker_open")
                self._state_machine.transition(
                    PipelineStage.FAILED, reason="Circuit breaker open — too many failures"
                )
                self._checkpoint_current_stage()
                break

            self._reasoning_cycles += 1
            stage = self._state_machine.current_stage

            self._checkpoint_current_stage()

            if stage == PipelineStage.EXPLORING:
                self._handle_exploring(raw_idea)
            elif stage == PipelineStage.CONJECTURING:
                self._handle_conjecturing(raw_idea)
            elif stage == PipelineStage.FORMALIZING:
                self._handle_formalizing(raw_idea)
            elif stage == PipelineStage.CHECKING_INTENT:
                self._handle_checking_intent(raw_idea)
            elif stage == PipelineStage.SEARCHING_COUNTEREXAMPLE:
                self._handle_searching_counterexample(raw_idea)
            elif stage == PipelineStage.PROVING:
                self._handle_proving(raw_idea)
            elif stage == PipelineStage.REFINING:
                self._handle_refining(raw_idea)

    def _handle_exploring(self, raw_idea: str) -> None:
        self._exploration_rounds += 1
        if self._exploration_rounds > self._config.max_exploration_rounds:
            self._state_machine.transition(
                PipelineStage.FAILED,
                reason=f"Max exploration rounds ({self._config.max_exploration_rounds}) reached",
            )
            return

        log.info("orchestrator_exploring", round=self._exploration_rounds)
        explorer = ExplorationAgent(
            llm_client=self._llm, lean_search=self._search
        )
        ctx = AgentContext(task=raw_idea)
        result = explorer.run(ctx)
        self._record_stage_usage(PipelineStage.EXPLORING, "exploration_agent", result.token_usage)

        if result.status != AgentStatus.SUCCESS or not result.result:
            self._circuit_breaker.record_failure()
            self._state_machine.transition(
                PipelineStage.FAILED, reason="Exploration failed"
            )
            return

        self._circuit_breaker.record_success()
        exploration = ExplorationResult.model_validate(result.result)
        for d in exploration.directions:
            self._memory.add_promising_direction(
                title=d.title,
                description=d.description,
                source=f"exploration_round_{self._exploration_rounds}",
                priority=1.0 - (d.ambition_level / 5.0),
            )

        self._state_machine.transition(
            PipelineStage.CONJECTURING, reason="Exploration complete"
        )
        self._state_machine.session_state.active_conjecture_index = None

    def _handle_conjecturing(self, raw_idea: str) -> None:
        log.info("orchestrator_conjecturing")
        generator = ConjectureGenerator(llm_client=self._llm)
        ctx = AgentContext(task=raw_idea)
        result = generator.run(ctx)
        self._record_stage_usage(PipelineStage.CONJECTURING, "conjecture_generator", result.token_usage)

        if result.status != AgentStatus.SUCCESS or not result.result:
            self._circuit_breaker.record_failure()
            self._state_machine.transition(
                PipelineStage.EXPLORING, reason="Conjecture generation failed"
            )
            return

        self._circuit_breaker.record_success()
        conjecture_set = ConjectureSet.model_validate(result.result)
        conjectures = conjecture_set.conjectures
        if not conjectures:
            self._state_machine.transition(
                PipelineStage.EXPLORING, reason="No conjectures generated"
            )
            return

        ranking = conjecture_set.ranking or list(range(len(conjectures)))
        found_untried = False

        for idx in ranking:
            if idx >= len(conjectures):
                continue
            conj = conjectures[idx]
            if self._memory.has_tried(conj.statement):
                continue
            if self._state_machine.session_state.conjectures_processed >= self._config.max_conjectures:
                break

            self._memory.record_conjecture(conj, ConjectureOutcome.PENDING)
            self._state_machine.session_state.active_conjecture_index = idx
            self._state_machine.session_state.conjectures_processed += 1
            self._active_conjecture = conj
            self._active_lean_statement = ""
            found_untried = True

            self._state_machine.transition(
                PipelineStage.FORMALIZING,
                reason=f"Processing conjecture: {conj.statement[:60]}",
                conjecture_index=idx,
            )
            break

        if not found_untried:
            if self._state_machine.session_state.conjectures_processed >= self._config.max_conjectures:
                self._state_machine.transition(
                    PipelineStage.FAILED,
                    reason=f"Max conjectures ({self._config.max_conjectures}) reached",
                )
            else:
                self._state_machine.transition(
                    PipelineStage.EXPLORING,
                    reason="All conjectures already tried",
                )

    def _handle_formalizing(self, raw_idea: str) -> None:
        conj = self._active_conjecture
        log.info("orchestrator_formalizing", conjecture=conj.statement[:60])

        pipeline = FormalizationPipeline(
            llm_client=self._llm, lean_repl=self._repl, lean_search=self._search
        )
        result = pipeline.run(conj.natural_language)

        if not result.success or not result.theorem:
            self._circuit_breaker.record_failure()
            self._memory.update_conjecture_outcome(
                conj.statement,
                ConjectureOutcome.FORMALIZATION_FAILED,
                failure_reason=result.failure_reason or "Formalization failed",
                stage_reached="formalizing",
            )
            self._state_machine.transition(
                PipelineStage.REFINING,
                reason=f"Formalization failed: {result.failure_reason}",
            )
            return

        self._circuit_breaker.record_success()
        self._active_lean_statement = result.theorem.lean_statement
        self._memory.update_conjecture_outcome(
            conj.statement,
            ConjectureOutcome.PENDING,
            lean_statement=result.theorem.lean_statement,
            stage_reached="checking_intent",
        )
        self._state_machine.transition(
            PipelineStage.CHECKING_INTENT, reason="Formalization succeeded"
        )

    def _handle_checking_intent(self, raw_idea: str) -> None:
        conj = self._active_conjecture
        lean_stmt = self._active_lean_statement
        log.info("orchestrator_checking_intent")

        informalizer = Informalizer(llm_client=self._llm)
        judge = IntentJudge(llm_client=self._llm, informalizer=informalizer)
        verdict = judge.judge(
            lean_code=lean_stmt,
            original_idea=raw_idea,
            conjecture=conj.natural_language,
        )
        self._record_stage_usage(
            PipelineStage.CHECKING_INTENT, "intent_judge", judge.cumulative_tokens
        )

        if verdict.overall_verdict == IntentVerdictType.INCORRECT:
            self._memory.update_conjecture_outcome(
                conj.statement,
                ConjectureOutcome.INTENT_MISMATCH,
                failure_reason=f"Intent mismatch: {'; '.join(verdict.all_concerns[:3])}",
                stage_reached="checking_intent",
            )
            self._state_machine.transition(
                PipelineStage.REFINING, reason="Intent verification failed"
            )
            return

        self._state_machine.transition(
            PipelineStage.SEARCHING_COUNTEREXAMPLE, reason="Intent verified"
        )

    def _handle_searching_counterexample(self, raw_idea: str) -> None:
        conj = self._active_conjecture
        lean_stmt = self._active_lean_statement
        log.info("orchestrator_searching_counterexample")

        searcher = CounterexampleSearcher(llm_client=self._llm, lean_repl=self._repl)
        cx_result = searcher.search(lean_code=lean_stmt, conjecture=conj.natural_language)
        self._record_stage_usage(
            PipelineStage.SEARCHING_COUNTEREXAMPLE,
            "counterexample_searcher",
            searcher.cumulative_tokens,
        )

        if cx_result.status == CounterexampleStatus.DISPROVED:
            desc = ""
            if cx_result.successful_counterexample:
                desc = cx_result.successful_counterexample.description
            self._memory.update_conjecture_outcome(
                conj.statement,
                ConjectureOutcome.DISPROVED,
                failure_reason=f"Counterexample: {desc}",
                stage_reached="searching_counterexample",
            )
            self._state_machine.transition(
                PipelineStage.REFINING, reason=f"Disproved: {desc}"
            )
            return

        self._state_machine.transition(
            PipelineStage.PROVING, reason="No counterexample found — plausible"
        )

    def _handle_proving(self, raw_idea: str) -> None:
        conj = self._active_conjecture
        lean_stmt = self._active_lean_statement
        log.info("orchestrator_proving", statement=lean_stmt[:60])

        pipeline = ProofPipeline(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_search=self._search,
            use_proof_critic=self._config.use_proof_critic,
            use_proof_detailer=self._config.use_proof_detailer,
        )
        result = pipeline.run(lean_stmt, conj.natural_language)

        if result.proved and result.final_proof:
            self._memory.update_conjecture_outcome(
                conj.statement,
                ConjectureOutcome.PROVED,
                proof_code=result.final_proof,
                stage_reached="proving",
            )
            log.info("orchestrator_proof_found", conjecture=conj.statement[:60])

            if self._has_more_conjectures_to_try():
                self._state_machine.transition(
                    PipelineStage.CONJECTURING,
                    reason="Proof found, trying more conjectures",
                )
            else:
                self._state_machine.transition(
                    PipelineStage.COMPLETE, reason="Proof found"
                )
            return

        self._memory.update_conjecture_outcome(
            conj.statement,
            ConjectureOutcome.PROOF_FAILED,
            failure_reason=result.failure_reason or "Proof search exhausted",
            stage_reached="proving",
        )
        self._state_machine.transition(
            PipelineStage.REFINING, reason="Proof search failed"
        )

    def _handle_refining(self, raw_idea: str) -> None:
        conj = self._active_conjecture
        log.info("orchestrator_refining", conjecture=conj.statement[:60])
        self._state_machine.session_state.refinements_attempted += 1

        if self._state_machine.session_state.refinements_attempted > self._config.max_refinements:
            log.info("orchestrator_refinement_limit_reached")
            if self._has_more_conjectures_to_try():
                self._state_machine.transition(
                    PipelineStage.CONJECTURING,
                    reason="Refinement limit reached, trying next conjecture",
                )
            else:
                self._state_machine.transition(
                    PipelineStage.EXPLORING,
                    reason="Refinement exhausted, exploring new directions",
                )
            return

        tc = self._find_tried_conjecture(conj.statement)
        failure_reason = tc.failure_reason if tc else "unknown"
        failure_outcome = tc.outcome.value if tc else "proof_failed"

        pipeline = RefinementPipeline(
            llm_client=self._llm,
            lean_repl=self._repl,
            lean_search=self._search,
            max_depth=1,
            generate_report=False,
        )
        result = pipeline.run(
            conjecture=conj,
            failure_reason=failure_reason,
            failure_outcome=failure_outcome,
            original_idea=raw_idea,
        )
        self._accumulate_tokens(result.total_token_usage)

        from agentic_research.models.refinement import RefinementStatus

        if result.status == RefinementStatus.PROVED and result.proved_variant:
            self._memory.record_conjecture(
                result.proved_variant,
                ConjectureOutcome.PROVED,
                proof_code=result.proof_code,
                stage_reached="refining",
            )
            log.info("orchestrator_refinement_proved")
            self._state_machine.transition(
                PipelineStage.COMPLETE, reason="Refined conjecture proved"
            )
            return

        if self._has_more_conjectures_to_try():
            self._state_machine.transition(
                PipelineStage.CONJECTURING,
                reason="Refinement exhausted, trying next conjecture",
            )
        else:
            self._state_machine.transition(
                PipelineStage.EXPLORING,
                reason="Refinement exhausted, exploring new directions",
            )

    def _has_more_conjectures_to_try(self) -> bool:
        return (
            self._state_machine.session_state.conjectures_processed
            < self._config.max_conjectures
        )

    def _find_tried_conjecture(self, statement: str) -> TriedConjecture | None:
        for tc in self._memory.data.tried_conjectures:
            if tc.conjecture.statement == statement:
                return tc
        return None

    def _record_stage_usage(
        self, stage: PipelineStage, agent_name: str, usage: TokenUsage
    ) -> None:
        self._stage_usages.append(StageTokenUsage(
            stage=stage, agent_name=agent_name, token_usage=usage
        ))
        self._accumulate_tokens(usage)
        self._cost_tracker.record_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=self._llm.model,
            cache_read_tokens=usage.cache_read_input_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens,
        )
        log.info(
            "stage_token_usage",
            stage=stage.value,
            agent=agent_name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    def _accumulate_tokens(self, usage: TokenUsage) -> None:
        self._total_tokens.input_tokens += usage.input_tokens
        self._total_tokens.output_tokens += usage.output_tokens
        self._total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self._total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens

    def _budget_exceeded(self) -> bool:
        if self._config.budget_limit_usd is None:
            return False
        cost = compute_cost(self._total_tokens)
        return cost.total_cost_usd >= self._config.budget_limit_usd

    def _build_result(self, raw_idea: str) -> ResearchSessionResult:
        cost = compute_cost(self._total_tokens)
        proved = self._memory.data.proved_conjectures()
        failed = self._memory.data.failed_conjectures()

        return ResearchSessionResult(
            session_id=self._session_id,
            raw_idea=raw_idea,
            proved_conjectures=proved,
            failed_conjectures=failed,
            partial_results=list(self._memory.data.partial_results),
            total_token_usage=self._total_tokens,
            cost_estimate=cost,
            final_stage=self._state_machine.current_stage,
            total_conjectures_tried=self._state_machine.session_state.conjectures_processed,
            total_refinements=self._state_machine.session_state.refinements_attempted,
            exploration_rounds=self._exploration_rounds,
        )
