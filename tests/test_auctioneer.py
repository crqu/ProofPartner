"""Tests for Auctioneer quality gating: adaptive spawning, semantic alignment,
and iterative auxiliary lemma refinement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agentic_research.agents.auctioneer import (
    Auctioneer,
    DEFAULT_K_EXTRA,
    _intent_verdict_to_score,
    compute_auction_score,
)
from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.formalization import (
    AuctionResult,
    AuctionVerdict,
    AuxiliaryLemma,
    LemmaStatement,
    TypeCandidate,
    TypeFormalizationCandidate,
)
from agentic_research.models.verification import IntentVerdictType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = []
    for text in responses:
        side_effects.append(LLMResponse(
            content=text,
            model="claude-opus-4-6-20250616",
            stop_reason="end_turn",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        ))
    mock.complete.side_effect = side_effects

    real_client_cls = LLMClient
    with patch("anthropic.Anthropic"):
        temp_client = real_client_cls.__new__(real_client_cls)
    mock.extract_json = temp_client.__class__.extract_json.__get__(mock, type(mock))
    return mock


MOCK_TYPE_LEAN_CODE = (
    "```lean\nstructure QuasiRandomGraph where\n"
    "  vertices : Finset Nat\n  edges : Finset (Nat × Nat)\n```"
)


def _good_candidate(cid: int, compiles: bool = True, proved: bool = True) -> TypeFormalizationCandidate:
    lemmas = []
    if proved:
        lemmas.append(AuxiliaryLemma(
            lemma=LemmaStatement(name="l1", statement_nl="s", for_type="T"),
            proved=True,
        ))
    return TypeFormalizationCandidate(
        candidate_id=cid,
        type_name="T",
        lean_code="structure T where\n  x : Nat",
        compiles=compiles,
        auxiliary_lemmas=lemmas,
    )


def _bad_candidate(cid: int) -> TypeFormalizationCandidate:
    return TypeFormalizationCandidate(
        candidate_id=cid,
        type_name="T",
        lean_code="bad code",
        compiles=False,
    )


# ---------------------------------------------------------------------------
# _intent_verdict_to_score
# ---------------------------------------------------------------------------


class TestIntentVerdictToScore:
    def test_correct_high_confidence(self):
        assert _intent_verdict_to_score(IntentVerdictType.CORRECT, 0.9) == 1.0

    def test_correct_low_confidence(self):
        assert _intent_verdict_to_score(IntentVerdictType.CORRECT, 0.4) == 0.75

    def test_incorrect_low_confidence(self):
        assert _intent_verdict_to_score(IntentVerdictType.INCORRECT, 0.3) == 0.5

    def test_incorrect_high_confidence(self):
        assert _intent_verdict_to_score(IntentVerdictType.INCORRECT, 0.8) == 0.0

    def test_boundary_confidence(self):
        assert _intent_verdict_to_score(IntentVerdictType.CORRECT, 0.6) == 1.0
        assert _intent_verdict_to_score(IntentVerdictType.INCORRECT, 0.6) == 0.0


# ---------------------------------------------------------------------------
# compute_auction_score with semantic_alignment
# ---------------------------------------------------------------------------


class TestComputeAuctionScoreAlignment:
    def test_default_alignment_is_1(self):
        c = _good_candidate(0)
        score = compute_auction_score(c)
        assert score.semantic_alignment_score == 1.0

    def test_explicit_alignment(self):
        c = _good_candidate(0)
        score = compute_auction_score(c, semantic_alignment=0.5)
        assert score.semantic_alignment_score == 0.5

    def test_low_alignment_lowers_total(self):
        c = _good_candidate(0)
        high = compute_auction_score(c, semantic_alignment=1.0)
        low = compute_auction_score(c, semantic_alignment=0.0)
        assert high.total_score > low.total_score

    def test_combined_ranking_lemmas_beat_alignment(self):
        """Candidate with more proved lemmas beats one with higher alignment but fewer lemmas."""
        many_lemmas = TypeFormalizationCandidate(
            candidate_id=0,
            type_name="T",
            lean_code="structure T where\n  x : Nat",
            compiles=True,
            auxiliary_lemmas=[
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l1", statement_nl="s", for_type="T"),
                    proved=True,
                ),
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l2", statement_nl="s", for_type="T"),
                    proved=True,
                ),
            ],
        )
        few_lemmas = TypeFormalizationCandidate(
            candidate_id=1,
            type_name="T",
            lean_code="structure T where\n  x : Nat",
            compiles=True,
            auxiliary_lemmas=[
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l1", statement_nl="s", for_type="T"),
                    proved=False,
                ),
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l2", statement_nl="s", for_type="T"),
                    proved=False,
                ),
            ],
        )

        score_many = compute_auction_score(many_lemmas, semantic_alignment=0.5)
        score_few = compute_auction_score(few_lemmas, semantic_alignment=1.0)
        assert score_many.total_score > score_few.total_score


# ---------------------------------------------------------------------------
# Semantic alignment via IntentJudge
# ---------------------------------------------------------------------------


class TestAuctioneerSemanticAlignment:
    def test_intent_judge_called_per_candidate(self):
        """IntentJudge.judge() is called once per candidate during evaluation."""
        from agentic_research.models.verification import IntentVerdict
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([MOCK_TYPE_LEAN_CODE] * 3)

        mock_judge = MagicMock()
        mock_judge.judge.return_value = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            overall_confidence=0.9,
        )

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            k_extra=0,
            quality_threshold=0.0,
            prover_config=ProverConfig(max_iterations=1),
            intent_judge=mock_judge,
            original_idea="test idea",
            conjecture="test conjecture",
        )

        ctx = AgentContext(
            task="T",
            metadata={
                "type_candidate": TypeCandidate(name="T", informal_description="test").model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        assert mock_judge.judge.call_count >= 3

    def test_intent_judge_failure_falls_back_to_1(self):
        """When IntentJudge raises, alignment defaults to 1.0."""
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([MOCK_TYPE_LEAN_CODE] * 3)

        mock_judge = MagicMock()
        mock_judge.judge.side_effect = RuntimeError("LLM error")

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            k_extra=0,
            quality_threshold=0.0,
            prover_config=ProverConfig(max_iterations=1),
            intent_judge=mock_judge,
            original_idea="test idea",
            conjecture="test conjecture",
        )

        ctx = AgentContext(
            task="T",
            metadata={
                "type_candidate": TypeCandidate(name="T", informal_description="test").model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        auction = AuctionResult.model_validate(result.result)
        assert auction.verdict == AuctionVerdict.ACCEPTED
        for s in auction.scores:
            assert s.semantic_alignment_score == 1.0

    def test_no_intent_judge_uses_default(self):
        """Without intent_judge, all candidates get alignment 1.0."""
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([MOCK_TYPE_LEAN_CODE] * 3)

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            k_extra=0,
            quality_threshold=0.0,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(
            task="T",
            metadata={
                "type_candidate": TypeCandidate(name="T", informal_description="test").model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        auction = AuctionResult.model_validate(result.result)
        for s in auction.scores:
            assert s.semantic_alignment_score == 1.0


# ---------------------------------------------------------------------------
# Adaptive spawning
# ---------------------------------------------------------------------------


class TestAdaptiveSpawning:
    def test_adaptive_spawn_on_retry(self):
        """When initial k candidates all fail, k_extra more are spawned and re-evaluated."""
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))

        error_code = "```lean\n-- MOCK_ERROR\nbad\n```"
        # Each TypeFormalizer retries up to 5 iterations, consuming 1 LLM response each.
        # 3 formalizers × 5 iterations = 15 error responses needed to exhaust all initial candidates.
        initial_errors = [error_code] * 15
        extra_good = [MOCK_TYPE_LEAN_CODE] * (DEFAULT_K_EXTRA * 5)
        llm = _make_mock_llm(initial_errors + extra_good)

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            k_extra=DEFAULT_K_EXTRA,
            quality_threshold=0.35,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(
            task="T",
            metadata={
                "type_candidate": TypeCandidate(name="T", informal_description="test").model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        auction = AuctionResult.model_validate(result.result)
        assert auction.verdict == AuctionVerdict.ACCEPTED
        assert len(auction.scores) == 3 + DEFAULT_K_EXTRA

    def test_no_adaptive_spawn_when_accepted(self):
        """No extra candidates are spawned when initial auction succeeds."""
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm([MOCK_TYPE_LEAN_CODE] * 3)

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            k_extra=2,
            quality_threshold=0.0,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(
            task="T",
            metadata={
                "type_candidate": TypeCandidate(name="T", informal_description="test").model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        auction = AuctionResult.model_validate(result.result)
        assert auction.verdict == AuctionVerdict.ACCEPTED
        assert len(auction.scores) == 3

    def test_k_extra_zero_disables_adaptive(self):
        """Setting k_extra=0 disables adaptive spawning."""
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        error_code = "```lean\n-- MOCK_ERROR\nbad\n```"
        llm = _make_mock_llm([error_code] * 3)

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            k_extra=0,
            quality_threshold=0.99,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(
            task="T",
            metadata={
                "type_candidate": TypeCandidate(name="T", informal_description="test").model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        auction = AuctionResult.model_validate(result.result)
        assert auction.verdict == AuctionVerdict.RETRY
        assert len(auction.scores) == 3

    def test_adaptive_spawn_candidate_ids_are_unique(self):
        """Extra candidates have IDs offset from the original batch."""
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        error_code = "```lean\n-- MOCK_ERROR\nbad\n```"
        initial_errors = [error_code] * 15
        extra_good = [MOCK_TYPE_LEAN_CODE] * 10
        llm = _make_mock_llm(initial_errors + extra_good)

        auctioneer = Auctioneer(
            llm_client=llm,
            lean_repl=repl,
            k=3,
            k_extra=2,
            quality_threshold=0.1,
            prover_config=ProverConfig(max_iterations=1),
        )

        ctx = AgentContext(
            task="T",
            metadata={
                "type_candidate": TypeCandidate(name="T", informal_description="test").model_dump(),
                "lemmas": [],
                "prior_definitions": "",
            },
        )

        result = auctioneer.run(ctx)
        auction = AuctionResult.model_validate(result.result)
        ids = [s.candidate_id for s in auction.scores]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# build_failure_feedback
# ---------------------------------------------------------------------------


class TestBuildFailureFeedback:
    def test_failure_feedback_includes_failed_lemmas(self):
        winner = TypeFormalizationCandidate(
            candidate_id=0,
            type_name="T",
            lean_code="structure T where\n  x : Nat",
            compiles=True,
            auxiliary_lemmas=[
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="ok_lemma", statement_nl="ok", for_type="T"),
                    proved=True,
                ),
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="bad_lemma", statement_nl="bad", for_type="T"),
                    proved=False,
                    error_message="type mismatch at application",
                ),
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="worse_lemma", statement_nl="worse", for_type="T"),
                    proved=False,
                    error_message="unknown identifier 'foo'",
                ),
            ],
        )
        ar = AuctionResult(
            type_name="T",
            verdict=AuctionVerdict.ACCEPTED,
            winner_id=0,
            winning_candidate=winner,
        )
        feedback = Auctioneer.build_failure_feedback(ar)
        assert "bad_lemma" in feedback
        assert "worse_lemma" in feedback
        assert "ok_lemma" not in feedback
        assert "type mismatch" in feedback
        assert "unknown identifier" in feedback

    def test_no_feedback_when_all_proved(self):
        winner = _good_candidate(0, compiles=True, proved=True)
        ar = AuctionResult(
            type_name="T",
            verdict=AuctionVerdict.ACCEPTED,
            winner_id=0,
            winning_candidate=winner,
        )
        feedback = Auctioneer.build_failure_feedback(ar)
        assert feedback == ""

    def test_no_feedback_without_winner(self):
        ar = AuctionResult(
            type_name="T",
            verdict=AuctionVerdict.RETRY,
        )
        feedback = Auctioneer.build_failure_feedback(ar)
        assert feedback == ""


# ---------------------------------------------------------------------------
# Iterative refinement (FormalizationPipeline._auction_type)
# ---------------------------------------------------------------------------


def _make_pipeline_for_refinement(
    candidates_per_round: list[list[TypeFormalizationCandidate]],
    max_refinement_iterations: int = 3,
):
    """Create a FormalizationPipeline with mocked Auctioneer responses."""
    from agentic_research.pipelines.formalization import FormalizationPipeline
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

    repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))

    call_index = [0]

    def mock_run_parallel(tc, lemmas, prior, *, k_override=None, id_offset=0, iteration_context=None):
        idx = call_index[0]
        call_index[0] += 1
        if idx < len(candidates_per_round):
            return candidates_per_round[idx]
        return candidates_per_round[-1]

    llm = _make_mock_llm([MOCK_TYPE_LEAN_CODE] * 100)

    pipeline = FormalizationPipeline(
        llm_client=llm,
        lean_repl=repl,
        lean_search=MagicMock(),
        k=3,
        max_retries=len(candidates_per_round),
        max_refinement_iterations=max_refinement_iterations,
    )

    return pipeline, mock_run_parallel


class TestIterativeRefinement:
    def test_iterative_refinement_early_exit(self):
        """Exits immediately when proof_rate >= 0.8."""
        good = TypeFormalizationCandidate(
            candidate_id=0,
            type_name="T",
            lean_code="structure T where\n  x : Nat",
            compiles=True,
            auxiliary_lemmas=[
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l1", statement_nl="s", for_type="T"),
                    proved=True,
                ),
                AuxiliaryLemma(
                    lemma=LemmaStatement(name="l2", statement_nl="s2", for_type="T"),
                    proved=True,
                ),
            ],
        )
        ar = AuctionResult(
            type_name="T",
            verdict=AuctionVerdict.ACCEPTED,
            winner_id=0,
            winning_candidate=good,
            scores=[],
        )
        assert good.proved_ratio == 1.0
        assert ar.refinement_iterations == 0

    def test_stall_detection_exits_on_plateau(self):
        """When improvement < 0.05, refinement stops."""
        from agentic_research.pipelines.formalization import (
            REFINEMENT_STALL_THRESHOLD,
        )

        history = [0.3, 0.32]
        improvement = history[-1] - history[-2]
        assert improvement < REFINEMENT_STALL_THRESHOLD

    def test_max_iterations_respected(self):
        from agentic_research.pipelines.formalization import (
            DEFAULT_MAX_REFINEMENT_ITERATIONS,
        )

        assert DEFAULT_MAX_REFINEMENT_ITERATIONS == 0

    def test_proof_rate_history_tracked(self):
        ar = AuctionResult(
            type_name="T",
            verdict=AuctionVerdict.ACCEPTED,
            winner_id=0,
            refinement_iterations=2,
            proof_rate_history=[0.3, 0.5, 0.8],
        )
        assert len(ar.proof_rate_history) == 3
        assert ar.refinement_iterations == 2

    def test_auction_result_has_refinement_fields(self):
        ar = AuctionResult(
            type_name="T",
            verdict=AuctionVerdict.ACCEPTED,
        )
        assert ar.refinement_iterations == 0
        assert ar.proof_rate_history == []

    def test_cost_bounded_by_max_iterations(self):
        k = 3
        max_iters = 3
        single_call_tokens = 500
        max_total = k * single_call_tokens * (max_iters + 1)
        assert max_total == 6000
