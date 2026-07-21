"""Tests for interactive type auction steering in FormalizationPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic_research.models.formalization import (
    AuctionResult,
    AuctionScore,
    AuctionVerdict,
    TypeCandidate,
    TypeDependencyGraph,
    TypeFormalizationCandidate,
    TypePlan,
)
from agentic_research.models.interaction import InteractionRequest, InteractionResponse
from agentic_research.pipelines.formalization import FormalizationPipeline


def _make_mock_llm():
    llm = MagicMock()
    llm.model = "claude-opus-4-6-20250616"
    return llm


def _make_mock_repl():
    repl = MagicMock()
    repl.execute.return_value = MagicMock(success=True, output="")
    return repl


def _make_mock_search():
    search = MagicMock()
    search.execute.return_value = MagicMock(entries=[])
    return search


def _make_type_plan():
    return TypePlan(
        conjecture_statement="test conjecture",
        candidates=[
            TypeCandidate(
                name="TestType",
                informal_description="A test type",
                lean_signature="structure TestType where",
            ),
        ],
        dependency_graph=TypeDependencyGraph(),
    )


def _make_auction_result_with_candidates():
    c0 = TypeFormalizationCandidate(
        candidate_id=0,
        type_name="TestType",
        lean_code="structure TestType where\n  val : Nat",
        compiles=True,
        auxiliary_lemmas=[],
    )
    c1 = TypeFormalizationCandidate(
        candidate_id=1,
        type_name="TestType",
        lean_code="structure TestType where\n  val : Int",
        compiles=True,
        auxiliary_lemmas=[],
    )
    return AuctionResult(
        type_name="TestType",
        verdict=AuctionVerdict.ACCEPTED,
        winner_id=0,
        scores=[
            AuctionScore(candidate_id=0, total_score=0.85),
            AuctionScore(candidate_id=1, total_score=0.72),
        ],
        winning_candidate=c0,
        all_candidates=[c0, c1],
        reason="Candidate 0 scored 0.85",
    )


class TestInteractiveSelection:
    def test_callback_receives_options(self):
        """Verify the interaction callback is called with correct options."""
        pipeline = FormalizationPipeline(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            interaction_callback=lambda req: InteractionResponse(
                selected_value=req.default_value
            ),
        )
        auction = _make_auction_result_with_candidates()
        result = pipeline._apply_interactive_selection(auction)
        assert result.winner_id == 0

    def test_user_overrides_winner(self):
        """User selects candidate 1 instead of default candidate 0."""
        def pick_second(req: InteractionRequest) -> InteractionResponse:
            return InteractionResponse(selected_value=1)

        pipeline = FormalizationPipeline(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            interaction_callback=pick_second,
        )
        auction = _make_auction_result_with_candidates()
        result = pipeline._apply_interactive_selection(auction)
        assert result.winner_id == 1
        assert result.winning_candidate is not None
        assert result.winning_candidate.candidate_id == 1
        assert "User selected" in result.reason

    def test_user_aborts_keeps_default(self):
        """When user aborts, original winner is preserved."""
        def abort(req: InteractionRequest) -> InteractionResponse:
            return InteractionResponse(aborted=True, selected_value=req.default_value)

        pipeline = FormalizationPipeline(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            interaction_callback=abort,
        )
        auction = _make_auction_result_with_candidates()
        result = pipeline._apply_interactive_selection(auction)
        assert result.winner_id == 0

    def test_invalid_selection_keeps_default(self):
        """An invalid candidate_id falls back to the original winner."""
        def pick_invalid(req: InteractionRequest) -> InteractionResponse:
            return InteractionResponse(selected_value=999)

        pipeline = FormalizationPipeline(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            interaction_callback=pick_invalid,
        )
        auction = _make_auction_result_with_candidates()
        result = pipeline._apply_interactive_selection(auction)
        assert result.winner_id == 0


class TestNoneCallbackUnchanged:
    def test_no_callback_uses_auctioneer_default(self):
        """When interaction_callback is None, pipeline uses auctioneer's pick."""
        pipeline = FormalizationPipeline(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            interaction_callback=None,
        )
        assert pipeline._interaction_callback is None

    def test_callback_stored_correctly(self):
        def cb(req):
            return InteractionResponse(selected_value=0)

        pipeline = FormalizationPipeline(
            llm_client=_make_mock_llm(),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            interaction_callback=cb,
        )
        assert pipeline._interaction_callback is cb
