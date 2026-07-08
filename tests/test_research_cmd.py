"""Tests for the 'research' CLI command.

All LLM calls and orchestrator internals are mocked — no real API calls.
Uses click.testing.CliRunner for CLI invocation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentic_research.cli.main import cli
from agentic_research.models.agents import TokenUsage
from agentic_research.models.session import (
    CostEstimate,
    PipelineStage,
    ResearchSessionResult,
    TriedConjecture,
)
from agentic_research.models.research import Conjecture


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_session_dir(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr("agentic_research.cli.main.SESSION_DIR", session_dir)
    return session_dir


def _make_result(
    *,
    final_stage: PipelineStage = PipelineStage.COMPLETE,
    proved: list[TriedConjecture] | None = None,
    failed: list[TriedConjecture] | None = None,
    session_id: str = "test-session-abc",
    total_conjectures: int = 3,
    total_refinements: int = 1,
    exploration_rounds: int = 1,
) -> ResearchSessionResult:
    return ResearchSessionResult(
        session_id=session_id,
        raw_idea="test idea",
        proved_conjectures=proved or [],
        failed_conjectures=failed or [],
        partial_results=[],
        total_token_usage=TokenUsage(input_tokens=1000, output_tokens=500),
        cost_estimate=CostEstimate(
            input_tokens=1000,
            output_tokens=500,
            input_cost_usd=0.015,
            output_cost_usd=0.0375,
        ),
        final_stage=final_stage,
        total_conjectures_tried=total_conjectures,
        total_refinements=total_refinements,
        exploration_rounds=exploration_rounds,
    )


def _patch_setup():
    """Return context managers that mock LLM, Lean REPL, and Lean search creation."""
    return (
        patch("agentic_research.cli.main._create_llm_client", return_value=MagicMock()),
        patch("agentic_research.cli.main._create_lean_repl", return_value=MagicMock()),
        patch("agentic_research.cli.main._create_lean_search", return_value=MagicMock()),
    )


class TestResearchArgParsing:
    def test_default_budget(self, runner):
        result = runner.invoke(cli, ["research", "--help"])
        assert result.exit_code == 0
        assert "20.00" in result.output

    def test_default_max_conjectures(self, runner):
        result = runner.invoke(cli, ["research", "--help"])
        assert "max-conjectures" in result.output

    def test_default_max_refinements(self, runner):
        result = runner.invoke(cli, ["research", "--help"])
        assert "max-refinements" in result.output

    def test_research_appears_in_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "research" in result.output

    def test_custom_budget_parsed(self, runner, tmp_session_dir):
        mock_result = _make_result()
        p1, p2, p3 = _patch_setup()
        with p1, p2, p3, patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator.run",
            return_value=mock_result,
        ), patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator._build_result",
            return_value=mock_result,
        ):
            result = runner.invoke(
                cli,
                ["research", "test idea", "--budget", "5.00"],
                input="y\n",
            )
        assert result.exit_code == 0


class TestResearchConfirmation:
    def test_decline_aborts(self, runner, tmp_session_dir):
        result = runner.invoke(cli, ["research", "test idea"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output

    def test_confirm_proceeds(self, runner, tmp_session_dir):
        mock_result = _make_result()
        p1, p2, p3 = _patch_setup()
        with p1, p2, p3, patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator.run",
            return_value=mock_result,
        ), patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator._build_result",
            return_value=mock_result,
        ):
            result = runner.invoke(
                cli, ["research", "test idea"], input="y\n"
            )
        assert result.exit_code == 0
        assert "RESEARCH COMPLETE" in result.output


class TestResearchResultDisplay:
    def test_complete_shows_success(self, runner, tmp_session_dir):
        mock_result = _make_result(final_stage=PipelineStage.COMPLETE)
        p1, p2, p3 = _patch_setup()
        with p1, p2, p3, patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator.run",
            return_value=mock_result,
        ), patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator._build_result",
            return_value=mock_result,
        ):
            result = runner.invoke(
                cli, ["research", "test idea"], input="y\n"
            )
        assert "RESEARCH COMPLETE" in result.output
        assert "Research Results" in result.output

    def test_failed_shows_incomplete(self, runner, tmp_session_dir):
        mock_result = _make_result(final_stage=PipelineStage.FAILED)
        p1, p2, p3 = _patch_setup()
        with p1, p2, p3, patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator.run",
            return_value=mock_result,
        ), patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator._build_result",
            return_value=mock_result,
        ):
            result = runner.invoke(
                cli, ["research", "test idea"], input="y\n"
            )
        assert "RESEARCH INCOMPLETE" in result.output

    def test_displays_stats_table(self, runner, tmp_session_dir):
        mock_result = _make_result(total_conjectures=4, total_refinements=2)
        p1, p2, p3 = _patch_setup()
        with p1, p2, p3, patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator.run",
            return_value=mock_result,
        ), patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator._build_result",
            return_value=mock_result,
        ):
            result = runner.invoke(
                cli, ["research", "test idea"], input="y\n"
            )
        assert "Stage reached" in result.output
        assert "Conjectures tried" in result.output
        assert "Total cost" in result.output

    def test_displays_proved_conjectures(self, runner, tmp_session_dir):
        proved = [
            TriedConjecture(
                conjecture=Conjecture(
                    statement="Every even number > 2 is the sum of two primes",
                    natural_language="Goldbach's conjecture",
                    confidence=0.9,
                    difficulty=5,
                ),
                lean_statement="theorem goldbach : True := trivial",
            ),
        ]
        mock_result = _make_result(proved=proved)
        p1, p2, p3 = _patch_setup()
        with p1, p2, p3, patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator.run",
            return_value=mock_result,
        ), patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator._build_result",
            return_value=mock_result,
        ):
            result = runner.invoke(
                cli, ["research", "test idea"], input="y\n"
            )
        assert "Proved Conjectures" in result.output
        assert "goldbach" in result.output

    def test_displays_session_resume_hint(self, runner, tmp_session_dir):
        mock_result = _make_result(session_id="sess-12345")
        p1, p2, p3 = _patch_setup()
        with p1, p2, p3, patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator.run",
            return_value=mock_result,
        ), patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator._build_result",
            return_value=mock_result,
        ):
            result = runner.invoke(
                cli, ["research", "test idea"], input="y\n"
            )
        assert "sess-12345" in result.output
        assert "resume" in result.output


class TestResearchKeyboardInterrupt:
    def test_keyboard_interrupt_shows_partial(self, runner, tmp_session_dir):
        mock_result = _make_result(final_stage=PipelineStage.EXPLORING)
        p1, p2, p3 = _patch_setup()
        with p1, p2, p3, patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator.run",
            side_effect=KeyboardInterrupt,
        ), patch(
            "agentic_research.orchestrator.engine.ResearchOrchestrator._build_result",
            return_value=mock_result,
        ):
            result = runner.invoke(
                cli, ["research", "test idea"], input="y\n"
            )
        assert "Interrupted" in result.output
        assert "Research Results" in result.output


class TestResearchSetupError:
    def test_setup_error_exits(self, runner, tmp_session_dir):
        with patch(
            "agentic_research.cli.main._create_llm_client",
            side_effect=Exception("ANTHROPIC_API_KEY not set"),
        ):
            result = runner.invoke(
                cli, ["research", "test idea"], input="y\n"
            )
        assert result.exit_code != 0
        assert "Setup error" in result.output
