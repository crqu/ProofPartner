"""Integration tests for the CLI commands.

All LLM calls and Lean backends are mocked — no real API calls are made.
Uses click.testing.CliRunner for CLI invocation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentic_research.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_session_dir(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr("agentic_research.cli.main.SESSION_DIR", session_dir)
    return session_dir


def _mock_llm_response(text: str = "mock response") -> MagicMock:
    text_block = MagicMock(type="text", text=text)
    response = MagicMock()
    response.content = [text_block]
    response.model = "claude-opus-4-6-20250616"
    response.stop_reason = "end_turn"
    response.usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return response


def _patch_anthropic():
    mock_response = _mock_llm_response(
        json.dumps({
            "concepts": [{"name": "prime numbers", "domain": "number theory"}],
            "directions": [
                {
                    "title": "Prime gaps",
                    "description": "Study gaps between consecutive primes",
                    "ambition_level": 3,
                }
            ],
            "domain": "number theory",
            "known_results": ["Prime Number Theorem"],
        })
    )
    patcher = patch("anthropic.Anthropic")
    mock_anthropic = patcher.start()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client
    return patcher, mock_client


class TestHelp:
    def test_help_shows_all_commands(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "explore" in result.output
        assert "formalize" in result.output
        assert "check" in result.output
        assert "prove" in result.output
        assert "status" in result.output
        assert "eval" in result.output
        assert "resume" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


def _mock_agent_result(*, success: bool = True, result_data: dict | None = None, error: str | None = None):
    from agentic_research.models.agents import AgentResult, AgentStatus, TokenUsage

    return AgentResult(
        agent_name="mock_agent",
        status=AgentStatus.SUCCESS if success else AgentStatus.FAILURE,
        result=result_data,
        error_message=error,
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


def _mock_token_usage():
    from agentic_research.models.agents import TokenUsage

    return TokenUsage(input_tokens=100, output_tokens=50)


class TestExploreCommand:
    def test_explore_happy_path(self, runner, tmp_session_dir):
        explore_data = {
            "raw_idea": "prime gaps",
            "domain": "number theory",
            "concepts": [{"name": "prime numbers"}],
            "directions": [
                {"title": "Gaps", "description": "Study gaps", "ambition_level": 3}
            ],
        }
        conj_data = {
            "conjectures": [
                {
                    "statement": "Every even number > 2 is the sum of two primes",
                    "natural_language": "Goldbach's conjecture",
                    "confidence": 0.95,
                    "difficulty": 5,
                    "novelty_score": 0.3,
                    "formalizability_score": 0.7,
                }
            ],
            "ranking": [0],
            "exploration_context": explore_data,
        }

        explore_result = _mock_agent_result(result_data=explore_data)
        conj_result = _mock_agent_result(result_data=conj_data)

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.agents.explorer.ExplorationAgent.run", return_value=explore_result),
            patch("agentic_research.agents.conjecturer.ConjectureGenerator.run", return_value=conj_result),
            patch("agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens", new_callable=lambda: property(lambda self: _mock_token_usage())),
            patch("agentic_research.agents.conjecturer.ConjectureGenerator.cumulative_tokens", new_callable=lambda: property(lambda self: _mock_token_usage())),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["explore", "prime gaps"])

        assert result.exit_code == 0
        assert "Cost Summary" in result.output
        assert "sum of two primes" in result.output

    def test_explore_budget_enforcement(self, runner, tmp_session_dir):
        explore_result = _mock_agent_result(result_data={"domain": "test"})

        mock_tokens = MagicMock()
        mock_tokens.input_tokens = 500_000
        mock_tokens.output_tokens = 500_000
        mock_tokens.cache_read_input_tokens = 0
        mock_tokens.cache_creation_input_tokens = 0

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.agents.explorer.ExplorationAgent.run", return_value=explore_result),
            patch("agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens", new_callable=lambda: property(lambda self: mock_tokens)),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["explore", "primes", "--budget", "0.01"])

        assert "Budget" in result.output or "budget" in result.output.lower() or "exceeded" in result.output.lower()

    def test_explore_setup_error_no_api_key(self, runner, tmp_session_dir):
        with patch(
            "agentic_research.cli.main._create_llm_client",
            side_effect=Exception("ANTHROPIC_API_KEY not set"),
        ):
            result = runner.invoke(cli, ["explore", "test idea"])

        assert result.exit_code != 0
        assert "Setup error" in result.output or "error" in result.output.lower()


class TestFormalizeCommand:
    def test_formalize_happy_path(self, runner, tmp_session_dir):
        from agentic_research.models.formalization import (
            ClaimCheckResult,
            ClaimCheckVerdict,
            FormalizationPipelineResult,
            TheoremFormalization,
        )
        from agentic_research.models.verification import (
            IntentVerdict,
            IntentVerdictType,
            PathVerdict,
            VerificationPath,
        )

        form_result = FormalizationPipelineResult(
            conjecture_nl="test conjecture",
            theorem=TheoremFormalization(
                conjecture_nl="test conjecture",
                lean_statement="theorem test : True := trivial",
                compiles=True,
            ),
            claim_check=ClaimCheckResult(
                verdict=ClaimCheckVerdict.PASS,
                original_statement="test",
                formalized_statement="test",
            ),
            success=True,
        )

        intent_verdict = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            path_verdicts=[
                PathVerdict(
                    path=VerificationPath.BLIND,
                    verdict=IntentVerdictType.CORRECT,
                    confidence=0.9,
                ),
            ],
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.formalization.FormalizationPipeline.run", return_value=form_result),
            patch("agentic_research.agents.intent_judge.IntentJudge.judge", return_value=intent_verdict),
            patch("agentic_research.agents.intent_judge.IntentJudge.cumulative_tokens", new_callable=lambda: property(lambda self: MagicMock(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0))),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["formalize", "every prime > 2 is odd"])

        assert result.exit_code == 0
        assert "CORRECT" in result.output
        assert "theorem test" in result.output
        assert "Cost Summary" in result.output

    def test_formalize_failure(self, runner, tmp_session_dir):
        from agentic_research.models.formalization import FormalizationPipelineResult

        form_result = FormalizationPipelineResult(
            conjecture_nl="bad conjecture",
            success=False,
            failure_reason="Could not parse mathematical content",
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.formalization.FormalizationPipeline.run", return_value=form_result),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["formalize", "gibberish"])

        assert result.exit_code != 0
        assert "failed" in result.output.lower() or "error" in result.output.lower()


class TestCheckCommand:
    def test_check_plausible(self, runner, tmp_session_dir):
        from agentic_research.models.verification import CounterexampleResult, CounterexampleStatus

        cx_result = CounterexampleResult(
            status=CounterexampleStatus.PLAUSIBLE,
            attempts_made=5,
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search", return_value=cx_result),
            patch("agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens", new_callable=lambda: property(lambda self: MagicMock(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0))),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["check", "theorem test : True := trivial"])

        assert result.exit_code == 0
        assert "PLAUSIBLE" in result.output

    def test_check_disproved(self, runner, tmp_session_dir):
        from agentic_research.models.verification import (
            CounterexampleCandidate,
            CounterexampleResult,
            CounterexampleStatus,
        )

        cx = CounterexampleCandidate(
            description="n=4 is a counterexample",
            lean_code="example : ¬ (4 > 5) := by omega",
            proves_negation=True,
        )
        cx_result = CounterexampleResult(
            status=CounterexampleStatus.DISPROVED,
            candidates_tried=[cx],
            successful_counterexample=cx,
            attempts_made=3,
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.agents.counterexample_searcher.CounterexampleSearcher.search", return_value=cx_result),
            patch("agentic_research.agents.counterexample_searcher.CounterexampleSearcher.cumulative_tokens", new_callable=lambda: property(lambda self: MagicMock(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0))),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["check", "theorem test : False := sorry"])

        assert result.exit_code == 0
        assert "DISPROVED" in result.output
        assert "n=4" in result.output


class TestProveCommand:
    def test_prove_confirmed_and_succeeds(self, runner, tmp_session_dir):
        from agentic_research.models.proof import ProofPipelineResult

        proof_result = ProofPipelineResult(
            statement="theorem test : True := trivial",
            proved=True,
            final_proof="trivial",
            claim_check_passed=True,
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.proof.ProofPipeline.run", return_value=proof_result),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["prove", "theorem test : True := trivial"], input="y\n")

        assert result.exit_code == 0
        assert "PROVED" in result.output
        assert "Cost Summary" in result.output

    def test_prove_declined(self, runner, tmp_session_dir):
        result = runner.invoke(cli, ["prove", "theorem test : True := trivial"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output

    def test_prove_failure(self, runner, tmp_session_dir):
        from agentic_research.models.proof import ProofPipelineResult

        proof_result = ProofPipelineResult(
            statement="theorem test : False := sorry",
            proved=False,
            failure_stage="recursive_prover",
            failure_reason="All strategies exhausted",
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.proof.ProofPipeline.run", return_value=proof_result),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["prove", "theorem test : False := sorry"], input="y\n")

        assert result.exit_code == 0
        assert "FAILED" in result.output
        assert "recursive_prover" in result.output

    def test_prove_passes_statement_nl(self, runner, tmp_session_dir):
        """prove_cmd must pass statement_nl to ProofPipeline.run so data-package detection works."""
        from agentic_research.models.proof import ProofPipelineResult

        proof_result = ProofPipelineResult(
            statement="DRO theorem statement",
            proved=True,
            final_proof="sorry",
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.proof.ProofPipeline.run", return_value=proof_result) as mock_run,
        ):
            mock_llm.return_value = MagicMock()
            stmt = "For all distributions in the DRO ambiguity set, the worst-case risk is bounded"
            result = runner.invoke(cli, ["prove", stmt], input="y\n")

        assert result.exit_code == 0
        mock_run.assert_called_once_with(lean_statement=stmt, statement_nl=stmt)

    def test_prove_timeout(self, runner, tmp_session_dir):
        from agentic_research.models.proof import ProofPipelineResult

        proof_result = ProofPipelineResult(
            statement="theorem test : True := trivial",
            proved=False,
            failure_stage="timeout",
            failure_reason="Exceeded time limit",
        )

        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            if call_count[0] <= 1:
                return 0.0
            return 700.0

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.pipelines.proof.ProofPipeline.run", return_value=proof_result),
            patch("agentic_research.cli.main.time.monotonic", side_effect=mock_monotonic),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(
                cli, ["prove", "test stmt", "--timeout", "600"], input="y\n"
            )

        assert result.exit_code == 0
        assert "Timeout" in result.output or "timeout" in result.output.lower()


class TestStatusCommand:
    def test_status_no_session(self, runner, tmp_session_dir):
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Session Status" in result.output
        assert "Session ID" in result.output

    def test_status_with_session(self, runner, tmp_session_dir):
        from agentic_research.memory.session import ResearchSessionMemory
        from agentic_research.models.research import Conjecture

        session = ResearchSessionMemory(session_id="test-session-123")
        conj = Conjecture(
            statement="P(n) for all n",
            natural_language="Test conjecture",
            confidence=0.8,
            difficulty=3,
        )
        session.record_conjecture(conj)
        session.save(tmp_session_dir / "current_session.json")

        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "test-session-123" in result.output


class TestResumeCommand:
    def test_resume_list_no_sessions(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr("agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", tmp_path / "empty")
        result = runner.invoke(cli, ["resume", "--list"])
        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_resume_list_with_sessions(self, runner, tmp_path, monkeypatch):
        ckpt_dir = tmp_path / "checkpoints"
        monkeypatch.setattr("agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", ckpt_dir)

        from agentic_research.models.session import (
            PipelineStage,
            SessionCheckpoint,
            SessionMemoryData,
            SessionState,
        )

        session_dir = ckpt_dir / "test-session-abc"
        session_dir.mkdir(parents=True)

        checkpoint = SessionCheckpoint(
            checkpoint_id="ckpt_1",
            stage=PipelineStage.PROVING,
            session_state=SessionState(raw_idea="test idea", stage=PipelineStage.PROVING),
            memory=SessionMemoryData(),
        )
        (session_dir / "ckpt_1.json").write_text(checkpoint.model_dump_json(indent=2))

        result = runner.invoke(cli, ["resume", "--list"])
        assert result.exit_code == 0
        assert "test-session-abc" in result.output
        assert "proving" in result.output

    def test_resume_no_session_id(self, runner):
        result = runner.invoke(cli, ["resume"])
        assert result.exit_code != 0
        assert "session ID" in result.output or "Error" in result.output

    def test_resume_invalid_session(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr("agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", tmp_path / "empty_ckpts")
        result = runner.invoke(cli, ["resume", "nonexistent-session"])
        assert result.exit_code != 0
        assert "No checkpoints found" in result.output

    def test_resume_valid_session(self, runner, tmp_path, monkeypatch):
        ckpt_dir = tmp_path / "checkpoints"
        monkeypatch.setattr("agentic_research.orchestrator.rollback.DEFAULT_CHECKPOINT_DIR", ckpt_dir)

        from agentic_research.models.session import (
            PipelineStage,
            ResearchSessionResult,
            SessionCheckpoint,
            SessionMemoryData,
            SessionState,
        )

        session_dir = ckpt_dir / "resume-session-1"
        session_dir.mkdir(parents=True)
        checkpoint = SessionCheckpoint(
            checkpoint_id="ckpt_1",
            stage=PipelineStage.PROVING,
            session_state=SessionState(raw_idea="test idea", stage=PipelineStage.PROVING),
            memory=SessionMemoryData(),
        )
        (session_dir / "ckpt_1.json").write_text(checkpoint.model_dump_json(indent=2))

        mock_result = ResearchSessionResult(
            session_id="resume-session-1",
            raw_idea="test idea",
            final_stage=PipelineStage.COMPLETE,
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.orchestrator.engine.ResearchOrchestrator.resume_from_checkpoint", return_value=mock_result),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["resume", "resume-session-1"])

        assert result.exit_code == 0
        assert "Resuming session" in result.output
        assert "RESEARCH COMPLETE" in result.output


class TestLeanNotFoundWarning:
    def test_formalize_warns_when_lean_missing(self, runner, tmp_session_dir):
        with patch("agentic_research.cli.main.shutil.which", return_value=None):
            result = runner.invoke(cli, ["formalize", "test conjecture"])
        assert "Warning: Lean 4 not found" in result.output

    def test_prove_warns_when_lean_missing(self, runner, tmp_session_dir):
        with patch("agentic_research.cli.main.shutil.which", return_value=None):
            result = runner.invoke(cli, ["prove", "theorem test : True := trivial"], input="n\n")
        assert "Warning: Lean 4 not found" in result.output

    def test_research_warns_when_lean_missing(self, runner, tmp_session_dir):
        with patch("agentic_research.cli.main.shutil.which", return_value=None):
            result = runner.invoke(cli, ["research", "test idea"], input="n\n")
        assert "Warning: Lean 4 not found" in result.output

    def test_explore_does_not_warn_about_lean(self, runner, tmp_session_dir):
        with (
            patch("agentic_research.cli.main.shutil.which", return_value=None),
            patch("agentic_research.cli.main._create_llm_client", side_effect=Exception("stop early")),
        ):
            result = runner.invoke(cli, ["explore", "test idea"])
        assert "Warning: Lean 4 not found" not in result.output

    def test_no_warning_when_lean_present(self, runner, tmp_session_dir):
        with patch("agentic_research.cli.main.shutil.which", return_value="/usr/bin/lean"):
            result = runner.invoke(cli, ["formalize", "test conjecture"])
        assert "Warning: Lean 4 not found" not in result.output


class TestInputValidation:
    def test_research_rejects_zero_budget(self, runner):
        result = runner.invoke(cli, ["research", "test idea", "--budget", "0"])
        assert result.exit_code != 0
        assert "greater than 0" in result.output or "Invalid" in result.output or "Error" in result.output

    def test_research_rejects_negative_budget(self, runner):
        result = runner.invoke(cli, ["research", "test idea", "--budget", "-5"])
        assert result.exit_code != 0

    def test_research_rejects_zero_max_conjectures(self, runner):
        result = runner.invoke(cli, ["research", "test idea", "--max-conjectures", "0"])
        assert result.exit_code != 0
        assert "greater than 0" in result.output or "Invalid" in result.output or "Error" in result.output

    def test_research_rejects_negative_max_conjectures(self, runner):
        result = runner.invoke(cli, ["research", "test idea", "--max-conjectures", "-1"])
        assert result.exit_code != 0

    def test_research_rejects_negative_max_refinements(self, runner):
        result = runner.invoke(cli, ["research", "test idea", "--max-refinements", "-1"])
        assert result.exit_code != 0
        assert "non-negative" in result.output or "Invalid" in result.output or "Error" in result.output

    def test_research_accepts_zero_max_refinements(self, runner, tmp_session_dir):
        """max-refinements = 0 is valid (no refinements)."""
        from agentic_research.models.session import PipelineStage, ResearchSessionResult

        mock_result = ResearchSessionResult(
            session_id="test",
            raw_idea="test idea",
            final_stage=PipelineStage.COMPLETE,
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_repl"),
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.orchestrator.engine.ResearchOrchestrator.run", return_value=mock_result),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(
                cli, ["research", "test idea", "--max-refinements", "0"], input="y\n"
            )

        assert result.exit_code == 0


class TestBudgetEnforcement:
    def test_explore_halts_on_budget(self, runner, tmp_session_dir):
        explore_result = _mock_agent_result(result_data={"domain": "test"})

        huge_tokens = MagicMock()
        huge_tokens.input_tokens = 1_000_000
        huge_tokens.output_tokens = 1_000_000
        huge_tokens.cache_read_input_tokens = 0
        huge_tokens.cache_creation_input_tokens = 0

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.agents.explorer.ExplorationAgent.run", return_value=explore_result),
            patch("agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens", new_callable=lambda: property(lambda self: huge_tokens)),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["explore", "test", "--budget", "0.001"])

        assert "Budget" in result.output or "budget" in result.output.lower() or "exceeded" in result.output.lower()


class TestErrorDisplay:
    def test_explore_shows_error_on_llm_failure(self, runner, tmp_session_dir):
        from agentic_research.models.agents import AgentResult, AgentStatus, TokenUsage

        failed_result = AgentResult(
            agent_name="exploration_agent",
            status=AgentStatus.ERROR,
            error_message="API rate limit exceeded",
            token_usage=TokenUsage(input_tokens=10, output_tokens=0),
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.cli.main._create_lean_search"),
            patch("agentic_research.agents.explorer.ExplorationAgent.run", return_value=failed_result),
            patch("agentic_research.agents.explorer.ExplorationAgent.cumulative_tokens", new_callable=lambda: property(lambda self: _mock_token_usage())),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["explore", "test"])

        assert result.exit_code != 0

    def test_check_shows_error_on_setup_failure(self, runner, tmp_session_dir):
        with patch(
            "agentic_research.cli.main._create_llm_client",
            side_effect=Exception("Missing ANTHROPIC_API_KEY"),
        ):
            result = runner.invoke(cli, ["check", "test"])

        assert result.exit_code != 0
        assert "Setup error" in result.output
