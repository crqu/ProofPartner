"""Tests for the eval runner."""

import time
from unittest.mock import MagicMock, patch

from agentic_research.eval.runner import (
    _SharedResources,
    _evaluate_proof_discovery,
    _sum_token_usage,
    run_eval,
)
from agentic_research.logging import configure_logging
from agentic_research.models.agents import TokenUsage
from agentic_research.models.eval import (
    BenchmarkSource,
    EvalConfig,
    EvalMode,
    Problem,
    ProblemDifficulty,
    ProblemSplit,
    ProofResult,
)
from agentic_research.models.proof import ProofPipelineResult


def _make_problem(pid: str = "test_problem", difficulty: ProblemDifficulty = ProblemDifficulty.AMC) -> Problem:
    return Problem(
        id=pid,
        name=pid,
        source=BenchmarkSource.MINIF2F,
        split=ProblemSplit.TEST,
        difficulty=difficulty,
        lean_statement="theorem test : True := by trivial",
        natural_language="Prove that True holds.",
    )


def _make_config(timeout: int = 600) -> EvalConfig:
    return EvalConfig(
        mode=EvalMode.PROOF_DISCOVERY,
        benchmark=BenchmarkSource.MINIF2F,
        split=ProblemSplit.TEST,
        timeout_seconds=timeout,
    )


def _make_shared() -> _SharedResources:
    return _SharedResources(
        llm_client=MagicMock(),
        lean_search=MagicMock(),
    )


class TestSumTokenUsage:
    def test_sums_all_four_fields(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=200,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=30,
        )
        assert _sum_token_usage(usage) == 380

    def test_zero_usage(self):
        assert _sum_token_usage(TokenUsage()) == 0


class TestEvaluateProofDiscoverySuccess:
    @patch("agentic_research.eval.runner.ProofPipeline")
    @patch("agentic_research.eval.runner.LeanRepl")
    def test_success_mapping(self, mock_repl_cls, mock_pipeline_cls):
        token_usage = TokenUsage(
            input_tokens=100, output_tokens=200,
            cache_creation_input_tokens=50, cache_read_input_tokens=30,
        )
        pipeline_result = ProofPipelineResult(
            statement="theorem test : True := by trivial",
            proved=True,
            final_proof="theorem test : True := by trivial",
            total_token_usage=token_usage,
        )
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = pipeline_result
        mock_pipeline_cls.return_value = mock_pipeline

        problem = _make_problem()
        config = _make_config()
        shared = _make_shared()

        result = _evaluate_proof_discovery(problem, config, shared)

        assert result.result == ProofResult.SUCCESS
        assert result.proof == "theorem test : True := by trivial"
        assert result.token_usage == 380
        assert result.attempts == 1
        assert result.duration_seconds > 0


class TestEvaluateProofDiscoveryFailure:
    @patch("agentic_research.eval.runner.ProofPipeline")
    @patch("agentic_research.eval.runner.LeanRepl")
    def test_failure_mapping(self, mock_repl_cls, mock_pipeline_cls):
        pipeline_result = ProofPipelineResult(
            statement="theorem test : True := by trivial",
            proved=False,
            failure_stage="proof_search",
            failure_reason="All strategies exhausted",
            total_token_usage=TokenUsage(input_tokens=500, output_tokens=100),
        )
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = pipeline_result
        mock_pipeline_cls.return_value = mock_pipeline

        result = _evaluate_proof_discovery(_make_problem(), _make_config(), _make_shared())

        assert result.result == ProofResult.FAILURE
        assert result.error_message == "All strategies exhausted"
        assert result.token_usage == 600
        assert result.proof is None


class TestEvaluateProofDiscoveryTimeout:
    @patch("agentic_research.eval.runner.ProofPipeline")
    @patch("agentic_research.eval.runner.LeanRepl")
    def test_timeout_mapping(self, mock_repl_cls, mock_pipeline_cls):
        def slow_run(*args, **kwargs):
            time.sleep(5)
            return ProofPipelineResult(statement="test", proved=False)

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = slow_run
        mock_pipeline_cls.return_value = mock_pipeline

        config = _make_config(timeout=1)
        result = _evaluate_proof_discovery(_make_problem(), config, _make_shared())

        assert result.result == ProofResult.TIMEOUT
        assert "Timeout" in (result.error_message or "")


class TestEvaluateProofDiscoveryError:
    @patch("agentic_research.eval.runner.ProofPipeline")
    @patch("agentic_research.eval.runner.LeanRepl")
    def test_error_mapping(self, mock_repl_cls, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = RuntimeError("API connection failed")
        mock_pipeline_cls.return_value = mock_pipeline

        result = _evaluate_proof_discovery(_make_problem(), _make_config(), _make_shared())

        assert result.result == ProofResult.ERROR
        assert "API connection failed" in (result.error_message or "")


class TestModelConfigPassthrough:
    def test_model_field_in_config(self):
        config = EvalConfig(
            mode=EvalMode.PROOF_DISCOVERY,
            model="claude-sonnet-4-20250514",
        )
        assert config.model == "claude-sonnet-4-20250514"

    def test_model_field_default_none(self):
        config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY)
        assert config.model is None


class TestRunEvalMockPipelineEndToEnd:
    @patch("agentic_research.eval.runner.LeanSearch")
    @patch("agentic_research.eval.runner.LLMClient")
    @patch("agentic_research.eval.runner.ProofPipeline")
    @patch("agentic_research.eval.runner.LeanRepl")
    def test_end_to_end(self, mock_repl_cls, mock_pipeline_cls, mock_llm_cls, mock_search_cls, tmp_path):
        configure_logging(json_output=False, level="WARNING")

        repo_dir = tmp_path / "miniF2F"
        (repo_dir / ".git").mkdir(parents=True)
        lean_dir = repo_dir / "MiniF2F" / "Test"
        lean_dir.mkdir(parents=True)
        lean_file = lean_dir / "Basic.lean"
        lean_file.write_text(
            "import Mathlib\n\n"
            "theorem test_one (n : Nat) : n = n := by rfl\n\n"
            "theorem test_two : True := by trivial\n"
        )

        call_count = 0

        def pipeline_run(lean_statement, statement_nl=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ProofPipelineResult(
                    statement=lean_statement,
                    proved=True,
                    final_proof=lean_statement + " by rfl",
                    total_token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                )
            return ProofPipelineResult(
                statement=lean_statement,
                proved=False,
                failure_reason="Could not prove",
                total_token_usage=TokenUsage(input_tokens=200, output_tokens=80),
            )

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = pipeline_run
        mock_pipeline_cls.return_value = mock_pipeline

        config = EvalConfig(
            mode=EvalMode.PROOF_DISCOVERY,
            benchmark=BenchmarkSource.MINIF2F,
            split=ProblemSplit.TEST,
            pass_k=1,
            data_dir=tmp_path,
        )

        report = run_eval(config)

        assert report.mode == EvalMode.PROOF_DISCOVERY
        assert report.aggregate.total == 2
        assert report.aggregate.successes == 1
        assert report.aggregate.failures == 1
        assert report.aggregate.pass_rate == 0.5

        assert report.by_difficulty is not None
        assert len(report.by_difficulty) > 0

        success_results = [r for r in report.results if r.result == ProofResult.SUCCESS]
        assert len(success_results) == 1
        assert success_results[0].token_usage == 150
