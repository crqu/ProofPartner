"""Tests for the mock backend guard (require_backend) and related CLI/model changes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from agentic_research.models.proof import ProofPipelineResult
from agentic_research.tools.lean_repl import ReplBackend, require_backend


class TestRequireBackend:
    def test_raises_when_mock_and_not_allowed(self):
        with patch("agentic_research.tools.lean_repl.detect_backend", return_value=ReplBackend.MOCK):
            with pytest.raises(RuntimeError, match="Lean 4 not found on PATH"):
                require_backend(allow_mock=False)

    def test_returns_mock_when_allowed(self):
        with patch("agentic_research.tools.lean_repl.detect_backend", return_value=ReplBackend.MOCK):
            result = require_backend(allow_mock=True)
            assert result == ReplBackend.MOCK

    def test_returns_subprocess_when_available(self):
        with patch("agentic_research.tools.lean_repl.detect_backend", return_value=ReplBackend.SUBPROCESS):
            result = require_backend(allow_mock=False)
            assert result == ReplBackend.SUBPROCESS

    def test_returns_lean_dojo_when_available(self):
        with patch("agentic_research.tools.lean_repl.detect_backend", return_value=ReplBackend.LEAN_DOJO):
            result = require_backend(allow_mock=False)
            assert result == ReplBackend.LEAN_DOJO


class TestCLIMockGuard:
    def test_prove_cmd_exits_without_lean(self):
        from agentic_research.cli.main import cli

        runner = CliRunner()
        with patch("agentic_research.tools.lean_repl.detect_backend", return_value=ReplBackend.MOCK):
            result = runner.invoke(cli, ["prove", "theorem test : True := trivial"])
            assert result.exit_code == 1
            assert "Lean 4 not found" in result.output or "Lean 4 not found" in (result.stderr_bytes or b"").decode()

    def test_prove_cmd_allows_mock_flag(self):
        from agentic_research.cli.main import cli

        runner = CliRunner()
        with patch("agentic_research.tools.lean_repl.detect_backend", return_value=ReplBackend.MOCK):
            result = runner.invoke(cli, ["prove", "--allow-mock", "theorem test : True := trivial"], input="n\n")
            # Should not exit with the mock guard error (exit code 1 from guard)
            # It will prompt for confirmation and we answer "n" → aborted
            assert "Lean 4 not found" not in result.output

    def test_check_cmd_exits_without_lean(self):
        from agentic_research.cli.main import cli

        runner = CliRunner()
        with patch("agentic_research.tools.lean_repl.detect_backend", return_value=ReplBackend.MOCK):
            result = runner.invoke(cli, ["check", "theorem test : True := trivial"])
            assert result.exit_code == 1

    def test_eval_cmd_exits_without_lean(self):
        from agentic_research.cli.main import cli

        runner = CliRunner()
        with patch("agentic_research.tools.lean_repl.detect_backend", return_value=ReplBackend.MOCK):
            result = runner.invoke(cli, ["eval", "miniF2F"])
            assert result.exit_code == 1


class TestProofPipelineResultFields:
    def test_has_backend_field(self):
        result = ProofPipelineResult(
            statement="test",
            backend="mock",
            verified=False,
        )
        assert result.backend == "mock"
        assert result.verified is False

    def test_default_values(self):
        result = ProofPipelineResult(statement="test")
        assert result.backend is None
        assert result.verified is True
