"""Tests for --interactive CLI flag and TTY guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentic_research.cli.main import cli, _build_interaction_callback
from agentic_research.models.interaction import (
    InteractionOption,
    InteractionRequest,
)


@pytest.fixture
def runner():
    return CliRunner()


class TestInteractiveFlag:
    def test_formalize_has_interactive_flag(self, runner):
        result = runner.invoke(cli, ["formalize", "--help"])
        assert result.exit_code == 0
        assert "--interactive" in result.output

    def test_prove_has_interactive_flag(self, runner):
        result = runner.invoke(cli, ["prove", "--help"])
        assert result.exit_code == 0
        assert "--interactive" in result.output

    def test_research_has_interactive_flag(self, runner):
        result = runner.invoke(cli, ["research", "--help"])
        assert result.exit_code == 0
        assert "--interactive" in result.output


class TestTTYGuard:
    @patch("agentic_research.cli.main.sys")
    def test_formalize_rejects_non_tty(self, mock_sys, runner):
        mock_sys.stdin.isatty.return_value = False
        mock_sys.exit = SystemExit
        result = runner.invoke(cli, ["formalize", "--interactive", "test conjecture"])
        assert result.exit_code != 0
        assert "TTY" in result.output

    @patch("agentic_research.cli.main.sys")
    def test_prove_rejects_non_tty(self, mock_sys, runner):
        mock_sys.stdin.isatty.return_value = False
        mock_sys.exit = SystemExit
        result = runner.invoke(cli, ["prove", "--interactive", "theorem test : True := by sorry"])
        assert result.exit_code != 0
        assert "TTY" in result.output

    @patch("agentic_research.cli.main.sys")
    def test_research_rejects_non_tty(self, mock_sys, runner):
        mock_sys.stdin.isatty.return_value = False
        mock_sys.exit = SystemExit
        result = runner.invoke(cli, ["research", "--interactive", "test idea"])
        assert result.exit_code != 0
        assert "TTY" in result.output


class TestBuildInteractionCallback:
    def test_callback_returns_callable(self):
        from rich.console import Console
        console = Console()
        cb = _build_interaction_callback(console)
        assert callable(cb)

    @patch("rich.prompt.IntPrompt.ask", return_value=2)
    def test_callback_selects_option(self, mock_ask):
        from rich.console import Console
        console = Console(file=MagicMock())
        cb = _build_interaction_callback(console)
        req = InteractionRequest(
            type="select",
            prompt="Pick one",
            options=[
                InteractionOption(label="A", value=10, score=0.9),
                InteractionOption(label="B", value=20, score=0.8),
                InteractionOption(label="C", value=30, score=0.7),
            ],
            default_value=10,
        )
        resp = cb(req)
        assert resp.selected_value == 20
        assert resp.aborted is False

    @patch("rich.prompt.IntPrompt.ask", return_value=0)
    def test_callback_abort_returns_default(self, mock_ask):
        from rich.console import Console
        console = Console(file=MagicMock())
        cb = _build_interaction_callback(console)
        req = InteractionRequest(
            type="select",
            prompt="Pick one",
            options=[
                InteractionOption(label="A", value=10, score=0.9),
            ],
            default_value=10,
        )
        resp = cb(req)
        assert resp.aborted is True
        assert resp.selected_value == 10

    @patch("rich.prompt.IntPrompt.ask", return_value=99)
    def test_callback_out_of_range_aborts(self, mock_ask):
        from rich.console import Console
        console = Console(file=MagicMock())
        cb = _build_interaction_callback(console)
        req = InteractionRequest(
            type="select",
            prompt="Pick one",
            options=[
                InteractionOption(label="A", value=10, score=0.9),
            ],
            default_value=10,
        )
        resp = cb(req)
        assert resp.aborted is True
