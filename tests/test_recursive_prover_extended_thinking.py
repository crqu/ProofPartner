"""Tests for extended thinking in RecursiveProver parent assembly."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic_research.agents.recursive_prover import RecursiveProver
from agentic_research.models.agents import (
    LLMResponse,
    ProverConfig,
    TokenUsage,
)
from agentic_research.models.proof import (
    LemmaTree,
    ProofNode,
)
from agentic_research.models.tools import CompilationResult, CompilationStatus, ToolStatus


def _mock_llm_response(content: str = "```lean\nsorry\n```") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


def _make_tree_with_children() -> LemmaTree:
    root = ProofNode(
        node_id="root",
        statement_nl="root statement",
        statement_lean="theorem root : True",
        depth=0,
        children=["child1"],
    )
    child = ProofNode(
        node_id="child1",
        statement_nl="child statement",
        statement_lean="theorem child1 : True",
        depth=1,
        parent_id="root",
    )
    return LemmaTree(
        root_id="root",
        nodes={"root": root, "child1": child},
        topological_order=["child1", "root"],
    )


def _make_ok_compilation() -> CompilationResult:
    return CompilationResult(
        status=ToolStatus.SUCCESS,
        compilation_status=CompilationStatus.OK,
        errors=[],
        warnings=[],
    )


class TestParentAssemblyExtendedThinking:
    def test_default_uses_extended_thinking(self) -> None:
        mock_llm = MagicMock()
        mock_repl = MagicMock()
        mock_repl.execute.return_value = _make_ok_compilation()
        mock_llm.complete.return_value = _mock_llm_response(
            "```lean\ntheorem root : True := by trivial\n```"
        )

        config = ProverConfig()
        assert config.parent_extended_thinking is True

        prover = RecursiveProver(
            llm_client=mock_llm,
            lean_repl=mock_repl,
            prover_config=config,
        )

        tree = _make_tree_with_children()
        tokens = TokenUsage()
        prover._prove_parent_with_children(tree, tree.nodes["root"], tokens)

        complete_call = mock_llm.complete.call_args
        assert complete_call.kwargs["use_extended_thinking"] is True
        assert complete_call.kwargs["thinking_budget"] == 40000

    def test_custom_thinking_budget_passed(self) -> None:
        mock_llm = MagicMock()
        mock_repl = MagicMock()
        mock_repl.execute.return_value = _make_ok_compilation()
        mock_llm.complete.return_value = _mock_llm_response(
            "```lean\ntheorem root : True := by trivial\n```"
        )

        config = ProverConfig(thinking_budget=80000)
        prover = RecursiveProver(
            llm_client=mock_llm,
            lean_repl=mock_repl,
            prover_config=config,
        )

        tree = _make_tree_with_children()
        tokens = TokenUsage()
        prover._prove_parent_with_children(tree, tree.nodes["root"], tokens)

        complete_call = mock_llm.complete.call_args
        assert complete_call.kwargs["thinking_budget"] == 80000

    def test_parent_extended_thinking_false_disables(self) -> None:
        mock_llm = MagicMock()
        mock_repl = MagicMock()
        mock_repl.execute.return_value = _make_ok_compilation()
        mock_llm.complete.return_value = _mock_llm_response(
            "```lean\ntheorem root : True := by trivial\n```"
        )

        config = ProverConfig(parent_extended_thinking=False)
        prover = RecursiveProver(
            llm_client=mock_llm,
            lean_repl=mock_repl,
            prover_config=config,
        )

        tree = _make_tree_with_children()
        tokens = TokenUsage()
        prover._prove_parent_with_children(tree, tree.nodes["root"], tokens)

        complete_call = mock_llm.complete.call_args
        assert complete_call.kwargs["use_extended_thinking"] is False
        assert "thinking_budget" not in complete_call.kwargs

    def test_prover_config_default_values(self) -> None:
        config = ProverConfig()
        assert config.parent_extended_thinking is True
        assert config.thinking_budget == 40000
        assert config.use_extended_thinking is False

    def test_thinking_budget_from_config_not_default(self) -> None:
        config = ProverConfig(thinking_budget=25000)
        assert config.thinking_budget == 25000

        mock_llm = MagicMock()
        mock_repl = MagicMock()
        mock_repl.execute.return_value = _make_ok_compilation()
        mock_llm.complete.return_value = _mock_llm_response(
            "```lean\ntheorem root : True := by trivial\n```"
        )

        prover = RecursiveProver(
            llm_client=mock_llm,
            lean_repl=mock_repl,
            prover_config=config,
        )

        tree = _make_tree_with_children()
        tokens = TokenUsage()
        prover._prove_parent_with_children(tree, tree.nodes["root"], tokens)

        complete_call = mock_llm.complete.call_args
        assert complete_call.kwargs["thinking_budget"] == 25000
