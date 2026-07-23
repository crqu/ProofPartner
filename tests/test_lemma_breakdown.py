"""Tests for LemmaBreakdown — best-of-k decomposition, scoring, and temperature variation.

All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agentic_research.agents.lemma_breakdown import LemmaBreakdown
from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.proof import LemmaTree, ProofNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_response(content: str, input_tokens: int = 100, output_tokens: int = 50) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_mock_llm(responses: list[str], input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = [
        _mock_llm_response(text, input_tokens, output_tokens) for text in responses
    ]
    mock.complete.side_effect = side_effects

    real_client_cls = LLMClient
    with patch("anthropic.Anthropic"):
        temp_client = real_client_cls.__new__(real_client_cls)
    mock.extract_json = temp_client.__class__.extract_json.__get__(mock, type(mock))
    return mock


def _make_decomposition_json(num_lemmas: int = 3) -> str:
    lemmas = []
    for i in range(num_lemmas):
        lemmas.append({
            "node_id": f"lemma_{i}",
            "statement_nl": f"Sub-lemma {i}",
            "depends_on": [],
        })
    topo = [f"lemma_{i}" for i in range(num_lemmas)]
    return json.dumps({"lemmas": lemmas, "topological_order": topo})


SMALL_TREE_JSON = _make_decomposition_json(2)
MEDIUM_TREE_JSON = _make_decomposition_json(4)
LARGE_TREE_JSON = _make_decomposition_json(8)


def _make_context() -> AgentContext:
    return AgentContext(
        task="Prove that n + m = m + n",
        metadata={
            "statement_lean": "theorem add_comm : ∀ n m : Nat, n + m = m + n := by sorry",
            "failed_attempts": "None",
        },
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoreDecomposition:
    def test_single_node_tree(self):
        tree = LemmaTree(
            root_id="root",
            nodes={"root": ProofNode(node_id="root", depth=0)},
            topological_order=["root"],
        )
        score = LemmaBreakdown._score_decomposition(tree)
        assert 0.0 < score <= 1.0

    def test_brevity_decreases_with_more_nodes(self):
        small = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", depth=0),
                "l1": ProofNode(node_id="l1", depth=1),
            },
            topological_order=["l1", "root"],
        )
        large = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", depth=0),
                **{f"l{i}": ProofNode(node_id=f"l{i}", depth=1) for i in range(10)},
            },
            topological_order=[f"l{i}" for i in range(10)] + ["root"],
        )
        small_score = LemmaBreakdown._score_decomposition(small)
        large_score = LemmaBreakdown._score_decomposition(large)
        assert small_score > large_score

    def test_balanced_tree_scores_higher(self):
        balanced = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", depth=0),
                "l1": ProofNode(node_id="l1", depth=1),
                "l2": ProofNode(node_id="l2", depth=1),
                "l3": ProofNode(node_id="l3", depth=1),
            },
            topological_order=["l1", "l2", "l3", "root"],
        )
        unbalanced = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", depth=0),
                "l1": ProofNode(node_id="l1", depth=1),
                "l2": ProofNode(node_id="l2", depth=2),
                "l3": ProofNode(node_id="l3", depth=3),
            },
            topological_order=["l3", "l2", "l1", "root"],
        )
        bal_score = LemmaBreakdown._score_decomposition(balanced)
        unbal_score = LemmaBreakdown._score_decomposition(unbalanced)
        assert bal_score > unbal_score

    def test_score_is_between_0_and_1(self):
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(node_id="root", depth=0),
                "l1": ProofNode(node_id="l1", depth=1),
                "l2": ProofNode(node_id="l2", depth=1),
            },
            topological_order=["l1", "l2", "root"],
        )
        score = LemmaBreakdown._score_decomposition(tree)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Best-of-k execution
# ---------------------------------------------------------------------------


class TestBestOfK:
    def test_best_of_k_generates_k_candidates(self):
        k = 3
        llm = _make_mock_llm([SMALL_TREE_JSON] * k)
        breakdown = LemmaBreakdown(llm_client=llm, decomposition_k=k)

        result = breakdown.run(_make_context())
        assert result.status == AgentStatus.SUCCESS
        assert llm.complete.call_count == k

    def test_best_of_k_selects_highest_score(self):
        k = 3
        llm = _make_mock_llm([
            LARGE_TREE_JSON,
            SMALL_TREE_JSON,
            MEDIUM_TREE_JSON,
        ])
        breakdown = LemmaBreakdown(llm_client=llm, decomposition_k=k)

        result = breakdown.run(_make_context())
        assert result.status == AgentStatus.SUCCESS
        tree = LemmaTree.model_validate(result.result)
        assert tree.decomposition_score is not None
        assert len(tree.nodes) == 3  # root + 2 from SMALL_TREE_JSON

    def test_temperature_variation_across_candidates(self):
        k = 3
        llm = _make_mock_llm([SMALL_TREE_JSON] * k)
        breakdown = LemmaBreakdown(llm_client=llm, decomposition_k=k)

        breakdown.run(_make_context())

        temperatures = []
        for call in llm.complete.call_args_list:
            temperatures.append(call.kwargs.get("temperature", call.args[0] if call.args else None))

        assert len(temperatures) == k
        expected = [0.3, 0.35, 0.4]
        for actual, exp in zip(temperatures, expected):
            assert abs(actual - exp) < 1e-6, f"Expected {exp}, got {actual}"

    def test_best_of_k_aggregates_token_usage(self):
        k = 3
        input_per_call = 200
        output_per_call = 100
        llm = _make_mock_llm(
            [SMALL_TREE_JSON] * k,
            input_tokens=input_per_call,
            output_tokens=output_per_call,
        )
        breakdown = LemmaBreakdown(llm_client=llm, decomposition_k=k)

        result = breakdown.run(_make_context())
        assert result.token_usage.input_tokens == k * input_per_call
        assert result.token_usage.output_tokens == k * output_per_call

    def test_single_candidate_fallback(self):
        llm = _make_mock_llm([SMALL_TREE_JSON])
        breakdown = LemmaBreakdown(llm_client=llm, decomposition_k=1)

        result = breakdown.run(_make_context())
        assert result.status == AgentStatus.SUCCESS
        assert llm.complete.call_count == 1

        tree = LemmaTree.model_validate(result.result)
        assert tree.decomposition_score is not None

    def test_default_k_is_1(self):
        llm = _make_mock_llm([SMALL_TREE_JSON])
        breakdown = LemmaBreakdown(llm_client=llm)
        assert breakdown._decomposition_k == 1

    def test_decomposition_score_populated_on_tree(self):
        llm = _make_mock_llm([SMALL_TREE_JSON])
        breakdown = LemmaBreakdown(llm_client=llm, decomposition_k=1)

        result = breakdown.run(_make_context())
        tree = LemmaTree.model_validate(result.result)
        assert tree.decomposition_score is not None
        assert tree.decomposition_score > 0.0

    def test_cost_within_bounds(self):
        k = 3
        input_per_call = 500
        output_per_call = 200
        llm = _make_mock_llm(
            [SMALL_TREE_JSON] * k,
            input_tokens=input_per_call,
            output_tokens=output_per_call,
        )
        breakdown = LemmaBreakdown(llm_client=llm, decomposition_k=k)
        result = breakdown.run(_make_context())

        assert result.token_usage.input_tokens <= k * input_per_call
        assert result.token_usage.output_tokens <= k * output_per_call
