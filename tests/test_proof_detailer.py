"""Tests for the ProofDetailer agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentic_research.agents.proof_detailer import ProofDetailer, compute_complexity_score
from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.proof import LemmaTree, ProofNode


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )


def _extract_json_helper(text: str):
    import re

    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = [_mock_llm_response(text) for text in responses]
    mock.complete.side_effect = side_effects
    mock.extract_json.side_effect = lambda text: _extract_json_helper(text)
    return mock


class TestComplexityScore:
    def test_simple_node_low_score(self):
        node = ProofNode(node_id="l1", statement_nl="x is positive")
        score = compute_complexity_score(node)
        assert score < 40

    def test_complex_node_high_score(self):
        node = ProofNode(
            node_id="l1",
            statement_nl=(
                "For all probability measures mu, there exists a coupling gamma "
                "such that the integral of the cost function under gamma equals "
                "the Wasserstein distance between the marginals, and this limit "
                "is clearly attained by a measurable transport map"
            ),
        )
        score = compute_complexity_score(node)
        assert score >= 40

    def test_quantifiers_increase_score(self):
        simple = ProofNode(node_id="l1", statement_nl="x equals y")
        quantified = ProofNode(
            node_id="l2",
            statement_nl="for all x there exists y such that for all z",
        )
        assert compute_complexity_score(quantified) > compute_complexity_score(simple)

    def test_cue_phrases_increase_score(self):
        without_cue = ProofNode(
            node_id="l1",
            statement_nl="the function is continuous on the closed interval" * 3,
        )
        with_cue = ProofNode(
            node_id="l2",
            statement_nl="clearly the function is continuous on the closed interval" * 3,
        )
        assert compute_complexity_score(with_cue) > compute_complexity_score(without_cue)


class TestProofDetailer:
    def test_skips_axiom_nodes(self):
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    children=["lemma_1"],
                ),
                "lemma_1": ProofNode(
                    node_id="lemma_1",
                    statement_nl="known result " * 50,
                    depth=1,
                    parent_id="root",
                    from_prior_work=True,
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        llm = _make_mock_llm([])
        detailer = ProofDetailer(llm_client=llm, complexity_threshold=0)
        ctx = AgentContext(
            task="detail",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = detailer.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        updated = LemmaTree.model_validate(result.result)
        assert updated.nodes["lemma_1"].proof_sketch_nl is None

    def test_details_complex_node(self):
        long_statement = (
            "For all probability measures mu on the measurable space, "
            "there exists a coupling that minimizes the integral of "
            "the cost function under the constraint that marginals match "
            "and the limit is attained clearly by standard arguments"
        )
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main theorem",
                    children=["lemma_1"],
                ),
                "lemma_1": ProofNode(
                    node_id="lemma_1",
                    statement_nl=long_statement,
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        detail_response = json.dumps({
            "needs_detailing": True,
            "reasoning": "complex coupling argument",
            "proof_sketch": [
                {"step_number": 1, "claim": "Construct the product space", "justification": "measurable_prod"},
                {"step_number": 2, "claim": "Apply inf over couplings", "justification": "iInf_le"},
                {"step_number": 3, "claim": "Show attainment", "justification": "isCompact_of_isClosed"},
            ],
        })
        llm = _make_mock_llm([detail_response])

        detailer = ProofDetailer(llm_client=llm, complexity_threshold=10)
        ctx = AgentContext(
            task="detail",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = detailer.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        updated = LemmaTree.model_validate(result.result)
        sketch = updated.nodes["lemma_1"].proof_sketch_nl
        assert sketch is not None
        assert "Step 1" in sketch
        assert "Step 3" in sketch

    def test_skips_simple_node(self):
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    children=["lemma_1"],
                ),
                "lemma_1": ProofNode(
                    node_id="lemma_1",
                    statement_nl="x > 0",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        llm = _make_mock_llm([])
        detailer = ProofDetailer(llm_client=llm, complexity_threshold=40)
        ctx = AgentContext(
            task="detail",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = detailer.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        updated = LemmaTree.model_validate(result.result)
        assert updated.nodes["lemma_1"].proof_sketch_nl is None

    def test_missing_tree_returns_failure(self):
        llm = _make_mock_llm([])
        detailer = ProofDetailer(llm_client=llm)
        ctx = AgentContext(task="detail", metadata={})
        result = detailer.run(ctx)
        assert result.status == AgentStatus.FAILURE

    def test_node_not_needing_detail(self):
        long_statement = (
            "For all elements x in the group, there exists an inverse y "
            "such that x * y = e and y * x = e for the identity element"
        )
        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main",
                    children=["lemma_1"],
                ),
                "lemma_1": ProofNode(
                    node_id="lemma_1",
                    statement_nl=long_statement,
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["lemma_1", "root"],
        )

        no_detail_response = json.dumps({
            "needs_detailing": False,
            "reasoning": "straightforward group theory",
        })
        llm = _make_mock_llm([no_detail_response])

        detailer = ProofDetailer(llm_client=llm, complexity_threshold=10)
        ctx = AgentContext(
            task="detail",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = detailer.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        updated = LemmaTree.model_validate(result.result)
        assert updated.nodes["lemma_1"].proof_sketch_nl is None
