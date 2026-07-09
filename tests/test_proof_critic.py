"""Tests for the ProofCritic agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentic_research.agents.proof_critic import ProofCritic
from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.proof import (
    CritiqueIssue,
    CritiqueIssueType,
    CritiqueResult,
    LemmaTree,
    ProofNode,
)


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


def _make_tree() -> LemmaTree:
    root = ProofNode(
        node_id="root",
        statement_nl="main theorem",
        statement_lean="theorem main := sorry",
        children=["lemma_1", "lemma_2"],
    )
    l1 = ProofNode(
        node_id="lemma_1",
        statement_nl="first sub-lemma: base case",
        depth=1,
        parent_id="root",
    )
    l2 = ProofNode(
        node_id="lemma_2",
        statement_nl="second sub-lemma: inductive step",
        depth=1,
        parent_id="root",
    )
    return LemmaTree(
        root_id="root",
        nodes={"root": root, "lemma_1": l1, "lemma_2": l2},
        topological_order=["lemma_1", "lemma_2", "root"],
    )


class TestProofCriticNoIssues:
    def test_no_issues_found(self):
        propose_response = '{"issues": []}'
        llm = _make_mock_llm([propose_response])

        critic = ProofCritic(llm_client=llm)
        result = critic.critique(
            tree=_make_tree(),
            statement_nl="main theorem",
            statement_lean="theorem main := sorry",
        )

        assert result.passed
        assert result.issues == []

    def test_via_agent_run(self):
        propose_response = '{"issues": []}'
        llm = _make_mock_llm([propose_response])

        critic = ProofCritic(llm_client=llm)
        ctx = AgentContext(
            task="main theorem",
            metadata={
                "lemma_tree": _make_tree().model_dump(),
                "statement_lean": "theorem main := sorry",
            },
        )
        result = critic.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        critique = CritiqueResult.model_validate(result.result)
        assert critique.passed


class TestProofCriticWithIssues:
    def test_blocking_issue_fails(self):
        propose_response = json.dumps({
            "issues": [{
                "issue_type": "unstated_hypothesis",
                "node_id": "lemma_1",
                "description": "Assumes x > 0 without stating it",
                "severity": "blocking",
                "suggested_fix": "Add x > 0 hypothesis",
            }]
        })
        confirm_response = json.dumps({
            "confirmed_issues": [{
                "issue_type": "unstated_hypothesis",
                "node_id": "lemma_1",
                "description": "Assumes x > 0 without stating it",
                "severity": "blocking",
                "suggested_fix": "Add x > 0 hypothesis",
                "confirmed": True,
            }]
        })

        llm = _make_mock_llm([propose_response, confirm_response])

        critic = ProofCritic(llm_client=llm)
        result = critic.critique(
            tree=_make_tree(),
            statement_nl="main theorem",
        )

        assert not result.passed
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == CritiqueIssueType.UNSTATED_HYPOTHESIS
        assert result.issues[0].confirmed

    def test_warning_issue_passes(self):
        propose_response = json.dumps({
            "issues": [{
                "issue_type": "unjustified_step",
                "node_id": "lemma_2",
                "description": "Step could use more justification",
                "severity": "warning",
                "suggested_fix": "Add intermediate step",
            }]
        })
        confirm_response = json.dumps({
            "confirmed_issues": [{
                "issue_type": "unjustified_step",
                "node_id": "lemma_2",
                "description": "Step could use more justification",
                "severity": "warning",
                "suggested_fix": "Add intermediate step",
                "confirmed": True,
            }]
        })

        llm = _make_mock_llm([propose_response, confirm_response])

        critic = ProofCritic(llm_client=llm)
        result = critic.critique(
            tree=_make_tree(),
            statement_nl="main theorem",
        )

        assert result.passed
        assert len(result.issues) == 1

    def test_proposed_but_refuted(self):
        propose_response = json.dumps({
            "issues": [{
                "issue_type": "hidden_case_split",
                "node_id": "lemma_1",
                "description": "Missing negative case",
                "severity": "blocking",
            }]
        })
        confirm_response = '{"confirmed_issues": []}'

        llm = _make_mock_llm([propose_response, confirm_response])

        critic = ProofCritic(llm_client=llm)
        result = critic.critique(
            tree=_make_tree(),
            statement_nl="main theorem",
        )

        assert result.passed
        assert result.issues == []


class TestProofCriticEdgeCases:
    def test_missing_tree_returns_failure(self):
        llm = _make_mock_llm([])
        critic = ProofCritic(llm_client=llm)
        ctx = AgentContext(task="main theorem", metadata={})
        result = critic.run(ctx)
        assert result.status == AgentStatus.FAILURE

    def test_unparseable_response(self):
        llm = _make_mock_llm(["not valid json at all"])
        critic = ProofCritic(llm_client=llm)
        result = critic.critique(
            tree=_make_tree(),
            statement_nl="main theorem",
        )
        assert result.passed
        assert result.issues == []


class TestCritiqueModels:
    def test_critique_issue_type_values(self):
        assert CritiqueIssueType.UNSTATED_HYPOTHESIS == "unstated_hypothesis"
        assert CritiqueIssueType.SWAPPED_QUANTIFIER == "swapped_quantifier"
        assert CritiqueIssueType.CIRCULAR_REASONING == "circular_reasoning"
        assert CritiqueIssueType.INCOMPLETE_DECOMPOSITION == "incomplete_decomposition"

    def test_critique_issue_defaults(self):
        issue = CritiqueIssue(
            issue_type=CritiqueIssueType.UNDEFINED_TERM,
            node_id="lemma_1",
            description="undefined term",
        )
        assert issue.severity == "warning"
        assert issue.suggested_fix == ""
        assert not issue.confirmed

    def test_critique_result_defaults(self):
        result = CritiqueResult()
        assert result.passed
        assert result.issues == []

    def test_critique_result_serialization(self):
        result = CritiqueResult(
            issues=[
                CritiqueIssue(
                    issue_type=CritiqueIssueType.WEAK_CHILD_LEMMA,
                    node_id="lemma_1",
                    description="too weak",
                    confirmed=True,
                )
            ],
            passed=False,
        )
        restored = CritiqueResult.model_validate(result.model_dump())
        assert not restored.passed
        assert len(restored.issues) == 1
        assert restored.issues[0].issue_type == CritiqueIssueType.WEAK_CHILD_LEMMA
