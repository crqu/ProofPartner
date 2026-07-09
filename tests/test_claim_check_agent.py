"""Tests for the ClaimCheck agent parse-failure default and verdict parsing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agentic_research.agents.claim_check import ClaimCheck
from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.formalization import ClaimCheckResult, ClaimCheckVerdict


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = []
    for text in responses:
        side_effects.append(LLMResponse(
            content=text,
            model="claude-opus-4-6-20250616",
            stop_reason="end_turn",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        ))
    mock.complete.side_effect = side_effects

    real_client_cls = LLMClient
    with patch("anthropic.Anthropic"):
        temp_client = real_client_cls.__new__(real_client_cls)
    mock.extract_json = temp_client.__class__.extract_json.__get__(mock, type(mock))

    return mock


def _run_claim_check(llm: MagicMock, *, use_llm_check: bool = True) -> ClaimCheckResult:
    checker = ClaimCheck(llm_client=llm, use_llm_check=use_llm_check)
    ctx = AgentContext(
        task="All primes > 2 are odd",
        metadata={
            "lean_code": "theorem primes_odd : True := sorry",
            "type_definitions": "",
        },
    )
    result = checker.run(ctx)
    assert result.status == AgentStatus.SUCCESS
    return ClaimCheckResult.model_validate(result.result)


class TestClaimCheckParseFailure:
    def test_claim_check_parse_failure_defaults_to_pass(self):
        llm = _make_mock_llm(["This is not valid JSON at all"])
        claim = _run_claim_check(llm)
        assert claim.verdict == ClaimCheckVerdict.PASS
        assert "parse error" in claim.reason.lower()

    def test_claim_check_valid_pass_verdict(self):
        response = json.dumps({
            "verdict": "pass",
            "reason": "ok",
            "statement_preserved": True,
        })
        llm = _make_mock_llm([response])
        claim = _run_claim_check(llm)
        assert claim.verdict == ClaimCheckVerdict.PASS
        assert claim.reason == "ok"
        assert claim.statement_preserved is True

    def test_claim_check_valid_fail_verdict(self):
        response = json.dumps({
            "verdict": "fail",
            "reason": "weakened",
            "statement_preserved": False,
        })
        llm = _make_mock_llm([response])
        claim = _run_claim_check(llm)
        assert claim.verdict == ClaimCheckVerdict.FAIL
        assert claim.reason == "weakened"
        assert claim.statement_preserved is False

    def test_claim_check_llm_disabled(self):
        llm = _make_mock_llm([])
        claim = _run_claim_check(llm, use_llm_check=False)
        assert claim.verdict == ClaimCheckVerdict.PASS
        assert "disabled" in claim.reason.lower()
        assert llm.complete.call_count == 0
