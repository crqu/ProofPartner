"""Tests for the agent framework (Phase 3).

All LLM calls are mocked — no real API calls are made.
Lean REPL uses mock backend for deterministic testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    LLMResponse,
    ProofAttempt,
    ProofAttemptStatus,
    ProverConfig,
    ProverResult,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# models/agents.py
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_defaults_zero(self):
        usage = TokenUsage()
        assert usage.total_tokens == 0
        assert usage.cache_creation_input_tokens == 0

    def test_serialization_roundtrip(self):
        usage = TokenUsage(input_tokens=10, output_tokens=20, cache_read_input_tokens=5)
        restored = TokenUsage.model_validate(usage.model_dump())
        assert restored == usage


class TestLLMResponse:
    def test_basic_fields(self):
        resp = LLMResponse(content="hello", model="claude-opus-4-6-20250616", stop_reason="end_turn")
        assert resp.content == "hello"
        assert resp.thinking is None

    def test_with_thinking(self):
        resp = LLMResponse(content="answer", thinking="Let me think...")
        assert resp.thinking == "Let me think..."


class TestAgentResult:
    def test_success(self):
        result = AgentResult(agent_name="test", status=AgentStatus.SUCCESS)
        assert result.attempts == 1
        assert result.error_message is None

    def test_serialization_roundtrip(self):
        result = AgentResult(
            agent_name="test",
            status=AgentStatus.FAILURE,
            result={"key": "value"},
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        restored = AgentResult.model_validate(result.model_dump())
        assert restored == result


class TestProverConfig:
    def test_defaults(self):
        config = ProverConfig()
        assert config.max_iterations == 5
        assert config.model == "claude-opus-4-6-20250616"
        assert config.temperature == 0.0

    def test_custom(self):
        config = ProverConfig(max_iterations=10, temperature=0.5)
        assert config.max_iterations == 10
        assert config.temperature == 0.5


class TestProofAttempt:
    def test_success_attempt(self):
        attempt = ProofAttempt(
            iteration=1,
            proof_code="theorem foo : True := trivial",
            status=ProofAttemptStatus.SUCCESS,
        )
        assert attempt.errors == []
        assert attempt.goals_remaining == []

    def test_error_attempt(self):
        attempt = ProofAttempt(
            iteration=2,
            proof_code="bad code",
            status=ProofAttemptStatus.COMPILATION_ERROR,
            errors=["unknown identifier"],
        )
        assert len(attempt.errors) == 1


class TestProverResult:
    def test_proved(self):
        result = ProverResult(
            statement="theorem foo : True",
            proved=True,
            final_proof="theorem foo : True := trivial",
        )
        assert result.proved
        assert result.failure_reason is None

    def test_not_proved(self):
        result = ProverResult(
            statement="theorem foo : True",
            proved=False,
            failure_reason="exhausted iterations",
        )
        assert not result.proved


# ---------------------------------------------------------------------------
# agents/base.py
# ---------------------------------------------------------------------------


class TestBaseAgent:
    def _make_agent(self, *, succeeds: bool = True, max_retries: int = 3):
        from agentic_research.agents.base import BaseAgent

        class _TestAgent(BaseAgent):
            def __init__(self, succeeds, max_retries):
                super().__init__(name="test_agent", max_retries=max_retries)
                self._succeeds = succeeds
                self.call_count = 0

            def _execute(self, context):
                self.call_count += 1
                if not self._succeeds:
                    raise RuntimeError("agent failed")
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.SUCCESS,
                    token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                )

        return _TestAgent(succeeds, max_retries)

    def test_successful_run(self):
        agent = self._make_agent(succeeds=True)
        ctx = AgentContext(task="prove something")
        result = agent.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        assert result.attempts == 1
        assert result.duration_seconds >= 0
        assert agent.call_count == 1

    def test_retry_exhaustion(self):
        agent = self._make_agent(succeeds=False, max_retries=2)
        ctx = AgentContext(task="prove something")
        result = agent.run(ctx)
        assert result.status == AgentStatus.MAX_RETRIES
        assert result.attempts == 2
        assert agent.call_count == 2
        assert "agent failed" in result.error_message

    def test_token_accumulation(self):
        agent = self._make_agent(succeeds=True)
        ctx = AgentContext(task="task")
        agent.run(ctx)
        agent.run(ctx)
        assert agent.cumulative_tokens.input_tokens == 20
        assert agent.cumulative_tokens.output_tokens == 10

    def test_properties(self):
        agent = self._make_agent(succeeds=True, max_retries=5)
        assert agent.name == "test_agent"
        assert agent.max_retries == 5


# ---------------------------------------------------------------------------
# agents/llm_client.py
# ---------------------------------------------------------------------------


class TestLLMClient:
    def test_missing_api_key_raises(self):
        from agentic_research.agents.llm_client import LLMClient, LLMClientError

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMClientError, match="ANTHROPIC_API_KEY"):
                LLMClient()

    def test_explicit_api_key(self):
        from agentic_research.agents.llm_client import LLMClient

        with patch("anthropic.Anthropic"):
            client = LLMClient(api_key="test-key-123")
            assert client.model == "claude-opus-4-6-20250616"

    def test_complete_basic(self):
        from agentic_research.agents.llm_client import LLMClient

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="proof code here")]
        mock_response.model = "claude-opus-4-6-20250616"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = LLMClient(api_key="test-key")
            result = client.complete(
                system="You are helpful.",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result.content == "proof code here"
            assert result.token_usage.input_tokens == 100
            assert result.token_usage.output_tokens == 50
            assert result.stop_reason == "end_turn"

    def test_complete_with_thinking(self):
        from agentic_research.agents.llm_client import LLMClient

        thinking_block = MagicMock(type="thinking", thinking="Deep thought...")
        text_block = MagicMock(type="text", text="Answer")
        mock_response = MagicMock()
        mock_response.content = [thinking_block, text_block]
        mock_response.model = "claude-opus-4-6-20250616"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(
            input_tokens=200, output_tokens=100,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = LLMClient(api_key="test-key")
            result = client.complete(
                messages=[{"role": "user", "content": "Think hard"}],
                use_extended_thinking=True,
            )

            assert result.thinking == "Deep thought..."
            assert result.content == "Answer"
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["temperature"] == 1
            assert "thinking" in call_kwargs

    def test_complete_with_cache(self):
        from agentic_research.agents.llm_client import LLMClient

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="cached")]
        mock_response.model = "claude-opus-4-6-20250616"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(
            input_tokens=50, output_tokens=25,
            cache_creation_input_tokens=200, cache_read_input_tokens=0,
        )

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = LLMClient(api_key="test-key")
            result = client.complete(
                system="System prompt",
                messages=[{"role": "user", "content": "Hello"}],
                use_cache=True,
            )

            assert result.token_usage.cache_creation_input_tokens == 200
            call_kwargs = mock_client.messages.create.call_args[1]
            assert isinstance(call_kwargs["system"], list)
            assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_extract_json_from_code_block(self):
        from agentic_research.agents.llm_client import LLMClient

        with patch("anthropic.Anthropic"):
            client = LLMClient(api_key="test-key")

        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = client.extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_raw(self):
        from agentic_research.agents.llm_client import LLMClient

        with patch("anthropic.Anthropic"):
            client = LLMClient(api_key="test-key")

        text = 'Here is the answer: {"proof": "by simp"}'
        result = client.extract_json(text)
        assert result == {"proof": "by simp"}

    def test_extract_json_array(self):
        from agentic_research.agents.llm_client import LLMClient

        with patch("anthropic.Anthropic"):
            client = LLMClient(api_key="test-key")

        text = 'Results: [1, 2, 3]'
        result = client.extract_json(text)
        assert result == [1, 2, 3]

    def test_extract_json_none_on_invalid(self):
        from agentic_research.agents.llm_client import LLMClient

        with patch("anthropic.Anthropic"):
            client = LLMClient(api_key="test-key")

        result = client.extract_json("no json here")
        assert result is None


# ---------------------------------------------------------------------------
# agents/prompt_templates.py
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_proof_attempt_template(self):
        from agentic_research.agents.prompt_templates import PROOF_ATTEMPT_TEMPLATE

        rendered = PROOF_ATTEMPT_TEMPLATE.format(statement="theorem foo : True := sorry")
        assert "theorem foo : True := sorry" in rendered

    def test_error_feedback_template(self):
        from agentic_research.agents.prompt_templates import ERROR_FEEDBACK_TEMPLATE

        rendered = ERROR_FEEDBACK_TEMPLATE.format(
            statement="theorem foo : True := sorry",
            previous_attempt="theorem foo : True := by omega",
            errors="type mismatch",
            goals="⊢ True",
        )
        assert "type mismatch" in rendered
        assert "omega" in rendered

    def test_conjecture_template(self):
        from agentic_research.agents.prompt_templates import CONJECTURE_GENERATION_TEMPLATE

        rendered = CONJECTURE_GENERATION_TEMPLATE.format(
            idea="prime gaps",
            context="number theory",
        )
        assert "prime gaps" in rendered

    def test_system_prompt_has_tactics(self):
        from agentic_research.agents.prompt_templates import LEAN4_PROVER_SYSTEM

        assert "simp" in LEAN4_PROVER_SYSTEM
        assert "omega" in LEAN4_PROVER_SYSTEM
        assert "Mathlib" in LEAN4_PROVER_SYSTEM


# ---------------------------------------------------------------------------
# agents/prover.py
# ---------------------------------------------------------------------------


def _make_mock_llm_client(responses: list[str]) -> MagicMock:
    """Create a mock LLM client returning the given proof texts in sequence."""
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = []
    for text in responses:
        side_effects.append(LLMResponse(
            content=f"```lean\n{text}\n```",
            model="claude-opus-4-6-20250616",
            stop_reason="end_turn",
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
        ))
    mock.complete.side_effect = side_effects
    return mock


class TestIterativeProver:
    def test_prove_success_first_try(self):
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm_client(["theorem foo : True := trivial"])

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=3),
        )

        ctx = AgentContext(task="theorem foo : True")
        result = prover.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        prover_result = ProverResult.model_validate(result.result)
        assert prover_result.proved
        assert prover_result.total_iterations == 1
        assert len(prover_result.attempts) == 1
        assert prover_result.attempts[0].status == ProofAttemptStatus.SUCCESS

    def test_prove_after_retry(self):
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm_client([
            "-- MOCK_ERROR\ntheorem foo : True := bad",
            "theorem foo : True := trivial",
        ])

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=5),
        )

        ctx = AgentContext(task="theorem foo : True")
        result = prover.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        prover_result = ProverResult.model_validate(result.result)
        assert prover_result.proved
        assert prover_result.total_iterations == 2
        assert prover_result.attempts[0].status == ProofAttemptStatus.COMPILATION_ERROR
        assert prover_result.attempts[1].status == ProofAttemptStatus.SUCCESS

    def test_prove_exhausts_iterations(self):
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm_client([
            "-- MOCK_ERROR\nbad1",
            "-- MOCK_ERROR\nbad2",
            "-- MOCK_ERROR\nbad3",
        ])

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=3),
        )

        ctx = AgentContext(task="theorem foo : True")
        result = prover.run(ctx)

        assert result.status == AgentStatus.FAILURE
        prover_result = ProverResult.model_validate(result.result)
        assert not prover_result.proved
        assert prover_result.total_iterations == 3
        assert len(prover_result.attempts) == 3
        assert "3 iterations" in prover_result.failure_reason

    def test_prove_incomplete_goals(self):
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm_client([
            "theorem foo : True := by sorry",
            "theorem foo : True := trivial",
        ])

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=5),
        )

        ctx = AgentContext(task="theorem foo : True")
        result = prover.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        prover_result = ProverResult.model_validate(result.result)
        assert prover_result.proved
        assert prover_result.attempts[0].status == ProofAttemptStatus.INCOMPLETE
        assert len(prover_result.attempts[0].goals_remaining) > 0

    def test_prover_token_tracking(self):
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm_client(["theorem foo : True := trivial"])

        prover = IterativeProver(
            llm_client=llm,
            lean_repl=repl,
            config=ProverConfig(max_iterations=3),
        )

        ctx = AgentContext(task="theorem foo : True")
        result = prover.run(ctx)

        assert result.token_usage.input_tokens == 50
        assert result.token_usage.output_tokens == 30

    def test_prover_config_defaults(self):
        from agentic_research.agents.prover import IterativeProver
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        llm = _make_mock_llm_client([])

        prover = IterativeProver(llm_client=llm, lean_repl=repl)
        assert prover.config.max_iterations == 5
        assert prover.config.model == "claude-opus-4-6-20250616"
        assert prover.name == "iterative_prover"


class TestExtractLeanCode:
    def test_from_code_block(self):
        from agentic_research.agents.prover import _extract_lean_code

        text = "Here is the proof:\n```lean\ntheorem foo := trivial\n```\nDone."
        assert _extract_lean_code(text) == "theorem foo := trivial"

    def test_raw_code(self):
        from agentic_research.agents.prover import _extract_lean_code

        text = "theorem foo := trivial"
        assert _extract_lean_code(text) == "theorem foo := trivial"

    def test_strips_whitespace(self):
        from agentic_research.agents.prover import _extract_lean_code

        text = "  \n  theorem foo := trivial  \n  "
        assert _extract_lean_code(text) == "theorem foo := trivial"
