"""Tests for the external prover backend (Leanstral / OpenAI-compatible API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner

from agentic_research.models.agents import TokenUsage
from agentic_research.models.external_prover import ExternalProverConfig, ExternalProverResult
from agentic_research.tools.external_prover import ExternalProverClient, _extract_proof_from_response


# ---------------------------------------------------------------------------
# models/external_prover.py
# ---------------------------------------------------------------------------


class TestExternalProverConfig:
    def test_defaults(self):
        cfg = ExternalProverConfig(api_url="http://localhost:8000")
        assert cfg.model_name == "leanstral-1.5"
        assert cfg.timeout == 120
        assert cfg.max_tokens == 8192
        assert cfg.api_key is None

    def test_custom(self):
        cfg = ExternalProverConfig(
            api_url="http://prover.example.com/v1",
            api_key="sk-test",
            model_name="leanstral-2.0",
            timeout=60,
            max_tokens=4096,
        )
        assert cfg.api_key == "sk-test"
        assert cfg.model_name == "leanstral-2.0"

    def test_serialization_roundtrip(self):
        cfg = ExternalProverConfig(api_url="http://localhost:8000", api_key="key")
        restored = ExternalProverConfig.model_validate(cfg.model_dump())
        assert restored == cfg


class TestExternalProverResult:
    def test_defaults(self):
        r = ExternalProverResult()
        assert not r.success
        assert r.proof_code is None
        assert r.error is None
        assert r.tokens_used.total_tokens == 0

    def test_success(self):
        r = ExternalProverResult(
            success=True,
            proof_code="by simp",
            tokens_used=TokenUsage(input_tokens=100, output_tokens=50),
        )
        assert r.success
        assert r.proof_code == "by simp"
        assert r.tokens_used.total_tokens == 150


# ---------------------------------------------------------------------------
# tools/external_prover.py — _extract_proof_from_response
# ---------------------------------------------------------------------------


class TestExtractProof:
    def test_lean_code_block(self):
        text = 'Here is the proof:\n```lean\nby simp [Nat.add_comm]\n```\nDone.'
        assert _extract_proof_from_response(text) == "by simp [Nat.add_comm]"

    def test_plain_code_block(self):
        text = '```\nby omega\n```'
        assert _extract_proof_from_response(text) == "by omega"

    def test_bare_text(self):
        text = "by ring"
        assert _extract_proof_from_response(text) == "by ring"

    def test_empty_code_block(self):
        text = "```lean\n\n```"
        assert _extract_proof_from_response(text) is None

    def test_no_content(self):
        assert _extract_proof_from_response("") is None


# ---------------------------------------------------------------------------
# tools/external_prover.py — ExternalProverClient
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> ExternalProverConfig:
    defaults = {"api_url": "http://localhost:8000/v1"}
    defaults.update(overrides)
    return ExternalProverConfig(**defaults)


_DUMMY_REQUEST = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")


def _ok_response(proof: str = "by simp", prompt_tokens: int = 50, completion_tokens: int = 30) -> httpx.Response:
    body = {
        "choices": [{"message": {"content": f"```lean\n{proof}\n```"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return httpx.Response(200, json=body, request=_DUMMY_REQUEST)


class TestExternalProverClient:
    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_successful_proof(self, mock_post):
        mock_post.return_value = _ok_response("by omega")
        client = ExternalProverClient(_make_config())
        result = client.prove("theorem foo : 1 + 1 = 2 := sorry")

        assert result.success
        assert result.proof_code == "by omega"
        assert result.tokens_used.input_tokens == 50
        assert result.tokens_used.output_tokens == 30
        assert result.error is None

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_api_key_sent(self, mock_post):
        mock_post.return_value = _ok_response()
        client = ExternalProverClient(_make_config(api_key="sk-test"))
        client.prove("theorem foo := sorry")

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-test"

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_no_api_key(self, mock_post):
        mock_post.return_value = _ok_response()
        client = ExternalProverClient(_make_config())
        client.prove("theorem foo := sorry")

        call_kwargs = mock_post.call_args
        assert "Authorization" not in call_kwargs.kwargs["headers"]

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_timeout(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timed out")
        client = ExternalProverClient(_make_config())
        result = client.prove("theorem foo := sorry")

        assert not result.success
        assert result.error == "Request timed out"

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_http_error(self, mock_post):
        mock_post.side_effect = httpx.HTTPStatusError(
            "error",
            request=httpx.Request("POST", "http://localhost:8000"),
            response=httpx.Response(500),
        )
        client = ExternalProverClient(_make_config())
        result = client.prove("theorem foo := sorry")

        assert not result.success
        assert "500" in result.error

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_malformed_json(self, mock_post):
        mock_post.return_value = httpx.Response(200, text="not json", headers={"content-type": "text/plain"}, request=_DUMMY_REQUEST)
        client = ExternalProverClient(_make_config())
        result = client.prove("theorem foo := sorry")

        assert not result.success
        assert "Malformed JSON" in result.error

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_no_choices(self, mock_post):
        mock_post.return_value = httpx.Response(200, json={"choices": [], "usage": {}}, request=_DUMMY_REQUEST)
        client = ExternalProverClient(_make_config())
        result = client.prove("theorem foo := sorry")

        assert not result.success
        assert "No choices" in result.error

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_no_proof_in_response(self, mock_post):
        body = {"choices": [{"message": {"content": ""}}], "usage": {}}
        mock_post.return_value = httpx.Response(200, json=body, request=_DUMMY_REQUEST)
        client = ExternalProverClient(_make_config())
        result = client.prove("theorem foo := sorry")

        assert not result.success
        assert "extract proof" in result.error

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_url_appends_chat_completions(self, mock_post):
        mock_post.return_value = _ok_response()
        client = ExternalProverClient(_make_config(api_url="http://localhost:8000/v1"))
        client.prove("theorem foo := sorry")

        called_url = mock_post.call_args.args[0]
        assert called_url == "http://localhost:8000/v1/chat/completions"

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_url_no_double_suffix(self, mock_post):
        mock_post.return_value = _ok_response()
        client = ExternalProverClient(_make_config(api_url="http://localhost:8000/v1/chat/completions"))
        client.prove("theorem foo := sorry")

        called_url = mock_post.call_args.args[0]
        assert called_url == "http://localhost:8000/v1/chat/completions"

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_budget_tokens_override(self, mock_post):
        mock_post.return_value = _ok_response()
        client = ExternalProverClient(_make_config(max_tokens=8192))
        client.prove("theorem foo := sorry", budget_tokens=2048)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# pipelines/proof.py — external prover integration
# ---------------------------------------------------------------------------


def _make_mock_repl():
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


def _make_mock_search():
    from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig
    return LeanSearch(SearchConfig(backend=SearchBackend.MOCK))


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient
    from agentic_research.models.agents import LLMResponse

    mock = MagicMock(spec=LLMClient)
    side_effects = [
        LLMResponse(
            content=text,
            model="claude-opus-4-6-20250616",
            stop_reason="end_turn",
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
        )
        for text in responses
    ]
    mock.complete.side_effect = side_effects
    return mock


class TestProofPipelineWithExternalProver:
    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_external_prover_success(self, mock_post):
        from agentic_research.pipelines.proof import ProofPipeline

        mock_post.return_value = _ok_response("by omega")
        config = _make_config()

        llm = _make_mock_llm([])
        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            use_external_prover=True,
            external_prover_config=config,
            use_claim_check=False,
        )

        result = pipeline.run("theorem foo : 1 + 1 = 2 := sorry")
        assert result.proved
        assert result.final_proof == "by omega"
        llm.complete.assert_not_called()

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_external_prover_failure_falls_back(self, mock_post):
        from agentic_research.pipelines.proof import ProofPipeline

        mock_post.side_effect = httpx.TimeoutException("timed out")
        config = _make_config()

        strategies_json = '{"strategies": [{"strategy_type": "direct", "description": "direct", "plausibility": 0.9, "relevant_lemmas": [], "key_tactics": ["simp"]}]}'
        llm = _make_mock_llm([
            strategies_json,
            "```lean\ntheorem foo : True := trivial\n```",
        ])

        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            use_external_prover=True,
            external_prover_config=config,
            use_claim_check=False,
            max_strategies=1,
        )

        result = pipeline.run("theorem foo : True")
        assert result.proved
        assert llm.complete.call_count >= 1

    @patch("agentic_research.tools.external_prover.httpx.post")
    def test_external_prover_tokens_accumulated(self, mock_post):
        from agentic_research.pipelines.proof import ProofPipeline

        mock_post.return_value = _ok_response("by simp", prompt_tokens=200, completion_tokens=100)
        config = _make_config()

        pipeline = ProofPipeline(
            llm_client=_make_mock_llm([]),
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            use_external_prover=True,
            external_prover_config=config,
            use_claim_check=False,
        )

        result = pipeline.run("theorem foo := sorry")
        assert result.total_token_usage.input_tokens == 200
        assert result.total_token_usage.output_tokens == 100

    def test_external_prover_disabled_uses_builtin(self):
        from agentic_research.pipelines.proof import ProofPipeline
        from agentic_research.models.agents import ProverConfig

        strategies_json = '{"strategies": [{"strategy_type": "direct", "description": "direct", "plausibility": 0.9, "relevant_lemmas": [], "key_tactics": ["simp"]}]}'
        llm = _make_mock_llm([
            strategies_json,
            "```lean\ntheorem foo : True := trivial\n```",
        ])

        pipeline = ProofPipeline(
            llm_client=llm,
            lean_repl=_make_mock_repl(),
            lean_search=_make_mock_search(),
            use_external_prover=False,
            use_claim_check=False,
            prover_config=ProverConfig(max_iterations=1),
            max_strategies=1,
        )

        result = pipeline.run("theorem foo : True")
        assert result.proved
        assert llm.complete.call_count >= 1


# ---------------------------------------------------------------------------
# cli/main.py — --backend option
# ---------------------------------------------------------------------------


class TestCLIBackendOption:
    def test_backend_builtin_default(self):
        from agentic_research.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["prove", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output
        assert "builtin" in result.output
        assert "leanstral" in result.output

    def test_backend_leanstral_missing_url(self):
        from agentic_research.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["prove", "--backend", "leanstral", "theorem foo := sorry"], env={"LEANSTRAL_API_URL": ""})
        assert result.exit_code != 0
        assert "LEANSTRAL_API_URL" in result.output

    def test_backend_leanstral_with_url(self):
        from agentic_research.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["prove", "--backend", "leanstral", "theorem foo := sorry"],
            input="n\n",
            env={"LEANSTRAL_API_URL": "http://localhost:8000/v1"},
        )
        assert "Backend: leanstral" in result.output

    def test_backend_builtin_explicit(self):
        from agentic_research.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["prove", "--backend", "builtin", "theorem foo := sorry"],
            input="n\n",
        )
        assert "Backend: builtin" in result.output
