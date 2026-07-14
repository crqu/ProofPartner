"""Tests for search-augmented leanification and adaptive thinking.

Covers:
- LemmaLeanifier with LeanSearch mock (search results in prompt)
- LemmaLeanifier without LeanSearch (backward compatibility)
- LemmaLeanifier search cache behaviour
- Adaptive thinking type in LLM client
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.proof import LemmaTree, ProofNode


def _mock_llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = [_mock_llm_response(text) for text in responses]
    mock.complete.side_effect = side_effects
    return mock


def _make_mock_repl():
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


def _make_mock_search():
    from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

    return LeanSearch(SearchConfig(backend=SearchBackend.MOCK))


def _make_tree() -> LemmaTree:
    return LemmaTree(
        root_id="root",
        nodes={
            "root": ProofNode(
                node_id="root",
                statement_nl="main theorem about addition",
                statement_lean="theorem main := sorry",
                children=["lemma_1"],
            ),
            "lemma_1": ProofNode(
                node_id="lemma_1",
                statement_nl="addition commutative natural numbers",
                depth=1,
                parent_id="root",
            ),
        },
        topological_order=["lemma_1", "root"],
    )


class TestLemmaLeanifierWithSearch:
    """LemmaLeanifier with LeanSearch mock — verify search results appear in prompt."""

    def test_search_results_in_prompt(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        tree = _make_tree()
        llm = _make_mock_llm([
            "```lean\ntheorem l1 : True := sorry\n```",
        ])
        repl = _make_mock_repl()
        search = _make_mock_search()

        agent = LemmaLeanifier(
            llm_client=llm, lean_repl=repl, lean_search=search,
        )
        ctx = AgentContext(
            task="leanify",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "Relevant Mathlib Theorems" in user_content
        assert "Nat.add_comm" in user_content


class TestLemmaLeanifierWithoutSearch:
    """LemmaLeanifier without LeanSearch (None) — backward compatibility."""

    def test_no_search_still_works(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        tree = _make_tree()
        llm = _make_mock_llm([
            "```lean\ntheorem l1 : True := sorry\n```",
        ])
        repl = _make_mock_repl()

        agent = LemmaLeanifier(llm_client=llm, lean_repl=repl)
        ctx = AgentContext(
            task="leanify",
            metadata={"lemma_tree": tree.model_dump()},
        )
        result = agent.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        call_args = llm.complete.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "Relevant Mathlib Theorems" not in user_content


class TestLemmaLeanifierSearchCache:
    """Verify same query string returns cached result without re-calling search."""

    def test_cache_prevents_duplicate_search(self):
        from agentic_research.agents.lemma_leanifier import LemmaLeanifier

        tree = LemmaTree(
            root_id="root",
            nodes={
                "root": ProofNode(
                    node_id="root",
                    statement_nl="main theorem",
                    statement_lean="theorem main := sorry",
                    children=["lemma_1", "lemma_2"],
                ),
                "lemma_1": ProofNode(
                    node_id="lemma_1",
                    statement_nl="addition commutative natural numbers",
                    depth=1,
                    parent_id="root",
                ),
                "lemma_2": ProofNode(
                    node_id="lemma_2",
                    statement_nl="addition commutative natural numbers",
                    depth=1,
                    parent_id="root",
                ),
            },
            topological_order=["lemma_1", "lemma_2", "root"],
        )

        llm = _make_mock_llm([
            "```lean\ntheorem l1 : True := sorry\n```",
            "```lean\ntheorem l2 : True := sorry\n```",
        ])
        repl = _make_mock_repl()
        search = _make_mock_search()
        original_execute = search.execute
        execute_calls: list[str] = []

        def tracking_execute(query):
            execute_calls.append(query)
            return original_execute(query)

        search.execute = tracking_execute

        agent = LemmaLeanifier(
            llm_client=llm, lean_repl=repl, lean_search=search,
        )
        ctx = AgentContext(
            task="leanify",
            metadata={"lemma_tree": tree.model_dump()},
        )
        agent.run(ctx)

        assert len(execute_calls) == 1


class TestAdaptiveThinking:
    """Verify the thinking type is 'adaptive' in LLM calls."""

    def _make_client(self):
        with patch("anthropic.Anthropic"):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                from agentic_research.agents.llm_client import LLMClient

                return LLMClient()

    def _mock_response(self):
        resp = MagicMock()
        resp.content = [MagicMock(type="text", text="ok")]
        resp.stop_reason = "end_turn"
        resp.model = "claude-opus-4-6-20250616"
        resp.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return resp

    def test_thinking_type_is_adaptive(self):
        client = self._make_client()
        client._client.messages.create = MagicMock(return_value=self._mock_response())

        client.complete(
            messages=[{"role": "user", "content": "hi"}],
            use_extended_thinking=True,
            thinking_budget=10000,
        )

        call_kwargs = client._client.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive"}

    def test_no_thinking_when_disabled(self):
        client = self._make_client()
        client._client.messages.create = MagicMock(return_value=self._mock_response())

        client.complete(
            messages=[{"role": "user", "content": "hi"}],
            use_extended_thinking=False,
        )

        call_kwargs = client._client.messages.create.call_args[1]
        assert "thinking" not in call_kwargs
