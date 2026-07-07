"""Tests for Phase 8: Conjecture Refinement Loop.

All LLM calls are mocked — no real API calls are made.
Lean REPL uses mock backend for deterministic testing.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from agentic_research.models.agents import (
    AgentContext,
    AgentStatus,
    LLMResponse,
    TokenUsage,
)
from agentic_research.models.refinement import (
    RefinementAttempt,
    RefinementHistory,
    RefinementOutcome,
    RefinementReport,
    RefinementResult,
    RefinementStatus,
    RefinementType,
)
from agentic_research.models.research import Conjecture


# ---------------------------------------------------------------------------
# models/refinement.py
# ---------------------------------------------------------------------------


class TestRefinementType:
    def test_values(self):
        assert RefinementType.WEAKENING == "weakening"
        assert RefinementType.STRENGTHENING == "strengthening"
        assert RefinementType.REFORMULATION == "reformulation"
        assert RefinementType.SPECIALIZATION == "specialization"


class TestRefinementOutcome:
    def test_values(self):
        assert RefinementOutcome.PROVED == "proved"
        assert RefinementOutcome.DISPROVED == "disproved"
        assert RefinementOutcome.FORMALIZATION_FAILED == "formalization_failed"
        assert RefinementOutcome.INTENT_MISMATCH == "intent_mismatch"
        assert RefinementOutcome.PROOF_FAILED == "proof_failed"
        assert RefinementOutcome.SKIPPED == "skipped"


class TestRefinementAttempt:
    def _make_conjecture(self, stmt: str = "test") -> Conjecture:
        return Conjecture(
            statement=stmt,
            natural_language=f"NL: {stmt}",
            confidence=0.5,
            difficulty=3,
        )

    def test_defaults(self):
        a = RefinementAttempt(
            original=self._make_conjecture("orig"),
            refined=self._make_conjecture("refined"),
            refinement_type=RefinementType.WEAKENING,
        )
        assert a.outcome == RefinementOutcome.SKIPPED
        assert a.failure_reason == ""
        assert a.proof_code is None
        assert a.depth == 0

    def test_proved(self):
        a = RefinementAttempt(
            original=self._make_conjecture("orig"),
            refined=self._make_conjecture("refined"),
            refinement_type=RefinementType.SPECIALIZATION,
            outcome=RefinementOutcome.PROVED,
            proof_code="theorem x := trivial",
            depth=2,
        )
        assert a.outcome == RefinementOutcome.PROVED
        assert a.proof_code is not None
        assert a.depth == 2

    def test_serialization_roundtrip(self):
        a = RefinementAttempt(
            original=self._make_conjecture("orig"),
            refined=self._make_conjecture("refined"),
            refinement_type=RefinementType.REFORMULATION,
            outcome=RefinementOutcome.DISPROVED,
            failure_reason="counterexample found",
        )
        restored = RefinementAttempt.model_validate(a.model_dump())
        assert restored.refinement_type == RefinementType.REFORMULATION
        assert restored.failure_reason == "counterexample found"


class TestRefinementHistory:
    def _make_conjecture(self, stmt: str = "test") -> Conjecture:
        return Conjecture(
            statement=stmt,
            natural_language=f"NL: {stmt}",
            confidence=0.5,
            difficulty=3,
        )

    def test_empty(self):
        h = RefinementHistory(original_idea="test idea")
        assert h.total_attempts == 0
        assert h.proved_variant is None

    def test_with_attempts(self):
        h = RefinementHistory(
            original_idea="test idea",
            original_conjecture=self._make_conjecture("original"),
            attempts=[
                RefinementAttempt(
                    original=self._make_conjecture("a"),
                    refined=self._make_conjecture("b"),
                    refinement_type=RefinementType.WEAKENING,
                    outcome=RefinementOutcome.DISPROVED,
                ),
                RefinementAttempt(
                    original=self._make_conjecture("b"),
                    refined=self._make_conjecture("c"),
                    refinement_type=RefinementType.SPECIALIZATION,
                    outcome=RefinementOutcome.PROVED,
                    proof_code="by trivial",
                ),
            ],
        )
        assert h.total_attempts == 2
        proved = h.proved_variant
        assert proved is not None
        assert proved.refined.statement == "c"
        assert proved.proof_code == "by trivial"

    def test_no_proved_variant(self):
        h = RefinementHistory(
            original_idea="test",
            attempts=[
                RefinementAttempt(
                    original=self._make_conjecture("a"),
                    refined=self._make_conjecture("b"),
                    refinement_type=RefinementType.WEAKENING,
                    outcome=RefinementOutcome.DISPROVED,
                ),
            ],
        )
        assert h.proved_variant is None


class TestRefinementReport:
    def test_defaults(self):
        r = RefinementReport()
        assert r.markdown_report == ""

    def test_with_content(self):
        h = RefinementHistory(original_idea="test idea")
        r = RefinementReport(
            markdown_report="# Report\nSome content",
            structured_history=h,
        )
        assert "Report" in r.markdown_report
        assert r.structured_history.original_idea == "test idea"


class TestRefinementResult:
    def _make_conjecture(self, stmt: str = "test") -> Conjecture:
        return Conjecture(
            statement=stmt,
            natural_language=f"NL: {stmt}",
            confidence=0.5,
            difficulty=3,
        )

    def test_proved(self):
        r = RefinementResult(
            status=RefinementStatus.PROVED,
            proved_variant=self._make_conjecture("proved variant"),
            proof_code="theorem x := trivial",
            max_depth_reached=2,
        )
        assert r.status == RefinementStatus.PROVED
        assert r.proved_variant is not None
        assert r.proof_code is not None
        assert r.max_depth_reached == 2

    def test_exhausted(self):
        r = RefinementResult(
            status=RefinementStatus.EXHAUSTED,
            max_depth_reached=3,
        )
        assert r.status == RefinementStatus.EXHAUSTED
        assert r.proved_variant is None

    def test_serialization_roundtrip(self):
        r = RefinementResult(
            status=RefinementStatus.PROVED,
            proved_variant=self._make_conjecture("x"),
            proof_code="by simp",
        )
        restored = RefinementResult.model_validate(r.model_dump())
        assert restored.status == RefinementStatus.PROVED
        assert restored.proved_variant.statement == "x"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    side_effects = [_mock_llm_response(text) for text in responses]
    mock.complete.side_effect = side_effects
    mock.extract_json.side_effect = lambda text: _extract_json_helper(text)
    return mock


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


def _make_mock_repl():
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

    return LeanRepl(ReplConfig(backend=ReplBackend.MOCK))


def _make_mock_search():
    from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

    return LeanSearch(SearchConfig(backend=SearchBackend.MOCK))


def _make_conjecture(stmt: str = "test", nl: str = "") -> Conjecture:
    return Conjecture(
        statement=stmt,
        natural_language=nl or f"NL: {stmt}",
        confidence=0.5,
        difficulty=3,
    )


# ---------------------------------------------------------------------------
# agents/conjecture_refiner.py
# ---------------------------------------------------------------------------


class TestConjectureRefiner:
    def _make_refiner_response(
        self, strategy: str, count: int = 2
    ) -> str:
        conjectures = []
        for i in range(count):
            conjectures.append({
                "statement": f"refined_{strategy}_{i}",
                "natural_language": f"Refined via {strategy} variant {i}",
                "confidence": 0.7,
                "difficulty": 2,
                "related_results": [],
                "novelty_score": 0.4,
                "formalizability_score": 0.8,
                "refinement_reasoning": f"Addresses failure via {strategy}",
            })
        return json.dumps({"refined_conjectures": conjectures})

    def test_weakening_strategy(self):
        from agentic_research.agents.conjecture_refiner import ConjectureRefiner

        response = self._make_refiner_response("weakening", 3)
        llm = _make_mock_llm([response])

        refiner = ConjectureRefiner(llm_client=llm)
        refined = refiner.refine(
            conjecture=_make_conjecture("all primes are even"),
            failure_reason="counterexample: 3",
            failure_outcome="disproved",
            original_idea="primes and evenness",
            strategy=RefinementType.WEAKENING,
        )

        assert len(refined) == 3
        assert all(isinstance(c, Conjecture) for c in refined)
        assert "weakening" in refined[0].statement

    def test_strengthening_strategy(self):
        from agentic_research.agents.conjecture_refiner import ConjectureRefiner

        response = self._make_refiner_response("strengthening", 2)
        llm = _make_mock_llm([response])

        refiner = ConjectureRefiner(llm_client=llm)
        refined = refiner.refine(
            conjecture=_make_conjecture("trivial statement"),
            failure_reason="too weak to be interesting",
            failure_outcome="proof_failed",
            original_idea="something deeper",
            strategy=RefinementType.STRENGTHENING,
        )

        assert len(refined) == 2

    def test_reformulation_strategy(self):
        from agentic_research.agents.conjecture_refiner import ConjectureRefiner

        response = self._make_refiner_response("reformulation", 2)
        llm = _make_mock_llm([response])

        refiner = ConjectureRefiner(llm_client=llm)
        refined = refiner.refine(
            conjecture=_make_conjecture("algebraic claim"),
            failure_reason="proof approach stuck",
            failure_outcome="proof_failed",
            original_idea="abstract algebra",
            strategy=RefinementType.REFORMULATION,
        )

        assert len(refined) == 2

    def test_specialization_strategy(self):
        from agentic_research.agents.conjecture_refiner import ConjectureRefiner

        response = self._make_refiner_response("specialization", 4)
        llm = _make_mock_llm([response])

        refiner = ConjectureRefiner(llm_client=llm)
        refined = refiner.refine(
            conjecture=_make_conjecture("for all groups G"),
            failure_reason="too general",
            failure_outcome="proof_failed",
            original_idea="group theory",
            strategy=RefinementType.SPECIALIZATION,
        )

        assert len(refined) == 4

    def test_max_4_variants(self):
        from agentic_research.agents.conjecture_refiner import ConjectureRefiner

        conjectures = [
            {
                "statement": f"variant_{i}",
                "natural_language": f"Variant {i}",
                "confidence": 0.5,
                "difficulty": 3,
            }
            for i in range(6)
        ]
        response = json.dumps({"refined_conjectures": conjectures})
        llm = _make_mock_llm([response])

        refiner = ConjectureRefiner(llm_client=llm)
        refined = refiner.refine(
            conjecture=_make_conjecture("x"),
            failure_reason="failed",
            failure_outcome="disproved",
            original_idea="idea",
            strategy=RefinementType.WEAKENING,
        )

        assert len(refined) <= 4

    def test_invalid_json_returns_empty(self):
        from agentic_research.agents.conjecture_refiner import ConjectureRefiner

        llm = _make_mock_llm(["not valid json at all"])
        refiner = ConjectureRefiner(llm_client=llm)
        refined = refiner.refine(
            conjecture=_make_conjecture("x"),
            failure_reason="failed",
            failure_outcome="disproved",
            original_idea="idea",
            strategy=RefinementType.WEAKENING,
        )
        assert refined == []

    def test_execute_via_context(self):
        from agentic_research.agents.conjecture_refiner import ConjectureRefiner

        response = self._make_refiner_response("weakening", 2)
        llm = _make_mock_llm([response])

        refiner = ConjectureRefiner(llm_client=llm)
        ctx = AgentContext(
            task="refine conjecture",
            metadata={
                "conjecture": _make_conjecture("bad claim").model_dump(),
                "failure_reason": "counterexample found",
                "failure_outcome": "disproved",
                "original_idea": "my idea",
                "strategy": "weakening",
            },
        )
        result = refiner.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        assert len(result.result["refined_conjectures"]) == 2

    def test_execute_missing_conjecture(self):
        from agentic_research.agents.conjecture_refiner import ConjectureRefiner

        llm = _make_mock_llm([])
        refiner = ConjectureRefiner(llm_client=llm)
        ctx = AgentContext(task="refine", metadata={})
        result = refiner.run(ctx)
        assert result.status == AgentStatus.FAILURE


# ---------------------------------------------------------------------------
# agents/refinement_reporter.py
# ---------------------------------------------------------------------------


class TestRefinementReporter:
    def test_generate_report(self):
        from agentic_research.agents.refinement_reporter import RefinementReporter

        llm = _make_mock_llm([
            "# Refinement Report\n\n## Original Conjecture\nTest conjecture.\n\n## Journey\nStep 1 failed.\n\n## Outcome\nExhausted."
        ])

        reporter = RefinementReporter(llm_client=llm)
        history = RefinementHistory(
            original_idea="test idea",
            original_conjecture=_make_conjecture("original"),
            attempts=[
                RefinementAttempt(
                    original=_make_conjecture("a"),
                    refined=_make_conjecture("b"),
                    refinement_type=RefinementType.WEAKENING,
                    outcome=RefinementOutcome.DISPROVED,
                    failure_reason="counterexample",
                    depth=1,
                ),
            ],
            final_result=RefinementStatus.EXHAUSTED,
        )

        report = reporter.generate_report(history)
        assert isinstance(report, RefinementReport)
        assert "Refinement Report" in report.markdown_report
        assert report.structured_history.total_attempts == 1

    def test_report_output_format(self):
        from agentic_research.agents.refinement_reporter import RefinementReporter

        llm = _make_mock_llm(["# Report\nContent here"])
        reporter = RefinementReporter(llm_client=llm)

        history = RefinementHistory(
            original_idea="idea",
            original_conjecture=_make_conjecture("x"),
            attempts=[
                RefinementAttempt(
                    original=_make_conjecture("x"),
                    refined=_make_conjecture("y"),
                    refinement_type=RefinementType.SPECIALIZATION,
                    outcome=RefinementOutcome.PROVED,
                    proof_code="by trivial",
                    depth=1,
                ),
            ],
            final_result=RefinementStatus.PROVED,
        )

        report = reporter.generate_report(history)
        assert report.markdown_report != ""
        assert report.structured_history.proved_variant is not None

    def test_execute_missing_history(self):
        from agentic_research.agents.refinement_reporter import RefinementReporter

        llm = _make_mock_llm([])
        reporter = RefinementReporter(llm_client=llm)
        ctx = AgentContext(task="report", metadata={})
        result = reporter.run(ctx)
        assert result.status == AgentStatus.FAILURE

    def test_format_attempts_empty(self):
        from agentic_research.agents.refinement_reporter import RefinementReporter

        llm = _make_mock_llm([])
        reporter = RefinementReporter(llm_client=llm)
        history = RefinementHistory(original_idea="test")
        text = reporter._format_attempts(history)
        assert "No refinement attempts" in text


# ---------------------------------------------------------------------------
# pipelines/refinement.py — max depth enforcement
# ---------------------------------------------------------------------------


class TestRefinementPipelineDepth:
    def test_max_depth_prevents_infinite_loop(self):
        from agentic_research.pipelines.refinement import RefinementPipeline

        refiner_response = json.dumps({
            "refined_conjectures": [{
                "statement": "refined_v",
                "natural_language": "Refined variant",
                "confidence": 0.7,
                "difficulty": 2,
            }]
        })

        responses = []
        for _ in range(20):
            responses.append(refiner_response)

        llm = _make_mock_llm(responses)
        repl = _make_mock_repl()
        search = _make_mock_search()

        pipeline = RefinementPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            max_depth=1,
            generate_report=False,
        )

        with patch.object(
            pipeline,
            "_evaluate_variant",
            return_value=(RefinementOutcome.PROOF_FAILED, None, "proof failed"),
        ):
            result = pipeline.run(
                conjecture=_make_conjecture("false claim"),
                failure_reason="counterexample",
                failure_outcome="disproved",
                original_idea="my idea",
            )

        assert result.status == RefinementStatus.EXHAUSTED
        assert result.max_depth_reached <= 2


class TestRefinementPipelineEndToEnd:
    def test_false_conjecture_weakened_to_proved(self):
        """End-to-end: false conjecture -> weakened -> proved."""
        from agentic_research.pipelines.refinement import RefinementPipeline

        weakened_response = json.dumps({
            "refined_conjectures": [{
                "statement": "weakened_statement",
                "natural_language": "Weakened variant that is provable",
                "confidence": 0.9,
                "difficulty": 1,
            }]
        })

        llm = _make_mock_llm([weakened_response])
        repl = _make_mock_repl()
        search = _make_mock_search()

        pipeline = RefinementPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            max_depth=3,
            generate_report=False,
        )

        with patch.object(
            pipeline,
            "_evaluate_variant",
            return_value=(RefinementOutcome.PROVED, "theorem x := trivial", ""),
        ):
            result = pipeline.run(
                conjecture=_make_conjecture(
                    "all even numbers > 2 are the sum of two primes",
                    nl="Goldbach conjecture",
                ),
                failure_reason="Cannot prove in general",
                failure_outcome="proof_failed",
                original_idea="Goldbach-type ideas",
            )

        assert result.status == RefinementStatus.PROVED
        assert result.proved_variant is not None
        assert result.proved_variant.statement == "weakened_statement"
        assert result.proof_code == "theorem x := trivial"
        assert result.history.total_attempts == 1

    def test_all_variants_fail_exhausts(self):
        """All refinement variants fail -> exhausted."""
        from agentic_research.pipelines.refinement import RefinementPipeline

        refiner_response = json.dumps({
            "refined_conjectures": [{
                "statement": "variant_a",
                "natural_language": "Variant A",
                "confidence": 0.6,
                "difficulty": 3,
            }]
        })

        llm = _make_mock_llm([refiner_response] * 10)
        repl = _make_mock_repl()
        search = _make_mock_search()

        pipeline = RefinementPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            max_depth=2,
            generate_report=False,
        )

        with patch.object(
            pipeline,
            "_evaluate_variant",
            return_value=(
                RefinementOutcome.FORMALIZATION_FAILED,
                None,
                "Cannot formalize",
            ),
        ):
            result = pipeline.run(
                conjecture=_make_conjecture("bad claim"),
                failure_reason="counterexample",
                failure_outcome="disproved",
                original_idea="idea",
            )

        assert result.status == RefinementStatus.EXHAUSTED
        assert result.proved_variant is None

    def test_disproved_variant_triggers_deeper_refinement(self):
        """A disproved variant should be queued for further refinement."""
        from agentic_research.pipelines.refinement import RefinementPipeline

        refiner_response_1 = json.dumps({
            "refined_conjectures": [{
                "statement": "depth1_variant",
                "natural_language": "Depth 1 variant",
                "confidence": 0.6,
                "difficulty": 3,
            }]
        })
        refiner_response_2 = json.dumps({
            "refined_conjectures": [{
                "statement": "depth2_variant",
                "natural_language": "Depth 2 variant",
                "confidence": 0.8,
                "difficulty": 2,
            }]
        })

        llm = _make_mock_llm([refiner_response_1, refiner_response_2])
        repl = _make_mock_repl()
        search = _make_mock_search()

        pipeline = RefinementPipeline(
            llm_client=llm,
            lean_repl=repl,
            lean_search=search,
            max_depth=3,
            generate_report=False,
        )

        call_count = [0]

        def mock_evaluate(conjecture, original_idea):
            call_count[0] += 1
            if call_count[0] == 1:
                return (RefinementOutcome.DISPROVED, None, "counterexample: n=3")
            return (RefinementOutcome.PROVED, "theorem x := trivial", "")

        with patch.object(pipeline, "_evaluate_variant", side_effect=mock_evaluate):
            result = pipeline.run(
                conjecture=_make_conjecture("false"),
                failure_reason="counterexample",
                failure_outcome="disproved",
                original_idea="idea",
            )

        assert result.status == RefinementStatus.PROVED
        assert result.history.total_attempts == 2
        assert result.max_depth_reached == 2


# ---------------------------------------------------------------------------
# Prompt templates (Phase 8)
# ---------------------------------------------------------------------------


class TestPhase8PromptTemplates:
    def test_refinement_user_template(self):
        from agentic_research.agents.prompt_templates import (
            CONJECTURE_REFINEMENT_USER_TEMPLATE,
        )

        rendered = CONJECTURE_REFINEMENT_USER_TEMPLATE.format(
            original_statement="all primes > 2 are odd",
            original_nl="Primes greater than 2 are odd",
            failure_outcome="disproved",
            failure_reason="counterexample: p=2",
            strategy="WEAKENING: add hypotheses",
            original_idea="prime number properties",
        )
        assert "all primes" in rendered
        assert "disproved" in rendered
        assert "WEAKENING" in rendered

    def test_refinement_report_user_template(self):
        from agentic_research.agents.prompt_templates import (
            REFINEMENT_REPORT_USER_TEMPLATE,
        )

        rendered = REFINEMENT_REPORT_USER_TEMPLATE.format(
            original_idea="my math idea",
            original_conjecture="all X are Y",
            attempts="Step 1: weakened, failed\nStep 2: specialized, proved",
            final_status="proved",
        )
        assert "my math idea" in rendered
        assert "proved" in rendered

    def test_refinement_system_prompt_exists(self):
        from agentic_research.agents.prompt_templates import (
            CONJECTURE_REFINEMENT_SYSTEM,
        )

        assert "Weakening" in CONJECTURE_REFINEMENT_SYSTEM
        assert "Strengthening" in CONJECTURE_REFINEMENT_SYSTEM
        assert "Reformulation" in CONJECTURE_REFINEMENT_SYSTEM
        assert "Specialization" in CONJECTURE_REFINEMENT_SYSTEM

    def test_report_system_prompt_exists(self):
        from agentic_research.agents.prompt_templates import (
            REFINEMENT_REPORT_SYSTEM,
        )

        assert "refinement journey" in REFINEMENT_REPORT_SYSTEM.lower()
