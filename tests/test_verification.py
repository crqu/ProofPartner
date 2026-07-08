"""Tests for Phase 6: Intent Judge + Counterexample Searcher.

All LLM calls are mocked — no real API calls are made.
Lean REPL uses mock backend for deterministic testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from agentic_research.models.agents import LLMResponse, TokenUsage
from agentic_research.models.verification import (
    CounterexampleCandidate,
    CounterexampleResult,
    CounterexampleStatus,
    InformalizationResult,
    IntentVerdict,
    IntentVerdictType,
    PathVerdict,
    VerificationPath,
)


# ---------------------------------------------------------------------------
# models/verification.py
# ---------------------------------------------------------------------------


class TestVerificationModels:
    def test_path_verdict_correct(self):
        pv = PathVerdict(
            path=VerificationPath.BLIND,
            verdict=IntentVerdictType.CORRECT,
            confidence=0.9,
        )
        assert pv.concerns == []
        assert pv.verdict == IntentVerdictType.CORRECT

    def test_path_verdict_incorrect_with_concerns(self):
        pv = PathVerdict(
            path=VerificationPath.ADVERSARIAL,
            verdict=IntentVerdictType.INCORRECT,
            concerns=["missing hypothesis", "wrong quantifier"],
            confidence=0.8,
        )
        assert len(pv.concerns) == 2

    def test_intent_verdict_has_concerns(self):
        v = IntentVerdict(
            overall_verdict=IntentVerdictType.INCORRECT,
            all_concerns=["issue 1"],
        )
        assert v.has_concerns

    def test_intent_verdict_no_concerns(self):
        v = IntentVerdict(overall_verdict=IntentVerdictType.CORRECT)
        assert not v.has_concerns

    def test_counterexample_candidate(self):
        c = CounterexampleCandidate(
            description="n=0",
            lean_code="example : ¬ P 0 := by decide",
            compilation_status="ok",
            proves_negation=True,
        )
        assert c.proves_negation

    def test_counterexample_result_disproved(self):
        ce = CounterexampleCandidate(description="n=0", proves_negation=True)
        r = CounterexampleResult(
            status=CounterexampleStatus.DISPROVED,
            successful_counterexample=ce,
            attempts_made=1,
        )
        assert r.is_disproved

    def test_counterexample_result_plausible(self):
        r = CounterexampleResult(
            status=CounterexampleStatus.PLAUSIBLE,
            attempts_made=5,
        )
        assert not r.is_disproved

    def test_informalization_result(self):
        r = InformalizationResult(
            lean_input="theorem foo : True := trivial",
            natural_language_output="The proposition True holds.",
        )
        assert "True" in r.natural_language_output

    def test_verification_path_values(self):
        assert VerificationPath.BLIND.value == "blind"
        assert VerificationPath.DIRECT.value == "direct"
        assert VerificationPath.ADVERSARIAL.value == "adversarial"
        assert VerificationPath.OPENAI.value == "openai"

    def test_serialization_roundtrip(self):
        verdict = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            path_verdicts=[
                PathVerdict(
                    path=VerificationPath.BLIND,
                    verdict=IntentVerdictType.CORRECT,
                    confidence=0.9,
                )
            ],
            adjudication_notes="all good",
        )
        restored = IntentVerdict.model_validate(verdict.model_dump())
        assert restored == verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )


def _make_mock_llm(responses: list[str]) -> MagicMock:
    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    mock.complete.side_effect = [_make_llm_response(r) for r in responses]
    mock.extract_json.side_effect = lambda text: LLMClient.extract_json(mock, text)

    real_extract = LLMClient.extract_json
    mock.extract_json = lambda text: real_extract(mock, text)
    return mock


def _make_mock_llm_with_json(responses: list[str]) -> MagicMock:
    """Build a mock LLM whose extract_json actually parses JSON."""
    import json
    import re

    from agentic_research.agents.llm_client import LLMClient

    mock = MagicMock(spec=LLMClient)
    mock.complete.side_effect = [_make_llm_response(r) for r in responses]

    def _real_extract(text):
        fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass
        for sc, ec in [("{", "}"), ("[", "]")]:
            start = text.find(sc)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == sc:
                    depth += 1
                elif text[i] == ec:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    mock.extract_json = _real_extract
    return mock


# ---------------------------------------------------------------------------
# agents/informalizer.py
# ---------------------------------------------------------------------------


class TestInformalizer:
    def test_informalize(self):
        from agentic_research.agents.informalizer import Informalizer

        llm = _make_mock_llm_with_json(
            ["For every natural number n, n plus zero equals n."]
        )
        informalizer = Informalizer(llm_client=llm)
        result = informalizer.informalize("theorem add_zero (n : Nat) : n + 0 = n := by simp")

        assert "natural number" in result.natural_language_output
        assert result.lean_input == "theorem add_zero (n : Nat) : n + 0 = n := by simp"

    def test_informalize_strips_comments(self):
        from agentic_research.agents.informalizer import Informalizer

        llm = _make_mock_llm_with_json(["Statement about True."])
        informalizer = Informalizer(llm_client=llm)

        code_with_comments = "-- AI generated\ntheorem foo : True := trivial"
        result = informalizer.informalize(code_with_comments)
        assert result.natural_language_output == "Statement about True."

    def test_informalize_via_agent_run(self):
        from agentic_research.agents.informalizer import Informalizer
        from agentic_research.models.agents import AgentContext, AgentStatus

        llm = _make_mock_llm_with_json(["Some math statement."])
        informalizer = Informalizer(llm_client=llm)
        ctx = AgentContext(task="theorem foo : True := trivial")
        result = informalizer.run(ctx)
        assert result.status == AgentStatus.SUCCESS
        assert result.result is not None


# ---------------------------------------------------------------------------
# agents/intent_judge.py
# ---------------------------------------------------------------------------

_CORRECT_JSON = '{"verdict": "correct", "concerns": [], "confidence": 0.95, "reasoning": "matches"}'
_INCORRECT_JSON = '{"verdict": "incorrect", "concerns": ["missing hypothesis"], "confidence": 0.8, "reasoning": "mismatch"}'
_CORRECT_WITH_DIMS_JSON = '{"verdict": "correct", "concerns": [], "confidence": 0.95, "reasoning": "matches", "type_fidelity": 0.9, "quantifier_accuracy": 0.85, "constraint_preservation": 0.8}'
_INCORRECT_WITH_DIMS_JSON = '{"verdict": "incorrect", "concerns": ["missing hypothesis"], "confidence": 0.8, "reasoning": "mismatch", "type_fidelity": 0.3, "quantifier_accuracy": 0.7, "constraint_preservation": 0.6}'


class TestIntentJudge:
    def _make_judge(self, llm_responses: list[str]):
        from agentic_research.agents.informalizer import Informalizer
        from agentic_research.agents.intent_judge import IntentJudge

        llm = _make_mock_llm_with_json(llm_responses)
        informalizer = Informalizer(llm_client=llm)
        return IntentJudge(llm_client=llm, informalizer=informalizer)

    def test_all_correct(self):
        judge = self._make_judge([
            "For all n, n + 0 = n.",
            _CORRECT_JSON,
            _CORRECT_JSON,
            _CORRECT_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem add_zero : ∀ n, n + 0 = n := by simp",
            original_idea="n plus zero equals n",
            conjecture="For all natural numbers n, n + 0 = n",
        )
        assert verdict.overall_verdict == IntentVerdictType.CORRECT
        assert not verdict.has_concerns
        assert len(verdict.path_verdicts) == 3

    def test_blind_path_catches_mismatch(self):
        judge = self._make_judge([
            "For all n, n times one equals n.",
            _INCORRECT_JSON,
            _CORRECT_JSON,
            _CORRECT_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem mul_one : ∀ n, n * 1 = n := by simp",
            original_idea="n plus zero equals n",
            conjecture="For all n, n + 0 = n",
        )
        assert verdict.overall_verdict == IntentVerdictType.INCORRECT
        assert verdict.has_concerns

    def test_adversarial_path_catches_issue(self):
        judge = self._make_judge([
            "Statement about primes.",
            _CORRECT_JSON,
            _CORRECT_JSON,
            _INCORRECT_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem prime_thing : True := trivial",
            original_idea="something about primes",
            conjecture="prime conjecture",
        )
        assert verdict.overall_verdict == IntentVerdictType.INCORRECT
        assert any(pv.path == VerificationPath.ADVERSARIAL for pv in verdict.path_verdicts)

    def test_direct_path_catches_issue(self):
        judge = self._make_judge([
            "Back translation.",
            _CORRECT_JSON,
            _INCORRECT_JSON,
            _CORRECT_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem t : True := trivial",
            original_idea="idea",
            conjecture="conjecture",
        )
        assert verdict.overall_verdict == IntentVerdictType.INCORRECT

    def test_all_paths_present(self):
        judge = self._make_judge([
            "Back translation.",
            _CORRECT_JSON,
            _CORRECT_JSON,
            _CORRECT_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem t : True := trivial",
            original_idea="idea",
            conjecture="conjecture",
        )
        paths = {pv.path for pv in verdict.path_verdicts}
        assert paths == {
            VerificationPath.BLIND,
            VerificationPath.DIRECT,
            VerificationPath.ADVERSARIAL,
        }

    def test_via_agent_run(self):
        from agentic_research.models.agents import AgentContext, AgentStatus

        judge = self._make_judge([
            "Back translation.",
            _CORRECT_JSON,
            _CORRECT_JSON,
            _CORRECT_JSON,
        ])
        ctx = AgentContext(
            task="theorem t : True := trivial",
            metadata={"original_idea": "idea", "conjecture": "conj"},
        )
        result = judge.run(ctx)
        assert result.status == AgentStatus.SUCCESS

    def test_unparseable_response_treated_as_incorrect(self):
        judge = self._make_judge([
            "Back translation.",
            "not valid json at all",
            _CORRECT_JSON,
            _CORRECT_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem t : True := trivial",
            original_idea="idea",
            conjecture="conjecture",
        )
        assert verdict.overall_verdict == IntentVerdictType.INCORRECT
        blind_pv = next(
            pv for pv in verdict.path_verdicts if pv.path == VerificationPath.BLIND
        )
        assert blind_pv.verdict == IntentVerdictType.INCORRECT

    def test_dimension_scores_parsed(self):
        judge = self._make_judge([
            "Back translation.",
            _CORRECT_WITH_DIMS_JSON,
            _CORRECT_WITH_DIMS_JSON,
            _CORRECT_WITH_DIMS_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem t : True := trivial",
            original_idea="idea",
            conjecture="conjecture",
        )
        for pv in verdict.path_verdicts:
            assert pv.type_fidelity == 0.9
            assert pv.quantifier_accuracy == 0.85
            assert pv.constraint_preservation == 0.8

    def test_missing_dimensions_default_to_half(self):
        judge = self._make_judge([
            "Back translation.",
            _CORRECT_JSON,
            _CORRECT_JSON,
            _CORRECT_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem t : True := trivial",
            original_idea="idea",
            conjecture="conjecture",
        )
        for pv in verdict.path_verdicts:
            assert pv.type_fidelity == 0.5
            assert pv.quantifier_accuracy == 0.5
            assert pv.constraint_preservation == 0.5

    def test_aggregated_dimensions_on_verdict(self):
        judge = self._make_judge([
            "Back translation.",
            _CORRECT_WITH_DIMS_JSON,
            _CORRECT_WITH_DIMS_JSON,
            _CORRECT_WITH_DIMS_JSON,
        ])
        verdict = judge.judge(
            lean_code="theorem t : True := trivial",
            original_idea="idea",
            conjecture="conjecture",
        )
        assert verdict.type_fidelity > 0.0
        assert verdict.quantifier_accuracy > 0.0
        assert verdict.constraint_preservation > 0.0
        assert verdict.overall_confidence > 0.0


# ---------------------------------------------------------------------------
# Dimension aggregation and passes_threshold
# ---------------------------------------------------------------------------


class TestDimensionAggregation:
    def test_adversarial_weighted_higher(self):
        from agentic_research.agents.intent_judge import _aggregate_dimensions

        verdicts = [
            PathVerdict(
                path=VerificationPath.BLIND,
                verdict=IntentVerdictType.CORRECT,
                type_fidelity=1.0,
                quantifier_accuracy=1.0,
                constraint_preservation=1.0,
            ),
            PathVerdict(
                path=VerificationPath.DIRECT,
                verdict=IntentVerdictType.CORRECT,
                type_fidelity=1.0,
                quantifier_accuracy=1.0,
                constraint_preservation=1.0,
            ),
            PathVerdict(
                path=VerificationPath.ADVERSARIAL,
                verdict=IntentVerdictType.CORRECT,
                type_fidelity=0.0,
                quantifier_accuracy=0.0,
                constraint_preservation=0.0,
            ),
        ]
        dims = _aggregate_dimensions(verdicts)
        # weights: 1.0 + 1.0 + 1.5 = 3.5
        # tf: (1.0 + 1.0 + 0.0*1.5) / 3.5 = 2.0/3.5 ≈ 0.5714
        assert abs(dims["type_fidelity"] - 2.0 / 3.5) < 0.01
        assert abs(dims["quantifier_accuracy"] - 2.0 / 3.5) < 0.01

    def test_uniform_scores(self):
        from agentic_research.agents.intent_judge import _aggregate_dimensions

        verdicts = [
            PathVerdict(
                path=VerificationPath.BLIND,
                verdict=IntentVerdictType.CORRECT,
                type_fidelity=0.8,
                quantifier_accuracy=0.8,
                constraint_preservation=0.8,
            ),
            PathVerdict(
                path=VerificationPath.DIRECT,
                verdict=IntentVerdictType.CORRECT,
                type_fidelity=0.8,
                quantifier_accuracy=0.8,
                constraint_preservation=0.8,
            ),
            PathVerdict(
                path=VerificationPath.ADVERSARIAL,
                verdict=IntentVerdictType.CORRECT,
                type_fidelity=0.8,
                quantifier_accuracy=0.8,
                constraint_preservation=0.8,
            ),
        ]
        dims = _aggregate_dimensions(verdicts)
        assert abs(dims["type_fidelity"] - 0.8) < 0.01
        assert abs(dims["overall_confidence"] - 0.8) < 0.01

    def test_empty_verdicts(self):
        from agentic_research.agents.intent_judge import _aggregate_dimensions

        dims = _aggregate_dimensions([])
        assert dims["type_fidelity"] == 0.5
        assert dims["overall_confidence"] == 0.5
        assert dims["passes"] is False


class TestPassesThreshold:
    def test_all_pass(self):
        v = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            type_fidelity=0.8,
            quantifier_accuracy=0.7,
            constraint_preservation=0.9,
            overall_confidence=0.8,
        )
        assert v.passes_threshold is True

    def test_one_dimension_low(self):
        v = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            type_fidelity=0.3,
            quantifier_accuracy=0.8,
            constraint_preservation=0.9,
            overall_confidence=0.7,
        )
        assert v.passes_threshold is False

    def test_overall_confidence_low(self):
        v = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            type_fidelity=0.5,
            quantifier_accuracy=0.5,
            constraint_preservation=0.5,
            overall_confidence=0.5,
        )
        assert v.passes_threshold is False

    def test_boundary_values_pass(self):
        v = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            type_fidelity=0.4,
            quantifier_accuracy=0.4,
            constraint_preservation=0.4,
            overall_confidence=0.6,
        )
        assert v.passes_threshold is True

    def test_boundary_just_below_fails(self):
        v = IntentVerdict(
            overall_verdict=IntentVerdictType.CORRECT,
            type_fidelity=0.39,
            quantifier_accuracy=0.4,
            constraint_preservation=0.4,
            overall_confidence=0.6,
        )
        assert v.passes_threshold is False

    def test_backward_compat_defaults(self):
        v = IntentVerdict(overall_verdict=IntentVerdictType.CORRECT)
        assert v.type_fidelity == 0.5
        assert v.quantifier_accuracy == 0.5
        assert v.constraint_preservation == 0.5
        assert v.overall_confidence == 0.5
        assert v.passes_threshold is False


# ---------------------------------------------------------------------------
# Feature flag: OpenAI path
# ---------------------------------------------------------------------------


class TestOpenAIFeatureFlag:
    def test_openai_disabled_by_default(self):
        from agentic_research.agents.intent_judge import _openai_enabled

        with patch.dict("os.environ", {}, clear=True):
            assert not _openai_enabled()

    def test_openai_enabled(self):
        from agentic_research.agents.intent_judge import _openai_enabled

        with patch.dict("os.environ", {"OPENAI_ENABLED": "true"}):
            assert _openai_enabled()

    def test_openai_enabled_values(self):
        from agentic_research.agents.intent_judge import _openai_enabled

        for val in ("true", "True", "TRUE", "1", "yes"):
            with patch.dict("os.environ", {"OPENAI_ENABLED": val}):
                assert _openai_enabled()

    def test_openai_disabled_values(self):
        from agentic_research.agents.intent_judge import _openai_enabled

        for val in ("false", "0", "no", ""):
            with patch.dict("os.environ", {"OPENAI_ENABLED": val}):
                assert not _openai_enabled()

    def test_openai_path_added_when_enabled(self):
        from agentic_research.agents.informalizer import Informalizer
        from agentic_research.agents.intent_judge import IntentJudge

        llm = _make_mock_llm_with_json([
            "Back translation.",
            _CORRECT_JSON,
            _CORRECT_JSON,
            _CORRECT_JSON,
        ])
        informalizer = Informalizer(llm_client=llm)
        judge = IntentJudge(llm_client=llm, informalizer=informalizer)

        mock_openai_verdict = PathVerdict(
            path=VerificationPath.OPENAI,
            verdict=IntentVerdictType.CORRECT,
            confidence=0.9,
        )
        with patch.dict("os.environ", {"OPENAI_ENABLED": "true"}):
            with patch.object(judge, "_run_openai_path", return_value=mock_openai_verdict):
                verdict = judge.judge(
                    lean_code="theorem t : True := trivial",
                    original_idea="idea",
                    conjecture="conjecture",
                )

        assert len(verdict.path_verdicts) == 4
        paths = {pv.path for pv in verdict.path_verdicts}
        assert VerificationPath.OPENAI in paths


# ---------------------------------------------------------------------------
# Adjudication logic
# ---------------------------------------------------------------------------


class TestAdjudication:
    def test_all_correct_no_concerns(self):
        from agentic_research.agents.intent_judge import _adjudicate

        verdicts = [
            PathVerdict(path=VerificationPath.BLIND, verdict=IntentVerdictType.CORRECT),
            PathVerdict(path=VerificationPath.DIRECT, verdict=IntentVerdictType.CORRECT),
            PathVerdict(path=VerificationPath.ADVERSARIAL, verdict=IntentVerdictType.CORRECT),
        ]
        result = _adjudicate(verdicts)
        assert result.overall_verdict == IntentVerdictType.CORRECT
        assert result.all_concerns == []

    def test_single_concern_triggers_incorrect(self):
        from agentic_research.agents.intent_judge import _adjudicate

        verdicts = [
            PathVerdict(path=VerificationPath.BLIND, verdict=IntentVerdictType.CORRECT),
            PathVerdict(
                path=VerificationPath.ADVERSARIAL,
                verdict=IntentVerdictType.INCORRECT,
                concerns=["missing condition"],
            ),
            PathVerdict(path=VerificationPath.DIRECT, verdict=IntentVerdictType.CORRECT),
        ]
        result = _adjudicate(verdicts)
        assert result.overall_verdict == IntentVerdictType.INCORRECT
        assert "missing condition" in result.all_concerns

    def test_multiple_concerns_aggregated(self):
        from agentic_research.agents.intent_judge import _adjudicate

        verdicts = [
            PathVerdict(
                path=VerificationPath.BLIND,
                verdict=IntentVerdictType.INCORRECT,
                concerns=["wrong scope"],
            ),
            PathVerdict(
                path=VerificationPath.ADVERSARIAL,
                verdict=IntentVerdictType.INCORRECT,
                concerns=["quantifier error", "type mismatch"],
            ),
            PathVerdict(path=VerificationPath.DIRECT, verdict=IntentVerdictType.CORRECT),
        ]
        result = _adjudicate(verdicts)
        assert result.overall_verdict == IntentVerdictType.INCORRECT
        assert len(result.all_concerns) == 3

    def test_adjudication_notes_populated(self):
        from agentic_research.agents.intent_judge import _adjudicate

        verdicts = [
            PathVerdict(
                path=VerificationPath.BLIND,
                verdict=IntentVerdictType.INCORRECT,
                concerns=["issue"],
            ),
            PathVerdict(path=VerificationPath.DIRECT, verdict=IntentVerdictType.CORRECT),
            PathVerdict(path=VerificationPath.ADVERSARIAL, verdict=IntentVerdictType.CORRECT),
        ]
        result = _adjudicate(verdicts)
        assert "blind" in result.adjudication_notes
        assert "refinement" in result.adjudication_notes


# ---------------------------------------------------------------------------
# agents/counterexample_searcher.py
# ---------------------------------------------------------------------------


class TestCounterexampleSearcher:
    def _make_searcher(self, llm_responses: list[str], repl_backend="mock"):
        from agentic_research.agents.counterexample_searcher import CounterexampleSearcher
        from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig

        llm = _make_mock_llm_with_json(llm_responses)
        repl = LeanRepl(ReplConfig(backend=ReplBackend.MOCK))
        return CounterexampleSearcher(llm_client=llm, lean_repl=repl, max_candidates=3)

    def test_plausible_when_no_counterexample(self):
        generation_response = '{"candidates": [{"description": "n=0", "values": "0", "reasoning": "edge case"}]}'
        formalization_response = '```lean\n-- MOCK_ERROR\nexample : False := sorry\n```'

        searcher = self._make_searcher([generation_response, formalization_response])
        result = searcher.search(
            lean_code="theorem foo : ∀ n, n + 0 = n := by simp",
            conjecture="For all n, n + 0 = n",
        )
        assert result.status == CounterexampleStatus.PLAUSIBLE
        assert result.successful_counterexample is None
        assert result.attempts_made == 1

    def test_disproved_when_counterexample_compiles(self):
        generation_response = '{"candidates": [{"description": "n=0", "values": "0", "reasoning": "edge"}]}'
        # Mock REPL: no sorry + no MOCK_ERROR = compiles with all goals closed
        formalization_response = '```lean\nexample : True := trivial\n```'

        searcher = self._make_searcher([generation_response, formalization_response])
        result = searcher.search(
            lean_code="theorem foo : False := sorry",
            conjecture="False holds",
        )
        assert result.status == CounterexampleStatus.DISPROVED
        assert result.successful_counterexample is not None
        assert result.successful_counterexample.proves_negation

    def test_multiple_candidates_stops_at_first_disproof(self):
        generation_response = '''{"candidates": [
            {"description": "n=0", "values": "0", "reasoning": "edge"},
            {"description": "n=1", "values": "1", "reasoning": "small"},
            {"description": "n=2", "values": "2", "reasoning": "another"}
        ]}'''
        # First fails, second succeeds
        fail_response = '```lean\n-- MOCK_ERROR\nbad\n```'
        success_response = '```lean\nexample : True := trivial\n```'

        searcher = self._make_searcher([
            generation_response,
            fail_response,
            success_response,
        ])
        result = searcher.search(
            lean_code="theorem foo : False := sorry",
            conjecture="False",
        )
        assert result.status == CounterexampleStatus.DISPROVED
        assert result.attempts_made == 2

    def test_all_candidates_fail(self):
        generation_response = '''{"candidates": [
            {"description": "n=0", "values": "0", "reasoning": "edge"},
            {"description": "n=1", "values": "1", "reasoning": "small"}
        ]}'''
        fail1 = '```lean\n-- MOCK_ERROR\nbad1\n```'
        fail2 = '```lean\n-- MOCK_ERROR\nbad2\n```'

        searcher = self._make_searcher([generation_response, fail1, fail2])
        result = searcher.search(
            lean_code="theorem foo : True := trivial",
            conjecture="True holds",
        )
        assert result.status == CounterexampleStatus.PLAUSIBLE
        assert result.attempts_made == 2

    def test_empty_candidates(self):
        searcher = self._make_searcher(['{"candidates": []}'])
        result = searcher.search(
            lean_code="theorem foo : True := trivial",
            conjecture="True",
        )
        assert result.status == CounterexampleStatus.PLAUSIBLE
        assert result.attempts_made == 0

    def test_invalid_json_response(self):
        searcher = self._make_searcher(["not json at all"])
        result = searcher.search(
            lean_code="theorem foo : True := trivial",
            conjecture="True",
        )
        assert result.status == CounterexampleStatus.PLAUSIBLE
        assert result.attempts_made == 0

    def test_via_agent_run(self):
        from agentic_research.models.agents import AgentContext, AgentStatus

        generation_response = '{"candidates": []}'
        searcher = self._make_searcher([generation_response])
        ctx = AgentContext(
            task="theorem foo : True := trivial",
            metadata={"conjecture": "True holds"},
        )
        result = searcher.run(ctx)
        assert result.status == AgentStatus.SUCCESS

    def test_max_candidates_respected(self):
        many_candidates = '{"candidates": [' + ", ".join(
            f'{{"description": "n={i}", "values": "{i}", "reasoning": "test"}}'
            for i in range(10)
        ) + "]}"
        fail_responses = ['```lean\n-- MOCK_ERROR\nbad\n```'] * 3

        searcher = self._make_searcher([many_candidates] + fail_responses)
        result = searcher.search(
            lean_code="theorem foo : True := trivial",
            conjecture="True",
        )
        assert result.attempts_made <= 3
