"""Tests for the Extractor agent and ExtractionResult models."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentic_research.agents.extractor import Extractor, MAX_INPUT_CHARS
from agentic_research.cli.main import cli
from agentic_research.models.agents import LLMResponse, TokenUsage
from agentic_research.models.extraction import (
    ExtractionResult,
    ExtractedDefinition,
    ExtractedLemma,
    ExtractedPriorWork,
    ExtractedTheorem,
)


SAMPLE_EXTRACTION_JSON = json.dumps({
    "paper_title": "Distributionally Robust Optimization",
    "paper_domain": "optimization",
    "theorems": [
        {
            "statement": "The worst-case risk over the ambiguity set equals the dual objective",
            "statement_latex": r"\sup_{Q \in \mathcal{U}} E_Q[l] = \inf_{\lambda} ...",
            "is_main": True,
            "section_ref": "Theorem 3.1",
        },
        {
            "statement": "The dual problem is convex",
            "is_main": False,
            "section_ref": "Proposition 3.2",
        },
    ],
    "definitions": [
        {
            "name": "Wasserstein ambiguity set",
            "informal_statement": "The set of distributions within Wasserstein distance r of P0",
            "depends_on": ["Wasserstein distance"],
            "in_mathlib": False,
        },
        {
            "name": "Wasserstein distance",
            "informal_statement": "The optimal transport distance between two measures",
            "depends_on": [],
            "in_mathlib": False,
        },
    ],
    "lemmas": [
        {
            "name": "Lemma 3.3",
            "informal_statement": "Strong duality holds under constraint qualification",
            "used_in_proof_of": "Theorem 3.1",
        },
    ],
    "prior_work": [
        {
            "citation": "Villani, 2009, Optimal Transport",
            "result_statement": "Kantorovich duality for optimal transport",
            "axiom_candidate": True,
        },
        {
            "citation": "Boyd & Vandenberghe, 2004",
            "result_statement": "Slater's condition implies strong duality",
            "axiom_candidate": False,
        },
    ],
})


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="claude-opus-4-6-20250616",
        stop_reason="end_turn",
        token_usage=TokenUsage(input_tokens=500, output_tokens=200),
    )


class TestExtractionResultModel:
    def test_empty_result(self) -> None:
        result = ExtractionResult()
        assert result.theorems == []
        assert result.definitions == []
        assert result.lemmas == []
        assert result.prior_work == []
        assert result.paper_title == ""
        assert result.paper_domain == ""

    def test_full_result(self) -> None:
        result = ExtractionResult(
            theorems=[ExtractedTheorem(statement="test", is_main=True)],
            definitions=[ExtractedDefinition(name="D1", informal_statement="def 1")],
            lemmas=[ExtractedLemma(name="L1", informal_statement="lemma 1")],
            prior_work=[ExtractedPriorWork(citation="Author 2024", result_statement="result")],
            paper_title="Test Paper",
            paper_domain="algebra",
        )
        assert len(result.theorems) == 1
        assert result.theorems[0].is_main is True
        assert result.paper_title == "Test Paper"

    def test_extracted_theorem_defaults(self) -> None:
        thm = ExtractedTheorem(statement="P implies Q")
        assert thm.statement_latex == ""
        assert thm.is_main is False
        assert thm.section_ref == ""

    def test_extracted_definition_defaults(self) -> None:
        defn = ExtractedDefinition(name="foo", informal_statement="a thing")
        assert defn.depends_on == []
        assert defn.in_mathlib is False

    def test_extracted_lemma_defaults(self) -> None:
        lem = ExtractedLemma(name="L", informal_statement="something")
        assert lem.used_in_proof_of == ""

    def test_extracted_prior_work_defaults(self) -> None:
        pw = ExtractedPriorWork(citation="X 2020", result_statement="result")
        assert pw.axiom_candidate is True

    def test_prior_work_axiom_candidate_false(self) -> None:
        pw = ExtractedPriorWork(
            citation="Basic", result_statement="basic fact", axiom_candidate=False
        )
        assert pw.axiom_candidate is False


class TestExtractor:
    def test_extract_from_latex(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(SAMPLE_EXTRACTION_JSON)
        mock_llm.extract_json.return_value = json.loads(SAMPLE_EXTRACTION_JSON)

        extractor = Extractor(llm_client=mock_llm)
        result = extractor.extract("\\begin{document}...\\end{document}")

        assert result.paper_title == "Distributionally Robust Optimization"
        assert result.paper_domain == "optimization"
        assert len(result.theorems) == 2
        assert result.theorems[0].is_main is True
        assert result.theorems[1].is_main is False
        assert len(result.definitions) == 2
        assert len(result.lemmas) == 1
        assert len(result.prior_work) == 2

    def test_empty_paper(self) -> None:
        mock_llm = MagicMock()
        empty_json = json.dumps({
            "paper_title": "",
            "paper_domain": "",
            "theorems": [],
            "definitions": [],
            "lemmas": [],
            "prior_work": [],
        })
        mock_llm.complete.return_value = _mock_llm_response(empty_json)
        mock_llm.extract_json.return_value = json.loads(empty_json)

        extractor = Extractor(llm_client=mock_llm)
        result = extractor.extract("no math here")

        assert result.theorems == []
        assert result.definitions == []
        assert result.lemmas == []
        assert result.prior_work == []

    def test_parse_failure_returns_empty(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response("not valid json at all")
        mock_llm.extract_json.return_value = None

        extractor = Extractor(llm_client=mock_llm)
        result = extractor.extract("some paper text")

        assert isinstance(result, ExtractionResult)
        assert result.theorems == []

    def test_input_truncation(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response("{}")
        mock_llm.extract_json.return_value = {}

        extractor = Extractor(llm_client=mock_llm)
        long_text = "x" * 200_000
        extractor.extract(long_text)

        call_args = mock_llm.complete.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[0]["content"]
        assert len(user_content) < MAX_INPUT_CHARS + 1000

    def test_is_main_filtering(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(SAMPLE_EXTRACTION_JSON)
        mock_llm.extract_json.return_value = json.loads(SAMPLE_EXTRACTION_JSON)

        extractor = Extractor(llm_client=mock_llm)
        result = extractor.extract("paper text")

        main_theorems = [t for t in result.theorems if t.is_main]
        assert len(main_theorems) == 1
        assert "worst-case risk" in main_theorems[0].statement

    def test_prior_work_axiom_classification(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(SAMPLE_EXTRACTION_JSON)
        mock_llm.extract_json.return_value = json.loads(SAMPLE_EXTRACTION_JSON)

        extractor = Extractor(llm_client=mock_llm)
        result = extractor.extract("paper text")

        axiom_candidates = [p for p in result.prior_work if p.axiom_candidate]
        non_axiom = [p for p in result.prior_work if not p.axiom_candidate]
        assert len(axiom_candidates) == 1
        assert "Kantorovich" in axiom_candidates[0].result_statement
        assert len(non_axiom) == 1
        assert "Slater" in non_axiom[0].result_statement

    def test_temperature_zero(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response("{}")
        mock_llm.extract_json.return_value = {}

        extractor = Extractor(llm_client=mock_llm)
        extractor.extract("text")

        call_args = mock_llm.complete.call_args
        assert call_args.kwargs["temperature"] == 0.0

    def test_execute_wraps_extract(self) -> None:
        from agentic_research.models.agents import AgentContext, AgentStatus

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_llm_response(SAMPLE_EXTRACTION_JSON)
        mock_llm.extract_json.return_value = json.loads(SAMPLE_EXTRACTION_JSON)

        extractor = Extractor(llm_client=mock_llm)
        ctx = AgentContext(task="paper text here")
        result = extractor.run(ctx)

        assert result.status == AgentStatus.SUCCESS
        assert result.result["paper_title"] == "Distributionally Robust Optimization"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_session_dir(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr("agentic_research.cli.main.SESSION_DIR", session_dir)
    return session_dir


class TestFormalizePaperCLI:
    def test_extract_only(self, runner, tmp_session_dir, tmp_path) -> None:
        tex_file = tmp_path / "paper.tex"
        tex_file.write_text(r"\begin{document}Test paper\end{document}")

        mock_result = ExtractionResult(
            paper_title="Test Paper",
            theorems=[ExtractedTheorem(statement="Main result", is_main=True)],
            definitions=[ExtractedDefinition(name="D1", informal_statement="def")],
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.agents.extractor.Extractor.extract", return_value=mock_result),
            patch(
                "agentic_research.agents.extractor.Extractor.cumulative_tokens",
                new_callable=lambda: property(
                    lambda self: TokenUsage(input_tokens=100, output_tokens=50)
                ),
            ),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(
                cli, ["formalize-paper", str(tex_file), "--extract-only"]
            )

        assert result.exit_code == 0
        assert "Test Paper" in result.output
        assert "Main result" in result.output
        assert "Cost Summary" in result.output

    def test_missing_file(self, runner) -> None:
        result = runner.invoke(cli, ["formalize-paper", "/nonexistent/paper.tex"])
        assert result.exit_code != 0

    def test_unsupported_file_type(self, runner, tmp_path) -> None:
        doc_file = tmp_path / "paper.docx"
        doc_file.write_text("not a tex file")

        result = runner.invoke(cli, ["formalize-paper", str(doc_file)])
        assert result.exit_code != 0
        assert "Unsupported" in result.output

    def test_empty_file(self, runner, tmp_path) -> None:
        tex_file = tmp_path / "empty.tex"
        tex_file.write_text("")

        result = runner.invoke(cli, ["formalize-paper", str(tex_file)])
        assert result.exit_code != 0
        assert "empty" in result.output.lower()

    def test_stub_formalization_message(self, runner, tmp_session_dir, tmp_path) -> None:
        tex_file = tmp_path / "paper.tex"
        tex_file.write_text(r"\begin{document}Paper content\end{document}")

        mock_result = ExtractionResult(
            theorems=[ExtractedTheorem(statement="Result", is_main=True)],
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.agents.extractor.Extractor.extract", return_value=mock_result),
            patch(
                "agentic_research.agents.extractor.Extractor.cumulative_tokens",
                new_callable=lambda: property(
                    lambda self: TokenUsage(input_tokens=100, output_tokens=50)
                ),
            ),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(cli, ["formalize-paper", str(tex_file)])

        assert result.exit_code == 0
        assert "not yet wired" in result.output

    def test_help_shows_formalize_paper(self, runner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert "formalize-paper" in result.output

    def test_theorem_index_out_of_range(self, runner, tmp_session_dir, tmp_path) -> None:
        tex_file = tmp_path / "paper.tex"
        tex_file.write_text(r"\begin{document}Paper\end{document}")

        mock_result = ExtractionResult(
            theorems=[ExtractedTheorem(statement="Only theorem", is_main=True)],
        )

        with (
            patch("agentic_research.cli.main._create_llm_client") as mock_llm,
            patch("agentic_research.agents.extractor.Extractor.extract", return_value=mock_result),
            patch(
                "agentic_research.agents.extractor.Extractor.cumulative_tokens",
                new_callable=lambda: property(
                    lambda self: TokenUsage(input_tokens=100, output_tokens=50)
                ),
            ),
        ):
            mock_llm.return_value = MagicMock()
            result = runner.invoke(
                cli, ["formalize-paper", str(tex_file), "--theorem-index", "5"]
            )

        assert result.exit_code != 0
        assert "out of range" in result.output
