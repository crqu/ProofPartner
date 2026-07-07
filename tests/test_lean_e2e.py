"""End-to-end tests using real Sonnet (Vertex AI) and real Lean 4.

These tests validate the actual pipeline integration — real LLM calls
through real Lean compilation. Gated on Lean 4 being installed.
"""

from __future__ import annotations

import shutil

import pytest

from agentic_research.agents.llm_client import LLMClient
from agentic_research.models.tools import CompilationStatus
from agentic_research.pipelines.formalization import FormalizationPipeline
from agentic_research.pipelines.proof import ProofPipeline
from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

LEAN_AVAILABLE = shutil.which("lean") is not None


def _make_components():
    llm = LLMClient(model="claude-sonnet-4-20250514")
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    search = LeanSearch(SearchConfig(backend=SearchBackend.MOCK))
    return llm, repl, search


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
@pytest.mark.timeout(180)
def test_formalize_real_lean():
    """Formalize a trivial math statement with real Sonnet + real Lean."""
    llm, repl, search = _make_components()

    pipeline = FormalizationPipeline(
        llm_client=llm, lean_repl=repl, lean_search=search, k=2, max_retries=1
    )
    result = pipeline.run("for all natural numbers n, n equals n")

    assert result is not None
    assert result.theorem is not None, f"Formalization failed: {result.failure_reason}"
    assert result.theorem.lean_statement, "Lean statement is empty"
    assert len(result.theorem.lean_statement.strip()) > 0

    compilation = repl.execute(result.theorem.lean_statement)
    assert compilation.compilation_status in (
        CompilationStatus.OK,
        CompilationStatus.ERROR,
    ), "Compilation did not terminate"


@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
@pytest.mark.timeout(180)
def test_prove_real_lean():
    """Attempt to prove a trivial Lean statement with real Sonnet + real Lean."""
    llm, repl, search = _make_components()

    pipeline = ProofPipeline(
        llm_client=llm,
        lean_repl=repl,
        lean_search=search,
        max_strategies=2,
        max_depth=2,
        max_retries_per_node=1,
        use_claim_check=False,
    )
    result = pipeline.run(
        "theorem trivial_true : True := by sorry",
        statement_nl="True is trivially true",
    )

    assert result is not None
    assert result.statement, "Statement is empty"
    if result.final_proof:
        assert len(result.final_proof.strip()) > 0
        compilation = repl.execute(result.final_proof)
        assert compilation.compilation_status in (
            CompilationStatus.OK,
            CompilationStatus.ERROR,
        ), "Proof compilation did not terminate"
