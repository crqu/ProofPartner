#!/usr/bin/env python3
"""E2E test: Putnam 2024 A1 with real Vertex AI + real Lean 4.

Runs the full ProofPipeline on PutnamBench/putnam_2024_a1 with extended
thinking enabled.  No mocks — requires CLAUDE_CODE_USE_VERTEX=1 and
Lean 4 on PATH.

Usage:
    python scripts/e2e_putnam_a1.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

PROBLEM_ID = "PutnamBench/putnam_2024_a1"

LEAN_HEADER = "import Mathlib"

LEAN_STATEMENT = """\
noncomputable abbrev putnam_2024_a1_solution : Set ℕ := {1}

theorem putnam_2024_a1 :
    {n : ℕ | 0 < n ∧ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 2*a^n + 3*b^n = 4*c^n}
      = putnam_2024_a1_solution := by sorry"""

TIMEOUT_SECONDS = 900

INPUT_PRICE_PER_MTOK = 15.0
OUTPUT_PRICE_PER_MTOK = 75.0
CACHE_WRITE_PRICE_PER_MTOK = 18.75
CACHE_READ_PRICE_PER_MTOK = 1.875


def _estimate_cost(usage) -> float:
    return (
        usage.input_tokens * INPUT_PRICE_PER_MTOK
        + usage.output_tokens * OUTPUT_PRICE_PER_MTOK
        + usage.cache_creation_input_tokens * CACHE_WRITE_PRICE_PER_MTOK
        + usage.cache_read_input_tokens * CACHE_READ_PRICE_PER_MTOK
    ) / 1_000_000


def main() -> None:
    from agentic_research.agents.llm_client import LLMClient
    from agentic_research.agents.nl_prover import NaturalLanguageProver
    from agentic_research.models.agents import ProverConfig
    from agentic_research.pipelines.proof import ProofPipeline
    from agentic_research.tools.lean_repl import (
        LeanRepl,
        ReplConfig,
        detect_backend,
        ReplBackend,
    )
    from agentic_research.tools.lean_search import (
        LeanSearch,
        SearchConfig,
        detect_search_backend,
        SearchBackend,
    )

    model = os.environ.get("AGENTIC_RESEARCH_MODEL", "claude-opus-4-6")

    print(f"Problem:  {PROBLEM_ID}")
    print(f"Model:    {model}")
    print(f"Timeout:  {TIMEOUT_SECONDS}s")
    print()

    # --- Lean backend ---
    repl_backend = detect_backend()
    if repl_backend == ReplBackend.MOCK:
        print("ERROR: No real Lean 4 backend found. Aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"Lean backend: {repl_backend.value}")

    lean_repl = LeanRepl(ReplConfig(backend=repl_backend, timeout_seconds=120))

    # --- Search backend ---
    search_backend = detect_search_backend()
    print(f"Search backend: {search_backend.value}")
    lean_search = LeanSearch(SearchConfig(backend=search_backend))

    # --- LLM client (auto-detects Vertex AI from env) ---
    llm_client = LLMClient(model=model, max_tokens=16384)
    print(f"LLM backend: {'vertex' if llm_client.is_vertex else 'direct'}")
    print()

    # --- Prover config ---
    prover_config = ProverConfig(
        use_extended_thinking=True,
        thinking_budget=10000,
    )

    # --- NL Prover ---
    nl_prover = NaturalLanguageProver(
        llm_client=llm_client,
        prover_config=prover_config,
    )

    # --- Pipeline ---
    pipeline = ProofPipeline(
        llm_client=llm_client,
        lean_repl=lean_repl,
        lean_search=lean_search,
        prover_config=prover_config,
        nl_prover=nl_prover,
        use_nl_proof_stage=True,
    )

    full_statement = f"{LEAN_HEADER}\n\n{LEAN_STATEMENT}"

    # --- Run with timeout ---
    result_holder: list = []
    error_holder: list[Exception] = []

    def _run() -> None:
        try:
            r = pipeline.run(
                lean_statement=full_statement,
                statement_nl="Find all positive integers n such that there exist positive integers a, b, c satisfying 2a^n + 3b^n = 4c^n.",
            )
            result_holder.append(r)
        except Exception as exc:
            error_holder.append(exc)

    start = time.monotonic()
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=TIMEOUT_SECONDS)
    elapsed = round(time.monotonic() - start, 3)

    print("=" * 60)
    print(f"Elapsed: {elapsed}s")
    print()

    if thread.is_alive():
        print("RESULT: TIMEOUT")
        print(json.dumps({
            "problem_id": PROBLEM_ID,
            "proved": False,
            "failure_reason": f"Timeout after {TIMEOUT_SECONDS}s",
            "elapsed_seconds": elapsed,
        }, indent=2))
        sys.exit(2)

    if error_holder:
        exc = error_holder[0]
        print(f"RESULT: ERROR — {exc}")
        print(json.dumps({
            "problem_id": PROBLEM_ID,
            "proved": False,
            "failure_reason": str(exc),
            "elapsed_seconds": elapsed,
        }, indent=2))
        sys.exit(3)

    result = result_holder[0]
    usage = result.total_token_usage
    cost = _estimate_cost(usage)

    print(f"Proved:         {result.proved}")
    print(f"Backend:        {result.backend}")
    print(f"Verified:       {result.verified}")
    print(f"Failure stage:  {result.failure_stage}")
    print(f"Failure reason: {result.failure_reason}")
    print()
    print(f"Input tokens:   {usage.input_tokens:,}")
    print(f"Output tokens:  {usage.output_tokens:,}")
    print(f"Cache write:    {usage.cache_creation_input_tokens:,}")
    print(f"Cache read:     {usage.cache_read_input_tokens:,}")
    print(f"Est. cost:      ${cost:.4f}")
    print()

    if result.final_proof:
        print("--- Final Proof ---")
        print(result.final_proof)
        print("--- End Proof ---")
    else:
        print("(no proof produced)")

    print()
    print(json.dumps({
        "problem_id": PROBLEM_ID,
        "proved": result.proved,
        "final_proof": result.final_proof,
        "failure_stage": result.failure_stage,
        "failure_reason": result.failure_reason,
        "backend": result.backend,
        "verified": result.verified,
        "elapsed_seconds": elapsed,
        "token_usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
        },
        "estimated_cost_usd": round(cost, 4),
    }, indent=2))

    sys.exit(0 if result.proved else 1)


if __name__ == "__main__":
    main()
