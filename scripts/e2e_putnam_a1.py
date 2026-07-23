#!/usr/bin/env python3
"""E2E validation: Putnam 2024 A1 with real Vertex AI + Lean 4.

Validates sorry-first skeleton validation, truncation handling, and
preamble propagation on the problem that originally exposed issues #110-112.

Usage:
    CLAUDE_CODE_USE_VERTEX=1 python scripts/e2e_putnam_a1.py

Exit codes:
    0 — proof succeeded
    1 — proof failed (with detailed failure taxonomy)
"""

from __future__ import annotations

import os
import signal
import sys
import time

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
)
log = structlog.get_logger(__name__)

TIMEOUT_SECONDS = 1800  # 30 minutes


def _timeout_handler(signum, frame):
    log.error("e2e_timeout", timeout_seconds=TIMEOUT_SECONDS)
    sys.exit(1)


def main() -> int:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    start = time.monotonic()

    log.info("e2e_start", problem="putnam_2024_a1")

    from agentic_research.eval.benchmarks import load_putnam_bench

    log.info("loading_putnam_bench")
    problem_set = load_putnam_bench()

    putnam_a1 = None
    for p in problem_set.problems:
        if "2024" in p.name and "a1" in p.name.lower():
            putnam_a1 = p
            break

    if putnam_a1 is None:
        log.error("problem_not_found", searched="putnam_2024_a1", total=len(problem_set.problems))
        if problem_set.problems:
            log.info("available_problems", names=[p.name for p in problem_set.problems[:10]])
        return 1

    log.info(
        "problem_loaded",
        name=putnam_a1.name,
        statement_len=len(putnam_a1.lean_statement),
        has_header=bool(putnam_a1.lean_header),
    )

    model = os.environ.get("AGENTIC_RESEARCH_MODEL", "claude-opus-4-6")
    log.info("initializing_pipeline", model=model)

    from agentic_research.agents.llm_client import LLMClient
    from agentic_research.agents.nl_prover import NaturalLanguageProver
    from agentic_research.models.agents import ProverConfig
    from agentic_research.pipelines.proof import ProofPipeline
    from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig
    from agentic_research.tools.lean_search import LeanSearch, SearchBackend, SearchConfig

    llm = LLMClient(model=model, max_tokens=16384)
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    search = LeanSearch(SearchConfig(backend=SearchBackend.LOOGLE))

    prover_config = ProverConfig(
        use_extended_thinking=True,
        max_iterations=3,
    )

    nl_prover = NaturalLanguageProver(
        llm_client=llm,
        prover_config=prover_config,
    )

    pipeline = ProofPipeline(
        llm_client=llm,
        lean_repl=repl,
        lean_search=search,
        prover_config=prover_config,
        max_depth=5,
        max_retries_per_node=3,
        use_claim_check=True,
        use_proof_critic=True,
        use_proof_detailer=True,
        nl_prover=nl_prover,
        use_nl_proof_stage=True,
    )

    lean_statement = putnam_a1.lean_statement
    if putnam_a1.lean_header:
        lean_statement = putnam_a1.lean_header + "\n\n" + lean_statement

    log.info("running_proof_pipeline", statement=lean_statement[:200])

    result = pipeline.run(
        lean_statement,
        statement_nl=putnam_a1.natural_language,
    )

    elapsed = round(time.monotonic() - start, 1)

    log.info(
        "e2e_result",
        proved=result.proved,
        failure_stage=result.failure_stage,
        failure_reason=result.failure_reason,
        claim_check_passed=result.claim_check_passed,
        backtrack_stages=result.backtrack_stages,
        backend=result.backend,
        verified=result.verified,
        elapsed_seconds=elapsed,
        total_input_tokens=result.total_token_usage.input_tokens,
        total_output_tokens=result.total_token_usage.output_tokens,
    )

    if result.proved:
        log.info("proof_succeeded", proof_len=len(result.final_proof or ""))
        print(f"\n{'='*60}")
        print(f"PROOF SUCCEEDED in {elapsed}s")
        print(f"{'='*60}")
        if result.final_proof:
            print(result.final_proof[:2000])
        return 0

    print(f"\n{'='*60}")
    print(f"PROOF FAILED in {elapsed}s")
    print(f"Stage: {result.failure_stage}")
    print(f"Reason: {result.failure_reason}")
    print(f"{'='*60}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
