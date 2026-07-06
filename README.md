# Agentic Mathematical Research Partner

An interactive agentic tool that takes rough mathematical ideas and transforms them into formal Lean 4 conjectures, discovers proofs from scratch, or refines conjectures when they turn out to be false.

Unlike autoformalization (translating existing proofs), this system is a *research partner* that explores, conjectures, and proves alongside the user through an interactive explore-conjecture-prove loop.

## Architecture

```
User's rough idea
       │
       ▼
┌──────────────┐
│  Exploration │ → identifies domain, relevant concepts, formalizations
│  Agent       │
└──────┬───────┘
       ▼
┌──────────────┐
│  Conjecture  │ → produces formal conjecture candidates
│  Generator   │
└──────┬───────┘
       ▼
┌──────────────┐
│  Type-First  │ → defines Lean types, validates via auxiliary lemmas
│  Formalizer  │
└──────┬───────┘
       ▼
┌──────────────┐
│  Intent      │ → verifies formalization captures user's idea
│  Judge       │
└──────┬───────┘
       ▼
┌──────────────┐
│ Counterexample│ → tries to disprove before investing in proof
│  Searcher    │
└──────┬───────┘
       ▼
   ┌───┴────┐
   │        │
survived  disproved → Conjecture Refiner → loop back
   │
   ▼
┌──────────┐
│  Proof   │ → recursive decomposition, iterative refinement
│  Search  │
└──────────┘
       │
       ▼
   Verified Lean Proof
```

## Package Structure

```
agentic_research/
├── agents/        # LLM-powered agents (prover, explorer, conjecturer, etc.)
├── tools/         # Lean 4 tool wrappers (REPL, search, lookup)
├── pipelines/     # Multi-agent pipelines
├── eval/          # Evaluation harness + benchmark loaders
├── orchestrator/  # Central orchestrator + state management
├── cli/           # Click CLI entry points
├── memory/        # Research session memory
└── models/        # Pydantic data models
```

## Setup

Requires Python 3.11+.

```bash
pip install -e ".[dev]"
```

## Evaluation

The eval harness supports three modes:

1. **Proof discovery** — given a Lean 4 statement, find a proof (miniF2F / PutnamBench)
2. **Conjecture quality** — score generated conjectures on formalizability, non-triviality, relevance
3. **End-to-end research** — given a rough idea, produce a verified Lean proof

```bash
# Run on miniF2F validation set
python -m agentic_research.eval.runner --mode proof_discovery --benchmark miniF2F --split valid --pass-k 1

# Sample 32 problems with a fixed seed
python -m agentic_research.eval.runner --benchmark miniF2F --sample-size 32 --seed 42

# Use the CLI
agentic-research eval miniF2F --mode proof_discovery --split valid --pass-k 8
```

## Benchmarks

- **miniF2F v2**: 488 problems (244 test + 244 validation) — competition math in Lean 4
- **PutnamBench**: 672 Putnam competition problems (stub loader, activated in later phases)

## Development

```bash
pytest                    # run tests
pytest -v tests/          # verbose
```
