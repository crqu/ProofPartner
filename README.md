# ProofPartner

An agentic mathematical research partner that transforms rough math ideas into formal Lean 4 conjectures and discovers proofs.

An interactive agentic tool that takes rough mathematical ideas and transforms them into formal Lean 4 conjectures, discovers proofs from scratch, or refines conjectures when they turn out to be false.

Unlike autoformalization (translating existing proofs), this system is a *research partner* that explores, conjectures, and proves alongside the user through an interactive explore-conjecture-prove loop.

## Key Features

1. **Explore-conjecture-prove loop** — go from rough ideas to verified Lean 4 proofs
2. **Type-first formalization** — defines Lean types before theorem statements, with auxiliary lemma validation
3. **Intent verification** — 3-path adversarial judge ensures formalization captures the user's original idea
4. **Counterexample search** — tries to disprove conjectures before investing in proof
5. **Conjecture refinement loop** — when proofs fail or counterexamples surface, automatically refines and retries
6. **Research session memory** — tiered hot/warm/cold memory tracks conjectures, directions, and partial results across sessions

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

## Setup

Requires Python 3.11+.

```bash
# Basic install
pip install -e ".[dev]"

# With Vertex AI support
pip install -e ".[dev]" && pip install 'anthropic[vertex]'
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Direct Anthropic API access |
| `CLAUDE_CODE_USE_VERTEX=1` | Enable Vertex AI backend |
| `ANTHROPIC_VERTEX_PROJECT_ID` | Google Cloud project for Vertex AI |
| `ANTHROPIC_VERTEX_REGION` | Vertex region (default: `us-east5`) |
| `AGENTIC_RESEARCH_MODEL` | Override default model (default: `claude-opus-4-6-20250616`) |

### Lean 4 (optional)

Install [elan](https://github.com/leanprover/elan) for real proof verification. Without it, Lean operations use mocked backends (sufficient for development and testing).

## CLI Usage

All commands support `--model` to override the LLM model and `--budget` to set a cost cap.

```bash
# Explore a rough mathematical idea and generate conjectures
agentic-research explore 'every sufficiently large even number is the sum of two primes' --budget 2.00

# Formalize a conjecture into Lean 4 with intent verification
agentic-research formalize 'the square root of 2 is irrational' --budget 3.00

# Search for counterexamples to a Lean 4 statement
agentic-research check 'theorem foo : ∀ n : Nat, n + 0 = n' --budget 2.00

# Attempt to prove a Lean 4 statement (interactive confirmation)
agentic-research prove 'theorem foo : ∀ n : Nat, n + 0 = n' --budget 10.00 --timeout 600

# Show current session state
agentic-research status

# Run benchmark evaluation
agentic-research eval miniF2F --mode proof_discovery --split valid --pass-k 8
```

## Production Hardening

- **Default budgets on all commands** — no unlimited operations; every command has a cost cap
- **Circuit breakers** — 5 consecutive failures halts the pipeline to prevent runaway spending
- **Tiered session memory** — hot/warm/cold tiers keep the most relevant context in working memory
- **Checkpointing at all 8 pipeline stages** — exploring, conjecturing, formalizing, checking intent, searching counterexamples, proving, refining, complete
- **Session resume** — `CheckpointManager` persists state so interrupted sessions can resume from the last checkpoint

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

## Competitive Landscape

No open-source tool combines conjecture generation from rough ideas, type-first formalization, intent verification, counterexample search, and automated proof discovery in a single pipeline. Existing systems like Hilbert (99.2% MiniF2F), FrenzyMath, and Google DeepMind's proof agents focus on proving existing formal statements — they don't take rough mathematical ideas and turn them into conjectures. This tool fills the gap between "I have a vague mathematical intuition" and "I have a verified Lean 4 proof."

## Development

```bash
pytest                    # run tests
pytest -v tests/          # verbose
ruff check agentic_research/ tests/  # lint
mypy agentic_research/    # type check (if installed)
```
