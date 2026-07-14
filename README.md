# ProofPartner

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-877%20passed-brightgreen.svg)]()

An agentic mathematical research partner that transforms rough mathematical ideas into formal Lean 4 conjectures and discovers proofs.

## Statement of Need

Existing theorem proving tools — [Hilbert](https://arxiv.org/abs/2502.11842), [ReProver](https://arxiv.org/abs/2306.15626), [LeanDojo](https://leandojo.org/) — require pre-formalized Lean 4 statements as input. Researchers with rough mathematical intuitions have no tool to go from *idea* to *formal proof*. ProofPartner fills this gap: it explores mathematical ideas, generates formal conjectures, verifies intent, searches for counterexamples, and discovers proofs, all in a single interactive pipeline.

ProofPartner adapts the *type-first formalization* framework and *auxiliary lemma validation* technique from [Moakhar et al. (2026)](https://arxiv.org/abs/2606.31134), extending them into a full agentic research loop with conjecture generation, intent verification, counterexample search, and proof discovery.

### Background Knowledge

No Lean 4 or formal verification experience is required. ProofPartner handles formalization automatically. Basic familiarity with mathematical concepts in your research area is sufficient. For those interested in learning Lean 4 directly, see [Theorem Proving in Lean 4](https://leanprover-community.github.io/lean4/theorem_proving_in_lean4/).

## Who Should Use This

**Good fit:**

- You have a rough mathematical idea and want to see it formalized as a Lean 4 conjecture
- You want iterative refinement — when proofs fail, ProofPartner automatically refines conjectures and retries
- You want intent verification — a 3-path adversarial judge ensures the formalization captures your original idea
- You want counterexample search before investing compute in proof attempts

**Not yet optimal for:**

- Proving pre-formalized Lean 4 statements — use [Hilbert](https://arxiv.org/abs/2502.11842) or [ReProver](https://arxiv.org/abs/2306.15626) instead
- Formalizing existing paper proofs — use dedicated autoformalization tools
- Safety-critical proof certification — use manual Lean 4 proof development

## Key Features

1. **Explore-conjecture-prove loop** — go from rough ideas to verified Lean 4 proofs
2. **Type-first formalization** — defines Lean types before theorem statements, with auxiliary lemma validation
3. **Intent verification** — 3-path adversarial judge ensures formalization captures the user's original idea
4. **Counterexample search** — tries to disprove conjectures before investing in proof
5. **Conjecture refinement loop** — when proofs fail or counterexamples surface, automatically refines and retries
6. **Research session memory** — tiered hot/warm/cold memory tracks conjectures, directions, and partial results across sessions

## Competitive Landscape

| User need | ProofPartner | Numina-Lean-Agent | Hilbert / ReProver | LeanDojo | DeepSeek-Prover |
|---|---|---|---|---|---|
| Start from rough idea → conjecture | **Yes** — explore + conjecture generation | No — requires formal input | No — requires formal input | No — requires formal input | No — requires formal input |
| Type-first formalization | **Yes** — defines types, then theorem | No | No | No | No |
| Intent verification | **Yes** — 3-path adversarial judge | No | No | No | No |
| Counterexample search | **Yes** — before proof investment | No | No | No | No |
| Prove pre-formalized statements | Supported | **Yes** — MCP-based interactive | **99.2% miniF2F** | **Yes** — retrieval-augmented | **Yes** — MCTS-based |
| Conjecture refinement on failure | **Yes** — automatic loop | Partial — user-driven | No | No | No |
| Interactive research sessions | **Yes** — checkpointed, resumable | **Yes** — MCP tool server | No | Partial | No |

*Note: 2026 systems (Goedel-Prover-V2, Kimina-Prover-72B, BFS-Prover) achieve 73–92% on miniF2F but operate only on pre-formalized statements.*

ProofPartner operates at **Stage 1** (idea → formal conjecture → proof), while most existing tools operate at **Stage 2** (formal statement → proof). Numina-Lean-Agent is the closest competitor with a similar interactive workflow. Use ProofPartner when you don't yet have a Lean 4 statement; use Hilbert or ReProver when you do.

## Evaluation

The eval harness supports three modes:

1. **Proof discovery** — given a Lean 4 statement, find a proof (miniF2F / PutnamBench)
2. **Conjecture quality** — score generated conjectures on formalizability, non-triviality, relevance
3. **End-to-end research** — given a rough idea, produce a verified Lean proof

```bash
# Run on miniF2F validation set (30 problems, extended thinking)
python -m agentic_research.eval.runner --benchmark miniF2F --split valid \
  --sample-size 30 --seed 42 --extended-thinking

# Run on PutnamBench
python -m agentic_research.eval.runner --benchmark PutnamBench --split valid \
  --sample-size 5 --seed 42 --extended-thinking
```

**Benchmarks:**

- **miniF2F**: 500 problems (256 valid + 244 test) — AMC/AIME/IMO competition math in Lean 4
- **PutnamBench**: 672 Putnam competition problems in Lean 4

**Results (miniF2F valid, pass@1, Claude Opus 4.6):**

| Sample | Pass Rate | Wilson 95% CI |
|---|---|---|
| 5 problems (seed=42) | 5/5 (100%) | [0.57, 1.00] |
| 30 problems (seed=42) | Pending | — |

*Benchmark run in progress — results will be updated when complete.*

## Setup

Requires Python 3.11+.

```bash
# Basic install
pip install -e ".[dev]"

# With Vertex AI support
pip install -e ".[dev,vertex]"
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Direct Anthropic API access |
| `CLAUDE_CODE_USE_VERTEX=1` | Enable Vertex AI backend |
| `ANTHROPIC_VERTEX_PROJECT_ID` | Google Cloud project for Vertex AI |
| `ANTHROPIC_VERTEX_REGION` | Vertex region (default: `us-east5`) |
| `AGENTIC_RESEARCH_MODEL` | Override default model (default: `claude-opus-4-6`). Use dateless IDs for Vertex AI compatibility |

### Lean 4 (optional)

Install [elan](https://github.com/leanprover/elan) for real proof verification:

```bash
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
```

Without Lean 4, the CLI still works for exploration, conjecture generation, and formalization
using mocked backends. Proof verification requires real Lean 4.
ProofPartner will warn you if Lean is not found when running formalize/prove/research commands.

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

# Run the full explore-conjecture-prove research loop
agentic-research research 'every sufficiently large even number is the sum of two primes' --budget 20.00

# Show current session state
agentic-research status

# Prove with ProofCritic + ProofDetailer (recommended for complex theorems)
agentic-research prove 'your theorem statement' --use-critic --use-detailer --budget 10.00

# Resume an interrupted research session
agentic-research resume <session-id>

# List available sessions to resume
agentic-research resume --list

# Override the LLM model (use dateless IDs for Vertex AI)
agentic-research --model claude-opus-4-6 explore 'my idea'
```

For a step-by-step walkthrough, see the [Tutorial](docs/TUTORIAL.md). For programmatic usage, see the [API Guide](docs/API.md).

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

For a detailed description of each stage, agent inventory, data flow, and cost control architecture, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Proof Pipeline

The proof pipeline follows the architecture from [Moakhar et al. (2026)](https://arxiv.org/abs/2606.31134):

1. **Automated Tactics** — tries `grind`, `simp_all`, `field_simp; ring`, `field_simp at *; nlinarith` (solves ~40% of AMC-level problems in seconds)
2. **ProofSearch** — iterative proving with extended thinking (2 strategies × 2 iterations)
3. **ProofCorrector** — analyzes compilation errors, suggests fixes, retries
4. **NaturalLanguageProver** — generates structured informal proof sketch before formalization
5. **ProofCritic** — adversarial loop until no logical gaps remain in the NL proof
6. **ProofDetailer** — expands NL proof into tactic-level steps (runs on ALL nodes)
7. **LemmaBreakdown** — decomposes into sub-lemmas guided by the validated NL proof + tactic hints
8. **LemmaLeanifier** — translates sub-lemmas to Lean 4 with Mathlib search context
9. **RecursiveProver** — parent-before-children proving with extended thinking and compiler error feedback

## Production Hardening

- **Default budgets on all commands** — no unlimited operations; every command has a cost cap
- **Circuit breakers** — 5 consecutive failures halts the pipeline to prevent runaway spending
- **Tiered session memory** — hot/warm/cold tiers keep the most relevant context in working memory
- **Checkpointing at all 8 pipeline stages** — exploring, conjecturing, formalizing, checking intent, searching counterexamples, proving, refining, complete
- **Session resume** — `CheckpointManager` persists state so interrupted sessions can resume from the last checkpoint
- **Verifier-guided self-correction** — Lean compiler errors fed back as structured feedback to retry loop (including root assembly retries)
- **Natural language proof stage** — generates informal proof sketches with adversarial critic loop before Lean formalization
- **Search-augmented leanification** — Mathlib search results injected into formalization prompts for better tactic selection
- **Progress tracking** — `rich.Progress` shows real-time pipeline stage, elapsed time, and cost
- **Data package injection** — domain-specific Lean 4 definitions auto-detected and injected for formalization quality

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
├── data_packages/ # Domain-specific Lean 4 preambles (DRO coupling, etc.)
└── models/        # Pydantic data models
```

## Development

```bash
pytest tests/ -v          # run tests
ruff check agentic_research/ tests/  # lint
mypy agentic_research/    # type check (if installed)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## Citation

If you use ProofPartner in your research, please cite:

```bibtex
@software{qu2026proofpartner,
  title     = {ProofPartner: An Agentic Mathematical Research Partner},
  author    = {Qu, Chengrui},
  year      = {2026},
  url       = {https://github.com/crqu/ProofPartner},
  version   = {0.1.0},
  license   = {MIT}
}
```

ProofPartner adapts the type-first formalization framework from the following work — please also cite:

```bibtex
@article{moakhar2026beyond,
  title   = {Beyond the Library: An Agentic Framework for Autoformalizing Research Mathematics},
  author  = {Soltani Moakhar, Arshia and Gholami, Iman and Springer, Max and JafariRaviz, Mahdi and Hajiaghayi, MohammadTaghi},
  year    = {2026},
  eprint  = {2606.31134},
  archiveprefix = {arXiv}
}
```

See also [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## Related Projects

- **[LeanDojo](https://leandojo.org/)** — retrieval-augmented theorem proving with Lean 4 interaction (NeurIPS 2023)
- **[ReProver](https://arxiv.org/abs/2306.15626)** — retrieval-augmented prover trained on Mathlib
- **[miniF2F](https://github.com/openai/miniF2F)** — cross-system benchmark for formal olympiad-level mathematics
- **[Mathlib](https://leanprover-community.github.io/mathlib4_docs/)** — Lean 4's comprehensive mathematics library
- **[Hilbert](https://arxiv.org/abs/2502.11842)** — 99.2% on miniF2F using whole-proof generation
- **[Numina-Lean-Agent](https://github.com/project-numina/numina-lean-agent)** — MCP-based agent for Lean 4, 100% on Putnam 2025 (closest Stage 1 competitor)
- **[Goedel-Prover-V2](https://arxiv.org/abs/2508.03613)** — 90.4% miniF2F with verifier-guided self-correction
- **[Kimina-Prover-72B](https://huggingface.co/AI-MO/Kimina-Prover-72B)** — RL-trained with structured `have` proofs, 92.2% miniF2F
- **[BFS-Prover](https://arxiv.org/abs/2502.03438)** — 73% miniF2F, validates simple search over MCTS

## Documentation

| Document | Description |
|---|---|
| [Quickstart](docs/QUICKSTART.md) | From zero to your first conjecture in 5 minutes |
| [Tutorial](docs/TUTORIAL.md) | Narrative walkthrough of a complete research session |
| [API Guide](docs/API.md) | Using ProofPartner as a Python library |
| [Architecture](docs/ARCHITECTURE.md) | Pipeline stages, agent inventory, data flow |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Model versions, cost estimates, hardware requirements |
| [FAQ](docs/FAQ.md) | Common questions and answers |
| [Glossary](docs/GLOSSARY.md) | Key terms and definitions |
| [Contributing](CONTRIBUTING.md) | How to contribute to ProofPartner |

## License

[MIT](LICENSE)

## Acknowledgments

This project was developed with assistance from Claude Code (Anthropic). AI tools were used for code generation, testing, and documentation. All outputs were reviewed and validated by human authors.
