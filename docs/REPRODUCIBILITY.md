# Reproducibility Guide

This document provides the information needed to reproduce ProofPartner results in a research setting: model versions, cost estimates, hardware requirements, and environment setup.

## Model Versions

**Default model:** `claude-opus-4-6-20250616`

Override via environment variable or CLI flag:

```bash
# Environment variable
export AGENTIC_RESEARCH_MODEL=claude-sonnet-4-20250514

# CLI flag (per-command)
agentic-research --model claude-sonnet-4-20250514 explore 'my idea'
```

**Model behavior notes:**

- Claude Opus 4.6 is the default and recommended model for all stages. It provides the best balance of mathematical reasoning and formalization quality.
- Claude Opus 4.5 has been validated for type-first formalization — it generated working DRO (distributionally robust optimization) type definitions.
- Smaller models (Sonnet) can be used for exploration and conjecture generation at lower cost, but may produce lower-quality formalizations.

**When reporting results, always specify:**

1. Model name and version (e.g., `claude-opus-4-6-20250616`)
2. Date of experiment (LLM behavior may change across API versions)
3. Temperature setting (default: 0.0 for most agents, 0.2 for exploration)

## Cost Estimates

Default budgets per command:

| Command | Default budget | Typical actual cost | What it does |
|---|---|---|---|
| `explore` | $2.00 | $0.05–$0.50 | Domain identification + conjecture generation |
| `formalize` | $3.00 | $0.50–$3.00 | Type-first formalization + intent verification |
| `check` | $2.00 | $0.10–$2.00 | Counterexample search |
| `prove` | $10.00 | $1.00–$10.00 | Full proof pipeline |
| `research` | $20.00 | $2.00–$20.00 | Complete explore → prove loop |

**Cost guidance:**

- A typical explore + formalize session costs $2–5.
- Proof search varies widely — simple theorems cost <$1, complex theorems may hit the $10 cap.
- All commands enforce budget caps via `CostTracker`. Operations halt when the budget is exceeded.
- Use `--budget` on any command to set a custom cap.

**Cost breakdown by token type:**

| Token type | Approximate rate (Opus 4.6) |
|---|---|
| Input tokens | $15 / 1M tokens |
| Output tokens | $75 / 1M tokens |
| Cache read | $1.50 / 1M tokens |
| Cache write | $18.75 / 1M tokens |

ProofPartner uses prompt caching for system prompts, which significantly reduces costs on repeated calls within the same session.

## Hardware Requirements

- **CPU:** Any modern x86-64 or ARM64 processor. No GPU required — LLM inference runs via the Anthropic API.
- **RAM:** Minimal beyond the Python process (~100–500 MB typical). No large local models.
- **Disk:** Session data stored in `.agentic_research/sessions/`, growing ~1–5 MB per session. Lean 4 + Mathlib installation requires ~5 GB if using real proof verification.
- **Network:** Internet connection required for Anthropic API calls and LeanSearch queries. Bandwidth usage is modest (text-only API calls).

## Lean 4 Setup

Lean 4 is **optional** for exploration, conjecture generation, and formalization. It is **required** for verified proof search and counterexample checking against the real Lean 4 kernel.

### Installing Lean 4

1. Install [elan](https://github.com/leanprover/elan) (the Lean version manager):

```bash
curl https://elan.lean-lang.org/install.sh -sSf | sh
```

2. ProofPartner has been tested with the leanprover/lean4 stable toolchain. Specify the toolchain in your Lean project's `lean-toolchain` file.

3. For Mathlib access (recommended for non-trivial formalization):

```bash
# In a Lean project with Mathlib dependency
lake build
```

### Without Lean 4

Without Lean installed, ProofPartner uses mocked backends. This is sufficient for:

- Exploring mathematical ideas
- Generating conjectures
- Running the formalization pipeline (type planning, theorem formalization)
- Development and testing

It is **not** sufficient for:

- Verifying that Lean statements compile
- Running real counterexample checks
- Producing verified proofs

## Python Environment

**Required:** Python 3.11+

For exact reproduction of a previous experiment:

```bash
# Save current environment
pip freeze > requirements-locked.txt

# Reproduce from locked requirements
pip install -r requirements-locked.txt
```

**Dependencies** (from `pyproject.toml`):

- `anthropic>=0.39.0,<1.0` — Anthropic API client
- `pydantic>=2.0.0,<3.0` — data model validation
- `click>=8.1.0,<9.0` — CLI framework
- `structlog>=24.0.0,<25.0` — structured logging
- `rich>=13.0.0,<14.0` — terminal output formatting

## LLM Non-Determinism

LLM outputs are inherently non-deterministic. Results will vary between runs even with identical inputs.

**To improve reproducibility:**

- Set temperature to 0.0 (this is the default for most agents)
- Report the exact model version used
- Report the date of your experiment
- Run multiple trials and report aggregate statistics

**In publications, include:**

```
Results were obtained using ProofPartner v0.1.0 with claude-opus-4-6-20250616 
(temperature 0.0, default budgets) on [date]. Due to LLM non-determinism, 
individual results may vary across runs.
```

## Vertex AI Configuration

For Google Cloud users:

```bash
export CLAUDE_CODE_USE_VERTEX=1
export ANTHROPIC_VERTEX_PROJECT_ID="my-gcp-project"
export ANTHROPIC_VERTEX_REGION="us-east5"  # optional, defaults to us-east5
```

Use dateless model IDs with Vertex AI (e.g., `claude-opus-4-6` instead of `claude-opus-4-6-20250616`). The client automatically converts between direct API and Vertex AI model ID formats.

## Further Reading

- **[Tutorial](TUTORIAL.md)** — step-by-step research session walkthrough
- **[API Guide](API.md)** — programmatic usage for batch experiments
- **[FAQ](FAQ.md)** — common questions including cost and model selection
- **[README](../README.md)** — project overview and setup
