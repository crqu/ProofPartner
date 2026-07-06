# Agentic Research Partner

## Project Overview

An interactive agentic tool that transforms rough mathematical ideas into formal Lean 4 conjectures and discovers proofs through an explore-conjecture-prove loop.

## Tech Stack

- Python 3.11+, pydantic v2, structlog, click, rich
- Lean 4 + Mathlib (via LeanDojo in later phases)
- Claude Opus 4.6 via Anthropic API

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Run eval harness
python -m agentic_research.eval.runner --mode proof_discovery --benchmark miniF2F --split valid

# CLI
agentic-research eval miniF2F --mode proof_discovery --split valid --pass-k 1
```

## Code Conventions

- All logging via `structlog` — use `from agentic_research.logging import get_logger`
- All data models via `pydantic.BaseModel` (v2)
- Type hints on all public functions
- Tests in `tests/` mirroring source structure
