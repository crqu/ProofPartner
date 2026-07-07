# Factory Configuration — Agentic Mathematical Research Partner

## Project

- **Name**: Agentic Mathematical Research Partner
- **Description**: Interactive agentic tool for math research using Lean 4 — transforms rough mathematical ideas into formal Lean 4 conjectures and discovers proofs through an explore-conjecture-prove loop
- **Language**: Python 3.11+
- **Framework**: Custom orchestrator (state machine)
- **Target branch**: master

## Eval Dimensions

- **tests**: Unit and integration test pass rate (`pytest tests/ -v`)
- **lint**: Code style and static analysis (`ruff check agentic_research/ tests/`)
- **type_check**: Type safety verification (`mypy agentic_research/`)
- **capability_surface**: Fraction of core pipeline stages implemented and functional
- **observability**: Structured logging coverage and cost tracking instrumentation

## Eval Command

```bash
pytest tests/ -v
```

## Mutable Surfaces

- `agentic_research/` — all source code
- `tests/` — test suite
- `pyproject.toml` — project configuration
- `factory.md` — this file
- `CLAUDE.md` — project conventions
- `README.md` — documentation

## Fixed Surfaces

- `.factory/` — factory infrastructure
- `data/benchmarks/` — downloaded benchmark data (read-only after download)
