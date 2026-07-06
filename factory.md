# Factory Configuration — Agentic Research Partner

## Eval Dimensions

- **proof_discovery_rate**: Fraction of benchmark problems (miniF2F/PutnamBench) where a valid Lean 4 proof is found within the attempt budget. Primary capability metric.
- **conjecture_quality**: Composite score of generated conjectures: formalizability (can it be expressed in Lean?), non-triviality (is it non-obvious?), relevance (does it capture the original idea?). Range 0-1.
- **compilation_rate**: Fraction of generated Lean 4 code that compiles successfully. Tracks the main bottleneck: writing valid Lean.

## Eval Command

```bash
python -m agentic_research.eval.runner --mode proof_discovery --benchmark miniF2F --split valid --pass-k 1
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
