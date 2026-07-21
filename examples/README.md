# ProofPartner — Demo Outputs

These are pre-recorded outputs from real ProofPartner runs, curated to show what the tool produces at each pipeline stage.

Each demo is a self-contained directory with the captured terminal output, any generated Lean 4 artifacts, a per-stage cost breakdown, and an annotated walkthrough explaining what happened.

## Demos

| # | Demo | Pipeline | Description |
|---|------|----------|-------------|
| 1 | [Putnam 2024 A1](01-putnam-proof/) | explore → formalize → prove | Full pipeline on a [PutnamBench](https://github.com/trishullab/PutnamBench) competition problem: determine all positive integers *n* with 2*a*^*n* + 3*b*^*n* = 4*c*^*n* |
| 2 | [Fibonacci & geometric probability](02-exploration-flow/) | explore | Exploration of the 2025 pick-up sticks discovery connecting Fibonacci numbers to the probability of forming geometric shapes with random parameters ([Scientific American, 2025](https://www.scientificamerican.com/article/students-find-hidden-fibonacci-sequence-in-classic-probability-puzzle/)) |

## How to read these

Each demo directory contains:

- **README.md** — Annotated walkthrough of what the pipeline did and what to look for
- **output.txt** — Captured terminal output (plain text, ANSI codes stripped)
- **cost.md** — Per-stage cost breakdown with token counts
- **theorem.lean** — (Demo 1 only) Generated Lean 4 artifact with metadata comments

## Version

Outputs were generated with ProofPartner v0.1.0, `claude-opus-4-6`, Lean 4 v4.33.0-rc1, Mathlib commit 27d317e.

## Non-determinism disclaimer

Actual outputs will vary — LLMs are non-deterministic. These demos represent typical successful runs. Your results may differ in conjecture ordering, proof strategies attempted, and total cost.
