# Cost Breakdown — Demo 1: Putnam 2024 A1

Model: `claude-opus-4-6` via Vertex AI

| Stage | Input Tokens | Output Tokens | Cost (USD) |
|-------|-------------|---------------|------------|
| Exploration (LeanSearch + LLM) | 437 | 2,386 | ~$0.19 |
| Conjecture Generation (2 LLM calls) | 2,556 | 4,282 | ~$0.35 |
| Formalization (type planning + theorem formalization, 5 iterations) | 4,750 | 2,287 | ~$0.38 |
| Proof Search (automated tactics + lemma decomposition + recursive prover) | ~4,200 | ~29,170 | ~$2.94 |
| **Total** | **~11,943** | **~38,125** | **$3.86** |

## Pipeline Result

- Exploration: **SUCCESS** — 5 conjectures generated, correctly identified answer {1}
- Formalization: **FAILED** — Lean REPL could not compile (Mathlib import errors in local environment)
- Proof Search: **FAILED** — recursive prover got stuck on set equality goal after 3 retries

## Timing

| Stage | Duration |
|-------|----------|
| Exploration | ~107s |
| Formalization | ~182s |
| Proof Search | ~502s |
| **Total** | **~791s (~13 min)** |

## Notes

The proof search correctly decomposed the problem into lemmas (n=1 existence, n=2 impossibility via mod-4 descent, n≥3 impossibility) but failed to formalize the individual lemmas into compiling Lean 4 code. This is a known limitation for competition-level problems — see the project's eval results on PutnamBench.
