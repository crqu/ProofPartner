# Cost Breakdown — Demo 1: Putnam 2024 A1

Model: `claude-opus-4-6` via Vertex AI

| Stage | LLM Calls | Input Tokens | Output Tokens | Cost (USD) |
|-------|-----------|-------------|---------------|------------|
| Exploration (LeanSearch + LLM) | 3 | 3,009 | 4,195 | ~$0.36 |
| Formalization (type planning + theorem formalizer + IntentJudge, 10 LLM calls) | 10 | 8,724 | 5,468 | ~$0.54 |
| Proof Search (automated tactics + lemma decomposition + recursive prover, 367 LLM calls) | 367 | 355,535 | 203,119 | ~$20.57 |
| **Total** | **380** | **~367,268** | **~212,782** | **$60.10** |

## Pipeline Result

- Exploration: **SUCCESS** — 5 conjectures generated, correctly identified answer {1}
- Formalization: **SUCCESS** — Lean 4 statement compiles on first iteration; IntentJudge verdict CORRECT (confidence 1.0)
- Proof Search: **FAILED** — recursive prover stuck on n≥2 impossibility lemma (`lemma_4_lemma_2`) after exhausting all retries; budget exceeded

## Timing

| Stage | Duration |
|-------|----------|
| Exploration | ~105s |
| Formalization | ~113s |
| Proof Search | ~4316s (72 min) |
| **Total** | **~4534s (~76 min)** |

## Notes

The proof search correctly decomposed the problem into lemmas and proved many leaf goals:
- **lemma_1** (n=1 witness): proved directly
- **lemma_2** sub-lemmas (b must be even): proved via sub-decomposition (lemma_2_lemma_1 through lemma_2_lemma_4)
- **lemma_3** sub-lemmas (a must be even): all 5 sub-lemmas proved
- **lemma_4** (n≥2 impossibility via infinite descent): partially proved — leaf lemmas at depth 3 succeeded, but `lemma_4_lemma_2` (the core descent step) remained stuck after repeated decomposition attempts

The recursive prover reached tree depth 4 (e.g., `lemma_4_lemma_2_lemma_4_lemma_3`) and ran 3622s before exhausting retries. Total cost was dominated by the proof search stage (~$59 of the $60.10 total).
