# Demo 1 — Putnam 2024 A1: Full Pipeline

## Problem

**Determine all positive integers *n* for which there exist positive integers *a*, *b*, *c* satisfying 2*a*^*n* + 3*b*^*n* = 4*c*^*n*.**

**Answer:** {1} — only *n* = 1 works.

**Source:** [PutnamBench](https://github.com/trishullab/PutnamBench) (`putnam_2024_a1`), from the 2024 William Lowell Putnam Mathematical Competition.

## Pipeline

This demo runs the full ProofPartner pipeline:

```
explore → formalize → prove
```

### Stage 1: Exploration

The Explorer agent analyzes the problem statement, identifies the number theory domain, and recognizes the Fermat-like structure of 2*a*^*n* + 3*b*^*n* = 4*c*^*n*.

**What to look for in the output:**
- Domain identification: the equation's resemblance to Fermat's Last Theorem (homogeneous Diophantine equation with degree *n*)
- Conjecture generation with confidence and difficulty scores
- Small-case analysis conjectures (testing *n* = 1, 2, 3 directly)
- Modular arithmetic conjectures (constraints mod 2, mod 3, mod 4)
- An analogy to FLT for *n* ≥ 2 (no solutions for large exponents)

### Stage 2: Formalization

The Formalizer agent translates the top conjecture into a Lean 4 statement using type-first formalization. The statement compiles on the first iteration, and the IntentJudge verifies it faithfully captures the original mathematical intent.

**What to look for in the output:**
- Type planning: the type planner identifies Lean types needed (ℕ, set-builder notation), with a JSON parse fallback and retry
- Lemma planning: 5 helper lemmas planned, covering 1 custom type (`DiophantineExponentSet`)
- Lean REPL compilation: the theorem formalizer produces a compiling statement on iteration 1 (`theorem_formalizer_done compiles=True iterations=1`), verified against Mathlib with the cache present
- IntentJudge verdict: `CORRECT` with confidence 1.0 across all dimensions (type fidelity, quantifier accuracy, constraint preservation)
- `formalization_pipeline_success` — the full type-first pipeline completes end-to-end

### Stage 3: Proof Search

The Prover agent searches for a proof, which requires two parts:

1. **n = 1 case:** Find positive integers *a*, *b*, *c* satisfying 2*a* + 3*b* = 4*c* (e.g., *a* = 1, *b* = 2, *c* = 2 gives 2 + 6 = 8 = 4·2)
2. **n ≥ 2 case:** Show no solution exists — via modular arithmetic arguments or descent, analogous to the Fermat's Last Theorem strategy

**What to look for in the output:**
- Budget confirmation prompt (proof search is the most expensive stage)
- Automated tactic attempts (tier1_combinator, aesop) failing before LLM-guided search
- Three proof strategies tried (direct, case_analysis, contradiction) — all fail on the monolithic goal
- Lemma decomposition: the pipeline breaks the theorem into 5 top-level lemmas
- Lemma leanification: all 5 lemmas successfully translated to compiling Lean 4 statements
- Recursive prover: `lemma_1` (n=1 witness) proved directly; `lemma_2`–`lemma_3` sub-lemmas (parity arguments) proved via further decomposition
- `lemma_4` (n≥2 impossibility) triggers deep recursive decomposition to tree depth 4, proving many leaf lemmas but getting stuck on the core descent step (`lemma_4_lemma_2`)
- 367 LLM calls over 72 minutes, $60.10 total cost — budget exceeded
- `PROOF FAILED` status — the recursive prover could not close all nodes

### Outcome: Partial Success

In this run, **exploration succeeded** (correctly identifying the answer as {1} with high confidence), **formalization succeeded** (producing a compiling Lean 4 statement with IntentJudge verdict CORRECT at confidence 1.0), but **proof search failed**. This is an honest result — Putnam competition problems are at the frontier of what automated provers can handle. The pipeline correctly:

- Identified the problem domain (Number Theory / Diophantine Equations)
- Generated the correct answer ({1}) as the top conjecture
- Formalized the statement into compiling Lean 4 on the first attempt
- Verified the formalization faithfully captures the mathematical intent
- Decomposed the proof into the right lemmas and proved many sub-goals
- But could not close the n≥2 impossibility argument via infinite descent

This demonstrates ProofPartner's strengths (exploration, conjecture generation, formalization with intent verification, lemma decomposition) and current limitations (formal proof of competition-level number theory via descent arguments).

### Mathematical approach

The key insight is that for *n* = 1, the linear Diophantine equation 2*a* + 3*b* = 4*c* has infinitely many positive integer solutions. For *n* ≥ 2, one can show via modular arithmetic (e.g., considering the equation mod 2 and mod 3) that the coefficients 2, 3, 4 prevent any solution, or apply deeper number-theoretic arguments related to Fermat's Last Theorem.

## Files

| File | Contents |
|------|----------|
| [output.txt](output.txt) | Captured terminal output from the full pipeline run (trimmed — proof search section summarized) |
| [theorem.lean](theorem.lean) | Generated Lean 4 artifact with metadata comments |
| [cost.md](cost.md) | Per-stage cost breakdown with token counts |
