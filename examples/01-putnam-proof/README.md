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

The Formalizer agent translates the top conjecture into a Lean 4 statement using type-first formalization. The IntentJudge then verifies that the formal statement faithfully captures the original mathematical intent.

**What to look for in the output:**
- The Lean 4 statement using set-builder notation, matching the PutnamBench format:
  ```lean
  theorem putnam_2024_a1 :
    {n : ℕ | 0 < n ∧ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 2*a^n + 3*b^n = 4*c^n}
      = {1}
  ```
- Intent verdict: `CORRECT` — confirming the formalization matches the problem

### Stage 3: Proof Search

The Prover agent searches for a proof, which requires two parts:

1. **n = 1 case:** Find positive integers *a*, *b*, *c* satisfying 2*a* + 3*b* = 4*c* (e.g., *a* = 1, *b* = 2, *c* = 2 gives 2 + 6 = 8 = 4·2)
2. **n ≥ 2 case:** Show no solution exists — via modular arithmetic arguments or descent, analogous to the Fermat's Last Theorem strategy

**What to look for in the output:**
- Budget confirmation prompt (proof search is the most expensive stage)
- Progress updates with running cost
- The prover's decomposition into lemmas (n=1 existence, n=2 mod-4 descent, n≥3 impossibility)
- `PROOF FAILED` status — the recursive prover got stuck on the set equality goal

### Outcome: Partial Success

In this run, **exploration succeeded** (correctly identifying the answer as {1} with high confidence) but **proof search failed**. This is an honest result — Putnam competition problems are at the frontier of what automated provers can handle. The pipeline correctly:

- Identified the problem domain (Number Theory / Diophantine Equations)
- Generated the correct answer ({1}) as conjecture #5
- Decomposed the proof into the right lemmas
- But could not formalize individual lemmas into compiling Lean 4 code

This demonstrates both ProofPartner's strengths (exploration, conjecture generation) and current limitations (formal proof of competition-level problems).

### Mathematical approach

The key insight is that for *n* = 1, the linear Diophantine equation 2*a* + 3*b* = 4*c* has infinitely many positive integer solutions. For *n* ≥ 2, one can show via modular arithmetic (e.g., considering the equation mod 2 and mod 3) that the coefficients 2, 3, 4 prevent any solution, or apply deeper number-theoretic arguments related to Fermat's Last Theorem.

## Files

| File | Contents |
|------|----------|
| [output.txt](output.txt) | Captured terminal output from the full pipeline run |
| [theorem.lean](theorem.lean) | Generated Lean 4 artifact with metadata comments |
| [cost.md](cost.md) | Per-stage cost breakdown with token counts |
