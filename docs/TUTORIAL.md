# Tutorial: A Complete Research Session

This tutorial walks through a complete research session with ProofPartner, from a rough mathematical idea to a formal Lean 4 conjecture. You'll learn what each command does, how to interpret the output, and what to do when things go wrong.

**Time:** ~30 minutes. **Cost:** ~$2–5 for exploration and formalization.

**Prerequisites:** ProofPartner installed (`pip install -e ".[dev]"`), an Anthropic API key set (`ANTHROPIC_API_KEY`). No Lean 4 installation required for exploration and formalization.

## Decision Tree: Where to Start

```
Do you have a rough mathematical idea?
  └─ Yes → Step 1: agentic-research explore
  
Do you have a precise natural-language conjecture?
  └─ Yes → Step 3: agentic-research formalize

Do you already have a Lean 4 statement?
  └─ Yes → Step 5: agentic-research check, then Step 6: agentic-research prove
```

## Step 1: Explore an Idea

Start with a rough mathematical idea. You don't need a precise statement — just a direction.

```bash
agentic-research explore 'every sufficiently large even number is the sum of two primes' --budget 2.00
```

**What happens:** The ExplorationAgent identifies the mathematical domain (number theory), searches Mathlib for related formalizations, and produces ranked research directions. Then the ConjectureGenerator produces formal conjecture candidates.

**Expected output:**

```
                     Generated Conjectures
┌───┬──────────────────────────────────────────┬────────────┬────────────┐
│ # │ Statement                                │ Confidence │ Difficulty │
├───┼──────────────────────────────────────────┼────────────┼────────────┤
│ 1 │ For all n > 2, 2n = p + q for primes... │       0.85 │          8 │
│ 2 │ Every even number > 4 has at least...    │       0.72 │          6 │
│ 3 │ ...                                      │       0.60 │          5 │
└───┴──────────────────────────────────────────┴────────────┴────────────┘

Domain: number_theory | Concepts found: 12 | Directions: 4

                Cost Summary
┌────────────┬────────────────┐
│ Total cost │ $0.0842        │
│ Budget     │ $2.00          │
│ Status     │ Within budget  │
└────────────┴────────────────┘
```

**How to read the output:**

- **Confidence** (0–1): How likely the conjecture is to be true, based on LLM assessment. Higher is better for proof attempts.
- **Difficulty** (1–10): Estimated difficulty of formalization and proof. Lower numbers are easier to prove.
- **Domain**: The identified mathematical area (e.g., `number_theory`, `analysis`, `algebra`).
- **Concepts found**: Number of related Mathlib concepts discovered via LeanSearch.
- **Directions**: Number of distinct research directions identified.

**Typical cost:** $0.05–$0.50.

## Step 2: Choose a Conjecture

Look at the ranked conjectures and pick one to formalize. Good strategies:

- **Start with high-confidence, low-difficulty** conjectures to validate your workflow
- **Avoid difficulty 9–10** conjectures on first attempts — these are likely open problems
- **Pick conjectures that match your research interest**, not just the highest-ranked one

## Step 3: Formalize a Conjecture

Take the natural-language conjecture and formalize it into Lean 4:

```bash
agentic-research formalize 'the square root of 2 is irrational' --budget 3.00
```

**What happens:** The FormalizationPipeline runs in stages:

1. **Type planning** — determines which Lean 4 types the conjecture needs
2. **Lemma planning** — generates auxiliary lemmas for validation
3. **Type formalization** — translates informal types to Lean 4 (k candidates, best-of-k selection)
4. **Theorem formalization** — produces the final Lean 4 theorem statement
5. **Intent verification** — the IntentJudge checks that the formalization captures your idea

This uses the *type-first formalization* approach from [Moakhar et al. (2026)](https://arxiv.org/abs/2606.31134).

**Expected output:**

```
Lean 4 Statement:
theorem sqrt2_irrational : Irrational (Real.sqrt 2) := by
  sorry

Intent Verdict: PASS
```

**How to interpret the intent verdict:**

- **PASS** — the 3-path judge confirms the Lean statement captures your conjecture. Proceed to counterexample check.
- **FAIL** — the formalization doesn't match your intent. The output includes specific concerns. Options:
  - Rephrase your conjecture more precisely and re-run `formalize`
  - Use `--budget` with a higher value to allow more formalization iterations

**Typical cost:** $0.50–$3.00.

## Step 4: Intent Verification Details

The IntentJudge uses three independent checks:

1. **Forward check** — does the Lean statement logically imply the informal conjecture?
2. **Backward check** — does the informal conjecture logically imply the Lean statement?
3. **Back-translation** — the Informalizer translates Lean 4 → natural language, which is compared against the original

If any check raises concerns, you'll see them listed. Common issues:

- **Scope mismatch** — the Lean statement is more specific or more general than intended
- **Missing conditions** — the formalization dropped an important precondition
- **Type mismatch** — the mathematical objects are formalized using the wrong Lean types

## Step 5: Check for Counterexamples

Before investing in proof search, check if the statement can be disproved:

```bash
agentic-research check 'theorem foo : ∀ n : Nat, n + 0 = n' --budget 2.00
```

**Expected output:**

```
PLAUSIBLE — no counterexample found

Candidates tried: 5
```

**How to interpret:**

- **PLAUSIBLE** — no counterexample was found. This doesn't prove the statement, but it's a good sign. Proceed to proof search.
- **DISPROVED** — a counterexample was found! The output shows the counterexample and its Lean 4 code. Options:
  - Return to Step 3 and refine the conjecture
  - Use `agentic-research research` for automatic refinement (the full loop handles this)

**Typical cost:** $0.10–$2.00.

## Step 6: Prove It

If the check returns PLAUSIBLE, attempt a proof:

```bash
agentic-research prove 'theorem foo : ∀ n : Nat, n + 0 = n' --budget 10.00 --timeout 600
```

**Note:** This is the most expensive operation. The CLI asks for confirmation before starting.

**What happens:** The ProofPipeline uses multiple strategies:

- Lemma decomposition into sub-goals
- Recursive proving (parent-before-children)
- Iterative refinement with Lean REPL feedback
- Direct proof search
- Final assembly and claim checking

**Budget/timeout guidance:**

| Theorem type | Suggested budget | Suggested timeout |
|---|---|---|
| Simple (Nat arithmetic) | $1–3 | 120s |
| Medium (basic algebra/analysis) | $3–10 | 300s |
| Hard (advanced number theory) | $10–20 | 600s |

**Typical cost:** $1–$10.

**Note:** Lean 4 must be installed (via [elan](https://github.com/leanprover/elan)) for verified proof search. Without it, proof search uses mocked backends and cannot produce verified proofs.

## Step 7: Handle Failure

When things go wrong, ProofPartner offers several paths:

**Counterexample found → automatic refinement:**

Use the full research loop instead of individual commands:

```bash
agentic-research research 'every sufficiently large even number is the sum of two primes' --budget 20.00
```

The orchestrator automatically detects counterexamples and triggers the ConjectureRefiner to produce modified conjectures, then re-enters the formalize → check → prove loop.

**Proof timeout → increase budget or decompose:**

```bash
# Increase budget and timeout
agentic-research prove 'theorem ...' --budget 20.00 --timeout 1200

# Or decompose into sub-lemmas manually, proving each separately
agentic-research prove 'lemma part1 : ...' --budget 5.00
agentic-research prove 'lemma part2 : ...' --budget 5.00
```

**Intent judge rejection → rephrase conjecture:**

If the intent judge rejects the formalization, look at the specific concerns and rephrase. Common fixes:

- Add explicit quantifiers: "for all n" instead of just "n"
- Specify the domain: "for natural numbers" or "for real numbers"
- Add boundary conditions: "for n > 2" instead of "for all n"

## Step 8: Check Session Status

```bash
agentic-research status
```

Shows your session's conjecture history, proof outcomes, and memory tier usage. Sessions are stored in `.agentic_research/sessions/` and persist between commands.

## The Full Research Loop

For a hands-off experience, use `agentic-research research` which runs the complete explore → conjecture → formalize → check → prove → refine loop:

```bash
agentic-research research 'the square root of 2 is irrational' --budget 20.00 --max-conjectures 5
```

This creates checkpoints at each stage. If interrupted, the session can be resumed (CLI `resume` command planned — see [issue #7](https://github.com/crqu/ProofPartner/issues/7)).

## Next Steps

- **[Quickstart](QUICKSTART.md)** — the condensed 5-minute version
- **[API Guide](API.md)** — use ProofPartner as a Python library in Jupyter notebooks or batch scripts
- **[Reproducibility](REPRODUCIBILITY.md)** — model versions, cost estimates, hardware requirements
- **[FAQ](FAQ.md)** — common questions and troubleshooting
- **[Architecture](ARCHITECTURE.md)** — detailed pipeline internals and agent inventory
