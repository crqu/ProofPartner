# Quickstart

Get from zero to your first conjecture in 5 minutes. For a narrative walkthrough of a complete research session, see [TUTORIAL.md](TUTORIAL.md).

## 1. Install

```bash
git clone <repo-url> && cd agentic-research
pip install -e ".[dev]"
```

## 2. Configure API Key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or for Vertex AI:

```bash
export CLAUDE_CODE_USE_VERTEX=1
export ANTHROPIC_VERTEX_PROJECT_ID="my-gcp-project"
```

## 3. Explore an Idea

```bash
agentic-research explore 'every sufficiently large even number is the sum of two primes'
```

This runs the ExplorationAgent and ConjectureGenerator. You'll see a table of ranked conjectures with confidence scores and difficulty estimates. Default budget: $2.00.

Expected output:

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

## 4. Formalize a Conjecture

Take one of the generated conjectures and formalize it into Lean 4:

```bash
agentic-research formalize 'the square root of 2 is irrational' --budget 3.00
```

This runs the FormalizationPipeline (type planning → type formalization → theorem formalization) and then the IntentJudge to verify the Lean statement captures your idea.

## 5. Check for Counterexamples

Before investing in proof search, check if the statement can be disproved:

```bash
agentic-research check 'theorem foo : ∀ n : Nat, n + 0 = n' --budget 2.00
```

## 6. Prove It

If the check returns PLAUSIBLE, attempt a proof:

```bash
agentic-research prove 'theorem foo : ∀ n : Nat, n + 0 = n' --budget 10.00 --timeout 600
```

This is the most expensive operation — it runs the full proof pipeline with recursive decomposition, iterative refinement, and Lean REPL verification.

## 7. Check Session Status

```bash
agentic-research status
```

Shows your session's conjecture history, proof outcomes, and memory tier usage.

## Next Steps

- **[Tutorial](TUTORIAL.md)** — narrative walkthrough of a complete research session (~30 min)
- **[API Guide](API.md)** — use ProofPartner as a Python library in Jupyter notebooks or batch scripts
- **[Architecture](ARCHITECTURE.md)** — full pipeline description and agent inventory
- **[Reproducibility](REPRODUCIBILITY.md)** — model versions, cost estimates, hardware requirements
- **[FAQ](FAQ.md)** — common questions and troubleshooting
- Run `agentic-research --help` for all CLI options
- Install [elan](https://github.com/leanprover/elan) for real Lean 4 proof verification (without it, Lean operations use mocked backends)
- Set `AGENTIC_RESEARCH_MODEL` to use a different Claude model
