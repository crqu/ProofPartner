# Demo 2 — Fibonacci & Geometric Probability: Exploration Flow

## Input

```
Explore connections between Fibonacci numbers and probability of forming
geometric shapes with random parameters
```

## Context

This demo explores a topic connected to a 2025 top-10 math discovery: the *pick-up sticks problem*. Given *n* random-length sticks, the probability that no three of them can form a triangle involves Fibonacci numbers — a surprising bridge between combinatorics, probability, and number theory.

The result was highlighted as one of the most significant mathematical discoveries of 2025, revealing deep connections between the Fibonacci sequence and geometric probability that were previously unknown.

## Pipeline

This demo runs only the exploration stage:

```
explore
```

### What the explore command shows

The Explorer agent takes a rough mathematical idea and produces:

1. **Ranked conjecture table** — 4–6 conjectures with confidence scores and difficulty ratings, spanning:
   - **Easy (difficulty 2–3):** Known Fibonacci identities in probabilistic context (e.g., F(*n*)² + F(*n*+1)² = F(2*n*+1), divisibility properties)
   - **Medium (difficulty 4–6):** Probability formulas involving Fibonacci ratios (e.g., P(no triangle from *n* uniform sticks) = F(2*n*+1)/F(2*n*+2))
   - **Hard (difficulty 7–8):** Asymptotic bounds and generalizations (e.g., *k*-simplex constraints yielding *k*-bonacci sequences)

2. **Domain identification** — Cross-domain nature: number theory, probability, combinatorics

3. **Cost summary** — Total cost and budget status

### What to look for

- **Diversity of conjectures:** The table should range from straightforward identities to genuinely novel connections, showing the Explorer's ability to generate across difficulty levels
- **Cross-domain connections:** Conjectures bridging number theory (Fibonacci properties) with probability (geometric shape formation) and combinatorics (counting arguments)
- **Confidence calibration:** Higher confidence for known results, lower for speculative generalizations
- **The pick-up sticks connection:** At least one conjecture should relate to the probability of triangle formation with random sticks, connecting back to the 2025 discovery

## Files

| File | Contents |
|------|----------|
| [output.txt](output.txt) | Captured terminal output from the exploration run |
| [cost.md](cost.md) | Per-stage cost breakdown with token counts |
