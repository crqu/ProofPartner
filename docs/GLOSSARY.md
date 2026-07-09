# Glossary

Key terms used throughout ProofPartner's documentation and codebase.

| Term | Definition |
|---|---|
| **Agentic** | An AI system that takes autonomous actions toward a goal, making decisions about what to do next rather than waiting for explicit instructions at each step. ProofPartner's agents (explorer, conjecturer, prover, etc.) each act autonomously within their stage of the pipeline. |
| **Auxiliary lemma** | A helper lemma generated alongside a type formalization to validate that the Lean 4 type definition is mathematically sound. Part of the type-first formalization approach. |
| **Circuit breaker** | A safety mechanism that halts the pipeline after 5 consecutive failures to prevent runaway API spending. Resets on any successful operation. |
| **Claim check** | A verification step that ensures a completed proof hasn't silently weakened hypotheses or proved a trivially weaker statement than intended. |
| **Conjecture refinement** | The process of modifying a conjecture after a counterexample is found or a proof attempt fails. ProofPartner's `ConjectureRefiner` agent produces refined variants and re-enters the formalization-proof loop. |
| **Counterexample search** | An automated search for inputs that disprove a conjecture. If found, the conjecture is marked DISPROVED and sent to refinement. If not found, the conjecture is marked PLAUSIBLE and proceeds to proof search. |
| **Formalization** | Translating informal mathematics (natural language) into machine-checkable code (Lean 4). ProofPartner uses type-first formalization: types are defined first, then the theorem statement is built on top. |
| **Intent verification** | Checking that a Lean 4 formalization captures the user's original mathematical idea. ProofPartner uses a 3-path adversarial approach: forward check, backward check, and back-translation comparison. |
| **Lean 4** | An interactive theorem prover and programming language. Lean 4 can verify that mathematical proofs are correct by checking them against its logical foundations. ProofPartner generates Lean 4 code. |
| **LeanSearch** | An API service that searches Lean 4's Mathlib library for existing definitions, theorems, and lemmas related to a query. Used by ProofPartner's exploration agent to find relevant formalizations. |
| **Mathlib** | Lean 4's comprehensive mathematics library, containing formalized definitions and theorems across algebra, analysis, number theory, topology, and more. ProofPartner searches Mathlib for relevant types and results. |
| **Proof term** | A complete proof expression in Lean 4 that can be type-checked by the Lean kernel. Unlike a tactic proof (which describes steps), a proof term directly represents the evidence for a proposition. |
| **Session memory** | ProofPartner's tiered memory system (hot/warm/cold) that tracks conjectures, research directions, and partial results across pipeline stages. Hot memory contains active context; cold memory stores completed results. |
| **Tactic** | A Lean 4 proof step that transforms the current proof state. Examples: `intro` (introduce a variable), `apply` (apply a lemma), `simp` (simplify), `sorry` (placeholder for an incomplete proof). |
| **Type-first formalization** | A formalization strategy that defines the Lean 4 types (mathematical structures) needed by a conjecture before attempting to write the theorem statement. Adapted from [Moakhar et al. (2026)](https://arxiv.org/abs/2606.31134). Produces more robust formalizations than direct statement translation. |
