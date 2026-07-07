# Architecture

## Pipeline Stages

The orchestrator (`ResearchOrchestrator`) drives an 8-stage state machine. Each stage can transition forward on success, sideways to refinement on failure, or halt on budget/circuit-breaker limits.

### 1. Exploring

`ExplorationAgent` takes a rough mathematical idea and identifies the relevant domain, Mathlib concepts, and promising research directions. It queries `LeanSearch` for related formalizations.

### 2. Conjecturing

`ConjectureGenerator` takes the exploration output and produces ranked conjecture candidates with confidence scores, difficulty estimates, and natural-language statements.

### 3. Formalizing

The `FormalizationPipeline` orchestrates three sub-agents:
- `TypePlanner` — analyzes the conjecture to determine which Lean types are needed
- `TypeFormalizer` — translates informal types to Lean 4, validated by the `Auctioneer` (best-of-k selection)
- `TheoremFormalizer` — produces the final Lean 4 theorem statement using the accepted types
- `LemmaPlanner` — generates auxiliary lemmas for validation

### 4. Checking Intent

`IntentJudge` verifies the formalization captures the user's original idea using a 3-path adversarial approach:
1. Forward check — does the Lean statement imply the informal conjecture?
2. Backward check — does the informal conjecture imply the Lean statement?
3. `Informalizer` back-translation — Lean 4 → natural language, compared against the original

### 5. Searching Counterexamples

`CounterexampleSearcher` tries to disprove the conjecture before investing in proof search. If disproved, the pipeline transitions to refinement.

### 6. Proving

The `ProofPipeline` uses multiple strategies:
- `LemmaBreakdown` — decomposes the goal into sub-lemmas
- `LemmaLeanifier` — translates sub-lemmas to Lean 4
- `RecursiveProver` — parent-before-children strategy
- `IterativeProver` — iterative refinement with Lean REPL feedback
- `ProofSearchAgent` — direct proof discovery
- `FlattenFinalize` — assembles sub-proofs into a complete Lean proof
- `ClaimCheck` — verifies the final proof hasn't silently weakened hypotheses

### 7. Refining

`RefinementPipeline` with `ConjectureRefiner` produces refined variants of failed conjectures. `RefinementReporter` generates human-readable reports of the refinement journey. Refinement can loop back to formalizing, conjecturing, or exploring.

### 8. Complete / Failed

Terminal states. A complete session has at least one verified Lean proof. A failed session has exhausted all refinement and exploration attempts.

## State Transitions

```
EXPLORING ──────────► CONJECTURING ──────► FORMALIZING
    ▲                      │  ▲                 │
    │                      │  │                 ▼
    │                      ▼  │          CHECKING_INTENT
    │                   FAILED │               │
    │                          │               ▼
    │                          │    SEARCHING_COUNTEREXAMPLE
    │                          │               │
    │                          │               ▼
    │                          └───────── PROVING
    │                                      │   │
    │                                      │   ▼
    └──────────── REFINING ◄───────────────┘ COMPLETE
```

Any stage can transition to FAILED. REFINING can loop back to FORMALIZING, CONJECTURING, or EXPLORING.

## Agent Inventory

| Agent | File | Purpose |
|---|---|---|
| `BaseAgent` | `agents/base.py` | Abstract base with run(context) → AgentResult protocol |
| `LLMClient` | `agents/llm_client.py` | Anthropic API wrapper (direct + Vertex AI) |
| `ExplorationAgent` | `agents/explorer.py` | Domain identification and concept search |
| `ConjectureGenerator` | `agents/conjecturer.py` | Ranked conjecture production |
| `TypePlanner` | `agents/type_planner.py` | Determines which Lean types a conjecture needs |
| `TypeFormalizer` | `agents/type_formalizer.py` | Translates informal types to Lean 4 |
| `Auctioneer` | `agents/auctioneer.py` | Best-of-k selection for type formalizations |
| `TheoremFormalizer` | `agents/theorem_formalizer.py` | Produces Lean 4 theorem statement from types |
| `LemmaPlanner` | `agents/lemma_planner.py` | Generates auxiliary lemmas for validation |
| `IntentJudge` | `agents/intent_judge.py` | 3-path adversarial intent verification |
| `Informalizer` | `agents/informalizer.py` | Back-translation: Lean 4 → natural language |
| `CounterexampleSearcher` | `agents/counterexample_searcher.py` | Tries to disprove conjectures |
| `ProofSearchAgent` | `agents/proof_search.py` | Direct proof discovery |
| `IterativeProver` | `agents/prover.py` | Iterative proof refinement with REPL feedback |
| `RecursiveProver` | `agents/recursive_prover.py` | Parent-before-children proving strategy |
| `LemmaBreakdown` | `agents/lemma_breakdown.py` | Decomposes proof goals into sub-lemmas |
| `LemmaLeanifier` | `agents/lemma_leanifier.py` | Translates sub-lemmas to Lean 4 |
| `FlattenFinalize` | `agents/flatten_finalize.py` | Assembles sub-proofs into complete proof |
| `ClaimCheck` | `agents/claim_check.py` | Verifies proof hasn't weakened hypotheses |
| `ConjectureRefiner` | `agents/conjecture_refiner.py` | Produces refined variants of failed conjectures |
| `RefinementReporter` | `agents/refinement_reporter.py` | Human-readable refinement journey reports |
| `PromptTemplates` | `agents/prompt_templates.py` | Centralized prompt management |

## Data Flow

```
                         ┌─────────────────────────────────────────────┐
                         │              ResearchOrchestrator           │
                         │  (state machine, checkpoints, cost control) │
                         └──────────────────┬──────────────────────────┘
                                            │
          ┌────────────┬────────────┬───────┴────────┬──────────────┐
          ▼            ▼            ▼                ▼              ▼
    ExplorationAgent  ConjectureGen  FormalizationPipeline  ProofPipeline  RefinementPipeline
          │            │            │                │              │
          │            │     ┌──────┴──────┐    ┌───┴────┐    ConjectureRefiner
          │            │     │  TypePlan   │    │ Lemma  │    RefinementReporter
          │            │     │  TypeForm   │    │ Break  │
          │            │     │  Auctioneer │    │ Lean   │
          │            │     │  TheoremForm│    │ Recur  │
          │            │     │  LemmaPlan  │    │ Iter   │
          │            │     │  IntentJudge│    │ Flatten│
          │            │     │  Informalzr │    │ Claim  │
          │            │     └─────────────┘    └────────┘
          │            │
          ▼            ▼
     LeanSearch    LLMClient ◄──── All agents use LLMClient
                       │
              ┌────────┴────────┐
              │ Anthropic API   │
              │ (direct/Vertex) │
              └─────────────────┘
```

## Cost Control Architecture

### CostTracker

Tracks token usage (input, output, cache read, cache write) and computes dollar cost per API call. Every CLI command wraps operations in a cost tracker and checks against the user's `--budget` flag.

### CircuitBreaker

Monitors consecutive failures across the pipeline. After 5 consecutive failures, the circuit opens and halts the pipeline to prevent runaway spending. Resets on any successful operation.

### RetryPolicy

Built into `LLMClient` — exponential backoff with configurable max retries (default: 3) and backoff ceiling (default: 30s).

### Budget Enforcement

Every CLI command has a default budget (`explore`: $2, `formalize`: $3, `check`: $2, `prove`: $10). The orchestrator checks cost after each agent call and halts if exceeded.

## Session Memory Tiers

`ResearchSessionMemory` maintains three tiers:

| Tier | Contents | Lifecycle |
|---|---|---|
| **Hot** | Active conjectures, current exploration context, recent proof attempts | Always in working memory |
| **Warm** | Partially explored directions, failed-but-informative conjectures | Loaded on demand |
| **Cold** | Completed proofs, abandoned directions, historical refinement traces | Archived, queryable |

Memory persists to `.agentic_research/sessions/` and is loaded when resuming a session.

## Checkpointing Architecture

`CheckpointManager` saves pipeline state at each of the 8 stages:

1. After exploration completes
2. After conjecture generation
3. After formalization
4. After intent verification
5. After counterexample search
6. After proof attempt
7. After refinement
8. On completion

Each checkpoint captures the full `SessionState` (current stage, active conjecture index, transition history) plus accumulated token usage. Sessions can resume from the last checkpoint after interruption.
