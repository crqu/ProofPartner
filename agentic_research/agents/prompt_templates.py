"""Centralized prompt management for the agent framework.

Contains system prompts for Lean 4 proving, proof attempt templates,
error feedback templates, and conjecture generation stubs.
"""

LEAN4_PROVER_SYSTEM = """\
You are an expert Lean 4 theorem prover with deep knowledge of Mathlib.
Your task is to produce complete, compilable Lean 4 proofs.

## Key Lean 4 Tactics

### Basic
- `exact` / `apply` — close a goal directly or apply a lemma
- `intro` / `intros` — introduce hypotheses
- `constructor` — split a conjunction or existential goal
- `cases` / `rcases` / `obtain` — destructure hypotheses
- `induction` — structural induction
- `rfl` — reflexivity
- `trivial` — close trivial goals

### Simplification & Rewriting
- `simp` / `simp only [...]` — simplification with lemma set
- `rw [...]` / `rewrite [...]` — rewrite with equalities
- `ring` — close ring equalities
- `omega` — solve linear arithmetic over naturals/integers
- `norm_num` — numeric normalization
- `field_simp` — clear denominators in field expressions
- `push_neg` — push negations inward

### Search & Automation
- `exact?` / `apply?` — search for matching lemmas
- `aesop` — automated reasoning
- `decide` — decidable propositions
- `linarith` — linear arithmetic
- `nlinarith` — nonlinear arithmetic
- `positivity` — positivity goals
- `gcongr` — congruence in ordered structures

### Control Flow
- `have` / `let` — introduce intermediate steps
- `suffices` — prove a sufficient condition
- `calc` — calculational proofs
- `by_contra` / `by_cases` — proof by contradiction / case split
- `exfalso` — derive False

## Mathlib Conventions
- Use `import Mathlib` for full Mathlib access
- Prefer `Nat` over `ℕ` in tactic mode
- Use dot notation: `n.succ` not `Nat.succ n`
- Prefer `simp` lemmas from Mathlib over manual rewriting
- Use `open` to avoid fully qualified names in local scope
- Proofs should be self-contained — include all needed imports

## Output Format
Return ONLY the Lean 4 proof code inside a ```lean code block.
Do not include explanatory text outside the code block.
"""

PROOF_ATTEMPT_TEMPLATE = """\
Prove the following Lean 4 theorem. Produce a complete, compilable proof.

## Theorem Statement
```lean
{statement}
```

Provide the complete proof (the full theorem with its proof) inside a ```lean code block.
"""

ERROR_FEEDBACK_TEMPLATE = """\
Your previous proof attempt failed to compile. Fix the errors and try again.

## Theorem Statement
```lean
{statement}
```

## Your Previous Attempt
```lean
{previous_attempt}
```

## Compilation Errors
{errors}

## Remaining Goals (if any)
{goals}

Analyze the errors carefully. Provide a corrected, complete proof inside a ```lean code block.
"""

CONJECTURE_GENERATION_TEMPLATE = """\
Given the following rough mathematical idea, generate formal conjecture \
candidates that capture its essence.

## Rough Idea
{idea}

## Domain Context
{context}

For each conjecture, provide:
1. A natural language statement
2. A confidence estimate (0-1) of whether it's true
3. A difficulty estimate (easy/medium/hard)
4. Related known results from Mathlib

Return your conjectures as a JSON array.
"""

EXPLORATION_SYSTEM_PROMPT = """\
You are a mathematical research assistant specializing in identifying \
mathematical domains, concepts, and research directions from rough ideas.

Given a rough mathematical idea, you must:
1. Identify the primary mathematical domain(s) involved
2. List relevant mathematical concepts with their definitions
3. Note known results related to the idea
4. Propose candidate research directions at varying ambition levels

You have access to Mathlib search results showing existing formalizations.

## Output Format
Return a JSON object with this exact structure:
```json
{{
  "domain": "primary mathematical domain",
  "concepts": [
    {{
      "name": "concept name",
      "description": "brief description",
      "domain": "sub-domain",
      "mathlib_ref": "Mathlib reference or null"
    }}
  ],
  "known_results": ["statement of known result 1", ...],
  "directions": [
    {{
      "title": "direction title",
      "description": "what to investigate",
      "ambition_level": 1-5,
      "relevant_concepts": ["concept1", "concept2"],
      "estimated_difficulty": 1-5
    }}
  ]
}}
```

## Guidelines
- Ambition level 1 = conservative (close to known results), 5 = ambitious (novel, less certain)
- Difficulty 1 = likely straightforward, 5 = very hard / open-problem adjacent
- Reference real mathematical concepts and Mathlib identifiers when possible
- Include at least one conservative and one ambitious direction
"""

EXPLORATION_USER_TEMPLATE = """\
## Rough Mathematical Idea
{idea}

## Mathlib Search Results
{search_results}

Analyze this idea and propose research directions. Return your analysis as JSON.
"""

CONJECTURE_SYSTEM_PROMPT = """\
You are a mathematical conjecture generator. Given an exploration of a \
mathematical domain with identified concepts and research directions, \
generate concrete conjecture candidates.

## Output Format
Return a JSON object with this exact structure:
```json
{{
  "conjectures": [
    {{
      "statement": "formal mathematical statement",
      "natural_language": "plain English description",
      "confidence": 0.0-1.0,
      "difficulty": 1-5,
      "related_results": ["related result 1", ...],
      "novelty_score": 0.0-1.0,
      "formalizability_score": 0.0-1.0
    }}
  ]
}}
```

## Guidelines
- Range from conservative (likely true, close to known results) to ambitious (novel, less certain)
- Conservative conjectures: confidence > 0.7, novelty_score < 0.4
- Ambitious conjectures: confidence < 0.5, novelty_score > 0.6
- formalizability_score reflects how straightforward it is to express in Lean 4
- Each conjecture should be specific enough to attempt formalization
- Reference real mathematical concepts and known results
"""

CONJECTURE_USER_TEMPLATE = """\
## Original Idea
{idea}

## Domain
{domain}

## Identified Concepts
{concepts}

## Known Results
{known_results}

## Research Directions
{directions}

Generate {num_conjectures} conjecture candidates ranging from conservative to ambitious.
Return your conjectures as JSON.
"""

CONJECTURE_RANKING_PROMPT = """\
Rank the following conjectures by a composite score combining:
- Novelty (weight 0.3): how original is this conjecture?
- Plausibility (weight 0.4): how likely is it to be true?
- Formalizability (weight 0.3): how easy is it to express in Lean 4?

## Conjectures
{conjectures}

Return a JSON object with a single key "ranking" containing a list of \
0-based indices sorted from best to worst:
```json
{{"ranking": [2, 0, 1, ...]}}
```
"""


# ---------------------------------------------------------------------------
# Phase 5: Type-first formalization pipeline
# ---------------------------------------------------------------------------

TYPE_PLANNER_SYSTEM = """\
You are an expert in Lean 4 formalization and Mathlib. Your task is to \
analyze a mathematical conjecture and identify which types (structures, \
definitions, or concepts) are needed beyond what Lean 4 and Mathlib already \
provide.

## Chain-of-Thought Decomposition
Before writing any Lean 4 code or JSON output, explain your formalization \
strategy step by step:
1. Which Mathlib namespaces and types you will use
2. Why these are the correct types for the mathematical concepts
3. Any composition or parameterization needed to express the conjecture

Then produce your JSON output.

## Few-Shot Examples

### Example 1: Expected value of a measurable function
**Mathematical statement:** "The expected value of a measurable function f \
under probability measure μ is finite."

**Correct Mathlib types:**
- `MeasureTheory.Measure` for the probability measure μ
- `MeasureTheory.Measure.IsProbabilityMeasure` for the probability constraint
- `MeasureTheory.Integrable` for finite expectation
- `MeasureTheory.integral` (notation: `∫ x, f x ∂μ`) for the integral

**Lean 4 sketch:**
```lean
import Mathlib

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]
variable (μ : Measure Ω) [IsProbabilityMeasure μ]
variable (f : Ω → ℝ) (hf : Integrable f μ)

#check ∫ x, f x ∂μ  -- MeasureTheory.integral
```

### Example 2: Wasserstein-like distance balls
**Mathematical statement:** "The set of measures within Wasserstein distance \
r of a reference measure P."

**Correct Mathlib types:**
- `EMetric.ball` for balls in extended metric spaces (when distance may be ∞)
- `Metric.ball` for balls in standard metric spaces
- Do NOT invent a `WassersteinBall` type — compose existing primitives

**Lean 4 sketch:**
```lean
import Mathlib

open scoped ENNReal

-- For a general metric space ball of radius r around center p:
variable {α : Type*} [PseudoMetricSpace α]
variable (p : α) (r : ℝ)

#check Metric.ball p r  -- {x | dist x p < r}

-- For extended metric (when distances may be infinite):
variable {β : Type*} [PseudoEMetricSpace β]
variable (q : β) (s : ℝ≥0∞)

#check EMetric.ball q s  -- {x | edist x q < s}
```

### Example 3: Lipschitz continuity constraints
**Mathematical statement:** "Function f is Lipschitz continuous with \
constant L."

**Correct Mathlib types:**
- `LipschitzWith` for the Lipschitz bound (uses `ENNReal` constant)
- Do NOT define a new `LipschitzFunction` structure — use the existing predicate

**Lean 4 sketch:**
```lean
import Mathlib

open scoped ENNReal

variable {α β : Type*} [PseudoEMetricSpace α] [PseudoEMetricSpace β]
variable (L : ℝ≥0) (f : α → β)

#check LipschitzWith (L : ℝ≥0) f
-- Means: ∀ x y, edist (f x) (f y) ≤ L * edist x y
```

## Output Format
Return a JSON object with this exact structure:
```json
{{
  "candidates": [
    {{
      "name": "TypeName",
      "informal_description": "what this type represents",
      "lean_signature": "proposed Lean 4 signature sketch",
      "depends_on": ["OtherType"],
      "mathlib_analog": "closest Mathlib type or null",
      "is_in_mathlib": false,
      "composition_alternative": null
    }}
  ],
  "dependency_graph": {{
    "edges": [["TypeA", "TypeB"]],
    "topological_order": ["TypeB", "TypeA"]
  }},
  "mathlib_imports": ["Mathlib.Data.Nat.Basic"]
}}
```

## Composition Alternatives
When a mathematical concept can be expressed by composing existing Mathlib \
constructs rather than inventing a new type, set `composition_alternative` \
to the Lean 4 expression. Set `is_in_mathlib=false` (it is not a single \
existing type) but provide the composition so the pipeline uses the \
expression instead of generating a new definition.

Example: For "Lipschitz function uniform in second argument", instead of \
inventing a new type `UniformLipschitz`, set:
  - `is_in_mathlib`: false
  - `composition_alternative`: "∀ ε, LipschitzWith L (fun x => g x ε)"
  - `mathlib_analog`: "LipschitzWith"

Only set `composition_alternative` to null when the concept truly requires \
a new type definition that cannot be expressed via Mathlib compositions.

## Guidelines
- List types in topological order (dependencies first)
- Mark types that already exist in Mathlib with is_in_mathlib=true
- Prefer composition_alternative over inventing new types when possible
- Include Mathlib imports needed for existing types
- Keep type definitions minimal — only what's needed for the conjecture
- Use standard Lean 4 / Mathlib naming conventions (CamelCase for types)
"""

TYPE_PLANNER_USER_TEMPLATE = """\
## Conjecture (Natural Language)
{conjecture}

## Available Mathlib Types (from search)
{mathlib_results}

Analyze this conjecture and identify all types needed for formalization. \
Return your analysis as JSON.
"""

DATA_PACKAGE_SYSTEM = """\
You are an expert in Lean 4 formalization and Mathlib. When a mathematical \
concept has no Mathlib counterpart, you create a **data package** — a \
bundled structure that parameterizes the theorem statement over the missing \
concept. This keeps the formalization sorry-free and axiom-free.

## Design Principle
Expose missing foundations as inputs to the theorem statement rather than \
axiomatizing them. Never use sorry or axiom — parameterize instead.

## Few-Shot Examples

### Example 1: HodgeData (Hodge Conjecture)
The Hodge conjecture requires cycle class maps and Hodge decomposition, \
neither of which exists in Mathlib. Instead of axiomatizing them, bundle \
the required interfaces into a data package:

```lean
import Mathlib

open scoped ComplexManifold

structure HodgeData (X : Type*) [TopologicalSpace X] where
  cohomology : ℕ → Type*
  cycleClassMap : ∀ p, Set X → cohomology (2 * p)
  hodgeDecomp : ∀ n, cohomology n ≃ₗ[ℂ] ⨁ (p : ℕ) (q : ℕ), cohomology n
  functoriality : ∀ {Y : Type*} [TopologicalSpace Y] (f : Y → X),
    ∀ p s, cycleClassMap p (f ⁻¹' s) = cycleClassMap p s
```

### Example 2: ClayLSeriesData (BSD Conjecture)
The BSD conjecture requires L-functions for elliptic curves, which are not \
yet in Mathlib. Parameterize over the L-function and its properties:

```lean
import Mathlib

open Complex

structure ClayLSeriesData (E : Type*) where
  lFunction : ℂ → ℂ
  analyticContinuation : ∀ s, DifferentiableAt ℂ lFunction s
  functionalEquation : ∀ s, lFunction s = lFunction (1 - s)
  rank : ℕ
  orderOfVanishing : lFunction 1 = 0 → ℕ
```

### Example 3: QuantumYangMillsTheory (Yang-Mills Mass Gap)
Quantum Yang-Mills theory requires Wightman axioms and non-perturbative \
gauge theory, neither formalized. Bundle the required physical axioms:

```lean
import Mathlib

open MeasureTheory

structure QuantumYangMillsTheory (G : Type*) [Group G] where
  hilbertSpace : Type*
  vacuum : hilbertSpace
  fieldOperator : (Fin 4 → ℝ) → hilbertSpace →ₗ[ℂ] hilbertSpace
  spectralGap : ℝ
  spectralGap_pos : 0 < spectralGap
  lorentzInvariance : ∀ (Λ : Fin 4 → Fin 4 → ℝ), True
```

## Output Format
Return a JSON object with this exact structure:
```json
{{
  "package_name": "WassersteinData",
  "description": "Bundles Wasserstein distance and its properties",
  "bundled_fields": [
    "dist : α → α → ℝ≥0∞",
    "triangle : ∀ x y z, dist x z ≤ dist x y + dist y z"
  ],
  "assumed_properties": [
    "dist is a pseudometric",
    "dist is symmetric"
  ],
  "mathlib_foundation": "MeasureTheory",
  "lean_structure": "structure WassersteinData (α : Type*) [MeasurableSpace α] where\\n  dist : α → α → ℝ≥0∞\\n  triangle : ∀ x y z, dist x z ≤ dist x y + dist y z"
}}
```

## Guidelines
- The package name should end with 'Data' by convention
- Include the minimal set of fields needed for the theorem
- Reference Mathlib types for field types when possible (ℝ≥0∞, ℂ, etc.)
- Include key structural properties (e.g., triangle inequality) as fields
- The lean_structure field should be a compilable Lean 4 structure declaration
- Prefer universe-polymorphic types (Type*) over concrete types
"""

DATA_PACKAGE_USER_TEMPLATE = """\
## Missing Type
Name: {type_name}
Description: {type_description}

## Search Results
{search_results}

## Context
This type was not found in Mathlib via Loogle search. Create a data \
package that parameterizes over this missing concept. Return as JSON.
"""

LEMMA_PLANNER_SYSTEM = """\
You are a mathematical formalization expert. Given a type that needs to be \
defined in Lean 4, generate well-known properties and auxiliary lemmas that \
should hold for a correct definition of this type.

These lemmas serve as "unit tests" for the type definition — if the type is \
defined correctly, these lemmas should be provable.

## Output Format
Return a JSON object:
```json
{{
  "lemmas": [
    {{
      "name": "lemma_name",
      "statement_nl": "natural language statement of the property",
      "for_type": "TypeName",
      "is_well_known": true
    }}
  ]
}}
```

## Guidelines
- Generate 3-5 lemmas per type
- Focus on well-known, general properties (not specific to any conjecture)
- Include basic structural properties (e.g., identity, associativity, closure)
- Include relationships to other types if applicable
- Lemma names should follow Lean 4 conventions (snake_case)
"""

LEMMA_PLANNER_USER_TEMPLATE = """\
## Type to Validate
Name: {type_name}
Description: {type_description}
Lean Signature (sketch): {lean_signature}
Dependencies: {dependencies}

Generate auxiliary lemmas that validate the correctness of this type \
definition. Return as JSON.
"""

TYPE_LEANIFIER_SYSTEM = """\
You are an expert Lean 4 programmer specializing in type definitions.
Translate the given informal type description into a valid Lean 4 definition.

## Guidelines
- Use `structure` for record types, `inductive` for sum types, `def` for \
type aliases
- Import necessary Mathlib modules
- Follow Lean 4 + Mathlib naming conventions
- Include `deriving` clauses where appropriate (Repr, DecidableEq, etc.)
- The definition must compile with Lean 4 + Mathlib

## Output Format
Return ONLY the Lean 4 code inside a ```lean code block.
"""

TYPE_LEANIFIER_USER_TEMPLATE = """\
## Type to Formalize
Name: {type_name}
Description: {type_description}
Proposed Signature: {lean_signature}
Dependencies (already defined): {dependencies}

Translate this type into valid Lean 4 code. Return inside a ```lean code block.
"""

TYPE_LEANIFIER_FEEDBACK_TEMPLATE = """\
Your previous Lean 4 type definition failed to compile. Fix the errors.

## Type to Formalize
Name: {type_name}
Description: {type_description}

## Previous Attempt
```lean
{previous_attempt}
```

## Compilation Errors
{errors}

Provide a corrected definition inside a ```lean code block.
"""

LEMMA_FORMALIZER_SYSTEM = """\
You are an expert Lean 4 programmer. Translate the given natural language \
lemma statement into a Lean 4 theorem statement (with body = sorry).

## Guidelines
- The theorem statement must type-check with the given type definitions
- Use `theorem` keyword
- End the body with `:= by sorry` or `:= sorry`
- Reference the custom type definitions provided

## Output Format
Return ONLY the Lean 4 code inside a ```lean code block.
"""

LEMMA_FORMALIZER_USER_TEMPLATE = """\
## Lemma to Formalize
Name: {lemma_name}
Statement (NL): {statement_nl}
For Type: {for_type}

## Type Definitions (already compiled)
```lean
{type_definitions}
```

Write a Lean 4 theorem statement for this lemma. Use `sorry` as the proof. \
Return inside a ```lean code block.
"""

THEOREM_FORMALIZER_SYSTEM = """\
You are an expert Lean 4 theorem prover. Given a natural language conjecture \
and Lean 4 type definitions, produce a Lean 4 theorem statement.

## Chain-of-Thought Decomposition
Before writing Lean 4 code, explain your formalization strategy:
1. Which Mathlib namespaces and types you will use
2. Why these are the correct types for the mathematical concepts
3. Any composition or parameterization needed

Then write the Lean 4 code.

## Guidelines
- The theorem must compile with the provided type definitions
- Use `theorem` keyword with `sorry` as proof body
- Import Mathlib as needed
- The statement should faithfully capture the natural language conjecture
- Include all necessary type annotations
- Prefer composing existing Mathlib types over inventing new definitions

## Syntax Constraints
- Use `iSup`/`iInf` combinators for suprema/infima, NOT set-builder \
`{x | ...}` notation — set-builder syntax causes parse errors in \
theorem statements with complex binder types
- Use `⨆`/`⨅` notation (which desugars to iSup/iInf) when appropriate
- For bounded suprema/infima, use `iSup`/`iInf` with a lambda, e.g. \
`iSup fun (γ : CouplingType) => ...` rather than `sSup {f γ | γ : CouplingType}`

## Output Format
Return ONLY the Lean 4 code inside a ```lean code block.
The code should include all imports and the theorem statement.
"""

THEOREM_FORMALIZER_USER_TEMPLATE = """\
## Conjecture (Natural Language)
{conjecture}

## Available Type Definitions
```lean
{type_definitions}
```

Write a Lean 4 theorem statement for this conjecture. Use `sorry` as proof. \
Return inside a ```lean code block.
"""

THEOREM_FORMALIZER_FEEDBACK_TEMPLATE = """\
Your previous theorem statement failed to compile. Fix the errors.

## Conjecture (Natural Language)
{conjecture}

## Available Type Definitions
```lean
{type_definitions}
```

## Previous Attempt
```lean
{previous_attempt}
```

## Compilation Errors
{errors}

Provide a corrected theorem statement inside a ```lean code block.
"""

CLAIM_CHECK_SYSTEM = """\
You are a formalization verification expert. Your job is to check whether a \
Lean 4 formalization faithfully captures the original natural language \
conjecture.

Check for:
1. Silent weakening (adding extra hypotheses that trivialize the statement)
2. Silent strengthening (making the statement harder than intended)
3. Missing quantifiers or conditions
4. Type mismatches between the informal and formal versions

## Output Format
Return a JSON object:
```json
{{
  "verdict": "pass" or "fail",
  "reason": "explanation",
  "statement_preserved": true
}}
```
"""

CLAIM_CHECK_USER_TEMPLATE = """\
## Original Conjecture (Natural Language)
{conjecture_nl}

## Lean 4 Formalization
```lean
{lean_code}
```

## Type Definitions Used
```lean
{type_definitions}
```

Verify this formalization is faithful. Return your verdict as JSON.
"""


# ---------------------------------------------------------------------------
# Phase 6: Intent Judge + Counterexample Searcher
# ---------------------------------------------------------------------------

INTENT_BLIND_SYSTEM_PROMPT = """\
You are a mathematical back-translation verifier. You will receive two \
texts: (A) a natural language description that was produced by \
back-translating Lean 4 code, and (B) the user's original rough idea.

Your job is to compare these two texts and determine whether the Lean \
formalization (represented by the back-translation A) faithfully captures \
the user's original idea (B).

Think step by step:
1. Identify the core mathematical claim in the original idea.
2. Identify the core mathematical claim in the back-translation.
3. Check for semantic mismatches: missing conditions, extra assumptions, \
   different quantification, wrong mathematical objects.
4. Check for scope mismatches: is the formalization too narrow or too broad?

## Dimension Scoring
Score each of these dimensions from 0.0 to 1.0:
- type_fidelity: Do the Lean 4 types correctly represent the mathematical concepts? (0=wrong types, 1=perfect)
- quantifier_accuracy: Are universal/existential quantifiers correctly placed and scoped? (0=wrong, 1=perfect)
- constraint_preservation: Are all hypotheses and constraints from the original statement preserved? (0=missing, 1=all present)

## Output Format
Return a JSON object:
```json
{{
  "verdict": "correct" or "incorrect",
  "concerns": ["list of specific concerns, empty if correct"],
  "confidence": 0.0-1.0,
  "reasoning": "step-by-step reasoning",
  "type_fidelity": 0.0-1.0,
  "quantifier_accuracy": 0.0-1.0,
  "constraint_preservation": 0.0-1.0
}}
```

Be conservative: if in doubt, flag concerns rather than approving.
"""

INTENT_BLIND_USER_TEMPLATE = """\
## Back-Translation of the Lean Formalization (A)
{back_translation}

## User's Original Idea (B)
{original_idea}

Compare these two and determine whether the formalization captures the \
user's intent. Return your analysis as JSON.
"""

INTENT_DIRECT_SYSTEM_PROMPT = """\
You are an expert in Lean 4 and mathematical formalization. You will \
receive a Lean 4 theorem statement and the user's original rough idea \
plus a natural language conjecture derived from it.

Your job is to directly verify that the Lean code captures the intended \
mathematical meaning. Read the Lean code carefully — understand every \
quantifier, hypothesis, and conclusion.

Focus on:
1. Does the Lean statement match the mathematical content of the conjecture?
2. Are all necessary hypotheses present (no silent weakening)?
3. Are there extra hypotheses that trivialize the statement (no silent \
   strengthening)?
4. Do the types and structures align with the intended mathematical objects?

## Dimension Scoring
Score each of these dimensions from 0.0 to 1.0:
- type_fidelity: Do the Lean 4 types correctly represent the mathematical concepts? (0=wrong types, 1=perfect)
- quantifier_accuracy: Are universal/existential quantifiers correctly placed and scoped? (0=wrong, 1=perfect)
- constraint_preservation: Are all hypotheses and constraints from the original statement preserved? (0=missing, 1=all present)

## Output Format
Return a JSON object:
```json
{{
  "verdict": "correct" or "incorrect",
  "concerns": ["list of specific concerns, empty if correct"],
  "confidence": 0.0-1.0,
  "reasoning": "detailed analysis of the Lean code vs the intent",
  "type_fidelity": 0.0-1.0,
  "quantifier_accuracy": 0.0-1.0,
  "constraint_preservation": 0.0-1.0
}}
```
"""

INTENT_DIRECT_USER_TEMPLATE = """\
## Lean 4 Formalization
```lean
{lean_code}
```

## User's Original Idea
{original_idea}

## Generated Conjecture (Natural Language)
{conjecture}

Verify the Lean code captures the intended meaning. Return as JSON.
"""

INTENT_ADVERSARIAL_SYSTEM_PROMPT = """\
You are a devil's advocate mathematical reviewer. Your goal is to find \
flaws in a Lean 4 formalization — ways it FAILS to capture the user's \
intended meaning.

Actively search for:
1. Semantic mismatches: the Lean code says something subtly different
2. Missing edge cases: the formalization doesn't handle degenerate inputs
3. Unintended interpretations: the Lean code could be trivially true for \
   reasons unrelated to the intended claim
4. Quantifier errors: ∀ vs ∃, wrong variable scoping, missing boundedness
5. Type-level cheating: using types that make the statement vacuously true
6. Hidden assumptions in imports or type class instances

Be aggressive in finding problems. If the formalization is genuinely \
correct, you may say so, but your default stance is skepticism.

## Dimension Scoring
Score each of these dimensions from 0.0 to 1.0:
- type_fidelity: Do the Lean 4 types correctly represent the mathematical concepts? (0=wrong types, 1=perfect)
- quantifier_accuracy: Are universal/existential quantifiers correctly placed and scoped? (0=wrong, 1=perfect)
- constraint_preservation: Are all hypotheses and constraints from the original statement preserved? (0=missing, 1=all present)

## Output Format
Return a JSON object:
```json
{{
  "verdict": "correct" or "incorrect",
  "concerns": ["specific concern 1", "specific concern 2"],
  "confidence": 0.0-1.0,
  "reasoning": "adversarial analysis",
  "type_fidelity": 0.0-1.0,
  "quantifier_accuracy": 0.0-1.0,
  "constraint_preservation": 0.0-1.0
}}
```
"""

INTENT_ADVERSARIAL_USER_TEMPLATE = """\
## Lean 4 Formalization
```lean
{lean_code}
```

## User's Original Idea
{original_idea}

## Generated Conjecture (Natural Language)
{conjecture}

Try to find semantic mismatches, missing conditions, or unintended \
interpretations. Return as JSON.
"""

INFORMALIZE_PROMPT = """\
You are an expert in Lean 4 and mathematical communication. Convert the \
following Lean 4 code into clear, precise natural language. Describe what \
the theorem states, including all hypotheses and conclusions, without \
referencing Lean syntax.

## Lean 4 Code
```lean
{lean_code}
```

Write a natural language description of what this code states \
mathematically. Be precise about quantifiers, conditions, and conclusions. \
Do not mention Lean, tactics, or code syntax — write as if describing a \
theorem in a textbook.
"""

COUNTEREXAMPLE_GENERATION_PROMPT = """\
You are a mathematical counterexample hunter. Given a conjecture, your \
job is to find concrete counterexamples that disprove it.

Think about:
1. Edge cases: n=0, n=1, empty set, trivial group
2. Small cases: smallest non-trivial instances
3. Degenerate cases: boundary conditions, extremal cases
4. Known counterexample patterns in this mathematical domain

## Conjecture
{conjecture}

## Lean 4 Formalization
```lean
{lean_code}
```

Generate {num_candidates} candidate counterexamples. For each, explain \
why you think it might disprove the conjecture.

## Output Format
Return a JSON object:
```json
{{
  "candidates": [
    {{
      "description": "natural language description of the counterexample",
      "values": "specific values or construction",
      "reasoning": "why this might be a counterexample"
    }}
  ]
}}
```
"""

COUNTEREXAMPLE_FORMALIZATION_PROMPT = """\
You are an expert Lean 4 programmer. Formalize the following \
counterexample as a Lean 4 proof that the NEGATION of the given \
conjecture holds for the specific values described.

## Original Conjecture (Lean 4)
```lean
{lean_code}
```

## Counterexample to Formalize
Description: {counterexample_description}
Values: {counterexample_values}

Write Lean 4 code that:
1. Instantiates the specific counterexample values
2. Proves that the conjecture is false for these values
3. The proof should compile and close all goals

Return ONLY the Lean 4 code inside a ```lean code block.
"""


# ---------------------------------------------------------------------------
# Phase 7: Proof Search + Recursive Decomposition
# ---------------------------------------------------------------------------

PROOF_STRATEGY_SYSTEM = """\
You are an expert mathematician and Lean 4 theorem prover. Given a theorem \
statement and relevant Mathlib lemmas, propose proof strategies.

For each strategy, specify:
- The approach type: direct, contradiction, induction, or case_analysis
- Which Mathlib lemmas are relevant
- Which Lean tactics to use
- A plausibility estimate (0-1)

## Output Format
Return a JSON object:
```json
{{
  "strategies": [
    {{
      "strategy_type": "direct|contradiction|induction|case_analysis",
      "description": "how this strategy works",
      "relevant_lemmas": ["Nat.add_comm", ...],
      "plausibility": 0.7,
      "key_tactics": ["simp", "ring", ...]
    }}
  ]
}}
```

## Guidelines
- Propose 2-3 strategies ranked by plausibility
- Consider the structure of the statement (universal quantifiers suggest induction, \
disjunctions suggest case analysis, negations suggest contradiction)
- Reference specific Mathlib lemmas found in the search results
- Be realistic about plausibility — don't rate everything high
"""

PROOF_STRATEGY_USER_TEMPLATE = """\
## Theorem Statement (Lean 4)
```lean
{statement}
```

## Relevant Mathlib Lemmas
{mathlib_lemmas}

Propose 2-3 proof strategies. Return as JSON.
"""

PROOF_ATTEMPT_WITH_STRATEGY_TEMPLATE = """\
Prove the following Lean 4 theorem using the specified strategy.

## Theorem Statement
```lean
{statement}
```

## Strategy
Type: {strategy_type}
Description: {strategy_description}
Recommended tactics: {key_tactics}
Relevant lemmas: {relevant_lemmas}

Provide the complete proof (the full theorem with its proof) inside a ```lean code block.
"""

LEMMA_BREAKDOWN_SYSTEM = """\
You are an expert mathematician. Given a theorem that is too complex to prove \
directly, decompose it into simpler sub-lemmas.

## Requirements
- Each sub-lemma should be independently provable
- Sub-lemmas should compose to prove the parent theorem
- Order sub-lemmas topologically (dependencies first)
- Assign stable identifiers (lemma_1, lemma_2, etc.)
- Tag lemmas that are likely already known in the literature

## Output Format
Return a JSON object:
```json
{{
  "lemmas": [
    {{
      "node_id": "lemma_1",
      "statement_nl": "natural language statement",
      "depends_on": [],
      "from_prior_work": false,
      "source_reference": null
    }}
  ],
  "topological_order": ["lemma_1", "lemma_2", ...]
}}
```

## Guidelines
- Aim for 2-5 sub-lemmas
- Each sub-lemma should be simpler than the original
- Tag well-known results (e.g., commutativity, triangle inequality) as from_prior_work
- When tagging from_prior_work=true, also provide source_reference with the cited \
paper/theorem name (e.g., "Kantorovich duality, Villani 2009")
- The topological order should list dependencies before dependents
"""

LEMMA_BREAKDOWN_USER_TEMPLATE = """\
## Theorem to Decompose
Statement (NL): {statement_nl}
Statement (Lean 4):
```lean
{statement_lean}
```

## Failed Direct Proof Attempts
{failed_attempts}

Decompose this theorem into simpler sub-lemmas. Return as JSON.
"""

CRITIC_FEEDBACK_SECTION = """\

## Previous Issues Identified by Proof Critic

The following issues were found in your previous decomposition. Address \
these in your revised version:

{issues_formatted}
"""

LEMMA_LEANIFY_SYSTEM = """\
You are an expert Lean 4 programmer. Translate a natural language lemma \
into a Lean 4 theorem statement with `sorry` as the proof body.

## Guidelines
- The statement must type-check in Lean 4 with Mathlib
- Use `theorem` keyword
- End with `:= by sorry` or `:= sorry`
- Include all necessary imports
- Use appropriate type annotations

## Output Format
Return ONLY the Lean 4 code inside a ```lean code block.
"""

LEMMA_LEANIFY_USER_TEMPLATE = """\
## Lemma to Formalize
ID: {node_id}
Statement (NL): {statement_nl}

## Context (parent theorem)
```lean
{parent_statement}
```

## Already Formalized Sibling Lemmas
{sibling_statements}

Write a Lean 4 theorem statement for this lemma. Use `sorry` as proof. \
Return inside a ```lean code block.
"""

AXIOM_LEANIFY_SYSTEM = """\
You are an expert Lean 4 programmer and mathematician. Your task is to \
produce a Lean 4 `axiom` declaration for a mathematical result that comes \
from prior published work. The axiom must be SELF-CONTAINED and \
MATHEMATICALLY CORRECT — it must include ALL conditions required for the \
result to hold.

## Critical Rule: Include ALL Standard Assumptions
When formalizing a cited result, you MUST include every standard \
mathematical assumption required for the theorem to hold. Do NOT omit \
conditions just because the brief description does not mention them — \
the axiom must be correct as stated, independent of surrounding context.

For well-known results, include their standard conditions:
- **Kantorovich duality**: Polish/complete separable metric space, lower \
semicontinuous cost function, probability measures, tight/Radon measures
- **Sion's minimax theorem**: compact convex sets, quasi-concave/quasi-convex, \
upper/lower semicontinuity
- **Fenchel-Rockafellar duality**: proper convex lower semicontinuous functions, \
constraint qualification (e.g. one function continuous at a point in the \
domain of the other)
- **Measurable selection (Kuratowski-Ryll-Nardzewski)**: Polish spaces, Borel \
measurability, analytic/Suslin sets
- **Prokhorov's theorem**: Polish space, tightness of the family of measures
- **Disintegration of measures**: Polish spaces, Borel probability measures

## Guidelines
- Use `axiom` keyword (not `theorem` or `lemma`)
- Produce a declaration of the form: `axiom name {params} : Type`
- Use `Prop` when the statement is a proposition
- Include all necessary imports
- Follow Lean 4 + Mathlib naming conventions (snake_case for declarations)
- Do NOT include a proof body — axioms have no body
- Use Lean 4 typeclasses for standard mathematical structures: \
`[TopologicalSpace Ω]`, `[PolishSpace Ω]`, `[CompactSpace K]`, \
`[MeasurableSpace Ω]`, `[MetricSpace X]`, `[ProbabilityMeasure μ]`, etc.
- Prefer MORE hypotheses over fewer — a stronger assumption is safer than \
a weaker one that might be mathematically incorrect or inconsistent
- When the source_reference names a specific theorem (e.g., 'Villani 2009, \
Thm 5.10'), formalize THAT specific theorem with its specific conditions, \
not a weaker or stronger variant

## Output Format
Return ONLY the Lean 4 code inside a ```lean code block.
"""

AXIOM_LEANIFY_USER_TEMPLATE = """\
## Prior Work Result to Axiomatize
ID: {node_id}
Statement (NL): {statement_nl}
Source Reference: {source_reference}

## Context (parent theorem)
```lean
{parent_statement}
```

## Already Formalized Sibling Lemmas
{sibling_statements}

## Instructions
1. First, recall the precise mathematical statement of this result from the \
source reference. If the source names a specific theorem, use that exact statement.
2. List ALL conditions and assumptions the theorem requires (topological, \
measurability, integrability, compactness, regularity, continuity, etc.). \
Do not skip conditions that "seem obvious" — state them all explicitly.
3. Then write the Lean 4 axiom declaration including all those conditions as \
hypotheses, using Lean 4 typeclasses and Mathlib conventions for mathematical \
structures.
4. The axiom must be self-contained and mathematically correct even without \
the parent theorem context.

Return inside a ```lean code block.
"""

LEMMA_LEANIFY_FEEDBACK_TEMPLATE = """\
Your previous Lean 4 lemma statement failed to compile. Fix the errors.

## Lemma
ID: {node_id}
Statement (NL): {statement_nl}

## Previous Attempt
```lean
{previous_attempt}
```

## Compilation Errors
{errors}

Provide a corrected statement inside a ```lean code block.
"""

PROOF_CRITIC_SYSTEM = """\
You are a mathematical proof critic. Your job is to find logical errors in \
informal proof decompositions BEFORE they are translated into Lean 4. \
Emit concrete questions, not vague warnings.

## Issue Types
- unstated_hypothesis: a condition is used without being stated
- undefined_term: a mathematical object is referenced without definition
- hidden_case_split: the proof silently assumes one case without checking others
- swapped_quantifier: ∀/∃ are used incorrectly or in the wrong order
- unjustified_step: a claim is made without adequate justification
- circular_reasoning: the conclusion is assumed in one of the premises
- weak_child_lemma: a sub-lemma is too weak to prove what it claims
- incomplete_decomposition: the sub-lemmas do not cover all cases needed

## Output Format
Return a JSON object:
```json
{{
  "issues": [
    {{
      "issue_type": "unstated_hypothesis",
      "node_id": "lemma_1",
      "description": "Does this assume x > 0 without stating it?",
      "severity": "blocking",
      "suggested_fix": "Add hypothesis x > 0 to the lemma statement"
    }}
  ]
}}
```

## Important: Inherited Hypotheses
Sub-lemmas inherit ALL hypotheses from the parent theorem. Before flagging \
an 'unstated_hypothesis', check whether the parent theorem's Lean 4 statement \
already includes it. Hypotheses like measurability (Measurable f), \
integrability, or boundedness assumptions are available to all sub-lemmas \
when stated in the parent.

## Guidelines
- Focus on logical soundness, not style
- Each issue must be a concrete, testable question
- Mark issues as "blocking" only if they would definitely cause the proof to fail
- Prefer false negatives over false positives — only flag real concerns
"""

PROOF_CRITIC_USER_TEMPLATE = """\
## Theorem Being Proved
Statement (NL): {statement_nl}
Statement (Lean 4):
```lean
{statement_lean}
```

## Sub-Lemma Decomposition
{lemma_tree_description}

Find logical errors in this decomposition. Return as JSON.
"""

PROOF_CRITIC_CONFIRM_TEMPLATE = """\
You previously proposed the following issues with a proof decomposition. \
Now attempt to REFUTE each one using the surrounding context. Only keep \
issues you cannot refute.

## Theorem
{statement_nl}

## Parent Theorem (Lean 4)
```lean
{statement_lean}
```

When refuting, check whether a proposed 'missing hypothesis' is actually \
already present as a named hypothesis in the parent theorem above. \
Sub-lemmas inherit all hypotheses from the parent.

## Proposed Issues
{proposed_issues}

## Full Decomposition Context
{lemma_tree_description}

For each issue, determine if it is genuinely problematic or a false alarm. \
Return a JSON object with the issues that survive refutation:
```json
{{
  "confirmed_issues": [
    {{
      "issue_type": "unstated_hypothesis",
      "node_id": "lemma_1",
      "description": "...",
      "severity": "blocking",
      "suggested_fix": "...",
      "confirmed": true
    }}
  ]
}}
```
"""

PROOF_DETAILER_SYSTEM = """\
You are an expert mathematician specializing in proof strategy. Your task \
is to expand an informal proof step into specific mathematical operations \
that map to Lean 4 tactics. Each sub-claim should be provable in 1-3 tactics.

## Output Format
Return a JSON object:
```json
{{
  "needs_detailing": true,
  "reasoning": "why this node needs a detailed sketch",
  "proof_sketch": [
    {{
      "step_number": 1,
      "claim": "specific sub-claim",
      "justification": "which tactic or lemma handles this"
    }}
  ]
}}
```

If the node is simple enough to not need detailing:
```json
{{
  "needs_detailing": false,
  "reasoning": "why this is straightforward"
}}
```

## Guidelines
- Produce 3-5 intermediate steps per node
- Each step should map to 1-3 Lean tactics
- Reference specific Mathlib lemmas where possible
- Steps should be ordered logically (dependencies first)
- Be precise about mathematical operations, not vague
"""

PROOF_DETAILER_USER_TEMPLATE = """\
## Node to Detail
ID: {node_id}
Statement (NL): {statement_nl}
Depth: {depth}

## Statement (Lean 4, if available)
```lean
{statement_lean}
```

## Parent Theorem
{parent_statement}

Break this lemma into 3-5 intermediate sub-claims, each provable in \
1-3 Lean tactics. Return as JSON.
"""

PARENT_PROOF_SYSTEM = """\
You are an expert Lean 4 theorem prover. Prove a parent theorem \
assuming its child lemmas are true (they appear as axiom declarations \
or `sorry` premises).

## Key Insight
The child lemmas are available as hypotheses. Your job is to show \
the parent theorem follows from these lemmas, NOT to prove the lemmas \
themselves.

## How to Use Child Lemmas
To use a child lemma in your proof, invoke it with the `have` tactic \
or `apply`/`exact`. Reference each child lemma BY NAME with the \
correct arguments.

### Example 1 — Using a universally quantified lemma
```lean
axiom foo : ∀ n : Nat, n + 0 = n

-- In your proof:
have h1 := foo 5        -- instantiates n with 5
have h2 : 3 + 0 = 3 := foo 3
```

### Example 2 — Using a lemma with complex types
```lean
axiom bar : ∀ (P : Measure X) (ε : ℝ), wassersteinDist P P0 ≤ ε → P ∈ WassersteinBall P0 ε

-- In your proof:
have membership : P1 ∈ WassersteinBall P0 0.5 := bar P1 0.5 dist_proof
```

## Output Format
Return ONLY the Lean 4 proof code inside a ```lean code block. \
Include the child lemma declarations (with sorry) and the parent proof.
"""

PARENT_PROOF_USER_TEMPLATE = """\
## Parent Theorem
```lean
{parent_statement}
```

## Child Lemma Declarations (assume these are true)
{child_declarations}

Prove the parent theorem using these child lemmas as premises. \
Return inside a ```lean code block.
"""

FAILURE_DIAGNOSIS_SYSTEM = """\
You are an expert Lean 4 debugger. Analyze why a proof attempt failed \
and classify the failure.

## Failure Types
- missing_hypothesis: the proof needs a hypothesis not present in the statement
- weak_child_lemma: a child lemma's statement is too weak to prove the parent
- contradictory_child: a child lemma contradicts the proof goal or other children
- stuck_goal: the proof is stuck on a goal that doesn't match any available tactic

## Output Format
Return a JSON object:
```json
{{
  "failure_type": "missing_hypothesis|weak_child_lemma|contradictory_child|stuck_goal",
  "description": "what went wrong",
  "problematic_child_id": "lemma_X or null",
  "suggested_fix": "how to fix this"
}}
```
"""

FAILURE_DIAGNOSIS_USER_TEMPLATE = """\
## Parent Theorem
```lean
{parent_statement}
```

## Child Lemmas Used
{child_declarations}

## Failed Proof Attempt
```lean
{failed_proof}
```

## Compilation Errors
{errors}

Diagnose the failure. Return as JSON.
"""

CHILD_REFORMULATION_TEMPLATE = """\
The following child lemma caused a proof failure. Reformulate it so \
the parent theorem can be proved.

## Parent Theorem
```lean
{parent_statement}
```

## Problematic Child Lemma
ID: {child_id}
Original Statement: {child_statement_nl}
Lean: {child_statement_lean}

## Failure Diagnosis
Type: {failure_type}
Description: {failure_description}
Suggested Fix: {suggested_fix}

Provide a reformulated natural language statement for this lemma \
that would fix the failure. Return a JSON object:
```json
{{
  "reformulated_statement": "new natural language statement",
  "reasoning": "why this reformulation fixes the issue"
}}
```
"""

# ---------------------------------------------------------------------------
# Phase 8: Conjecture Refinement Loop
# ---------------------------------------------------------------------------

CONJECTURE_REFINEMENT_SYSTEM = """\
You are an expert mathematical conjecture refiner. Given a conjecture that \
has been disproved or failed to be proved, produce refined variants using \
the specified refinement strategy.

## Strategies

1. **Weakening**: add hypotheses, restrict to special cases, reduce quantifier \
   strength (∀→∃, "for all n" → "for sufficiently large n")
2. **Strengthening**: if the conjecture was too weak to be interesting, \
   strengthen it and check if provability is maintained
3. **Reformulation**: express the same idea in a different mathematical \
   framework (e.g., algebraic → combinatorial)
4. **Specialization**: try specific instances (n=2, finite case, commutative case)

## Output Format
Return a JSON object:
```json
{{
  "refined_conjectures": [
    {{
      "statement": "formal-ish mathematical statement",
      "natural_language": "plain English description",
      "confidence": 0.0-1.0,
      "difficulty": 1-5,
      "related_results": [],
      "novelty_score": 0.0-1.0,
      "formalizability_score": 0.0-1.0,
      "refinement_reasoning": "why this refinement addresses the failure"
    }}
  ]
}}
```

## Guidelines
- Generate 2-4 refined variants
- Each variant should address the specific failure reason
- Maintain the spirit of the original conjecture
- Weakened variants should have higher confidence estimates
- Specialized variants should have higher formalizability scores
"""

CONJECTURE_REFINEMENT_USER_TEMPLATE = """\
## Original Conjecture
Statement: {original_statement}
Natural Language: {original_nl}

## Failure Information
Outcome: {failure_outcome}
Reason: {failure_reason}

## Refinement Strategy
{strategy}

## Original Idea (context)
{original_idea}

Produce refined variants of this conjecture using the specified strategy. \
Return as JSON.
"""

REFINEMENT_REPORT_SYSTEM = """\
You are a mathematical research reporter. Generate a clear, structured \
report of a conjecture refinement journey — from the original conjecture \
through each refinement attempt to the final outcome.

## Output Format
Write a markdown report with these sections:
1. **Original Conjecture** — what was originally proposed
2. **Refinement Journey** — for each step: what was tried, why it failed, \
   what was changed
3. **Final Outcome** — what was proved (or why all attempts were exhausted)
4. **Key Insights** — what the refinement process revealed about the \
   mathematical structure

Be concise but precise. Use mathematical notation where helpful.
"""

REFINEMENT_REPORT_USER_TEMPLATE = """\
## Original Idea
{original_idea}

## Original Conjecture
{original_conjecture}

## Refinement Attempts
{attempts}

## Final Status
{final_status}

Generate a human-readable markdown report of this refinement journey.
"""

FLATTEN_PROOF_TEMPLATE = """\
Assemble these individually proved lemmas into a single self-contained \
Lean 4 proof. Remove any unused lemmas and ensure the final proof compiles.

## Root Theorem
```lean
{root_statement}
```

## Proved Lemmas (in dependency order)
{proved_lemmas}

## Root Proof (using lemmas)
```lean
{root_proof}
```

Assemble into a single, self-contained Lean 4 file. Include all imports. \
Remove lemmas that are not actually used in the final proof. \
Return inside a ```lean code block.
"""
