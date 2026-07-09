# API Guide

ProofPartner can be used as a Python library in addition to the CLI. This enables batch processing, Jupyter notebook integration, and custom research pipelines.

API examples use the actual class interfaces. For the authoritative API, see the source code and type hints — all public functions have type annotations.

## Example 1: Single Exploration

Run the exploration and conjecture generation pipeline on a mathematical idea:

```python
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.explorer import ExplorationAgent
from agentic_research.agents.conjecturer import ConjectureGenerator
from agentic_research.models.agents import AgentContext, AgentStatus
from agentic_research.models.research import ExplorationResult, ConjectureSet
from agentic_research.tools.lean_search import LeanSearch

llm = LLMClient(model="claude-opus-4-6-20250616")
lean_search = LeanSearch()

# Step 1: Explore the idea
explorer = ExplorationAgent(llm_client=llm, lean_search=lean_search)
ctx = AgentContext(task="prime gaps in arithmetic progressions")
result = explorer.run(ctx)

if result.status == AgentStatus.SUCCESS:
    exploration = ExplorationResult.model_validate(result.result)
    print(f"Domain: {exploration.domain}")
    print(f"Concepts: {len(exploration.concepts)}")
    print(f"Directions: {len(exploration.directions)}")
    for d in exploration.directions:
        print(f"  - {d.title} (difficulty: {d.estimated_difficulty})")

# Step 2: Generate conjectures
generator = ConjectureGenerator(llm_client=llm)
conj_ctx = AgentContext(
    task="prime gaps in arithmetic progressions",
    metadata={"exploration_result": result.result},
)
conj_result = generator.run(conj_ctx)

if conj_result.status == AgentStatus.SUCCESS:
    conjectures = ConjectureSet.model_validate(conj_result.result)
    for conj in conjectures.conjectures:
        print(f"  [{conj.confidence:.2f}] {conj.statement}")
```

## Example 2: Formalization Pipeline

Run the type-first formalization pipeline on a natural-language conjecture:

```python
from agentic_research.agents.llm_client import LLMClient
from agentic_research.pipelines.formalization import FormalizationPipeline
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

llm = LLMClient(model="claude-opus-4-6-20250616")
lean_repl = LeanRepl()
lean_search = LeanSearch()

pipeline = FormalizationPipeline(
    llm_client=llm,
    lean_repl=lean_repl,
    lean_search=lean_search,
)

result = pipeline.run(conjecture_nl="the square root of 2 is irrational")

if result.success and result.theorem:
    print(f"Lean statement:\n{result.theorem.lean_statement}")
    print(f"Compiles: {result.theorem.compiles}")
    print(f"Iterations used: {result.theorem.iterations_used}")
    if result.type_formalization:
        print(f"Types defined: {len(result.type_formalization.accepted_types)}")
else:
    print(f"Failed at: {result.failure_stage}")
    print(f"Reason: {result.failure_reason}")
```

## Example 3: Full Research Orchestrator

Run the complete explore-conjecture-prove loop with budget control:

```python
from agentic_research.agents.llm_client import LLMClient
from agentic_research.orchestrator.engine import ResearchOrchestrator
from agentic_research.models.session import OrchestratorConfig, PipelineStage
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

llm = LLMClient(model="claude-opus-4-6-20250616")
lean_repl = LeanRepl()
lean_search = LeanSearch()

config = OrchestratorConfig(
    budget_limit_usd=15.00,
    max_conjectures=3,
    max_refinements=2,
)

orchestrator = ResearchOrchestrator(
    llm_client=llm,
    lean_repl=lean_repl,
    lean_search=lean_search,
    config=config,
)

result = orchestrator.run("the square root of 2 is irrational")

print(f"Final stage: {result.final_stage.value}")
print(f"Cost: ${result.cost_estimate.total_cost_usd:.4f}")
print(f"Conjectures tried: {result.total_conjectures_tried}")
print(f"Proofs found: {len(result.proved_conjectures)}")

for tc in result.proved_conjectures:
    print(f"\n  Proved: {tc.conjecture.statement}")
    if tc.proof_code:
        print(f"  Proof:\n{tc.proof_code}")
```

## Example 4: Batch Processing with Checkpointing

Process multiple conjectures with simple file-based checkpointing:

```python
import json
from pathlib import Path
from agentic_research.agents.llm_client import LLMClient
from agentic_research.pipelines.formalization import FormalizationPipeline
from agentic_research.tools.lean_repl import LeanRepl
from agentic_research.tools.lean_search import LeanSearch

llm = LLMClient()
lean_repl = LeanRepl()
lean_search = LeanSearch()

pipeline = FormalizationPipeline(
    llm_client=llm, lean_repl=lean_repl, lean_search=lean_search,
)

checkpoint = Path("batch_progress.json")
results = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}

conjectures = [
    "every continuous function on [0,1] is bounded",
    "the square root of 2 is irrational",
    "there are infinitely many primes",
]

for i, conj in enumerate(conjectures):
    if str(i) in results:
        print(f"Skipping [{i}] (already processed)")
        continue

    print(f"Processing [{i}]: {conj}")
    result = pipeline.run(conjecture_nl=conj)
    results[str(i)] = {
        "input": conj,
        "success": result.success,
        "lean_statement": result.theorem.lean_statement if result.theorem else None,
        "failure_reason": result.failure_reason,
    }
    checkpoint.write_text(json.dumps(results, indent=2))
    print(f"  Success: {result.success}")
```

## Example 5: Using Individual Agents

Access agents directly for fine-grained control:

```python
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.counterexample_searcher import CounterexampleSearcher
from agentic_research.agents.intent_judge import IntentJudge
from agentic_research.agents.informalizer import Informalizer
from agentic_research.models.verification import CounterexampleStatus
from agentic_research.tools.lean_repl import LeanRepl

llm = LLMClient()
lean_repl = LeanRepl()

# Counterexample search
searcher = CounterexampleSearcher(llm_client=llm, lean_repl=lean_repl)
cx_result = searcher.search(
    lean_code="theorem foo : ∀ n : Nat, n + 0 = n",
    conjecture="n plus zero equals n for all natural numbers",
)
print(f"Status: {cx_result.status.value}")  # "plausible" or "disproved"

# Intent verification
informalizer = Informalizer(llm_client=llm)
judge = IntentJudge(llm_client=llm, informalizer=informalizer)
verdict = judge.judge(
    lean_code="theorem foo : ∀ n : Nat, n + 0 = n",
    original_idea="n plus zero equals n",
    conjecture="for all natural numbers n, n + 0 = n",
)
print(f"Verdict: {verdict.overall_verdict.value}")  # "correct" or "incorrect"
if verdict.all_concerns:
    for concern in verdict.all_concerns:
        print(f"  Concern: {concern}")
```

## Key Output Types

### ResearchSessionResult

Returned by `ResearchOrchestrator.run()`:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Unique session identifier |
| `raw_idea` | `str` | The original input idea |
| `proved_conjectures` | `list[TriedConjecture]` | Conjectures that were successfully proved |
| `failed_conjectures` | `list[TriedConjecture]` | Conjectures that failed at some stage |
| `total_token_usage` | `TokenUsage` | Cumulative token counts |
| `cost_estimate` | `CostEstimate` | Dollar cost breakdown |
| `final_stage` | `PipelineStage` | Stage the pipeline reached (e.g., `COMPLETE`, `FAILED`) |
| `total_conjectures_tried` | `int` | Number of conjectures processed |
| `total_refinements` | `int` | Number of refinement attempts |
| `exploration_rounds` | `int` | Number of exploration rounds |

### FormalizationPipelineResult

Returned by `FormalizationPipeline.run()`:

| Field | Type | Description |
|---|---|---|
| `conjecture_nl` | `str` | The input natural-language conjecture |
| `success` | `bool` | Whether formalization succeeded |
| `theorem` | `TheoremFormalization | None` | The formalized theorem (if successful) |
| `type_formalization` | `TypeFormalizationResult | None` | Type formalization details |
| `failure_stage` | `str | None` | Stage where failure occurred |
| `failure_reason` | `str | None` | Human-readable failure description |
| `total_token_usage` | `TokenUsage` | Token counts for the pipeline |

### AgentResult

Returned by all agents via `BaseAgent.run()`:

| Field | Type | Description |
|---|---|---|
| `agent_name` | `str` | Name of the agent |
| `status` | `AgentStatus` | `SUCCESS`, `MAX_RETRIES`, or `ERROR` |
| `result` | `dict | None` | Agent-specific output data |
| `error_message` | `str | None` | Error details (on failure) |
| `token_usage` | `TokenUsage` | Token counts for this agent call |
| `duration_seconds` | `float` | Wall-clock time |
| `attempts` | `int` | Number of attempts used |

## Further Reading

- **[Tutorial](TUTORIAL.md)** — narrative walkthrough using the CLI
- **[Architecture](ARCHITECTURE.md)** — pipeline stages, agent inventory, data flow
- **[Reproducibility](REPRODUCIBILITY.md)** — model versions and cost estimates
- **[README](../README.md)** — project overview and setup
