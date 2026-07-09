# ProofPartner

## Quick Start

```bash
pip install -e ".[dev]"

# With Vertex AI support
pip install -e ".[dev,vertex]"
```

### Environment Variables

- `ANTHROPIC_API_KEY` — direct Anthropic API access
- `CLAUDE_CODE_USE_VERTEX=1` + `ANTHROPIC_VERTEX_PROJECT_ID` — Vertex AI backend
- `AGENTIC_RESEARCH_MODEL` — override default model (default: `claude-opus-4-6-20250616`)

## Commands

```bash
# Tests
pytest tests/ -v

# Lint
ruff check agentic_research/ tests/

# Type check (if mypy installed)
mypy agentic_research/

# CLI (use --model to override LLM model for any command)
agentic-research --help
agentic-research --model claude-sonnet-4-20250514 explore 'my idea'

# Eval harness
python -m agentic_research.eval.runner --mode proof_discovery --benchmark miniF2F --split valid
```

## Project Structure

```
agentic_research/
├── agents/           # LLM-powered agents (explorer, conjecturer, prover, intent judge, etc.)
│   ├── base.py       # BaseAgent with retry logic and token tracking
│   ├── llm_client.py # Anthropic API wrapper
│   └── ...           # 20 specialized agents
├── cli/              # Click-based CLI entry point
├── eval/             # Benchmark evaluation harness (miniF2F, PutnamBench)
├── memory/           # Research session memory (conjectures, directions, preferences)
├── models/           # Pydantic data models (agents, formalization, proof, session, etc.)
├── orchestrator/     # State machine engine, checkpointing, rollback
│   ├── engine.py     # ResearchOrchestrator — main loop with 8 stages
│   ├── state.py      # ResearchStage enum and transitions
│   └── rollback.py   # CheckpointManager for session recovery
├── pipelines/        # Multi-agent pipelines (formalization, proof, refinement)
└── tools/            # Lean 4 integration (REPL, search, lookup)
tests/                # 13 test files, 359+ tests
```

## Coding Conventions

- **Data models**: Pydantic `BaseModel` (v2) at all boundaries
- **Logging**: `structlog` via `from agentic_research.logging import get_logger`
- **Type hints**: Required on all public functions
- **Tests**: Mock Lean backends and LLM clients — real Lean 4 requires `@pytest.mark.lean_required`
- **Imports**: Use absolute imports from `agentic_research`
