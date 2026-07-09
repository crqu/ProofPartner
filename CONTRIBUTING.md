# Contributing to ProofPartner

Thank you for your interest in contributing to ProofPartner! This document explains how to set up your development environment, the project's coding conventions, and how to submit changes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/crqu/ProofPartner.git
cd ProofPartner

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# (Optional) Install Vertex AI support
pip install -e ".[dev,vertex]"
```

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=agentic_research --cov-report=term-missing

# Skip tests that require Lean 4
pytest tests/ -v -m "not lean_required"
```

### Linting

```bash
ruff check agentic_research/ tests/
```

### Type Checking

```bash
mypy agentic_research/
```

## Code Conventions

- **Data models:** Use Pydantic `BaseModel` (v2) at all module boundaries
- **Logging:** Use `structlog` via `from agentic_research.logging import get_logger`
- **Type hints:** Required on all public functions
- **Imports:** Use absolute imports from `agentic_research` (e.g., `from agentic_research.agents.base import BaseAgent`)
- **Testing:** Mock Lean backends and LLM clients in tests. Tests requiring a real Lean 4 installation must be marked with `@pytest.mark.lean_required`

## How to Contribute

### Bug Reports

Open an issue on [GitHub](https://github.com/crqu/ProofPartner/issues) with:

- A clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- Python version, OS, and ProofPartner version (`agentic-research --version`)

### Feature Requests

Open an issue describing the feature, its motivation, and how it fits into the existing pipeline. For large architectural changes, please discuss in an issue before starting implementation.

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Ensure all tests pass: `pytest tests/ -v`
5. Ensure lint passes: `ruff check agentic_research/ tests/`
6. Commit with a descriptive message
7. Push to your fork and open a pull request

**PR checklist:**

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Lint passes (`ruff check agentic_research/ tests/`)
- [ ] New features include tests
- [ ] Public functions have type annotations
- [ ] Description explains what changed and why

### What We Welcome

- Bug fixes
- Documentation improvements
- Test coverage improvements
- Small features that fit within the existing architecture
- New agent implementations that follow the `BaseAgent` protocol

### Scope

For large changes (new pipeline stages, alternative LLM backends, major refactors), please open an issue first to discuss the approach. This helps avoid duplicate work and ensures the change fits the project's direction.

## Project Structure

```
agentic_research/
├── agents/        # LLM-powered agents (BaseAgent subclasses)
├── cli/           # Click CLI entry points
├── eval/          # Evaluation harness + benchmark loaders
├── memory/        # Research session memory (hot/warm/cold tiers)
├── models/        # Pydantic data models
├── orchestrator/  # State machine, checkpointing, cost tracking
├── pipelines/     # Multi-agent pipelines (formalization, proof, refinement)
└── tools/         # Lean 4 integration (REPL, search, lookup)
tests/             # Test files (~586 tests)
docs/              # Documentation
```

## License

By contributing to ProofPartner, you agree that your contributions will be licensed under the [MIT License](LICENSE).

## Questions?

Open an issue on [GitHub](https://github.com/crqu/ProofPartner/issues) or check the [FAQ](docs/FAQ.md).
