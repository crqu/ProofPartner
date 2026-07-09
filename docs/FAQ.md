# Frequently Asked Questions

## How is ProofPartner different from Hilbert, ReProver, or LeanDojo?

ProofPartner starts from **rough mathematical ideas** (Stage 2) — you describe an intuition in natural language, and ProofPartner generates formal conjectures, verifies they match your intent, checks for counterexamples, and attempts proofs. Hilbert, ReProver, and LeanDojo start from **pre-formalized Lean 4 statements** (Stage 1) — they need a precise formal statement as input.

Use ProofPartner when you don't yet have a Lean 4 statement. Use Hilbert or ReProver when you already have one and need a proof.

See the [competitive landscape table](../README.md#competitive-landscape) in the README for a detailed comparison.

## Do I need Lean 4 installed?

**No** for exploration, conjecture generation, and formalization. ProofPartner uses mocked Lean backends for these stages, which is sufficient for generating conjectures and Lean 4 statements.

**Yes** for verified proof search and counterexample checking. These require the Lean 4 kernel to validate proofs and check statements. Install Lean 4 via [elan](https://github.com/leanprover/elan).

See the [Reproducibility Guide](REPRODUCIBILITY.md#lean-4-setup) for installation instructions.

## How much does it cost?

All ProofPartner commands run via the Anthropic API and incur token costs. Typical costs:

| Operation | Typical cost |
|---|---|
| Explore an idea | $0.05–$0.50 |
| Formalize a conjecture | $0.50–$3.00 |
| Check for counterexamples | $0.10–$2.00 |
| Proof search | $1.00–$10.00 |
| Full research loop | $2.00–$20.00 |

Every command has a `--budget` flag that sets a hard cost cap. The default budgets are: explore $2, formalize $3, check $2, prove $10, research $20.

See the [Reproducibility Guide](REPRODUCIBILITY.md#cost-estimates) for detailed cost breakdowns.

## What models are supported?

ProofPartner supports Claude models via the Anthropic API or Google Cloud Vertex AI:

- **Default:** `claude-opus-4-6-20250616`
- **Override:** Set `AGENTIC_RESEARCH_MODEL` environment variable, or use `--model` on any CLI command

```bash
# Use a different model
agentic-research --model claude-sonnet-4-20250514 explore 'my idea'
```

Currently Anthropic-only. Support for additional LLM providers is on the roadmap.

## My proof search timed out or hit the budget cap

Several options:

1. **Increase the budget and timeout:**

```bash
agentic-research prove 'theorem ...' --budget 20.00 --timeout 1200
```

2. **Decompose the theorem** into smaller lemmas and prove each separately. Complex theorems are easier to prove when broken into sub-goals.

3. **Try a different model** — different models may find different proof strategies:

```bash
agentic-research --model claude-opus-4-6-20250616 prove 'theorem ...'
```

4. **Use the full research loop** which includes automatic refinement:

```bash
agentic-research research 'your idea' --budget 20.00
```

## Can I resume an interrupted session?

Session checkpointing is built into the orchestrator — state is saved at each pipeline stage. A CLI `resume` command is planned (see [issue #7](https://github.com/crqu/ProofPartner/issues/7)).

Programmatically, you can resume via the API:

```python
orchestrator.resume_from_checkpoint(checkpoint_id)
```

## How do I cite ProofPartner?

See [CITATION.cff](../CITATION.cff) in the repository root for machine-readable citation metadata. BibTeX entries are available in the [README Citation section](../README.md#citation).

Please also cite the foundational work that ProofPartner builds on:

> Soltani Moakhar, A., Gholami, I., Springer, M., JafariRaviz, M., & Hajiaghayi, M. (2026). Beyond the Library: An Agentic Framework for Autoformalizing Research Mathematics. arXiv:2606.31134.

## Can I use a different LLM provider?

Currently, ProofPartner supports the Anthropic API (direct and Vertex AI) only. The `LLMClient` class in `agents/llm_client.py` is the single point of integration. OpenAI support is gated behind the `OPENAI_ENABLED` feature flag but is not yet fully implemented.

## What does "type-first formalization" mean?

Instead of directly translating a natural-language conjecture into a Lean 4 theorem statement, ProofPartner first identifies and formalizes the *types* (mathematical structures) needed, validates them with auxiliary lemmas, and only then builds the theorem statement on top of the accepted types. This approach is adapted from [Moakhar et al. (2026)](https://arxiv.org/abs/2606.31134) and produces more robust formalizations.

See the [Architecture](ARCHITECTURE.md) document for details on each pipeline stage.

## What are the eval benchmarks?

ProofPartner includes an evaluation harness with two benchmarks:

- **miniF2F v2** — 488 competition-level math problems in Lean 4 (244 test + 244 validation)
- **PutnamBench** — 672 Putnam competition problems (stub loader)

The eval harness supports three modes: proof discovery, conjecture quality, and end-to-end research. Baseline results are pending.

```bash
agentic-research eval miniF2F --mode proof_discovery --split valid --pass-k 8
```

## How do I use ProofPartner in a Jupyter notebook?

Use the Python API directly — see the [API Guide](API.md) for examples. All agents, pipelines, and the orchestrator can be imported and used programmatically.

## Where are sessions stored?

Session data is stored in `.agentic_research/sessions/` in the current working directory. Each session includes conjecture history, proof outcomes, and memory tier data. Sessions persist between CLI commands.

## Further Reading

- **[Tutorial](TUTORIAL.md)** — complete research session walkthrough
- **[API Guide](API.md)** — programmatic Python usage
- **[Reproducibility](REPRODUCIBILITY.md)** — model versions, costs, hardware
- **[Glossary](GLOSSARY.md)** — key terms and definitions
- **[Architecture](ARCHITECTURE.md)** — pipeline internals
- **[README](../README.md)** — project overview
