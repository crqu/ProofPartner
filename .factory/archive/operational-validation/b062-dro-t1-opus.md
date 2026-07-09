# Operational Validation: B-062 — DRO T=1 Formalization with Opus Model

**Date:** 2026-07-09
**Branch:** exp-025-dro-opus
**Issue:** #34
**Hypothesis:** H1 (EXPLOIT, operational) — test whether model capability (Opus vs Sonnet) is the formalization bottleneck
**Cost:** $0.0000 / $3.00 budget
**Duration:** ~25s total (two model attempts)

## Environment

| Component | Status |
|-----------|--------|
| Lean 4 | Found (`~/.elan/bin/lean`) |
| Mathlib cache | Present (proofpartner-lean/.lake/packages/mathlib/) |
| Loogle search | Operational (responded in ~350ms) |
| API backend | Vertex AI (project: `itpc-gcp-ai-eng-claude`, region: `us-east5`) |
| Direct API key | NOT SET |

## Attempt 1: Opus (claude-opus-4-6-20250616)

```bash
agentic-research --model claude-opus-4-6-20250616 formalize \
  'For the single-period (T=1) case, the distributionally robust optimization problem satisfies: sup_{Q in B_r(P)} E_Q[f(x,xi)] = inf_{lambda >= 0} {lambda * r + E_P[sup_{xi_prime} {f(x,xi_prime) - lambda * c(xi, xi_prime)}]} where B_r(P) is the Wasserstein ball of radius r around empirical measure P, f is the cost function, and c is the ground metric'
```

**Result: BLOCKED — Model not available on Vertex AI**

The model `claude-opus-4-6@20250616` returned HTTP 404 from Vertex AI endpoint. The type_planner agent exhausted all retries (2 attempts x 3 retries = 6 total API calls), all returning 404.

Error: `Publisher model projects/itpc-gcp-ai-eng-claude/locations/us-east5/publishers/anthropic/models/claude-opus-4-6@20250616 was not found or your project does not have access to it.`

## Attempt 2: Sonnet 4.5 Fallback (claude-sonnet-4-5-20250620)

```bash
agentic-research --model claude-sonnet-4-5-20250620 formalize '<same statement>'
```

**Result: BLOCKED — Model not available on Vertex AI**

Same 404 error for `claude-sonnet-4-5@20250620`. The Vertex project `itpc-gcp-ai-eng-claude` has lost access to all Claude models tested:

| Model | Status |
|-------|--------|
| claude-opus-4-6-20250616 | 404 Not Found |
| claude-sonnet-4-5-20250620 | 404 Not Found |
| claude-sonnet-4-20250514 | 404 Not Found (deprecated) |
| claude-haiku-4-5-20251001 | 404 Not Found |
| claude-3-5-sonnet-v2@20241022 | 404 Not Found |
| claude-3-5-sonnet@20240620 | 404 Not Found |

Multiple regions tested (us-east5, us-central1, europe-west1, europe-west4) — all returned 404.

## Verdict: BLOCKED — No API Access

The experiment cannot be executed. The Vertex AI project `itpc-gcp-ai-eng-claude` does not currently have access to any Claude model in any tested region. No direct Anthropic API key (`ANTHROPIC_API_KEY`) is configured as a fallback.

**Note:** Previous successful experiments (B-042 series on 2026-07-08) used `claude-sonnet-4@20250514` on the same Vertex project. That model has since been deprecated (EOL: 2026-06-15) and access appears to have been revoked project-wide.

## Prerequisites to Unblock

1. **Option A:** Set `ANTHROPIC_API_KEY` environment variable with a valid direct API key that supports Opus
2. **Option B:** Re-enable Claude model access in the Vertex AI project `itpc-gcp-ai-eng-claude` — the model publisher agreement or billing may need renewal
3. **Option C:** Update `ANTHROPIC_VERTEX_PROJECT_ID` to a project that has active Claude model access

## Pipeline Infrastructure Validation

Despite the API failure, the pipeline infrastructure itself is operational:
- Package installed and CLI functional
- Lean REPL initialized (mock backend)
- Loogle search responded successfully (~350ms per call)
- Type planner agent correctly orchestrated tool calls before LLM requests
- Retry logic and error handling worked as designed
