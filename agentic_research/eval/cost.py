"""Cost estimation for eval runs based on token usage and model pricing."""

from __future__ import annotations

from agentic_research.models.agents import TokenUsage

PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-5": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
}

_DEFAULT_MODEL = "claude-opus-4-6"


def estimate_cost(token_usage: TokenUsage, model: str = _DEFAULT_MODEL) -> float:
    """Estimate cost in USD from token usage. Prices per million tokens."""
    prices = PRICING.get(model, PRICING[_DEFAULT_MODEL])
    return (
        token_usage.input_tokens * prices["input"] / 1_000_000
        + token_usage.output_tokens * prices["output"] / 1_000_000
        + token_usage.cache_read_input_tokens * prices["cache_read"] / 1_000_000
        + token_usage.cache_creation_input_tokens * prices["cache_write"] / 1_000_000
    )
