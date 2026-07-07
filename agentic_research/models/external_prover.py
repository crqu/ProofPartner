"""Pydantic models for external prover backends."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_research.models.agents import TokenUsage


class ExternalProverConfig(BaseModel):
    """Configuration for an external prover API endpoint."""

    api_url: str = Field(description="Base URL of the OpenAI-compatible API")
    api_key: str | None = Field(default=None, description="API key for authentication")
    model_name: str = Field(default="leanstral-1.5", description="Model name to request")
    timeout: int = Field(default=120, ge=1, description="Request timeout in seconds")
    max_tokens: int = Field(default=8192, ge=1, description="Max tokens for the response")


class ExternalProverResult(BaseModel):
    """Result from an external prover attempt."""

    success: bool = False
    proof_code: str | None = None
    error: str | None = None
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)
