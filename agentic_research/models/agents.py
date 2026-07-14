"""Pydantic models for the agent framework."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token counts for a single LLM request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """Response from an LLM call."""

    content: str = ""
    model: str = ""
    stop_reason: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    thinking: str | None = None


class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    MAX_RETRIES = "max_retries"


class AgentResult(BaseModel):
    """Result returned by every agent run."""

    agent_name: str
    status: AgentStatus
    result: dict | None = None
    error_message: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    attempts: int = 1
    duration_seconds: float = 0.0


class AgentContext(BaseModel):
    """Input context passed to an agent's run method."""

    task: str = Field(description="The main task/query for the agent")
    history: list[str] = Field(default_factory=list, description="Prior conversation turns or attempt history")
    metadata: dict = Field(default_factory=dict, description="Additional context (problem ID, config overrides, etc.)")


class ProverConfig(BaseModel):
    """Configuration for the iterative prover agent."""

    max_iterations: int = Field(default=2, ge=1, description="Max proof refinement iterations")
    model: str = Field(default="claude-opus-4-6-20250616", description="LLM model ID")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=16384, ge=1)
    use_extended_thinking: bool = False
    lean_timeout_seconds: int = Field(default=60, ge=1)


class ProofAttemptStatus(str, Enum):
    SUCCESS = "success"
    COMPILATION_ERROR = "compilation_error"
    INCOMPLETE = "incomplete"
    TIMEOUT = "timeout"
    ERROR = "error"


class ProofAttempt(BaseModel):
    """A single proof attempt within the iterative prover."""

    iteration: int
    proof_code: str
    status: ProofAttemptStatus
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    goals_remaining: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class ProverResult(BaseModel):
    """Full result from the iterative prover."""

    statement: str
    proved: bool = False
    final_proof: str | None = None
    attempts: list[ProofAttempt] = Field(default_factory=list)
    total_iterations: int = 0
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    failure_reason: str | None = None
