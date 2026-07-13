"""Pydantic models for the evaluation system."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class BenchmarkSource(str, Enum):
    MINIF2F = "miniF2F"
    PUTNAM_BENCH = "PutnamBench"


class ProblemSplit(str, Enum):
    TEST = "test"
    VALIDATION = "valid"


class ProblemDifficulty(str, Enum):
    AMC = "amc"
    AIME = "aime"
    MATHD = "mathd"
    IMO = "imo"
    PUTNAM = "putnam"
    UNKNOWN = "unknown"


class Problem(BaseModel):
    """A single benchmark problem — a Lean 4 theorem statement to prove."""

    id: str
    name: str
    source: BenchmarkSource
    split: ProblemSplit
    difficulty: ProblemDifficulty = ProblemDifficulty.UNKNOWN
    lean_header: str = Field(default="", description="Import statements and open namespaces")
    lean_statement: str = Field(description="The Lean 4 theorem statement (without proof)")
    natural_language: str = Field(default="", description="Natural language description if available")
    file_path: str = Field(default="", description="Original file path in the benchmark repo")


class ProblemSet(BaseModel):
    """A collection of benchmark problems."""

    name: str
    source: BenchmarkSource
    problems: list[Problem] = Field(default_factory=list)

    @property
    def test_problems(self) -> list[Problem]:
        return [p for p in self.problems if p.split == ProblemSplit.TEST]

    @property
    def validation_problems(self) -> list[Problem]:
        return [p for p in self.problems if p.split == ProblemSplit.VALIDATION]


class EvalMode(str, Enum):
    PROOF_DISCOVERY = "proof_discovery"
    CONJECTURE_QUALITY = "conjecture_quality"
    END_TO_END = "end_to_end"


class ProofResult(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class ProblemResult(BaseModel):
    """Result of evaluating a single problem."""

    problem_id: str
    mode: EvalMode
    result: ProofResult
    proof: str | None = None
    attempts: int = 0
    duration_seconds: float = 0.0
    error_message: str | None = None
    token_usage: int = 0


class ConjectureScore(BaseModel):
    """Scoring for conjecture quality evaluation."""

    formalizability: float = Field(ge=0.0, le=1.0, description="Can this be expressed in Lean 4?")
    non_triviality: float = Field(ge=0.0, le=1.0, description="Is this non-obvious?")
    relevance: float = Field(ge=0.0, le=1.0, description="Does it capture the original idea?")

    @property
    def composite(self) -> float:
        return (self.formalizability + self.non_triviality + self.relevance) / 3.0


class WilsonInterval(BaseModel):
    """Wilson score confidence interval for a proportion."""

    lower: float
    upper: float
    center: float
    n: int
    successes: int


class AggregateStats(BaseModel):
    """Aggregate statistics across a set of problem results."""

    total: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    errors: int = 0
    pass_rate: float = 0.0
    wilson_ci: WilsonInterval | None = None
    mean_attempts: float = 0.0
    mean_duration_seconds: float = 0.0
    total_tokens: int = 0


class ScoreReport(BaseModel):
    """Complete evaluation report."""

    mode: EvalMode
    benchmark: str
    split: ProblemSplit | None = None
    results: list[ProblemResult] = Field(default_factory=list)
    aggregate: AggregateStats = Field(default_factory=AggregateStats)
    conjecture_scores: list[ConjectureScore] | None = None
    by_difficulty: dict[str, AggregateStats] | None = None


class EvalConfig(BaseModel):
    """Configuration for an evaluation run."""

    mode: EvalMode
    benchmark: BenchmarkSource = BenchmarkSource.MINIF2F
    split: ProblemSplit = ProblemSplit.VALIDATION
    pass_k: int = Field(default=1, ge=1, description="Number of attempts per problem (pass@k)")
    timeout_seconds: int = Field(default=1800, description="Timeout per problem")
    sample_size: int | None = Field(default=None, description="Subset of problems to evaluate")
    seed: int = Field(default=0, description="Random seed for sampling")
    data_dir: Path = Field(default=Path("data/benchmarks"), description="Where to store benchmark data")
    model: str | None = Field(default=None, description="LLM model for proof attempts")
    use_extended_thinking: bool = Field(default=True, description="Enable extended thinking for proof search")
    thinking_budget: int = Field(default=10000, description="Token budget for extended thinking")
    max_critic_retries: int = Field(default=3, description="Max proof critic retry rounds")
    use_intent_judge: bool = Field(default=True, description="Enable intent judge for type formalization")
