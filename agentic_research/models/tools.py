"""Pydantic models for the Lean 4 tool layer."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


class ToolResult(BaseModel):
    """Base result returned by every tool invocation."""

    tool_name: str
    status: ToolStatus
    duration_seconds: float = 0.0
    error_message: str | None = None


class CompilationStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class ProofGoal(BaseModel):
    """A single proof goal reported by the Lean REPL."""

    goal: str = Field(description="The goal statement (target type)")
    hypotheses: list[str] = Field(default_factory=list, description="Local hypotheses in scope")


class CompilationResult(ToolResult):
    """Result of compiling Lean 4 code via the REPL."""

    tool_name: str = "lean_repl"
    compilation_status: CompilationStatus = CompilationStatus.ERROR
    goals: list[ProofGoal] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    lean_output: str = ""
    all_goals_closed: bool = False


class SearchResultEntry(BaseModel):
    """A single search hit from Mathlib."""

    name: str = Field(description="Fully qualified Lean name")
    type_signature: str = Field(default="", description="Type signature")
    doc_string: str = Field(default="", description="Documentation string")
    module: str = Field(default="", description="Source module in Mathlib")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResult(ToolResult):
    """Result of searching Mathlib for lemmas/theorems."""

    tool_name: str = "lean_search"
    query: str = ""
    entries: list[SearchResultEntry] = Field(default_factory=list)
    total_results: int = 0
    truncated: bool = False


class LookupResult(ToolResult):
    """Result of looking up a specific Lean/Mathlib identifier."""

    tool_name: str = "lean_lookup"
    identifier: str = ""
    type_signature: str = ""
    doc_string: str = ""
    module: str = ""
    source_url: str = ""
    found: bool = False


class CleanResult(ToolResult):
    """Result of stripping AI-generated comments from Lean code."""

    tool_name: str = "hint_cleaner"
    original_code: str = ""
    cleaned_code: str = ""
    comments_removed: int = 0
