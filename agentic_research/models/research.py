"""Pydantic models for the exploration and conjecture generation pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Concept(BaseModel):
    """A mathematical concept identified during exploration."""

    name: str = Field(description="Name of the concept (e.g., 'graph coloring')")
    description: str = Field(default="", description="Brief description of the concept")
    domain: str = Field(default="", description="Mathematical domain (e.g., 'combinatorics')")
    mathlib_ref: str | None = Field(default=None, description="Mathlib reference if found")


class ResearchDirection(BaseModel):
    """A candidate research direction proposed by the Exploration Agent."""

    title: str = Field(description="Short title for the direction")
    description: str = Field(description="What this direction investigates")
    ambition_level: int = Field(
        ge=1, le=5,
        description="1=conservative (close to known results), 5=ambitious (novel)",
    )
    relevant_concepts: list[str] = Field(default_factory=list)
    estimated_difficulty: int = Field(
        default=3, ge=1, le=5,
        description="1=easy, 5=very hard",
    )


class ExplorationResult(BaseModel):
    """Structured output from the Exploration Agent."""

    raw_idea: str = Field(description="The original rough idea from the user")
    domain: str = Field(description="Primary mathematical domain identified")
    concepts: list[Concept] = Field(default_factory=list)
    known_results: list[str] = Field(default_factory=list)
    directions: list[ResearchDirection] = Field(default_factory=list)


class Conjecture(BaseModel):
    """A single conjecture candidate."""

    statement: str = Field(description="Formal-ish mathematical statement")
    natural_language: str = Field(description="Plain English description")
    confidence: float = Field(ge=0.0, le=1.0, description="Estimated probability of being true")
    difficulty: int = Field(ge=1, le=5, description="Estimated difficulty to prove (1=easy, 5=very hard)")
    related_results: list[str] = Field(default_factory=list)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    formalizability_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def composite_score(self) -> float:
        return (self.novelty_score + self.confidence + self.formalizability_score) / 3.0


class ConjectureSet(BaseModel):
    """Structured output from the Conjecture Generator."""

    conjectures: list[Conjecture] = Field(default_factory=list)
    ranking: list[int] = Field(default_factory=list, description="Indices into conjectures, best first")
    exploration_context: ExplorationResult | None = None
