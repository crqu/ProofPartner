"""Pydantic models for paper-level extraction (Extractor agent output)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedTheorem(BaseModel):
    """A theorem extracted from a research paper."""

    statement: str
    statement_latex: str = ""
    is_main: bool = False
    section_ref: str = ""


class ExtractedDefinition(BaseModel):
    """A definition extracted from a research paper."""

    name: str
    informal_statement: str
    depends_on: list[str] = Field(default_factory=list)
    in_mathlib: bool = False


class ExtractedLemma(BaseModel):
    """A supporting lemma extracted from a research paper."""

    name: str
    informal_statement: str
    used_in_proof_of: str = ""


class ExtractedPriorWork(BaseModel):
    """A prior result cited in a research paper."""

    citation: str
    result_statement: str
    axiom_candidate: bool = True


class ExtractionResult(BaseModel):
    """Complete extraction output from a research paper."""

    theorems: list[ExtractedTheorem] = Field(default_factory=list)
    definitions: list[ExtractedDefinition] = Field(default_factory=list)
    lemmas: list[ExtractedLemma] = Field(default_factory=list)
    prior_work: list[ExtractedPriorWork] = Field(default_factory=list)
    paper_title: str = ""
    paper_domain: str = ""
