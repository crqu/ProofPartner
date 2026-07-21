"""Pydantic models for interactive pipeline steering."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class InteractionOption(BaseModel):
    """A single option presented to the user during interactive steering."""

    label: str = Field(description="Human-readable label for this option")
    value: Any = Field(description="Value returned when this option is selected")
    score: float = Field(default=0.0, description="Score or ranking metric for this option")


class InteractionRequest(BaseModel):
    """Request for user input at a pipeline decision point."""

    type: Literal["select"] = Field(description="Interaction type")
    prompt: str = Field(description="Question or instruction shown to the user")
    options: list[InteractionOption] = Field(default_factory=list)
    default_value: Any = Field(default=None, description="Value used if the user aborts or skips")


class InteractionResponse(BaseModel):
    """User's response to an InteractionRequest."""

    selected_value: Any = Field(default=None, description="The value the user selected")
    aborted: bool = Field(default=False, description="True if the user chose to abort/skip")
