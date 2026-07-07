"""Unified LLM client wrapping the Anthropic API.

Supports Claude Opus 4.6 with extended thinking, prompt caching,
structured output extraction, and token usage tracking.

Feature flag OPENAI_ENABLED gates optional GPT integration (disabled by default).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

from agentic_research.logging import get_logger
from agentic_research.models.agents import LLMResponse, TokenUsage

log = get_logger(__name__)

DEFAULT_MODEL = "claude-opus-4-6-20250616"
OPENAI_ENABLED = os.environ.get("OPENAI_ENABLED", "false").lower() in ("true", "1", "yes")


class LLMClientError(Exception):
    """Raised when the LLM client encounters an unrecoverable error."""


class LLMRetryExhaustedError(LLMClientError):
    """Raised when all retry attempts have been exhausted."""


class LLMClient:
    """Unified LLM client for the agent framework."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise LLMClientError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it or pass api_key to LLMClient."
            )

        import anthropic
        self._client = anthropic.Anthropic(api_key=resolved_key)
        log.info("llm_client_init", model=model)

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        system: str | None = None,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_extended_thinking: bool = False,
        thinking_budget: int = 10000,
        use_cache: bool = False,
    ) -> LLMResponse:
        """Send a completion request to the Anthropic API."""
        resolved_max = max_tokens or self._max_tokens
        resolved_temp = temperature if temperature is not None else self._temperature

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": resolved_max,
        }

        if system:
            if use_cache:
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                kwargs["system"] = system

        if use_extended_thinking:
            kwargs["temperature"] = 1
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        else:
            kwargs["temperature"] = resolved_temp

        log.info(
            "llm_request",
            model=self._model,
            message_count=len(messages),
            max_tokens=resolved_max,
            extended_thinking=use_extended_thinking,
        )

        response = self._call_with_retries(kwargs)

        content_text = ""
        thinking_text = None
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                content_text = block.text

        usage = response.usage
        token_usage = TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

        log.info(
            "llm_response",
            model=self._model,
            stop_reason=response.stop_reason,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
        )

        return LLMResponse(
            content=content_text,
            model=response.model,
            stop_reason=response.stop_reason,
            token_usage=token_usage,
            thinking=thinking_text,
        )

    def _call_with_retries(self, kwargs: dict[str, Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    break
                delay = min(
                    self._backoff_base * math.pow(2, attempt - 1),
                    self._backoff_max,
                )
                log.warning(
                    "llm_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    delay=delay,
                    error=str(exc),
                )
                time.sleep(delay)

        raise LLMRetryExhaustedError(
            f"All {self._max_retries} retries exhausted: {last_exc}"
        ) from last_exc

    def extract_json(self, text: str) -> dict | list | None:
        """Extract JSON from LLM response text (code blocks or raw)."""
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

        return None
