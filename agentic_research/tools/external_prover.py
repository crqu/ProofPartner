"""External prover client for OpenAI-compatible proof backends (e.g. Leanstral)."""

from __future__ import annotations

import re

import httpx

from agentic_research.logging import get_logger
from agentic_research.models.agents import TokenUsage
from agentic_research.models.external_prover import ExternalProverConfig, ExternalProverResult

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a Lean 4 proof assistant. Given a Lean 4 theorem statement, "
    "produce a complete proof. Return ONLY the proof wrapped in a ```lean code block. "
    "Do not include the original statement — only the proof body (the tactic block or term)."
)


def _extract_proof_from_response(text: str) -> str | None:
    """Extract Lean 4 proof code from a markdown code block in the response."""
    match = re.search(r"```(?:lean)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        return code if code else None
    stripped = text.strip()
    if stripped and not stripped.startswith("```"):
        return stripped
    return None


class ExternalProverClient:
    """Sends formal Lean 4 statements to an OpenAI-compatible API for proof search."""

    def __init__(self, config: ExternalProverConfig) -> None:
        self._config = config

    def prove(self, statement: str, budget_tokens: int | None = None) -> ExternalProverResult:
        """Attempt to prove a Lean 4 statement via the external API."""
        max_tokens = budget_tokens if budget_tokens is not None else self._config.max_tokens

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        payload = {
            "model": self._config.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Prove the following Lean 4 statement:\n\n{statement}"},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        url = self._config.api_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        log.info(
            "external_prover_request",
            url=url,
            model=self._config.model_name,
            statement_len=len(statement),
        )

        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            log.warning("external_prover_timeout", timeout=self._config.timeout)
            return ExternalProverResult(error="Request timed out")
        except httpx.HTTPStatusError as exc:
            log.warning("external_prover_http_error", status=exc.response.status_code)
            return ExternalProverResult(error=f"HTTP {exc.response.status_code}")
        except httpx.HTTPError as exc:
            log.warning("external_prover_error", error=str(exc))
            return ExternalProverResult(error=str(exc))

        try:
            data = response.json()
        except Exception:
            log.warning("external_prover_malformed_json")
            return ExternalProverResult(error="Malformed JSON response")

        usage = data.get("usage", {})
        tokens = TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

        choices = data.get("choices", [])
        if not choices:
            log.warning("external_prover_no_choices")
            return ExternalProverResult(error="No choices in response", tokens_used=tokens)

        content = choices[0].get("message", {}).get("content", "")
        proof_code = _extract_proof_from_response(content)

        if not proof_code:
            log.warning("external_prover_no_proof_extracted")
            return ExternalProverResult(
                error="Could not extract proof from response",
                tokens_used=tokens,
            )

        log.info("external_prover_success", proof_len=len(proof_code))
        return ExternalProverResult(success=True, proof_code=proof_code, tokens_used=tokens)
