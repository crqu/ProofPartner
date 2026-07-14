"""Tests for eval runner configuration and ProverConfig threading."""

from agentic_research.models.agents import ProverConfig
from agentic_research.models.eval import EvalConfig, EvalMode


def test_eval_config_defaults():
    """EvalConfig new fields have correct defaults."""
    config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY)
    assert config.use_extended_thinking is True
    assert config.thinking_budget == 10000
    assert config.max_critic_retries == 3
    assert config.use_intent_judge is True
    assert config.timeout_seconds == 600


def test_prover_config_from_eval_config():
    """ProverConfig is constructed with use_extended_thinking from EvalConfig."""
    config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY, use_extended_thinking=True)
    prover_config = ProverConfig(use_extended_thinking=config.use_extended_thinking)
    assert prover_config.use_extended_thinking is True


def test_prover_config_extended_thinking_disabled():
    """ProverConfig respects use_extended_thinking=False from EvalConfig."""
    config = EvalConfig(mode=EvalMode.PROOF_DISCOVERY, use_extended_thinking=False)
    prover_config = ProverConfig(use_extended_thinking=config.use_extended_thinking)
    assert prover_config.use_extended_thinking is False
