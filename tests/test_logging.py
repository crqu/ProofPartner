"""Tests for structlog configuration."""

import json

from agentic_research.logging import configure_logging, get_logger


def test_configure_logging_json(capsys):
    configure_logging(json_output=True, level="DEBUG")
    logger = get_logger("test")
    logger.info("hello", key="value")
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.strip())
    assert parsed["event"] == "hello"
    assert parsed["key"] == "value"
    assert parsed["level"] == "info"
    assert "timestamp" in parsed


def test_get_logger():
    logger = get_logger("my_module")
    assert logger is not None
