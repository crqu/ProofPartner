"""Tests for Phase 3: Context and memory hardening.

Covers tiered session memory, context size monitoring, stage history
compression, and cache_prefix on the base agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_research.agents.base import BaseAgent
from agentic_research.memory.context_monitor import ContextSizeMonitor
from agentic_research.memory.session import (
    HOT_TIER_SIZE,
    ResearchSessionMemory,
    WARM_TIER_SIZE,
    WarmConjecture,
)
from agentic_research.models.agents import AgentContext, AgentResult
from agentic_research.models.research import Conjecture
from agentic_research.models.session import ConjectureOutcome


def _make_conjecture(idx: int, statement: str | None = None) -> Conjecture:
    return Conjecture(
        statement=statement or f"Conjecture {idx}",
        natural_language=f"Conjecture {idx} in plain English",
        confidence=0.7,
        difficulty=3,
    )


# ---------------------------------------------------------------------------
# Tier transitions
# ---------------------------------------------------------------------------


class TestTierTransitions:
    def test_first_three_stay_in_hot(self) -> None:
        mem = ResearchSessionMemory(session_id="test-hot")
        for i in range(3):
            mem.record_conjecture(_make_conjecture(i))
        assert len(mem.data.tried_conjectures) == HOT_TIER_SIZE
        assert len(mem.warm_conjectures) == 0
        assert len(mem.cold_conjectures) == 0

    def test_fourth_demotes_oldest_to_warm(self) -> None:
        mem = ResearchSessionMemory(session_id="test-warm")
        for i in range(4):
            mem.record_conjecture(_make_conjecture(i))
        assert len(mem.data.tried_conjectures) == HOT_TIER_SIZE
        assert len(mem.warm_conjectures) == 1
        assert mem.warm_conjectures[0].statement == "Conjecture 0"

    def test_warm_fills_before_cold(self) -> None:
        mem = ResearchSessionMemory(session_id="test-warm-fill")
        for i in range(HOT_TIER_SIZE + WARM_TIER_SIZE):
            mem.record_conjecture(_make_conjecture(i))
        assert len(mem.data.tried_conjectures) == HOT_TIER_SIZE
        assert len(mem.warm_conjectures) == WARM_TIER_SIZE
        assert len(mem.cold_conjectures) == 0

    def test_overflow_warm_demotes_to_cold(self) -> None:
        mem = ResearchSessionMemory(session_id="test-cold")
        count = HOT_TIER_SIZE + WARM_TIER_SIZE + 3
        for i in range(count):
            mem.record_conjecture(_make_conjecture(i))
        assert len(mem.data.tried_conjectures) == HOT_TIER_SIZE
        assert len(mem.warm_conjectures) == WARM_TIER_SIZE
        assert len(mem.cold_conjectures) == 3
        assert mem.total_conjecture_count == count

    def test_warm_summary_truncates_long_failure_reason(self) -> None:
        from agentic_research.models.session import TriedConjecture, PipelineStage
        tc = TriedConjecture(
            conjecture=_make_conjecture(0),
            outcome=ConjectureOutcome.PROOF_FAILED,
            failure_reason="x" * 200,
            stage_reached=PipelineStage.PROVING,
        )
        wc = WarmConjecture.from_tried(tc)
        assert len(wc.failure_reason) <= 120

    def test_has_tried_checks_all_tiers(self) -> None:
        mem = ResearchSessionMemory(session_id="test-tried")
        for i in range(HOT_TIER_SIZE + WARM_TIER_SIZE + 5):
            mem.record_conjecture(_make_conjecture(i))
        assert mem.has_tried("Conjecture 0")
        assert mem.has_tried(f"Conjecture {HOT_TIER_SIZE}")
        assert mem.has_tried(f"Conjecture {HOT_TIER_SIZE + WARM_TIER_SIZE + 4}")
        assert not mem.has_tried("Never recorded")


# ---------------------------------------------------------------------------
# Memory compaction / overflow archiving
# ---------------------------------------------------------------------------


class TestMemoryCompaction:
    def test_compact_is_idempotent(self) -> None:
        mem = ResearchSessionMemory(session_id="test-idem")
        for i in range(6):
            mem.record_conjecture(_make_conjecture(i))
        warm_before = len(mem.warm_conjectures)
        cold_before = len(mem.cold_conjectures)
        mem.compact()
        assert len(mem.warm_conjectures) == warm_before
        assert len(mem.cold_conjectures) == cold_before

    def test_archive_on_overflow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        max_c = 20
        mem = ResearchSessionMemory(session_id="test-archive", max_conjectures=max_c)
        for i in range(max_c + 5):
            mem.record_conjecture(_make_conjecture(i))
        assert mem.total_conjecture_count <= max_c
        archive_file = tmp_path / f"session_archive_{mem.session_id}.json"
        assert archive_file.exists()
        archived = json.loads(archive_file.read_text())
        assert len(archived) >= 5

    def test_archive_appends_on_subsequent_overflows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        max_c = 15
        mem = ResearchSessionMemory(session_id="test-append", max_conjectures=max_c)
        for i in range(max_c + 3):
            mem.record_conjecture(_make_conjecture(i))
        archive_file = tmp_path / f"session_archive_{mem.session_id}.json"
        first_count = len(json.loads(archive_file.read_text()))
        for i in range(max_c + 3, max_c + 6):
            mem.record_conjecture(_make_conjecture(i))
        second_count = len(json.loads(archive_file.read_text()))
        assert second_count > first_count

    def test_total_count_tracks_all_tiers(self) -> None:
        mem = ResearchSessionMemory(session_id="test-count")
        for i in range(20):
            mem.record_conjecture(_make_conjecture(i))
        assert mem.total_conjecture_count == 20


# ---------------------------------------------------------------------------
# Context size estimation
# ---------------------------------------------------------------------------


class TestContextSizeMonitor:
    def test_initial_estimate_is_zero(self) -> None:
        mon = ContextSizeMonitor()
        assert mon.context_tokens_estimate == 0

    def test_update_increases_estimate(self) -> None:
        mon = ContextSizeMonitor()
        mon.update("a" * 400)
        assert mon.context_tokens_estimate == 100

    def test_set_total_overrides(self) -> None:
        mon = ContextSizeMonitor()
        mon.update("a" * 400)
        mon.set_total(800)
        assert mon.context_tokens_estimate == 200

    def test_warning_emitted_at_8k(self, capfd: pytest.CaptureFixture[str]) -> None:
        mon = ContextSizeMonitor()
        mon.update("a" * (8_000 * 4))
        assert 8_000 in mon._warnings_emitted

    def test_warning_emitted_at_16k(self) -> None:
        mon = ContextSizeMonitor()
        mon.update("a" * (16_000 * 4))
        assert 16_000 in mon._warnings_emitted

    def test_warning_emitted_at_32k(self) -> None:
        mon = ContextSizeMonitor()
        mon.update("a" * (32_000 * 4))
        assert 32_000 in mon._warnings_emitted

    def test_warning_only_emitted_once(self) -> None:
        mon = ContextSizeMonitor()
        mon.update("a" * (8_000 * 4))
        mon.update("a" * 100)
        assert mon._warnings_emitted == {8_000}

    def test_reset_clears_state(self) -> None:
        mon = ContextSizeMonitor()
        mon.update("a" * (8_000 * 4))
        mon.reset()
        assert mon.context_tokens_estimate == 0
        assert len(mon._warnings_emitted) == 0


# ---------------------------------------------------------------------------
# Stage history compression
# ---------------------------------------------------------------------------


class TestStageHistoryCompression:
    def test_compress_stage_history_adds_summary(self) -> None:
        mem = ResearchSessionMemory(session_id="test-compress")
        mem.compress_stage_history("exploring", 5, 2, 1, "Conjecture A")
        assert len(mem.stage_summaries) == 1
        assert "5 conjectures tried" in mem.stage_summaries[0]
        assert "2 disproved" in mem.stage_summaries[0]
        assert "1 proved" in mem.stage_summaries[0]
        assert "Conjecture A" in mem.stage_summaries[0]

    def test_multiple_stage_compressions(self) -> None:
        mem = ResearchSessionMemory(session_id="test-multi-compress")
        mem.compress_stage_history("exploring", 3, 1, 0, "C1")
        mem.compress_stage_history("proving", 2, 0, 2, "C2")
        assert len(mem.stage_summaries) == 2


# ---------------------------------------------------------------------------
# Cache prefix on BaseAgent
# ---------------------------------------------------------------------------


class _TestAgent(BaseAgent):
    SYSTEM_PROMPT = "You are a mathematical research assistant."

    def _execute(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError


class _TestAgentNoPrompt(BaseAgent):
    def _execute(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError


class TestCachePrefix:
    def test_cache_prefix_returns_system_prompt(self) -> None:
        agent = _TestAgent(name="test-agent")
        assert agent.cache_prefix == "You are a mathematical research assistant."

    def test_cache_prefix_empty_when_no_prompt(self) -> None:
        agent = _TestAgentNoPrompt(name="test-agent-no-prompt")
        assert agent.cache_prefix == ""


# ---------------------------------------------------------------------------
# Save / Load round-trip with tiers
# ---------------------------------------------------------------------------


class TestSaveLoadWithTiers:
    def test_round_trip_preserves_tiers(self, tmp_path: Path) -> None:
        mem = ResearchSessionMemory(session_id="test-save", max_conjectures=50)
        for i in range(20):
            mem.record_conjecture(
                _make_conjecture(i),
                outcome=ConjectureOutcome.PROOF_FAILED,
                failure_reason=f"Failed attempt {i}",
            )
        mem.compress_stage_history("exploring", 20, 10, 2, "Conjecture 19")

        save_path = tmp_path / "session.json"
        mem.save(save_path)

        loaded = ResearchSessionMemory.load(save_path)
        assert loaded.session_id == "test-save"
        assert len(loaded.data.tried_conjectures) == len(mem.data.tried_conjectures)
        assert len(loaded.warm_conjectures) == len(mem.warm_conjectures)
        assert len(loaded.cold_conjectures) == len(mem.cold_conjectures)
        assert loaded.total_conjecture_count == mem.total_conjecture_count
        assert loaded.stage_summaries == mem.stage_summaries

    def test_summary_includes_tier_info(self) -> None:
        mem = ResearchSessionMemory(session_id="test-summary")
        for i in range(20):
            mem.record_conjecture(_make_conjecture(i))
        s = mem.summary()
        assert s["hot_tier"] == HOT_TIER_SIZE
        assert s["warm_tier"] == WARM_TIER_SIZE
        assert s["cold_tier"] == 20 - HOT_TIER_SIZE - WARM_TIER_SIZE
