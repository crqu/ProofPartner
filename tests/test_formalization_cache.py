"""Tests for the cross-session formalization cache."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agentic_research.cache.formalization_cache import CachedFormalization, FormalizationCache


def _make_entry(
    type_name: str = "MyType",
    lean_toolchain: str = "v4.8.0",
    lean_code: str = "structure MyType where\n  x : Nat",
) -> CachedFormalization:
    return CachedFormalization(
        type_name=type_name,
        type_signature="structure MyType where",
        lean_code=lean_code,
        lean_toolchain=lean_toolchain,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


class TestFormalizationCache:
    def test_cache_roundtrip(self, tmp_path):
        cache = FormalizationCache(db_path=tmp_path / "cache.db")
        entry = _make_entry()
        cache.put(entry)
        result = cache.get("MyType", "v4.8.0")
        assert result is not None
        assert result.type_name == "MyType"
        assert result.lean_code == entry.lean_code
        assert result.lean_toolchain == "v4.8.0"
        cache.close()

    def test_cache_miss_returns_none(self, tmp_path):
        cache = FormalizationCache(db_path=tmp_path / "cache.db")
        result = cache.get("NonExistent", "v4.8.0")
        assert result is None
        cache.close()

    def test_cache_toolchain_filter(self, tmp_path):
        cache = FormalizationCache(db_path=tmp_path / "cache.db")
        entry = _make_entry(lean_toolchain="v4.8.0")
        cache.put(entry)
        result = cache.get("MyType", "v4.9.0")
        assert result is None
        cache.close()

    def test_cache_reuse_count_increments(self, tmp_path):
        cache = FormalizationCache(db_path=tmp_path / "cache.db")
        entry = _make_entry()
        cache.put(entry)
        r1 = cache.get("MyType", "v4.8.0")
        assert r1 is not None
        assert r1.reuse_count == 1
        r2 = cache.get("MyType", "v4.8.0")
        assert r2 is not None
        assert r2.reuse_count == 2
        cache.close()

    def test_cache_invalidate_toolchain(self, tmp_path):
        cache = FormalizationCache(db_path=tmp_path / "cache.db")
        cache.put(_make_entry(lean_toolchain="v4.8.0"))
        cache.put(_make_entry(type_name="OtherType", lean_toolchain="v4.8.0"))
        count = cache.invalidate_toolchain("v4.8.0")
        assert count == 2
        assert cache.get("MyType", "v4.8.0") is None
        cache.close()

    def test_cache_upsert(self, tmp_path):
        cache = FormalizationCache(db_path=tmp_path / "cache.db")
        cache.put(_make_entry(lean_code="version1"))
        cache.put(_make_entry(lean_code="version2"))
        entries = cache.list_entries()
        assert len(entries) == 1
        assert entries[0].lean_code == "version2"
        cache.close()

    def test_cache_multiple_types(self, tmp_path):
        cache = FormalizationCache(db_path=tmp_path / "cache.db")
        cache.put(_make_entry(type_name="TypeA"))
        cache.put(_make_entry(type_name="TypeB"))
        cache.put(_make_entry(type_name="TypeC"))
        entries = cache.list_entries()
        assert len(entries) == 3
        cache.close()

    def test_cache_default_path(self):
        with patch("agentic_research.cache.formalization_cache.Path.mkdir"):
            with patch("agentic_research.cache.formalization_cache.sqlite3.connect") as mock_conn:
                mock_conn.return_value = MagicMock()
                mock_conn.return_value.row_factory = None
                FormalizationCache(db_path=None)
                call_args = mock_conn.call_args[0][0]
                assert ".proofpartner" in call_args
                assert "formalization_cache.db" in call_args

    def test_cache_cross_session(self, tmp_path):
        """Simulate cross-session reuse by creating two separate cache instances."""
        db_path = tmp_path / "cache.db"
        cache1 = FormalizationCache(db_path=db_path)
        cache1.put(_make_entry())
        cache1.close()

        cache2 = FormalizationCache(db_path=db_path)
        result = cache2.get("MyType", "v4.8.0")
        assert result is not None
        assert result.lean_code == "structure MyType where\n  x : Nat"
        cache2.close()

    def test_cache_injection_into_prior_definitions(self, tmp_path):
        """Integration test: cached type skips auction in FormalizationPipeline."""
        from agentic_research.models.formalization import (
            TypeCandidate,
            TypeDependencyGraph,
            TypePlan,
        )

        db_path = tmp_path / "cache.db"
        cache = FormalizationCache(db_path=db_path)
        cache.put(_make_entry(type_name="CachedType", lean_code="-- cached code"))

        llm = MagicMock()
        lean_repl = MagicMock()
        lean_search = MagicMock()

        from agentic_research.pipelines.formalization import FormalizationPipeline

        pipeline = FormalizationPipeline(
            llm_client=llm,
            lean_repl=lean_repl,
            lean_search=lean_search,
            formalization_cache=cache,
            lean_toolchain="v4.8.0",
        )

        candidate = TypeCandidate(name="CachedType", informal_description="test type")
        type_plan = TypePlan(
            conjecture_statement="test conjecture",
            candidates=[candidate],
            dependency_graph=TypeDependencyGraph(),
        )

        result = pipeline._run_type_formalization(type_plan, {})

        assert len(result.accepted_types) == 1
        assert result.accepted_types[0].type_name == "CachedType"
        assert result.accepted_types[0].lean_code == "-- cached code"
        cache.close()

    def test_cache_population_on_acceptance(self, tmp_path):
        """Integration test: auction acceptance populates cache."""
        from agentic_research.models.formalization import (
            AuctionResult,
            AuctionVerdict,
            TypeCandidate,
            TypeDependencyGraph,
            TypeFormalizationCandidate,
            TypePlan,
        )

        db_path = tmp_path / "cache.db"
        cache = FormalizationCache(db_path=db_path)

        llm = MagicMock()
        lean_repl = MagicMock()
        lean_search = MagicMock()
        search_result_mock = MagicMock()
        search_result_mock.entries = ["something"]
        lean_search.execute.return_value = search_result_mock

        from agentic_research.pipelines.formalization import FormalizationPipeline

        pipeline = FormalizationPipeline(
            llm_client=llm,
            lean_repl=lean_repl,
            lean_search=lean_search,
            formalization_cache=cache,
            lean_toolchain="v4.8.0",
        )

        winner = TypeFormalizationCandidate(
            candidate_id=1,
            type_name="NewType",
            lean_code="structure NewType where\n  val : Int",
            compiles=True,
        )
        auction_result = AuctionResult(
            type_name="NewType",
            verdict=AuctionVerdict.ACCEPTED,
            winning_candidate=winner,
        )

        with patch.object(pipeline, "_auction_type", return_value=auction_result):
            candidate = TypeCandidate(name="NewType", informal_description="a new type")
            type_plan = TypePlan(
                conjecture_statement="test conjecture",
                candidates=[candidate],
                dependency_graph=TypeDependencyGraph(),
            )
            pipeline._run_type_formalization(type_plan, {})

        cached = cache.get("NewType", "v4.8.0")
        assert cached is not None
        assert cached.lean_code == "structure NewType where\n  val : Int"
        cache.close()
