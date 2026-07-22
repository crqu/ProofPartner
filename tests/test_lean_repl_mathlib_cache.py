"""Tests for _SubprocessBackend._ensure_mathlib_cache()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import subprocess

import pytest

from agentic_research.tools.lean_repl import ReplConfig, ReplBackend, _SubprocessBackend


@pytest.fixture
def backend():
    cfg = ReplConfig(backend=ReplBackend.SUBPROCESS)
    return _SubprocessBackend(cfg)


class TestEnsureMathlibCache:
    def test_skips_when_oleans_present(self, backend: _SubprocessBackend, tmp_path: Path):
        with patch.object(type(backend), "_MATHLIB_OLEANS_DIR", tmp_path):
            backend._ensure_mathlib_cache()

        assert backend._mathlib_cache_checked is True

    def test_runs_lake_exe_cache_get_when_missing(self, backend: _SubprocessBackend, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        proc_result = MagicMock(returncode=0, stdout="", stderr="")
        with (
            patch.object(type(backend), "_MATHLIB_OLEANS_DIR", missing),
            patch.object(type(backend), "_LAKE_PROJECT_DIR", tmp_path),
            patch("agentic_research.tools.lean_repl.subprocess.run", return_value=proc_result) as mock_run,
        ):
            backend._ensure_mathlib_cache()

        mock_run.assert_called_once_with(
            ["lake", "exe", "cache", "get"],
            capture_output=True,
            text=True,
            timeout=_SubprocessBackend._CACHE_DOWNLOAD_TIMEOUT,
            cwd=str(tmp_path),
        )

    def test_only_runs_once(self, backend: _SubprocessBackend, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        proc_result = MagicMock(returncode=0, stdout="", stderr="")
        with (
            patch.object(type(backend), "_MATHLIB_OLEANS_DIR", missing),
            patch.object(type(backend), "_LAKE_PROJECT_DIR", tmp_path),
            patch("agentic_research.tools.lean_repl.subprocess.run", return_value=proc_result) as mock_run,
        ):
            backend._ensure_mathlib_cache()
            backend._ensure_mathlib_cache()

        assert mock_run.call_count == 1

    def test_handles_download_failure(self, backend: _SubprocessBackend, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        proc_result = MagicMock(returncode=1, stdout="", stderr="download failed")
        with (
            patch.object(type(backend), "_MATHLIB_OLEANS_DIR", missing),
            patch.object(type(backend), "_LAKE_PROJECT_DIR", tmp_path),
            patch("agentic_research.tools.lean_repl.subprocess.run", return_value=proc_result),
        ):
            backend._ensure_mathlib_cache()

        assert backend._mathlib_cache_checked is True

    def test_handles_timeout(self, backend: _SubprocessBackend, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        with (
            patch.object(type(backend), "_MATHLIB_OLEANS_DIR", missing),
            patch.object(type(backend), "_LAKE_PROJECT_DIR", tmp_path),
            patch(
                "agentic_research.tools.lean_repl.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="lake", timeout=600),
            ),
        ):
            backend._ensure_mathlib_cache()

        assert backend._mathlib_cache_checked is True

    def test_handles_os_error(self, backend: _SubprocessBackend, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        with (
            patch.object(type(backend), "_MATHLIB_OLEANS_DIR", missing),
            patch.object(type(backend), "_LAKE_PROJECT_DIR", tmp_path),
            patch(
                "agentic_research.tools.lean_repl.subprocess.run",
                side_effect=OSError("lake not found"),
            ),
        ):
            backend._ensure_mathlib_cache()

        assert backend._mathlib_cache_checked is True

    def test_compile_with_lake_calls_ensure_cache(self, backend: _SubprocessBackend, tmp_path: Path):
        lean_dir = tmp_path / "ProofPartner"
        lean_dir.mkdir()
        with (
            patch.object(type(backend), "_LAKE_PROJECT_DIR", tmp_path),
            patch.object(backend, "_ensure_mathlib_cache") as mock_ensure,
            patch("agentic_research.tools.lean_repl.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            backend._compile_with_lake("import Mathlib\n-- test", timeout=30)

        mock_ensure.assert_called_once()

    def test_compile_with_lake_skips_cache_without_mathlib(self, backend: _SubprocessBackend, tmp_path: Path):
        lean_dir = tmp_path / "ProofPartner"
        lean_dir.mkdir()
        with (
            patch.object(type(backend), "_LAKE_PROJECT_DIR", tmp_path),
            patch.object(backend, "_ensure_mathlib_cache") as mock_ensure,
            patch("agentic_research.tools.lean_repl.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            backend._compile_with_lake("theorem x : True := trivial", timeout=30)

        mock_ensure.assert_not_called()
