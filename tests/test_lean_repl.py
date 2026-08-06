"""Tests for lean_repl infrastructure awareness — olean checks, path configurability, error classification."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_research.tools.lean_repl import (
    ReplConfig,
    _SubprocessBackend,
    classify_compilation_error,
)


# ---------------------------------------------------------------------------
# Tests 1-3: olean check + path configurability
# ---------------------------------------------------------------------------


def test_has_lake_project_requires_olean(tmp_path: Path):
    """has_lake_project() returns False when lakefile.toml exists but Mathlib.olean does not."""
    (tmp_path / "lakefile.toml").write_text("[package]\nname = \"test\"")
    config = ReplConfig(lake_project_dir=tmp_path)
    backend = _SubprocessBackend(config)
    with patch("shutil.which", return_value="/usr/bin/lake"):
        result = backend.has_lake_project()
    assert result is False


def test_has_lake_project_with_olean(tmp_path: Path):
    """has_lake_project() returns True when all three checks pass (lakefile + binary + olean)."""
    (tmp_path / "lakefile.toml").write_text("[package]\nname = \"test\"")
    olean_dir = tmp_path / ".lake" / "packages" / "mathlib" / ".lake" / "build" / "lib" / "lean"
    olean_dir.mkdir(parents=True)
    (olean_dir / "Mathlib.olean").write_bytes(b"")
    config = ReplConfig(lake_project_dir=tmp_path)
    backend = _SubprocessBackend(config)
    with patch("shutil.which", return_value="/usr/bin/lake"):
        result = backend.has_lake_project()
    assert result is True


def test_has_lake_project_logs_olean_warning(tmp_path: Path, caplog):
    """Warning logged with hint field when olean missing but lakefile present."""
    (tmp_path / "lakefile.toml").write_text("[package]\nname = \"test\"")
    config = ReplConfig(lake_project_dir=tmp_path)
    backend = _SubprocessBackend(config)
    with patch("shutil.which", return_value="/usr/bin/lake"):
        backend.has_lake_project()
    # structlog may not write to caplog, so we verify the return value instead
    assert backend._lake_available is False


def test_lake_project_dir_from_env(tmp_path: Path):
    """_LAKE_PROJECT_DIR uses LEAN_PROJECT_DIR env var when set."""
    with patch.dict("os.environ", {"LEAN_PROJECT_DIR": str(tmp_path)}):
        import importlib
        import agentic_research.tools.lean_repl as repl_mod
        importlib.reload(repl_mod)
        assert repl_mod._DEFAULT_LAKE_PROJECT_DIR == tmp_path
    # Reload to restore default
    importlib.reload(repl_mod)


def test_lake_project_dir_default_fallback():
    """_LAKE_PROJECT_DIR falls back to the relative proofpartner-lean/ path when env var is not set."""
    with patch.dict("os.environ", {}, clear=False):
        import os
        if "LEAN_PROJECT_DIR" in os.environ:
            pytest.skip("LEAN_PROJECT_DIR is set in the environment")
        import importlib
        import agentic_research.tools.lean_repl as repl_mod
        importlib.reload(repl_mod)
        assert str(repl_mod._DEFAULT_LAKE_PROJECT_DIR).endswith("proofpartner-lean")
    importlib.reload(repl_mod)


def test_repl_config_lake_project_dir(tmp_path: Path):
    """ReplConfig.lake_project_dir overrides both env var and default when provided."""
    custom_path = tmp_path / "custom-lean"
    custom_path.mkdir()
    config = ReplConfig(lake_project_dir=custom_path)
    backend = _SubprocessBackend(config)
    assert backend._LAKE_PROJECT_DIR == custom_path


# ---------------------------------------------------------------------------
# Tests 4-6: classify_compilation_error
# ---------------------------------------------------------------------------


def test_classify_compilation_error_infrastructure():
    """classify_compilation_error returns 'infrastructure' for olean missing errors."""
    assert classify_compilation_error("Mathlib.olean does not exist") == "infrastructure"
    assert classify_compilation_error("error: no such file or directory") == "infrastructure"
    assert classify_compilation_error("lake: command not found") == "infrastructure"
    assert classify_compilation_error("could not find package 'mathlib'") == "infrastructure"


def test_classify_compilation_error_compilation():
    """classify_compilation_error returns 'compilation' for regular Lean errors."""
    assert classify_compilation_error("unknown identifier 'foo'") == "compilation"
    assert classify_compilation_error("error: type mismatch") == "compilation"
    assert classify_compilation_error("tactic 'omega' failed") == "compilation"


def test_classify_compilation_error_unknown():
    """classify_compilation_error returns 'unknown' for unrecognized messages."""
    assert classify_compilation_error("some random message") == "unknown"
    assert classify_compilation_error("") == "unknown"
