"""Tests for Mathlib integration via the proofpartner-lean lake project."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_research.models.tools import CompilationStatus
from agentic_research.tools.lean_repl import LeanRepl, ReplBackend, ReplConfig, _SubprocessBackend

LEAN_AVAILABLE = shutil.which("lean") is not None
LAKE_PROJECT = Path(__file__).parent.parent / "proofpartner-lean"


def test_lakefile_exists():
    """The proofpartner-lean project ships with a lakefile.toml."""
    lakefile = LAKE_PROJECT / "lakefile.toml"
    assert lakefile.is_file(), f"Missing {lakefile}"
    content = lakefile.read_text()
    assert "mathlib" in content
    assert 'scope = "leanprover-community"' in content


def test_lean_toolchain_exists():
    """The lean-toolchain file pins the Lean version."""
    toolchain = LAKE_PROJECT / "lean-toolchain"
    assert toolchain.is_file(), f"Missing {toolchain}"
    content = toolchain.read_text().strip()
    assert content.startswith("leanprover/lean4:")


def test_compile_uses_temp_file():
    """_compile_with_lake uses a unique temp file, not shared Scratch.lean."""
    config = ReplConfig(backend=ReplBackend.SUBPROCESS)
    backend = _SubprocessBackend(config)
    lean_dir = LAKE_PROJECT / "ProofPartner"
    before = set(lean_dir.glob("*.lean"))
    # After _compile_with_lake (mocked subprocess), no new .lean files should remain
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("P", (), {
            "stdout": "", "stderr": "", "returncode": 0,
        })()
        backend._lake_available = True
        backend._compile_with_lake("-- test", timeout=10)
    after = set(lean_dir.glob("*.lean"))
    assert before == after, "Temp file was not cleaned up"


def test_subprocess_backend_detects_lake_project():
    """_SubprocessBackend.has_lake_project() returns True when lakefile and lake binary exist."""
    config = ReplConfig(backend=ReplBackend.SUBPROCESS)
    backend = _SubprocessBackend(config)
    with patch("shutil.which", return_value="/usr/bin/lake"):
        assert backend.has_lake_project() is True


def test_subprocess_backend_fallback_without_lake_project():
    """When lake project is absent, has_lake_project() returns False."""
    config = ReplConfig(backend=ReplBackend.SUBPROCESS)
    backend = _SubprocessBackend(config)
    fake_path = Path("/nonexistent/proofpartner-lean")
    with patch.object(type(backend), "_LAKE_PROJECT_DIR", fake_path):
        backend._lake_available = None
        assert backend.has_lake_project() is False


def test_subprocess_backend_fallback_without_lake_binary():
    """When lake binary is missing, has_lake_project() returns False."""
    config = ReplConfig(backend=ReplBackend.SUBPROCESS)
    backend = _SubprocessBackend(config)
    with patch("shutil.which", return_value=None):
        backend._lake_available = None
        assert backend.has_lake_project() is False


@pytest.mark.lean_required
@pytest.mark.skipif(not LEAN_AVAILABLE, reason="Lean 4 not installed")
@pytest.mark.timeout(120)
def test_mathlib_import_compiles():
    """Compile a Lean file importing a Mathlib module via the lake project."""
    repl = LeanRepl(ReplConfig(backend=ReplBackend.SUBPROCESS))
    code = (
        "import Mathlib.Topology.MetricSpace.Lipschitz\n"
        "#check LipschitzWith\n"
    )
    result = repl.execute(code)
    assert result.compilation_status == CompilationStatus.OK, (
        f"Mathlib compilation failed: {result.errors}"
    )
