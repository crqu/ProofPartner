"""Lean 4 REPL wrapper for incremental code compilation.

Supports three backends:
  1. LeanDojo Dojo environment (preferred, when installed)
  2. Direct subprocess calls to ``lean`` / ``lake`` (fallback)
  3. Mock mode for testing without Lean installed
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
from typing import Any

from agentic_research.logging import get_logger
from agentic_research.models.tools import (
    CompilationResult,
    CompilationStatus,
    ProofGoal,
    ToolStatus,
)
from agentic_research.tools.base import BaseTool

log = get_logger(__name__)


class ReplBackend(str, Enum):
    LEAN_DOJO = "lean_dojo"
    SUBPROCESS = "subprocess"
    MOCK = "mock"


@dataclass
class ReplConfig:
    backend: ReplBackend = ReplBackend.MOCK
    lean_executable: str = "lean"
    timeout_seconds: int = 60
    working_dir: str | None = None
    lake_env: str | None = None
    extra_args: list[str] = field(default_factory=list)


def detect_backend() -> ReplBackend:
    """Auto-detect the best available backend."""
    try:
        import lean_dojo  # noqa: F401
        return ReplBackend.LEAN_DOJO
    except ImportError:
        pass
    if shutil.which("lean"):
        return ReplBackend.SUBPROCESS
    return ReplBackend.MOCK


def _parse_lean_errors(output: str) -> tuple[list[str], list[str]]:
    """Extract error and warning messages from Lean compiler output."""
    errors: list[str] = []
    warnings: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if " error:" in line or stripped.startswith("error"):
            errors.append(stripped)
        elif " warning:" in line or stripped.startswith("warning"):
            warnings.append(stripped)
    return errors, warnings


def _parse_goals(output: str) -> list[ProofGoal]:
    """Extract proof goals from Lean output (unsolved goals message)."""
    goals: list[ProofGoal] = []
    goal_block = re.search(r"unsolved goals\n(.*?)(?:\n\n|\Z)", output, re.DOTALL)
    if not goal_block:
        return goals

    current_hyps: list[str] = []
    for line in goal_block.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("⊢"):
            goal_text = line[1:].strip()
            goals.append(ProofGoal(goal=goal_text, hypotheses=list(current_hyps)))
            current_hyps = []
        else:
            current_hyps.append(line)

    return goals


class _MockBackend:
    """Deterministic mock: succeeds for ``sorry``-free code, fails otherwise."""

    def compile(self, code: str, timeout: int) -> CompilationResult:
        has_sorry = "sorry" in code
        has_error_marker = "-- MOCK_ERROR" in code
        if has_error_marker:
            return CompilationResult(
                status=ToolStatus.SUCCESS,
                compilation_status=CompilationStatus.ERROR,
                errors=["mock compilation error"],
                lean_output="error: mock compilation error",
            )
        if has_sorry:
            return CompilationResult(
                status=ToolStatus.SUCCESS,
                compilation_status=CompilationStatus.OK,
                warnings=["declaration uses 'sorry'"],
                lean_output="warning: declaration uses 'sorry'",
                goals=[ProofGoal(goal="<mock goal>", hypotheses=[])],
                all_goals_closed=False,
            )
        return CompilationResult(
            status=ToolStatus.SUCCESS,
            compilation_status=CompilationStatus.OK,
            lean_output="",
            all_goals_closed=True,
        )


class _SubprocessBackend:
    def __init__(self, config: ReplConfig) -> None:
        self._config = config

    def compile(self, code: str, timeout: int) -> CompilationResult:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False, dir=self._config.working_dir
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            cmd = [self._config.lean_executable, tmp_path] + self._config.extra_args
            env = os.environ.copy()
            if self._config.lake_env:
                env["LAKE_PATH"] = self._config.lake_env

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=self._config.working_dir,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            combined = f"{stdout}\n{stderr}".strip()

            errors, warnings = _parse_lean_errors(combined)
            goals = _parse_goals(combined)
            comp_status = CompilationStatus.OK if proc.returncode == 0 else CompilationStatus.ERROR
            all_closed = comp_status == CompilationStatus.OK and not goals

            return CompilationResult(
                status=ToolStatus.SUCCESS,
                compilation_status=comp_status,
                errors=errors,
                warnings=warnings,
                goals=goals,
                lean_output=combined,
                all_goals_closed=all_closed,
            )
        except subprocess.TimeoutExpired:
            return CompilationResult(
                status=ToolStatus.TIMEOUT,
                compilation_status=CompilationStatus.TIMEOUT,
                error_message=f"Lean compilation timed out after {timeout}s",
            )
        finally:
            os.unlink(tmp_path)


class _LeanDojoBackend:
    """Backend using LeanDojo's Dojo environment for incremental compilation."""

    def __init__(self, config: ReplConfig) -> None:
        self._config = config

    def compile(self, code: str, timeout: int) -> CompilationResult:
        try:
            from lean_dojo import Dojo, TacticState, ProofFinished, LeanGitRepo, Theorem  # type: ignore[import-untyped]
        except ImportError:
            return CompilationResult(
                status=ToolStatus.UNAVAILABLE,
                compilation_status=CompilationStatus.ERROR,
                error_message="lean-dojo is not installed",
            )

        try:
            repo = LeanGitRepo(
                "https://github.com/leanprover-community/mathlib4",
                "master",
            )
            theorem = Theorem(repo, "Mathlib", "placeholder")
            with Dojo(theorem, timeout=timeout) as (dojo, init_state):
                state = dojo.run_tac(init_state, code)
                if isinstance(state, ProofFinished):
                    return CompilationResult(
                        status=ToolStatus.SUCCESS,
                        compilation_status=CompilationStatus.OK,
                        all_goals_closed=True,
                    )
                if isinstance(state, TacticState):
                    goals = [
                        ProofGoal(goal=g) for g in str(state).split("\n") if g.strip()
                    ]
                    return CompilationResult(
                        status=ToolStatus.SUCCESS,
                        compilation_status=CompilationStatus.OK,
                        goals=goals,
                        all_goals_closed=False,
                    )
                return CompilationResult(
                    status=ToolStatus.SUCCESS,
                    compilation_status=CompilationStatus.ERROR,
                    errors=[str(state)],
                    lean_output=str(state),
                )
        except Exception as exc:
            return CompilationResult(
                status=ToolStatus.ERROR,
                compilation_status=CompilationStatus.ERROR,
                error_message=str(exc),
            )


_ReplBackendImpl = _MockBackend | _SubprocessBackend | _LeanDojoBackend

_BACKENDS: dict[ReplBackend, Callable[[ReplConfig], _ReplBackendImpl]] = {
    ReplBackend.MOCK: lambda cfg: _MockBackend(),
    ReplBackend.LEAN_DOJO: lambda cfg: _LeanDojoBackend(cfg),
    ReplBackend.SUBPROCESS: lambda cfg: _SubprocessBackend(cfg),
}


class LeanRepl(BaseTool):
    """Lean 4 REPL tool — compile code and get structured feedback."""

    _name = "lean_repl"

    def __init__(self, config: ReplConfig | None = None) -> None:
        if config is None:
            config = ReplConfig(backend=detect_backend())
        self._config = config
        self._backend: _MockBackend | _SubprocessBackend | _LeanDojoBackend = _BACKENDS[config.backend](config)
        log.info("lean_repl_init", backend=config.backend.value)

    @property
    def backend(self) -> ReplBackend:
        return self._config.backend

    def execute(self, code: str) -> CompilationResult:
        result = super().execute(code)
        if isinstance(result, CompilationResult):
            return result
        return CompilationResult(
            status=result.status,
            compilation_status=CompilationStatus.ERROR,
            error_message=result.error_message,
        )

    def _run(self, input_data: Any) -> CompilationResult:
        code = str(input_data)
        return self._backend.compile(code, self._config.timeout_seconds)
