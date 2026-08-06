"""Benchmark loaders for miniF2F v2 and PutnamBench."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from agentic_research.logging import get_logger
from agentic_research.models.eval import (
    BenchmarkSource,
    Problem,
    ProblemDifficulty,
    ProblemSet,
    ProblemSplit,
)

log = get_logger(__name__)

MINIF2F_REPO = "https://github.com/google-deepmind/miniF2F.git"
MINIF2F_LEAN4_DIR = "MiniF2F"
PUTNAM_BENCH_REPO = "https://github.com/trishullab/PutnamBench.git"
PUTNAM_LEAN4_DIR = "lean4/src"


def _infer_difficulty(name: str) -> ProblemDifficulty:
    """Infer problem difficulty from its name prefix."""
    lower = name.lower()
    for prefix, difficulty in [
        ("amc", ProblemDifficulty.AMC),
        ("aime", ProblemDifficulty.AIME),
        ("mathd", ProblemDifficulty.MATHD),
        ("imo", ProblemDifficulty.IMO),
    ]:
        if lower.startswith(prefix):
            return difficulty
    return ProblemDifficulty.UNKNOWN


def _infer_split(file_path: str) -> ProblemSplit:
    """Infer split from the file path."""
    if "Test" in file_path or "test" in file_path:
        return ProblemSplit.TEST
    return ProblemSplit.VALIDATION


def _extract_docstring(content: str, theorem_pos: int) -> str:
    """Extract the Lean 4 docstring (/-- ... -/) immediately preceding a theorem.

    Finds the last /-- ... -/ block before theorem_pos.
    """
    before = content[:theorem_pos]
    matches = list(re.finditer(r"/--\s*(.*?)\s*-/", before, re.DOTALL))
    if matches:
        last = matches[-1]
        between = before[last.end():]
        if between.strip() == "":
            lines = last.group(1).strip().splitlines()
            return "\n".join(line.strip() for line in lines)
    return ""


_HEADER_KEYWORDS = frozenset([
    "import", "open", "def", "noncomputable", "set_option", "variable",
    "instance", "section", "namespace", "attribute", "abbrev",
])


def _parse_lean4_file(path: Path, source: BenchmarkSource) -> list[Problem]:
    """Parse a Lean 4 file and extract theorem statements."""
    content = path.read_text(encoding="utf-8")
    problems: list[Problem] = []

    header_lines: list[str] = []
    theorem_pattern = re.compile(
        r"^(theorem|lemma)\s+(\w+)(.+?)(?=^(?:theorem|lemma)\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    in_docstring = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("/--"):
            in_docstring = True
            if "-/" in stripped[3:]:
                in_docstring = False
            continue
        if in_docstring:
            if "-/" in stripped:
                in_docstring = False
            continue
        if stripped.startswith("theorem") or stripped.startswith("lemma"):
            break
        first_word = stripped.split()[0] if stripped.split() else ""
        if first_word in _HEADER_KEYWORDS or stripped == "":
            header_lines.append(line)

    header = "\n".join(header_lines).strip()

    solution_defs: dict[str, str] = {}
    solution_pattern = re.compile(
        r"^((?:noncomputable\s+)?(?:abbrev|def)\s+(putnam_\w+_solution\b).+?:=\s*sorry)",
        re.MULTILINE | re.DOTALL,
    )
    for sol_match in solution_pattern.finditer(content):
        sol_block = sol_match.group(1).strip()
        sol_name = sol_match.group(2)
        problem_stem = sol_name.replace("_solution", "")
        solution_defs[problem_stem] = sol_block

    solution_answers: dict[str, str] = {}
    comment_answer_pattern = re.compile(
        r"(?:abbrev|def)\s+(\w+_solution)\b.*?:=[^\n]*sorry\s*\n\s*--\s*(.+)",
    )
    for ca_match in comment_answer_pattern.finditer(content):
        solution_answers[ca_match.group(1)] = ca_match.group(2).strip()

    for match in theorem_pattern.finditer(content):
        keyword = match.group(1)
        name = match.group(2)
        rest = match.group(3)

        full_text = f"{keyword} {name}{rest}".strip()

        statement = _extract_statement(full_text)

        natural_language = _extract_docstring(content, match.start())

        sol_def = solution_defs.get(name)
        if sol_def:
            sol_name = name + "_solution"
            answer = solution_answers.get(sol_name)
            if answer:
                sol_def = sol_def.replace(":= sorry", f":= {answer}", 1)
            statement = f"{sol_def}\n\n{statement}"

        split = _infer_split(str(path))
        difficulty = _infer_difficulty(name)

        problems.append(
            Problem(
                id=f"{source.value}/{name}",
                name=name,
                source=source,
                split=split,
                difficulty=difficulty,
                lean_header=header,
                lean_statement=statement,
                natural_language=natural_language,
                file_path=str(path),
            )
        )

    return problems


def _extract_statement(full_text: str) -> str:
    """Extract the theorem statement without the proof body.

    Handles := by, := sorry, :=\nsorry (PutnamBench), and where clauses.
    """
    text = re.sub(r'answer\s*\(([^)]+)\)', r'(\1)', full_text)
    normalized = re.sub(r":=\s+", ":= ", text)

    for marker in [":= by", ":= sorry", ":=by"]:
        idx = normalized.find(marker)
        if idx != -1:
            return normalized[:idx].strip() + " := by sorry"

    if ":= " in normalized:
        idx = normalized.find(":= ")
        return normalized[:idx].strip() + " := sorry"

    return normalized.rstrip().rstrip(":").strip() + " := by sorry"


def _clone_repo(repo_url: str, target_dir: Path) -> None:
    """Clone a git repository if it doesn't already exist."""
    if target_dir.exists() and (target_dir / ".git").exists():
        log.info("benchmark_repo_exists", path=str(target_dir))
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    log.info("cloning_benchmark_repo", url=repo_url, target=str(target_dir))

    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    log.info("benchmark_repo_cloned", path=str(target_dir))


def load_minif2f(data_dir: Path | None = None) -> ProblemSet:
    """Load the miniF2F v2 benchmark (488 problems: 244 test + 244 validation).

    Downloads the repository if not already present.
    """
    if data_dir is None:
        data_dir = Path("data/benchmarks")

    repo_dir = data_dir / "miniF2F"
    _clone_repo(MINIF2F_REPO, repo_dir)

    lean4_dir = repo_dir / MINIF2F_LEAN4_DIR
    if not lean4_dir.exists():
        raise FileNotFoundError(
            f"miniF2F Lean 4 directory not found at {lean4_dir}. "
            "The repository structure may have changed."
        )

    problems: list[Problem] = []
    for lean_file in sorted(lean4_dir.rglob("*.lean")):
        parsed = _parse_lean4_file(lean_file, BenchmarkSource.MINIF2F)
        problems.extend(parsed)

    log.info(
        "minif2f_loaded",
        total=len(problems),
        test=sum(1 for p in problems if p.split == ProblemSplit.TEST),
        valid=sum(1 for p in problems if p.split == ProblemSplit.VALIDATION),
    )

    return ProblemSet(
        name="miniF2F-v2",
        source=BenchmarkSource.MINIF2F,
        problems=problems,
    )


def load_putnam_bench(data_dir: Path | None = None) -> ProblemSet:
    """Load PutnamBench problems (672 problems).

    Stub loader — downloads the repository but parsing is deferred to a later phase.
    """
    if data_dir is None:
        data_dir = Path("data/benchmarks")

    repo_dir = data_dir / "PutnamBench"

    try:
        _clone_repo(PUTNAM_BENCH_REPO, repo_dir)
    except subprocess.CalledProcessError:
        log.warning("putnam_bench_clone_failed", msg="PutnamBench download failed — stub will return empty set")
        return ProblemSet(
            name="PutnamBench",
            source=BenchmarkSource.PUTNAM_BENCH,
            problems=[],
        )

    lean4_dir = repo_dir / PUTNAM_LEAN4_DIR
    problems: list[Problem] = []

    if lean4_dir.exists():
        for lean_file in sorted(lean4_dir.rglob("*.lean")):
            parsed = _parse_lean4_file(lean_file, BenchmarkSource.PUTNAM_BENCH)
            for p in parsed:
                p.difficulty = ProblemDifficulty.PUTNAM
            problems.extend(parsed)

    log.info("putnam_bench_loaded", total=len(problems))

    return ProblemSet(
        name="PutnamBench",
        source=BenchmarkSource.PUTNAM_BENCH,
        problems=problems,
    )
