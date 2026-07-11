"""DRO coupling data package — pre-built Lean 4 preamble for
distributionally robust optimization formalization.

Follows the Zhang-Yang-Gao finite-support proof strategy which reduces
DRO duality to one-dimensional Legendre-Fenchel conjugation (no
Kantorovich duality or Sion minimax needed).

Definitions use Mathlib's Measure.fst/Measure.snd for marginal constraints,
ProbabilityMeasure for probability, and iSup/iInf lattice combinators
for suprema/infima.
"""

from __future__ import annotations

import re

from agentic_research.data_packages import register

LEAN_PREAMBLE = """\
import Mathlib

open MeasureTheory Set ENNReal

noncomputable section

/-!
# DRO Coupling Definitions

Coupling-based definitions for Wasserstein distance and balls,
grounded in Mathlib's measure theory library.
-/

variable {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]

/-- A coupling of two probability measures is a joint measure whose
    marginals (via Measure.fst and Measure.snd) match the given measures. -/
structure Coupling (μ ν : Measure Ω) where
  joint : Measure (Ω × Ω)
  is_probability : IsProbabilityMeasure joint
  fst_marginal : joint.fst = μ
  snd_marginal : joint.snd = ν

/-- Wasserstein distance between two measures, defined as the infimum of
    expected transport cost over all couplings. Uses iInf combinator. -/
def wassersteinDist (cost : Ω → Ω → ℝ≥0∞) (μ ν : Measure Ω) : ℝ≥0∞ :=
  iInf fun (γ : Coupling μ ν) =>
    ∫⁻ p, cost p.1 p.2 ∂γ.joint

/-- Wasserstein ball of radius r around reference measure P — the set of
    all measures within Wasserstein distance r. -/
def wassersteinBall (cost : Ω → Ω → ℝ≥0∞) (P : Measure Ω) (r : ℝ≥0∞) :
    Set (Measure Ω) :=
  {Q | wassersteinDist cost P Q ≤ r}

/-! ## Pre-built axioms for standard properties -/

-- Identity coupling: the diagonal measure couples P with itself
axiom identity_coupling_exists {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  (cost : Ω → Ω → ℝ≥0∞) (P : Measure Ω) [IsProbabilityMeasure P] :
  ∃ (γ : Coupling P P), ∫⁻ p, cost p.1 p.2 ∂γ.joint = 0

-- Wasserstein distance: self-distance is zero
axiom wassersteinDist_self {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  (cost : Ω → Ω → ℝ≥0∞) (P : Measure Ω) [IsProbabilityMeasure P]
  (hcost : ∀ x, cost x x = 0) :
  wassersteinDist cost P P = 0

-- Wasserstein distance: non-negativity (trivial from ENNReal)
axiom wassersteinDist_nonneg {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  (cost : Ω → Ω → ℝ≥0∞) (P Q : Measure Ω) :
  0 ≤ wassersteinDist cost P Q

-- Wasserstein ball: reference measure is in its own ball
axiom self_mem_wassersteinBall {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  (cost : Ω → Ω → ℝ≥0∞) (P : Measure Ω) [IsProbabilityMeasure P]
  (r : ℝ≥0∞) (hcost : ∀ x, cost x x = 0) :
  P ∈ wassersteinBall cost P r

-- Wasserstein ball: monotonicity in radius
axiom wassersteinBall_mono {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  (cost : Ω → Ω → ℝ≥0∞) (P : Measure Ω) {r s : ℝ≥0∞} (hrs : r ≤ s) :
  wassersteinBall cost P r ⊆ wassersteinBall cost P s

-- Wasserstein ball: membership characterization
axiom mem_wassersteinBall_iff {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  (cost : Ω → Ω → ℝ≥0∞) (P Q : Measure Ω) (r : ℝ≥0∞) :
  Q ∈ wassersteinBall cost P r ↔ wassersteinDist cost P Q ≤ r

-- Coupling: marginal constraints
axiom coupling_fst {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  (μ ν : Measure Ω) (γ : Coupling μ ν) :
  γ.joint.fst = μ

axiom coupling_snd {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  (μ ν : Measure Ω) (γ : Coupling μ ν) :
  γ.joint.snd = ν

-- Wasserstein distance: triangle inequality
axiom wassersteinDist_triangle {Ω : Type*} [MeasurableSpace Ω] [TopologicalSpace Ω]
  [MetricSpace Ω] (cost : Ω → Ω → ℝ≥0∞) (P Q R : Measure Ω) :
  wassersteinDist cost P R ≤ wassersteinDist cost P Q + wassersteinDist cost Q R

end
"""

MATHLIB_IMPORTS: list[str] = [
    "Mathlib.MeasureTheory.Measure.MeasureSpace",
    "Mathlib.MeasureTheory.Measure.Prod",
    "Mathlib.MeasureTheory.Integral.Lebesgue",
    "Mathlib.Order.CompleteLattice",
    "Mathlib.Topology.MetricSpace.Basic",
]

DESCRIPTION = (
    "Coupling-based Wasserstein distance and ball definitions for DRO, "
    "using Mathlib's Measure.fst/Measure.snd marginal constraints and "
    "iInf lattice combinator."
)


def _parse_axioms(preamble: str) -> dict[str, str]:
    """Extract axiom name -> full declaration from the preamble text."""
    axioms: dict[str, str] = {}
    lines = preamble.split("\n")
    i = 0
    while i < len(lines):
        m = re.match(r"^axiom\s+(\w+)\b", lines[i])
        if m:
            name = m.group(1)
            decl_lines = [lines[i]]
            i += 1
            while i < len(lines) and lines[i].startswith("  "):
                decl_lines.append(lines[i])
                i += 1
            axioms[name] = "\n".join(decl_lines)
        else:
            i += 1
    return axioms


LEAN_AXIOMS: dict[str, str] = _parse_axioms(LEAN_PREAMBLE)

_AXIOM_KEYWORDS: dict[str, list[str]] = {
    "identity_coupling_exists": ["identity", "coupling", "diagonal", "couples"],
    "wassersteinDist_self": ["wasserstein", "distance", "self", "zero"],
    "wassersteinDist_nonneg": ["wasserstein", "distance", "nonneg", "non-negative"],
    "self_mem_wassersteinBall": ["wasserstein", "ball", "self", "reference", "member"],
    "wassersteinBall_mono": ["wasserstein", "ball", "monoton", "radius"],
    "mem_wassersteinBall_iff": ["wasserstein", "ball", "membership", "characteriz"],
    "coupling_fst": ["coupling", "marginal", "first", "fst"],
    "coupling_snd": ["coupling", "marginal", "second", "snd"],
    "wassersteinDist_triangle": ["wasserstein", "distance", "triangle", "inequality"],
}


@register("dro_coupling")
class DROCouplingPackage:
    """Pre-built Lean 4 preamble for DRO formalization."""

    name: str = "dro_coupling"
    description: str = DESCRIPTION

    @staticmethod
    def lean_preamble() -> str:
        return LEAN_PREAMBLE

    @staticmethod
    def mathlib_imports() -> list[str]:
        return list(MATHLIB_IMPORTS)

    @staticmethod
    def provided_definitions() -> list[str]:
        return ["Coupling", "wassersteinDist", "wassersteinBall"]

    @staticmethod
    def provided_axioms() -> dict[str, str]:
        """Return mapping of axiom names to their full Lean 4 declarations."""
        return dict(LEAN_AXIOMS)

    @staticmethod
    def axiom_keywords() -> dict[str, list[str]]:
        """Return mapping of axiom names to matching keywords."""
        return dict(_AXIOM_KEYWORDS)
