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
