"""Pre-built data packages for domain-specific formalization.

Each data package provides a Lean 4 preamble with correct Mathlib imports
and definitions for a specific mathematical domain, avoiding LLM
hallucination of Mathlib identifiers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_research.data_packages.dro_coupling import DROCouplingPackage


_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register a data package class by domain name."""
    def decorator(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_package(name: str) -> "DROCouplingPackage | None":
    """Look up a data package by domain name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        return None
    return cls()


def available_packages() -> list[str]:
    """Return names of all registered data packages."""
    return list(_REGISTRY.keys())


def _ensure_registered() -> None:
    """Import submodules so their @register decorators fire."""
    import agentic_research.data_packages.dro_coupling  # noqa: F401


_ensure_registered()
