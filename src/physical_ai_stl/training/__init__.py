# ruff: noqa: I001
from __future__ import annotations

"""Public entrypoints for :mod:`physical_ai_stl.training`.

This small shim provides a *lazy* import layer around the :mod:`grids` utilities
so users can write::

    from physical_ai_stl.training import grid1d, grid2d

without importing heavy optional dependencies (e.g., :mod:`torch`) until the
attributes are actually accessed.  The implementation follows :pep:`562`
(module-level ``__getattr__``/``__dir__``) and re-exports a curated surface
API while keeping import time tiny.

Design goals
============
- **Friendly errors.** If PyTorch is missing, we raise a clear, actionable
  message suggesting the extra: ``pip install "physical-ai-stl[torch]"``.
- **Zero overhead on subsequent lookups.** Resolved attributes are cached in
  ``globals()``.
- **Great editor support.** Under :data:`typing.TYPE_CHECKING` we provide proper
  imports so that static analyzers and IDEs see the real symbols.

References
----------
The lazy import mechanism uses PEP 562 (``__getattr__`` and ``__dir__`` at the
module level).
"""

from difflib import get_close_matches
import importlib
from typing import Any, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Lazily importable submodules
# ---------------------------------------------------------------------------

# Map of lazily importable submodules.
_LAZY_MODULES: dict[str, str] = {
    "grids": "physical_ai_stl.training.grids",
}

# Forwarded attributes: provide nice `from physical_ai_stl.training import grid1d`.
# These will be resolved from the `grids` module on first access.
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    # original API
    "grid1d": ("grids", "grid1d"),
    "grid2d": ("grids", "grid2d"),
    # new generators / helpers
    "grid3d": ("grids", "grid3d"),
    "spacing1d": ("grids", "spacing1d"),
    "spacing2d": ("grids", "spacing2d"),
    # samplers
    "sample_interior_1d": ("grids", "sample_interior_1d"),
    "sample_interior_2d": ("grids", "sample_interior_2d"),
    "sample_boundary_1d": ("grids", "sample_boundary_1d"),
    "sample_boundary_2d": ("grids", "sample_boundary_2d"),
    # simple domains
    "Box1D": ("grids", "Box1D"),
    "Box2D": ("grids", "Box2D"),
}

# A few ergonomic aliases we support at runtime (kept out of __all__).
# These let users write common variations without us committing to them as API.
_ALIASES: dict[str, str] = {
    # casing / underscores
    "Grid1D": "grid1d",
    "Grid2D": "grid2d",
    "Grid3D": "grid3d",
    "spacing_1d": "spacing1d",
    "spacing_2d": "spacing2d",
    "sample_interior1d": "sample_interior_1d",
    "sample_interior2d": "sample_interior_2d",
    "sample_boundary1d": "sample_boundary_1d",
    "sample_boundary2d": "sample_boundary_2d",
    "box1d": "Box1D",
    "box2d": "Box2D",
}

# ---------------------------------------------------------------------------
# What star-import exposes
# ---------------------------------------------------------------------------

# What `from physical_ai_stl.training import *` exposes.
# (Note: star-import will import `grids` once to retrieve forwarded names.)
__all__ = sorted({*list(_LAZY_MODULES.keys()), *list(_FORWARD_ATTRS.keys())})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_module(mod_path: str):  # pragma: no cover - trivial wrapper
    """Import *mod_path* with a friendly hint if optional deps are missing."""
    try:
        return importlib.import_module(mod_path)
    except ImportError as e:  # Provide a friendlier nudge for torch missing.
        msg = str(e).lower()
        if "torch" in msg or "pytorch" in msg:
            raise ImportError(
                "The 'grids' utilities require PyTorch. "
                'Install the extra with: pip install "physical-ai-stl[torch]"'
            ) from e
        raise


def _resolve_forward(name: str) -> Any:
    """Resolve a forwarded attribute *name* and cache it in ``globals()``."""
    mod_name, attr = _FORWARD_ATTRS[name]
    module = __getattr__(mod_name)  # ensure submodule is imported & cached
    value = getattr(module, attr)
    globals()[name] = value  # cache for subsequent lookups
    return value


# ---------------------------------------------------------------------------
# PEP 562 hooks
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:  # pragma: no cover - exercised implicitly
    # Submodule?
    if name in _LAZY_MODULES:
        module = _import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    # Aliases supported at runtime (not part of the public API)
    if name in _ALIASES:
        canonical = _ALIASES[name]
        return __getattr__(canonical)
    # Forwarded attribute?
    if name in _FORWARD_ATTRS:
        return _resolve_forward(name)
    # Nice error with a "did you mean" suggestion.
    candidates = list(__all__) + list(_ALIASES.keys())
    suggestion = get_close_matches(name, candidates, n=1)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.{hint}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    # Expose both already-bound globals and the lazy attributes
    return sorted(set(globals().keys()) | set(__all__))


# Help IDEs/type-checkers without paying runtime import cost.
if TYPE_CHECKING:  # pragma: no cover
    from . import grids as grids  # noqa: F401
    from .grids import (  # noqa: F401
        Box1D,
        Box2D,
        grid1d,
        grid2d,
        grid3d,
        sample_boundary_1d,
        sample_boundary_2d,
        sample_interior_1d,
        sample_interior_2d,
        spacing1d,
        spacing2d,
    )
