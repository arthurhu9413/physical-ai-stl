# ruff: noqa: I001
from __future__ import annotations

"""
Lightweight physics helpers (PDE/ODE demos) with zero-cost imports.

- **Lazy submodules**: `diffusion1d`, `heat2d`
- **Forwarded helpers** (resolved on first access):
  - 1‑D diffusion: `pde_residual`, `residual_loss`, `boundary_loss`,
    `Interval1D`, `sine_ic`, `sine_solution`, `make_dirichlet_mask_1d`
  - 2‑D heat: `residual_heat2d`, `bc_ic_heat2d`, `SquareDomain2D`,
    `gaussian_ic`, `make_dirichlet_mask`

### Power-user toggles (optional)
Set via environment variables *before* importing this package:

- `PHYSICAL_AI_STL_EAGER_IMPORTS=1`
    Eagerly import submodules and resolve forwarded attributes.
    Helpful for IDE indexing or interactive exploration.
- `PHYSICAL_AI_STL_STRICT_INIT=1`
    Validates that all forwarded attributes exist (implies eager imports).
    Fails fast if an export mapping is incorrect.

These toggles never run during normal use; by default everything is lazy
and incurs near‑zero overhead at import time.
"""

from typing import Any, TYPE_CHECKING
import importlib
import os
from difflib import get_close_matches

# ----- Lazily exposed submodules ------------------------------------------------

_LAZY_MODULES: dict[str, str] = {
    "diffusion1d": "physical_ai_stl.physics.diffusion1d",
    "heat2d": "physical_ai_stl.physics.heat2d",
}

# Forwarded attributes: provide nice `from physical_ai_stl.physics import pde_residual`.
# These are fetched from their submodules on first access and then cached here.
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    # 1‑D diffusion
    "pde_residual": ("diffusion1d", "pde_residual"),
    "residual_loss": ("diffusion1d", "residual_loss"),
    "boundary_loss": ("diffusion1d", "boundary_loss"),
    "Interval1D": ("diffusion1d", "Interval1D"),
    "sine_ic": ("diffusion1d", "sine_ic"),
    "sine_solution": ("diffusion1d", "sine_solution"),
    "make_dirichlet_mask_1d": ("diffusion1d", "make_dirichlet_mask_1d"),
    # 2‑D heat
    "residual_heat2d": ("heat2d", "residual_heat2d"),
    "bc_ic_heat2d": ("heat2d", "bc_ic_heat2d"),
    "SquareDomain2D": ("heat2d", "SquareDomain2D"),
    "gaussian_ic": ("heat2d", "gaussian_ic"),
    "make_dirichlet_mask": ("heat2d", "make_dirichlet_mask"),
}

# What `from physical_ai_stl.physics import *` exposes.
__all__ = sorted({*list(_LAZY_MODULES.keys()), *list(_FORWARD_ATTRS.keys())})


# ----- Internal helpers ---------------------------------------------------------

def _truthy_env(var: str) -> bool:  # pragma: no cover - trivial
    return os.getenv(var, "").strip().lower() in {"1", "true", "yes", "on"}


def _import_module(mod_path: str) -> Any:  # pragma: no cover - tiny wrapper
    """
    Import a submodule with a friendlier error if PyTorch is missing.
    We detect both top-level and transitive `torch` import failures.
    """
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "") or ""
        if missing == "torch" or missing.startswith("torch."):
            raise ImportError(
                "The physics helpers require PyTorch (module 'torch' not found). "
                'Install the extra with: pip install "physical-ai-stl[torch]"'
            ) from e
        raise
    except ImportError as e:
        # Offer a friendlier nudge if the failure was due to missing torch.
        msg = str(e).lower()
        if "torch" in msg or "pytorch" in msg:
            raise ImportError(
                "The physics helpers require PyTorch. "
                'Install the extra with: pip install "physical-ai-stl[torch]"'
            ) from e
        raise


def _load_submodule(name: str) -> Any:
    module = _import_module(_LAZY_MODULES[name])
    globals()[name] = module  # cache in module globals for subsequent lookups
    return module


def _load_forward(name: str) -> Any:
    mod_name, attr = _FORWARD_ATTRS[name]
    module = globals().get(mod_name)
    if module is None:
        module = _load_submodule(mod_name)
    value = getattr(module, attr)
    globals()[name] = value  # cache resolved attribute
    return value


# ----- Lazy attribute access (PEP 562) ------------------------------------------

def __getattr__(name: str) -> Any:  # pragma: no cover - exercised implicitly
    # Submodule?
    if name in _LAZY_MODULES:
        return _load_submodule(name)
    # Forwarded attribute?
    if name in _FORWARD_ATTRS:
        return _load_forward(name)
    # Friendly suggestions for typos
    candidates = list(__all__)
    near = get_close_matches(name, candidates, n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(repr(c) for c in near)}?" if near else ""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.{hint}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    # Expose both already‑bound globals and the lazy attributes
    return sorted(list(globals().keys()) + list(__all__))


# ----- Optional: eager validation / IDE-friendly imports ------------------------

# Eager import if the user requested it (IDE indexing or strict CI sanity check).
if _truthy_env("PHYSICAL_AI_STL_STRICT_INIT") or _truthy_env("PHYSICAL_AI_STL_EAGER_IMPORTS"):  # pragma: no cover
    strict = _truthy_env("PHYSICAL_AI_STL_STRICT_INIT")

    # Import submodules first
    for _m in list(_LAZY_MODULES.keys()):
        try:
            _load_submodule(_m)
        except Exception:
            if strict:
                raise

    # Then validate that forwarded names actually resolve (strict mode only)
    if strict:
        missing: list[str] = []
        for _name, (_mod, _attr) in _FORWARD_ATTRS.items():
            try:
                _load_forward(_name)
            except Exception as _e:  # collect all failures in one readable message
                missing.append(f"{_name}  <-  {_mod}.{_attr}  ({_e.__class__.__name__}: {_e})")
        if missing:
            bullet = "\n  - "
            raise ImportError(
                "physical_ai_stl.physics: forward export validation failed:"
                f"{bullet}" + f"{bullet}".join(missing)
            )


# ----- Static imports for IDEs / type checkers only ----------------------------

if TYPE_CHECKING:  # pragma: no cover
    from . import diffusion1d as diffusion1d, heat2d as heat2d  # noqa: F401
    from .diffusion1d import (  # noqa: F401
        boundary_loss,
        Interval1D,
        make_dirichlet_mask_1d,
        pde_residual,
        residual_loss,
        sine_ic,
        sine_solution,
    )
    from .heat2d import (  # noqa: F401
        bc_ic_heat2d,
        gaussian_ic,
        make_dirichlet_mask,
        residual_heat2d,
        SquareDomain2D,
    )
