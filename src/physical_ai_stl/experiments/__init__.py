# ruff: noqa: I001
# isort: skip_file
from __future__ import annotations

"""
physical_ai_stl.experiments
===========================

Light‑weight registry and lazy‑import shim for runnable experiments.

**Why this exists.**
- Keep import time and base dependencies tiny (only stdlib) while experiments
  can optionally depend on PyTorch and monitoring toolkits.
- Provide a *stable*, typed surface for scripts, notebooks, and tests:
  ``names()``, ``available()``, ``get_runner()``, ``run()``, ``register()``, ``about()``.
- Forward the most common symbols (e.g., ``run_diffusion1d``) without importing
  heavy modules up front.
- Allow extension via third‑party plugins discovered from the optional
  ``physical_ai_stl.experiments`` entry‑point group.

This design matches the course goal of rapidly comparing Physics‑ML frameworks
(Neuromancer, PhysicsNeMo, TorchPhysics) together with STL/STREL monitoring,
without forcing any particular stack until it is actually used.
"""

from collections.abc import Callable, Mapping
from importlib import import_module as _import_module
from importlib import util as _import_util
from typing import Any, Protocol, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Built‑ins exposed directly (lazy submodules)
# ---------------------------------------------------------------------------

# NB: Kept explicit for IDEs and static analyzers; dynamic derivation is avoided
# to prevent accidental surface changes when plugins are present.
__all__ = ['diffusion1d', 'heat2d']

# ---------------------------------------------------------------------------
# Registry (name → "module:function" | callable)
# ---------------------------------------------------------------------------

# Default experiments shipped with the package
_EXPERIMENTS: dict[str, str | Callable[..., Any]] = {
    'diffusion1d': 'physical_ai_stl.experiments.diffusion1d:run_diffusion1d',
    'heat2d':      'physical_ai_stl.experiments.heat2d:run_heat2d',
}

# One‑line blurbs for docs/printing without importing heavy modules
_DOCS: dict[str, str] = {
    'diffusion1d': '1‑D diffusion (heat) PINN with optional STL penalty.',
    'heat2d':      '2‑D heat‑equation PINN with optional STL/STREL penalty.',
}

# Submodules that we expose attributes from on demand
_LAZY_MODULES: dict[str, str] = {
    'diffusion1d': 'physical_ai_stl.experiments.diffusion1d',
    'heat2d':      'physical_ai_stl.experiments.heat2d',
}

# Forwarded names for nice imports like:
#   from physical_ai_stl.experiments import run_diffusion1d, Heat2DConfig
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    'run_diffusion1d': ('diffusion1d', 'run_diffusion1d'),
    'Diffusion1DConfig': ('diffusion1d', 'Diffusion1DConfig'),
    'run_heat2d': ('heat2d', 'run_heat2d'),
    'Heat2DConfig': ('heat2d', 'Heat2DConfig'),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_has_torch_cached: bool | None = None


def _has_torch() -> bool:
    """Cheap probe for PyTorch availability (cached)."""
    global _has_torch_cached
    if _has_torch_cached is None:
        _has_torch_cached = _import_util.find_spec('torch') is not None
    return _has_torch_cached


def _import_with_friendly_error(mod_path: str):
    """
    Import *mod_path* but rewrite the most likely ImportError into a helpful,
    actionable message focused on optional extras used in this project.
    """
    try:
        return _import_module(mod_path)
    except ImportError as e:  # pragma: no cover - tiny UX shim
        msg = str(e).lower()
        # Common case: user forgot to install the torch extra.
        if 'torch' in msg or 'pytorch' in msg or (_import_util.find_spec('torch') is None):
            raise ImportError(
                "This experiment requires PyTorch. Install the optional extra:\n"
                '  pip install "physical-ai-stl[torch]"\n'
                "or use the provided extras file:\n"
                "  pip install -r requirements-extra.txt"
            ) from e
        raise


def _split_target(target: str) -> tuple[str, str]:
    if ':' not in target:
        raise ValueError(f"Expected 'module:function' but got {target!r}.")
    mod, func = target.split(':', 1)
    return mod, func


def _normalize_name(name: str) -> str:
    return name.lower().strip()


def _maybe_discover_plugins() -> None:
    """
    Populate the registry from entry points.

    Third‑party packages can contribute experiments by exposing entry points in
    group ``physical_ai_stl.experiments`` with names mapping to either a
    callable or a ``\"module:function\"`` string. Example (pyproject.toml):

        [project.entry-points."physical_ai_stl.experiments"]
        burgers1d = "my_pkg.exp.burgers:run"

    Discovery is best‑effort and never raises.
    """
    try:  # Python 3.10+
        from importlib import metadata as _md  # type: ignore[attr-defined]
        eps = _md.entry_points()
        candidates = eps.select(group='physical_ai_stl.experiments') if hasattr(eps, 'select') else eps.get('physical_ai_stl.experiments', [])  # type: ignore[call-arg, index]
        for ep in candidates or ():
            key = _normalize_name(ep.name)
            if key not in _EXPERIMENTS:
                _EXPERIMENTS[key] = ep.value  # keep lazy until first use
    except Exception:
        # Silently ignore: plugin discovery is optional and should never break imports.
        return


# Discover once on import (no heavy work is done; values remain strings)
_maybe_discover_plugins()


class Runner(Protocol):
    """Callable signature accepted by the registry."""
    def __call__(self, cfg: Mapping[str, Any] | dict[str, Any]) -> Any:  # pragma: no cover - typing only
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def names() -> list[str]:
    """Return the sorted list of registered experiment keys."""
    return sorted(_EXPERIMENTS.keys())


def available() -> dict[str, bool]:
    """
    Quick availability probe for each registered experiment.

    For the built‑in experiments, availability is equivalent to ``torch`` being
    importable; plugins may override by registering a callable that performs its
    own checks.
    """
    has_torch = _has_torch()
    return {name: has_torch if not callable(_EXPERIMENTS[name]) else True
            for name in _EXPERIMENTS.keys()}


def register(name: str, target: str | Callable[..., Any]) -> None:
    """
    Add a new experiment mapping.

    Parameters
    ----------
    name:
        Registry key. Stored case‑insensitively.
    target:
        Either a callable ``fn(cfg)`` or a string ``\"module:function\"`` which
        will be lazily imported on first use.
    """
    _EXPERIMENTS[_normalize_name(name)] = target


def get_runner(name: str) -> Runner:
    """
    Resolve *name* to a runnable callable, importing on demand.

    This function memoizes the resolution: after the first call, a
    ``\"module:function\"`` string is replaced by the actual callable in the
    registry to avoid repeated attribute lookups.
    """
    key = _normalize_name(name)
    if key not in _EXPERIMENTS:
        opts = ', '.join(sorted(_EXPERIMENTS))
        raise KeyError(f"Unknown experiment {name!r}. Available: {opts}.")
    target = _EXPERIMENTS[key]
    if callable(target):
        fn = target  # type: ignore[assignment]
    else:
        mod_path, func_name = _split_target(target)
        mod = _import_with_friendly_error(mod_path)
        try:
            fn = getattr(mod, func_name)
        except AttributeError as e:  # pragma: no cover - rare
            raise AttributeError(f"Module {mod_path!r} has no attribute {func_name!r}.") from e
        # Memoize: subsequent calls skip import+getattr
        _EXPERIMENTS[key] = fn  # type: ignore[assignment]
    if not callable(fn):
        raise TypeError(f"Registered target for {name!r} is not callable: {fn!r}")
    return fn  # type: ignore[return-value]


def run(name: str, cfg: Mapping[str, Any] | dict[str, Any]) -> Any:
    """Lookup *name* in the registry and execute it with configuration *cfg*."""
    fn = get_runner(name)
    return fn(cfg)


def about() -> str:
    """Human‑readable summary of experiments and coarse availability."""
    width = max((len(n) for n in _EXPERIMENTS), default=0)
    avail = available()
    lines = ['experiments:']
    for n in names():
        tag = 'yes' if avail.get(n, False) else 'no '
        blurb = _DOCS.get(n, '')
        lines.append(f"  {n.ljust(width)}  {tag}  {blurb}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lazy attribute access (PEP 562)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    # Submodule passthrough (e.g., experiments.diffusion1d)
    if name in _LAZY_MODULES:
        mod = _import_with_friendly_error(_LAZY_MODULES[name])
        globals()[name] = mod
        return mod
    # Forward common symbols (e.g., run_diffusion1d)
    if name in _FORWARD_ATTRS:
        mod_key, obj_name = _FORWARD_ATTRS[name]
        mod = _import_with_friendly_error(_LAZY_MODULES[mod_key])
        obj = getattr(mod, obj_name)
        globals()[name] = obj  # cache for subsequent attribute access
        return obj
    # Expose utility callables without adding to __all__.
    if name in {'names', 'available', 'get_runner', 'run', 'register', 'about'}:
        return globals()[name]
    raise AttributeError(f"module 'physical_ai_stl.experiments' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    base = set(globals().keys())
    return sorted(base | set(__all__) | set(_FORWARD_ATTRS.keys()) | set(_LAZY_MODULES.keys())
                  | {'names', 'available', 'get_runner', 'run', 'register', 'about'})


# ---------------------------------------------------------------------------
# Static imports for type checkers only (avoid runtime import cost)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from . import diffusion1d as diffusion1d  # noqa: F401
    from . import heat2d as heat2d  # noqa: F401
    from .diffusion1d import Diffusion1DConfig as Diffusion1DConfig  # noqa: F401
    from .diffusion1d import run_diffusion1d as run_diffusion1d  # noqa: F401
    from .heat2d import Heat2DConfig as Heat2DConfig  # noqa: F401
    from .heat2d import run_heat2d as run_heat2d  # noqa: F401
