# ruff: noqa: I001
from __future__ import annotations

"""
Unified monitoring front-end for STL and spatio‑temporal logics.

This package exposes three *optional* backends behind a tiny, dependency‑aware
facade:

- **RTAMT** (``rtamt``) for *exact* (non‑differentiable) Signal Temporal Logic
  robustness on discrete‑ and dense‑time traces. Recommended for offline
  evaluation and crisp pass/fail checks.
- **MoonLight** (``moonlight``) for monitoring *spatio‑temporal* properties
  (STREL) over grids/graphs; Python wrapper around the Java engine.
- **Soft / differentiable STL** (``stl_soft``) implemented in PyTorch for
  smooth, backprop‑friendly penalties to *nudge* learning.  (No third‑party
  dependency beyond PyTorch.)

The module is designed to be import‑cheap even when optional dependencies are
missing. All imports are *lazy* and dependency checks are explicit and helpful.
"""

import importlib
import importlib.metadata as _metadata
import importlib.util as _import_util
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, TYPE_CHECKING, overload, Literal

# -----------------------------------------------------------------------------
# Public surface (declared up front for tools/IDEs)
# -----------------------------------------------------------------------------

__all__ = [
    # Submodules (lazy)
    "moonlight_helper",
    "rtamt_monitor",
    "stl_soft",

    # Capability / environment probes
    "available_backends",
    "ensure",
    "get_backend",
    "about",
    "prefer_backend",
    "is_available",

    # Convenience wrappers (safe defaults; do not import heavy deps at module import)
    "monitor_global_upper_bound",
    "monitor_response_within",

    # Re-exports (lazy)
    "load_script_from_file",
    "get_monitor",
    "build_grid_graph",
    "field_to_signal",
    "stl_always_upper_bound",
    "stl_response_within",
    "evaluate_series",
    "evaluate_multi",
    "satisfied",
    "STLPenalty",
    "softmin",
    "softmax",
    "soft_and",
    "soft_or",
    "soft_not",
    "pred_leq",
    "pred_geq",
    "pred_abs_leq",
    "pred_linear_leq",
    "always",
    "eventually",
    "always_window",
    "eventually_window",
    "shift_left",
]

# Map "logical backend name" -> loader module
_BACKENDS: Mapping[str, str] = {
    "rtamt": "physical_ai_stl.monitoring.rtamt_monitor",
    "moonlight": "physical_ai_stl.monitoring.moonlight_helper",
    "soft": "physical_ai_stl.monitoring.stl_soft",
}

# For *availability* checks we probe the true external distributions.
# Keys here must include any extra runtime deps a backend requires.
_OPT_DEPS: Mapping[str, str] = {
    "rtamt": "rtamt",
    "moonlight": "moonlight",  # Python bindings for MoonLight (Java)
    "torch": "torch",          # required by stl_soft
}

# Submodules we expose directly (lazy)
_SUBMODULES: Mapping[str, str] = {
    "moonlight_helper": _BACKENDS["moonlight"],
    "rtamt_monitor": _BACKENDS["rtamt"],
    "stl_soft": _BACKENDS["soft"],
}

# Friendly, dependency‑aware re‑exports (name -> "module:object")
_REEXPORTS: Mapping[str, str] = {
    # MoonLight helpers
    "load_script_from_file": f'{_BACKENDS["moonlight"]}:load_script_from_file',
    "get_monitor":           f'{_BACKENDS["moonlight"]}:get_monitor',
    "build_grid_graph":      f'{_BACKENDS["moonlight"]}:build_grid_graph',
    "field_to_signal":       f'{_BACKENDS["moonlight"]}:field_to_signal',

    # RTAMT helpers
    "stl_always_upper_bound": f'{_BACKENDS["rtamt"]}:stl_always_upper_bound',
    "stl_response_within":    f'{_BACKENDS["rtamt"]}:stl_response_within',
    "evaluate_series":        f'{_BACKENDS["rtamt"]}:evaluate_series',
    "evaluate_multi":         f'{_BACKENDS["rtamt"]}:evaluate_multi',
    "satisfied":              f'{_BACKENDS["rtamt"]}:satisfied',

    # Smooth (differentiable) STL semantics
    "STLPenalty":     f'{_BACKENDS["soft"]}:STLPenalty',
    "softmin":        f'{_BACKENDS["soft"]}:softmin',
    "softmax":        f'{_BACKENDS["soft"]}:softmax',
    "soft_and":       f'{_BACKENDS["soft"]}:soft_and',
    "soft_or":        f'{_BACKENDS["soft"]}:soft_or',
    "soft_not":       f'{_BACKENDS["soft"]}:soft_not',
    "pred_leq":       f'{_BACKENDS["soft"]}:pred_leq',
    "pred_geq":       f'{_BACKENDS["soft"]}:pred_geq',
    "pred_abs_leq":   f'{_BACKENDS["soft"]}:pred_abs_leq',
    "pred_linear_leq":f'{_BACKENDS["soft"]}:pred_linear_leq',
    "always":         f'{_BACKENDS["soft"]}:always',
    "eventually":     f'{_BACKENDS["soft"]}:eventually',
    "always_window":  f'{_BACKENDS["soft"]}:always_window',
    "eventually_window": f'{_BACKENDS["soft"]}:eventually_window',
    "shift_left":     f'{_BACKENDS["soft"]}:shift_left',
}

if TYPE_CHECKING:  # pragma: no cover - imported only for static type checkers
    from . import moonlight_helper as moonlight_helper  # noqa: F401
    from . import rtamt_monitor as rtamt_monitor        # noqa: F401
    from . import stl_soft as stl_soft                  # noqa: F401


# -----------------------------------------------------------------------------
# Capability probing (fast; does not import heavy modules)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _probe(mod_name: str) -> tuple[bool, str | None]:
    """Return (is_available, version_str_or_None) for an optional dependency.

    Uses :mod:`importlib.util.find_spec` to avoid importing heavy modules.
    Falls back to reading ``__version__`` when distribution metadata is missing.
    """
    if _import_util.find_spec(mod_name) is None:
        return False, None
    dist = _OPT_DEPS.get(mod_name) or mod_name
    try:
        ver = _metadata.version(dist)  # type: ignore[arg-type]
    except Exception:
        try:
            m = importlib.import_module(mod_name)
            ver = getattr(m, "__version__", None)
        except Exception:
            ver = None
    return True, ver


@lru_cache(maxsize=1)
def available_backends() -> dict[str, dict[str, bool | str | None]]:
    """Return a cached availability report for all monitoring backends.

    Example::

        >>> from physical_ai_stl.monitoring import available_backends, about
        >>> available_backends()
        {'rtamt': {'available': False, 'version': None},
         'moonlight': {'available': True, 'version': '0.7.0'},
         'soft': {'available': True, 'version': '2.4.1'}}
        >>> print(about())
        physical_ai_stl.monitoring backends:
          rtamt      no  (-)
          moonlight  yes (0.7.0)
          soft       yes (2.4.1)

    Notes
    -----
    ``soft`` is considered available iff PyTorch is importable.
    """
    report: dict[str, dict[str, bool | str | None]] = {}
    for name in ("rtamt", "moonlight"):
        ok, ver = _probe(name)
        report[name] = {"available": ok, "version": ver}
    ok, ver = _probe("torch")
    report["soft"] = {"available": ok, "version": ver}
    return report


def is_available(name: str) -> bool:
    """Quick boolean check for a backend (``'rtamt'|'moonlight'|'soft'``)."""
    rep = available_backends()
    key = "soft" if name.lower().strip() == "soft" else name.lower().strip()
    if key not in rep:
        raise KeyError(f"Unknown backend {name!r}. Expected one of: {', '.join(rep)}.")
    return bool(rep[key]["available"])


def ensure(*backends: str) -> None:
    """Ensure selected backends are importable; raise a helpful ``ImportError``.

    Examples
    --------
    >>> ensure("soft")         # requires PyTorch
    >>> ensure("rtamt")        # requires the 'rtamt' package
    >>> ensure("moonlight")    # requires the 'moonlight' package and Java
    """
    missing: list[str] = []
    versions: dict[str, str | None] = {}
    for b in backends:
        key = "torch" if b == "soft" else b
        ok, ver = _probe(key)
        versions[b] = ver
        if not ok:
            missing.append(b)
    if missing:
        # Helpful hints per backend
        hints: dict[str, str] = {
            "rtamt": (
                "Install with:  pip install rtamt\n"
                "Docs: https://arxiv.org/abs/2005.11827"
            ),
            "moonlight": (
                "Install Python bindings and ensure a Java runtime is available:\n"
                "  pip install moonlight\n"
                "Paper: https://link.springer.com/article/10.1007/s10009-023-00710-5"
            ),
            "soft": (
                "Install PyTorch for differentiable STL:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu  # (CPU wheel)"
            ),
        }
        msg = ["Optional monitoring backend(s) missing: " + ", ".join(missing)]
        for b in missing:
            msg.append(f"- {b}: " + hints.get(b, "please install the required package(s)."))
        raise ImportError("\n".join(msg))


def get_backend(name: str) -> Any:
    """Import and return the backend helper module by name.

    Parameters
    ----------
    name:
        One of ``'rtamt'``, ``'moonlight'``, or ``'soft'`` (case‑insensitive).

    Raises
    ------
    KeyError
        If the backend name is unknown.
    ImportError
        If required packages are missing.
    """
    key = name.lower().strip()
    if key not in _BACKENDS:
        raise KeyError(f"Unknown backend {name!r}. Expected one of: {', '.join(_BACKENDS)}.")
    # Quick dependency check before import (keeps errors helpful)
    ensure("soft" if key == "soft" else key)
    mod = importlib.import_module(_BACKENDS[key])
    return mod


def about() -> str:
    """Human‑readable summary of backend availability and versions."""
    rep = available_backends()
    width = max(len(k) for k in rep) if rep else 7
    lines = ["physical_ai_stl.monitoring backends:"]
    for name in ("rtamt", "moonlight", "soft"):
        avail = rep[name]["available"]
        ver = rep[name]["version"]
        lines.append(f"  {name.ljust(width)}  {'yes' if avail else 'no ':<3} ({ver or '-'})")
    return "\n".join(lines)


def prefer_backend(prefer: Iterable[str] = ("rtamt", "moonlight", "soft")) -> str:
    """Pick the first available backend from a preference order.

    Example
    -------
    >>> prefer_backend()         # 'rtamt' if installed; else 'moonlight' or 'soft'
    >>> prefer_backend(('soft',))  # enforce differentiable fallback
    """
    rep = available_backends()
    for cand in prefer:
        key = cand.lower().strip()
        if key not in _BACKENDS:
            continue
        if rep.get("soft" if key == "soft" else key, {}).get("available"):
            return key
    raise ImportError("No monitoring backend available. Try: pip install rtamt moonlight torch")


# -----------------------------------------------------------------------------
# High‑level convenience wrappers (keep them tiny; reuse helpers in submodules)
# -----------------------------------------------------------------------------

def _values_from_series(series: Iterable[float] | Iterable[tuple[float, float]]) -> list[float]:
    """Return the value sequence from a series accepted by :mod:`rtamt` helpers.

    Accepts either a regular list of values ``[v0, v1, ...]`` (uniform sampling)
    or a timestamped list ``[(t0, v0), (t1, v1), ...]``. Only the values are
    returned; timestamps—when present—are ignored by the soft semantics path.
    """
    it = iter(series)
    try:
        first = next(it)
    except StopIteration:
        return []
    vals: list[float] = []
    # Timestamped?
    if isinstance(first, (tuple, list)) and len(first) == 2:
        t0, v0 = first  # noqa: F841 - t0 unused
        vals.append(float(v0))
        for _t, v in it:  # type: ignore[misc]
            vals.append(float(v))
        return vals
    # Regular values
    vals.append(float(first))  # type: ignore[arg-type]
    for v in it:
        vals.append(float(v))
    return vals


def monitor_global_upper_bound(
    series: Iterable[float] | Iterable[tuple[float, float]],
    u_max: float,
    *,
    var: str = "u",
    dt: float = 1.0,
    backend: Literal["auto", "rtamt", "soft"] = "auto",
    time_semantics: Literal["dense", "discrete"] = "dense",
) -> float:
    """Robustness of the STL formula ``G ( {var} ≤ u_max )`` on a time series.

    Chooses a backend automatically (default preference: RTAMT → soft).
    This is a thin, dependency‑aware wrapper around
    :func:`physical_ai_stl.monitoring.rtamt_monitor.stl_always_upper_bound` and
    :mod:`physical_ai_stl.monitoring.stl_soft`.

    Parameters
    ----------
    series:
        Signal values either as ``[v0, v1, ...]`` with uniform step ``dt`` or
        as timestamped pairs ``[(t0, v0), (t1, v1), ...]``.
    u_max:
        Upper bound in the atomic predicate.
    var, dt, time_semantics:
        Passed through to the RTAMT helper when that backend is used.
    backend:
        ``'auto'`` (prefer RTAMT if installed, else soft), ``'rtamt'``, or ``'soft'``.

    Returns
    -------
    float
        Quantitative robustness (≥0 ⇒ satisfied; <0 ⇒ violated).
    """
    key = prefer_backend(("rtamt", "soft")) if backend == "auto" else backend
    if key == "rtamt":
        rt = get_backend("rtamt")
        spec = rt.stl_always_upper_bound(var=var, u_max=float(u_max), time_semantics=time_semantics)
        return float(rt.evaluate_series(spec, var, series, dt=dt))  # type: ignore[no-any-return]
    # soft semantics
    ensure("soft")
    # Local import to keep module import cheap
    from . import stl_soft as _stl  # type: ignore
    import torch as _torch  # type: ignore
    vals = _values_from_series(series)
    if not vals:
        return float("inf")
    u = _torch.tensor(vals, dtype=_torch.float32)
    margins = _stl.pred_leq(u, float(u_max))
    rob = _stl.always(margins)
    return float(rob.item())


def monitor_response_within(
    req: Iterable[float] | Iterable[tuple[float, float]],
    resp: Iterable[float] | Iterable[tuple[float, float]],
    *,
    var_req: str = "req",
    var_resp: str = "resp",
    within: float = 1.0,
    dt: float = 1.0,
    backend: Literal["auto", "rtamt", "soft"] = "auto",
    time_semantics: Literal["dense", "discrete"] = "dense",
) -> float:
    """Robustness for the response property ``G ( req → F_[0,within] resp )``.

    The *soft* path implements a smooth approximation using max/min
    aggregations; the *rtamt* path uses the exact STL monitor.

    Parameters
    ----------
    req, resp:
        Two time series with the same sampling; each can be ``[v0, ...]`` or
        ``[(t0, v0), ...]``.
    within:
        Non‑negative time window for the *eventually* operator.
    backend, var_*, dt, time_semantics:
        See :func:`monitor_global_upper_bound`.
    """
    key = prefer_backend(("rtamt", "soft")) if backend == "auto" else backend
    if key == "rtamt":
        rt = get_backend("rtamt")
        spec = rt.stl_response_within(var_req=var_req, var_resp=var_resp, within=float(within), time_semantics=time_semantics)
        # evaluate_multi accepts a mapping {name: series}
        return float(rt.evaluate_multi(spec, {var_req: req, var_resp: resp}, dt=dt))  # type: ignore[no-any-return]
    # soft semantics
    ensure("soft")
    from . import stl_soft as _stl  # type: ignore
    import torch as _torch  # type: ignore
    r = _torch.tensor(_values_from_series(req), dtype=_torch.float32)
    s = _torch.tensor(_values_from_series(resp), dtype=_torch.float32)
    # Interpret req/resp as *margins*: req ≥ 0 means the request is active,
    # resp ≥ 0 means the response condition holds. A simple differentiable
    # approximation of G ( req → F_[0,w] resp ) is:
    #   G ( max(-req,  F_[0,w](resp)) )
    # where max implements the implication a → b ≡ (¬a) ∨ b.
    w = int(max(1, round(float(within) / float(dt))))
    # Eventually over a causal sliding window of length w
    fw = _stl.eventually_window(s, w)
    impl = _torch.maximum(-r, fw)
    rob = _stl.always(impl)
    return float(rob.item())


# -----------------------------------------------------------------------------
# Lazy attribute access & dir() shims
# -----------------------------------------------------------------------------

class _MissingBackendProxy:
    """Lightweight placeholder for an optional backend submodule.

    Accessing any attribute raises a helpful ImportError, but merely importing
    :mod:`physical_ai_stl.monitoring` will not fail in environments where the
    optional dependency is not installed.
    """
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, _: str) -> Any:  # pragma: no cover - trivial
        raise ImportError(
            f"Optional monitoring backend '{self._name}' is not available. "
            f"Install its dependencies and try again. "
            f"See: physical_ai_stl.monitoring.about()"
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<missing backend '{self._name}'; install dependencies to use>"


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    # Submodule requested?
    if name in _SUBMODULES:
        # Map submodule name back to backend key
        backend_key = "moonlight" if name == "moonlight_helper" else ("rtamt" if name == "rtamt_monitor" else "soft")
        if is_available(backend_key):
            mod = importlib.import_module(_SUBMODULES[name])
            globals()[name] = mod
            return mod
        # Not available: return a lightweight proxy that raises on use
        proxy = _MissingBackendProxy(backend_key)
        globals()[name] = proxy  # cache
        return proxy
    # Re-export requested?
    if name in _REEXPORTS:
        mod_name, obj_name = _REEXPORTS[name].split(":")
        # Determine which backend this object belongs to
        if mod_name.endswith(".rtamt_monitor"):
            backend_key = "rtamt"
        elif mod_name.endswith(".moonlight_helper"):
            backend_key = "moonlight"
        else:
            backend_key = "soft"
        if is_available(backend_key):
            obj = getattr(importlib.import_module(mod_name), obj_name)
            globals()[name] = obj
            return obj
        # Missing backend: return a stub callable (or descriptor) that raises on use
        def _missing(*_a: Any, **_k: Any) -> Any:  # pragma: no cover - trivial
            raise ImportError(
                f"Optional monitoring backend '{backend_key}' is not available. "
                f"Install its dependencies and try again. "
                f"See: physical_ai_stl.monitoring.about()"
            )
        globals()[name] = _missing
        return _missing
    raise AttributeError(f"module 'physical_ai_stl.monitoring' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()) + list(_REEXPORTS.keys()))
