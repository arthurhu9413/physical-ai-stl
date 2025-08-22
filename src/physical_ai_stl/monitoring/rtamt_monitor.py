from __future__ import annotations

"""Thin, dependency-tolerant helpers around **RTAMT** for STL monitoring.
----------------------------------------------------------------------------

This module provides small utilities that make it easier to:
- create common STL specifications (with a choice of dense or discrete time),
- normalize a variety of time‑series inputs to the timestamped shape RTAMT expects,
- evaluate single/multi‑signal traces against a compiled specification, and
- post‑process robustness to a boolean satisfaction value.

Design principles
-----------------
* **Optional dependency**: importing this file does *not* import :mod:`rtamt`
  unless you actually build/evaluate a spec. If RTAMT is missing, an informative
  error explains how to install it.
* **Version resilience**: RTAMT’s API has seen minor variations across releases
  (e.g., availability of :class:`StlDenseTimeSpecification`, accepted signatures
  of :meth:`evaluate`). The helpers below try the modern signature first and
  fall back to older ones automatically. See the RTAMT README and papers for
  the canonical API and semantics.  
  References: GitHub README (usage of *StlDiscreteTimeSpecification* /
  *StlDenseTimeSpecification* and the :py:meth:`evaluate`/:py:meth:`update`
  split), and the 2025 journal article describing offline/online monitors.  
  (We cite these in the project docs; see usage notes there.)

The goal is to keep this module **small**, **predictable**, and **fast** while
remaining friendly to student projects that explore “physical‑AI + STL” ideas.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

# ----- Internal utilities ----------------------------------------------------

_RTAMT = None  # cached module to avoid repeated imports


def _import_rtamt():
    """Import and cache :mod:`rtamt` on first use.

    Returns
    -------
    module
        The imported :mod:`rtamt` module.

    Raises
    ------
    ImportError
        If RTAMT is not installed.
    """
    global _RTAMT
    if _RTAMT is not None:
        return _RTAMT
    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "rtamt is not installed. Install it with 'pip install rtamt' "
            "or see https://github.com/nickovic/rtamt for build notes."
        ) from e
    _RTAMT = rtamt
    return rtamt


# Treat "time series" inputs liberally:
# - [v0, v1, ...] with uniform dt
# - [(t0, v0), (t1, v1), ...] explicit timestamps
TimeSeries = Iterable[float] | Iterable[tuple[float, float]]


def _normalize_series(series: TimeSeries, dt: float | None) -> list[tuple[float, float]]:
    """Normalize a numeric series to a list of ``(t, value)`` pairs.

    Accepted inputs
    ---------------
    * **Regular samples**: ``[v0, v1, ...]`` interpreted with constant ``dt``.
    * **Explicit timestamps**: ``[(t0, v0), (t1, v1), ...]``.

    Notes
    -----
    * Timestamps are cast to ``float`` and **sorted** in ascending order.
      If duplicate timestamps are present, the **last** sample at that time
      is kept (common in piecewise‑constant traces).
    * When using the regular‑sampling form, ``dt`` must be positive.
    """
    it = iter(series)
    try:
        first = next(it)
    except StopIteration:
        return []

    # Explicit timestamps given?
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        t0, v0 = float(first[0]), float(first[1])
        out: list[tuple[float, float]] = [(t0, float(v0))]
        for el in it:
            t, v = el  # type: ignore[misc]
            out.append((float(t), float(v)))

        # Ensure ascending timestamps and coalesce exact duplicates by keeping the last.
        out.sort(key=lambda tv: tv[0])
        dedup: list[tuple[float, float]] = []
        for t, v in out:
            if dedup and t == dedup[-1][0]:
                dedup[-1] = (t, v)
            else:
                dedup.append((t, v))
        return dedup

    # Otherwise: treat as regularly sampled values.
    step = 1.0 if dt is None else float(dt)
    if step <= 0.0:
        raise ValueError("'dt' must be > 0 for regularly sampled series.")
    out = [(0.0, float(first))]
    k = 1
    for v in it:
        out.append((k * step, float(v)))
        k += 1
    return out


def _coerce_scalar(rob: object) -> float:
    """Coerce various return shapes from RTAMT to a plain ``float``.

    RTAMT typically returns a numeric scalar for offline :meth:`evaluate` and
    online :meth:`update`. Older releases (or bindings) sometimes return
    small containers (e.g., ``[rob]`` or ``[(t, rob)]``). This function
    converts those cases to a single robustness number.
    """
    # Fast path: already a numeric scalar (float/int/NumPy scalar).
    try:
        return float(rob)  # type: ignore[arg-type]
    except Exception:
        pass

    # Common container fallbacks.
    if isinstance(rob, (list, tuple)):
        if not rob:
            return 0.0
        first = rob[0]
        # Shape: [(t, value), ...] → take the value
        if isinstance(first, (list, tuple)):
            return float(first[1] if len(first) > 1 else first[0])
        # Shape: [value, ...]
        return float(first)
    # Last attempt (custom numeric types).
    return float(rob)  # type: ignore[arg-type]


# ----- Spec builders ---------------------------------------------------------

def _new_spec(time_semantics: Literal["dense", "discrete"] = "dense") -> Any:
    """Return a new RTAMT STL specification object for the requested semantics.

    This prefers the modern classes (``StlDenseTimeSpecification``,
    ``StlDiscreteTimeSpecification``) and falls back if needed.
    """
    rtamt = _import_rtamt()
    if time_semantics == "dense":
        Spec = (
            getattr(rtamt, "StlDenseTimeSpecification", None)
            or getattr(rtamt, "StlDenseTimeOfflineSpecification", None)
        )
        if Spec is None:
            raise RuntimeError("Your RTAMT distribution lacks dense‑time support.")
        return Spec()
    elif time_semantics == "discrete":
        try:
            return rtamt.StlDiscreteTimeSpecification()
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Your RTAMT distribution lacks discrete‑time support.") from e
    else:  # pragma: no cover - guarded by type hints
        raise ValueError("time_semantics must be 'dense' or 'discrete'.")


def build_stl_spec(
    spec_text: str,
    *,
    var_types: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    io_types: Mapping[str, str] | None = None,
    time_semantics: Literal["dense", "discrete"] = "dense",
) -> Any:
    """Convenience builder for an STL specification.

    Parameters
    ----------
    spec_text:
        STL formula text. Use RTAMT’s syntax, e.g.:
        ``"always ((s <= 1.0) and (rise(s) -> eventually[0:2](s >= 0.5)))"``
    var_types:
        Variable declarations as ``{"name": "float"}`` or a list of
        ``[(name, "float"), ...]``. RTAMT commonly expects variables typed
        as ``"float"``.
    io_types:
        Optional map of variable names to IO roles ``{"input"|"output"}``.
        This is used by the IA‑STL variant in RTAMT; if unavailable in the
        installed version, the calls are silently skipped.
    time_semantics:
        ``"dense"`` (default) or ``"discrete"``.

    Returns
    -------
    Any
        A parsed RTAMT specification object ready for :meth:`evaluate`.
    """
    spec = _new_spec(time_semantics=time_semantics)

    # Declarations
    if isinstance(var_types, Mapping):
        decl_items = list(var_types.items())
    else:
        decl_items = list(var_types)
    for name, ty in decl_items:
        spec.declare_var(str(name), str(ty))

    # IO types (best‑effort; optional in several RTAMT versions)
    if io_types:
        set_io = getattr(spec, "set_var_io_type", None)
        if callable(set_io):
            for name, role in io_types.items():
                try:
                    set_io(str(name), str(role))
                except Exception:
                    # Ignore silently if unsupported or the role is invalid in this build.
                    pass

    spec.spec = spec_text
    spec.parse()
    return spec


def stl_always_upper_bound(
    var: str = "u",
    u_max: float = 1.0,
    *,
    time_semantics: Literal["dense", "discrete"] = "dense",
):
    """Return a compiled ``always (var <= u_max)`` spec.

    This is a tiny helper used in demos/tests; it declares a single ``float``
    variable and parses the formula. See RTAMT’s README for the meaning of
    :math:`\textbf{G}` (always) robustness under dense/discrete time.  
    Example of the same pattern in the official docs shows using
    :class:`StlDenseTimeSpecification` or :class:`StlDiscreteTimeSpecification` and
    calling :py:meth:`parse` then :py:meth:`evaluate`.  
    """
    spec = _new_spec(time_semantics=time_semantics)
    spec.declare_var(var, "float")
    spec.spec = f"always ({var} <= {float(u_max)})"
    spec.parse()
    return spec


def stl_response_within(
    var: str,
    boundary: str,
    theta: float,
    tau: int,
    *,
    time_semantics: Literal["dense", "discrete"] = "dense",
):
    """Bounded response: whenever ``boundary >= theta``, ``var`` responds within ``tau``.

    The spec encodes: ``always ( (boundary >= theta) -> eventually[0:tau] (var >= theta) )``.
    This mirrors the *request–grant* examples from the RTAMT repository.
    """
    spec = _new_spec(time_semantics=time_semantics)
    spec.declare_var(var, "float")
    spec.declare_var(boundary, "float")
    spec.spec = (
        f"always ( ({boundary} >= {float(theta)}) -> "
        f"eventually[0:{int(tau)}] ({var} >= {float(theta)}) )"
    )
    spec.parse()
    return spec


# ----- Evaluation helpers ----------------------------------------------------

def _evaluate_with_fallbacks(spec: Any, payload: Any) -> Any:
    """Call ``spec.evaluate`` trying several historical signatures.

    Modern RTAMT accepts a map ``{name: [(t, v), ...], ...}``.
    Many examples use positional pairs ``['x', series], ['y', series]`` instead.
    A very old variant expects two lists: ``([names], [series_list])``.
    """
    # 1) Mapping form (newer and most ergonomic)
    try:
        return spec.evaluate(payload)
    except Exception:
        pass

    # 2) Positional pairs: evaluate(*[['x', series], ['y', series]])
    if isinstance(payload, Mapping):
        pairs = [[k, payload[k]] for k in payload]
        try:
            return spec.evaluate(*pairs)  # type: ignore[misc]
        except Exception:
            pass

        # 3) Two‑list legacy fallback
        try:
            names = list(payload.keys())
            series_list = [payload[k] for k in names]
            return spec.evaluate(names, series_list)
        except Exception:
            pass

    # If payload was already pairs (our multi‑variable fallback), try 2) & 3).
    if isinstance(payload, (list, tuple)) and payload and isinstance(payload[0], (list, tuple)):
        try:
            return spec.evaluate(*payload)  # type: ignore[misc]
        except Exception:
            try:
                names = [p[0] for p in payload]
                series_list = [p[1] for p in payload]
                return spec.evaluate(names, series_list)
            except Exception:
                pass

    # Give up with a helpful message.
    raise TypeError("Unsupported evaluate() signature for your RTAMT build.")


def evaluate_series(
    spec: Any,
    var: str,
    series: TimeSeries,
    *,
    dt: float = 1.0,
) -> float:
    """Evaluate robustness for a **single** variable ``var`` time series.

    Parameters
    ----------
    spec:
        Parsed RTAMT specification (e.g., from :func:`stl_always_upper_bound` or
        :func:`build_stl_spec`).
    var:
        The variable name as declared in the spec.
    series:
        The values for ``var`` (regular samples or explicit timestamps).
    dt:
        Sampling period for the regular‑samples form. Ignored if ``series``
        already carries timestamps.

    Returns
    -------
    float
        Robustness of the formula, coerced to a plain ``float`` regardless of
        the underlying RTAMT return shape.
    """
    ts = _normalize_series(series, dt)
    # Try the modern fastest path first: dict mapping.
    payload = {var: ts}
    rob = _evaluate_with_fallbacks(spec, payload)
    return _coerce_scalar(rob)


def evaluate_multi(
    spec: Any,
    data: Mapping[str, TimeSeries] | Sequence[tuple[str, TimeSeries]],
    *,
    dt: float | Mapping[str, float] = 1.0,
) -> float:
    """Evaluate robustness for a **multi‑variable** trace.

    Parameters
    ----------
    spec:
        Parsed RTAMT specification.
    data:
        Either a mapping ``{name: series, ...}`` or a list of ``[(name, series)]``.
        Each series can be regular‑sampled or timestamped; see :func:`_normalize_series`.
    dt:
        Either a global sampling period (applied to all regular‑sampled series), or
        a per‑variable mapping ``{name: dt}``.

    Returns
    -------
    float
        Robustness as a plain ``float``.
    """
    if isinstance(data, Mapping):
        items = list(data.items())
    else:
        items = list(data)

    # Build dt map
    if isinstance(dt, Mapping):
        dt_map: Mapping[str, float] = dt
    else:
        dt_map = {name: float(dt) for name, _ in items}

    # Normalize all series to timestamped lists.
    series_map: dict[str, list[tuple[float, float]]] = {
        name: _normalize_series(s, dt_map.get(name))
        for name, s in items
    }

    # Try evaluate with the most ergonomic modern signature first (plus fallbacks).
    rob = _evaluate_with_fallbacks(spec, series_map)
    return _coerce_scalar(rob)


def satisfied(robustness: float) -> bool:
    """Return ``True`` iff the STL robustness is **non‑negative**.

    This mirrors the convention used in RTAMT and in the project tests.
    Keep the threshold at ``0.0`` (no epsilon) to avoid masking small violations.
    """
    return float(robustness) >= 0.0


__all__ = [
    "build_stl_spec",
    "stl_always_upper_bound",
    "stl_response_within",
    "evaluate_series",
    "evaluate_multi",
    "satisfied",
]
