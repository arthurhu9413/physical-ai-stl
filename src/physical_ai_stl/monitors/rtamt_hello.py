from __future__ import annotations

"""Minimal, version‑robust RTAMT “hello world” monitor.
------------------------------------------------------

This module exposes :func:`stl_hello_offline`, a tiny self‑check used by the
test-suite to verify that the optional **RTAMT** dependency is wired up
correctly.  The function evaluates the discrete‑time STL property

    G (u <= 1.0)

on a short trace ``u = [0.2, 0.4, 1.1]`` sampled at unit period and returns
the *offline robustness* as a plain ``float``.  For this property the expected
robustness is ``min_t (1.0 - u(t)) = -0.1``.

The implementation is defensive across RTAMT releases:
* It tries the documented “list of [name, series] pairs” calling style first,
  then older variants, and finally a mapping‑style signature if present.
* It normalizes the result into a scalar even if a particular version returns
  a sequence (e.g., time‑stamped robustness values).
"""

from typing import Any, Iterable, Mapping
import math


def _as_float(x: Any) -> float | None:
    """Best‑effort cast to ``float``; return ``None`` on failure."""
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _reduce_min(seq: Iterable[Any]) -> float | None:
    """Return the minimum ``float`` value found in ``seq``.

    The sequence may contain plain numbers, robustness values, or
    time‑stamped pairs ``(t, value)`` / ``[t, value]``. Elements that cannot
    be coerced to floats are ignored.  ``None`` is returned if no numeric
    value is found.
    """
    m = math.inf
    found = False
    for item in seq:
        val: float | None = None
        # Common shape from RTAMT: sequence of (t, value) pairs.
        if isinstance(item, (list, tuple)) and item:
            # Prefer the second component when present; otherwise the first.
            candidate = item[1] if len(item) > 1 else item[0]
            val = _as_float(candidate)
        else:
            val = _as_float(item)
        if val is not None:
            found = True
            if val < m:
                m = val
    return m if found else None


def _coerce_scalar(rob: Any) -> float:
    """Normalize RTAMT outputs (across versions) to a single ``float``.

    RTAMT historically returned either a scalar robustness value or a
    sequence (often time‑stamped).  Newer releases typically return a
    scalar for offline evaluation, but we keep this tolerant adapter for
    compatibility and tests.
    """
    # Fast path: already a scalar (int/float/NumPy scalar).
    val = _as_float(rob)
    if val is not None:
        return val

    # Mapping shape (e.g., {name: series} or {'out': values}). Reduce across values.
    if isinstance(rob, Mapping):
        mins: list[float] = []
        for v in rob.values():
            try:
                mins.append(_coerce_scalar(v))
            except Exception:
                continue
        if mins:
            return min(mins)
        raise TypeError("Could not coerce mapping return from RTAMT to float.")

    # Iterable shape: list/tuple of numbers or time‑stamped pairs.
    if isinstance(rob, (list, tuple)):
        reduced = _reduce_min(rob)
        if reduced is not None:
            return reduced

    # Last attempt (e.g., custom numeric types that behave like scalars).
    val = _as_float(rob)
    if val is not None:
        return val
    raise TypeError(f"Unsupported RTAMT return type: {type(rob)!r}")


def stl_hello_offline() -> float:
    """Return robustness of ``G (u <= 1.0)`` for a tiny discrete‑time trace.

    The trace is ``u(0)=0.2, u(1)=0.4, u(2)=1.1`` sampled every 1s.  The
    function imports :mod:`rtamt` lazily so the rest of the package remains
    importable when RTAMT is not installed.
    """
    # Import inside the function to allow the test to skip when RTAMT is missing.
    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency is missing.
        raise ImportError("rtamt not installed") from e

    # Build a minimal discrete‑time specification: G (u <= 1.0).
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var("u", "float")
    spec.spec = "always (u <= 1.0)"
    spec.parse()

    # Simple, safe‑to‑evaluate time series (t, u(t)) with unit sampling.
    ts: list[tuple[int, float]] = [(0, 0.2), (1, 0.4), (2, 1.1)]

    # Call evaluate() in a version‑robust way. Prefer the documented pair form.
    try:
        rob = spec.evaluate(["u", ts])  # README-documented signature
    except Exception:
        try:
            # Very old variant: names list + series list.
            rob = spec.evaluate(["u"], [ts])
        except Exception:
            # Some builds accept a mapping {var: series}.
            rob = spec.evaluate({"u": ts})

    return _coerce_scalar(rob)


__all__ = ["stl_hello_offline"]
