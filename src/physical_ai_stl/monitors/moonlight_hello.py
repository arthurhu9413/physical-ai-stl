# ruff: noqa: I001
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


# Keep the demo script small and explicit; terminate statements with semicolons
# to match the Script Syntax documented upstream.
# We define a tiny temporal script with two real-valued signals (x, y) and a single
# formula named `future` that states: "eventually within the next 0.4 time units,
# x is *strictly less* than y".  Using a bounded window keeps the example robust
# on short traces and matches the MoonLight wiki syntax.
_SCRIPT: str = (
    "signal { real x; real y; }\n"
    "domain boolean;\n"
    "formula cmp = (x < y);\n"
    "formula future = eventually[0, 0.4](cmp);\n"
)


def _to_list_floats(a: Sequence[float]) -> list[float]:
    # Converting via list() ensures types are native Python floats (not np.float64).
    return [float(v) for v in a]


def _monitor_with_best_effort(
    mon: Any, t: Sequence[float], x: Sequence[float], y: Sequence[float]
) -> np.ndarray:
    time_list = _to_list_floats(t)
    sig_pairs: list[tuple[float, float]] = list(zip(_to_list_floats(x), _to_list_floats(y)))

    try:
        # Preferred signature (documented on the MoonLight wiki)
        out = mon.monitor(time_list, sig_pairs)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: pack (t, x, y) into a single matrix for older interfaces.
        data = np.column_stack([t, x, y]).astype(float, copy=False)
        out = mon.monitor(data)  # type: ignore[misc]

    # MoonLight returns something iterable of pairs [time, value]; make it a stable ndarray.
    arr = np.array(out, dtype=float)
    # Most versions already return shape (N, 2). If a flat vector sneaks through, fix it.
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    return arr


def temporal_hello() -> np.ndarray:
    """Run a minimal MoonLight temporal monitor over a tiny toy signal.

    Returns
    -------
    numpy.ndarray
        An array of shape (N, 2) with columns [time, boolean_value] under the
        Boolean semantics (1.0 for true, 0.0 for false).
    """
    # Import lazily so the rest of the project can be installed/tested without Java.
    try:
        from moonlight import ScriptLoader  # type: ignore
    except Exception as e:  # pragma: no cover - exercised in tests by import blocking
        raise ImportError(
            "moonlight (Python interface) is not available. Install with 'pip install moonlight' "
            "and ensure a compatible Java runtime (Java 21+) is on PATH."
        ) from e

    # Tiny signal: five evenly-spaced samples on [0, 0.8].
    t = np.arange(0.0, 1.0, 0.2, dtype=float)  # [0.0, 0.2, 0.4, 0.6, 0.8]
    x = np.sin(t)
    y = np.cos(t)

    # Load the script and get the monitor.
    mls = ScriptLoader.loadFromText(_SCRIPT)
    try:
        # Explicitly set Boolean domain in case the loader's default changes.
        mls.setBooleanDomain()  # type: ignore[attr-defined]
    except Exception:
        # If the script already fixes the domain or the API lacks this method, ignore.
        pass
    mon = mls.getMonitor("future")

    return _monitor_with_best_effort(mon, t, x, y)


__all__ = ["temporal_hello"]
