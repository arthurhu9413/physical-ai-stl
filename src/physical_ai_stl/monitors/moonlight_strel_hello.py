from __future__ import annotations

"""
MoonLight STREL “hello world”:

This module demonstrates end‑to‑end spatio‑temporal monitoring with
[MoonLight](https://github.com/MoonLightSuite/moonlight) on a tiny grid. It
evaluates the **containment** property from the demo script
``scripts/specs/contain_hotspot.mls``:

    contain := eventually (globally ( !(somewhere(hot)) ))

Intuition: a transient hotspot appears somewhere on the grid; the property
holds if, *eventually*, the hotspot disappears everywhere for some period.

Design goals
------------
- **Version‑robust.** Prefer the central helpers in
  :mod:`physical_ai_stl.monitoring.moonlight_helper` when available, and fall
  back to small local shims otherwise. This keeps the rest of the repository
  independent of MoonLight/JNI details.
- **Portable.** Works when the .mls script is packaged with the repo, but
  gracefully falls back to a small inline script if the file is missing (e.g.,
  when installed from PyPI).
- **Minimal surface.** Single public function :func:`strel_hello` returning a
  NumPy array for easy testing/printing.

Returns
-------
A ``np.ndarray`` view of MoonLight’s raw monitor output for the formula
``contain``.  The exact shape depends on the installed MoonLight version and
domain semantics; most common is ``(N, 2)`` with node (or time) in col 0 and
truth/robustness value in col 1.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np

# Prefer the shared helpers if present (keeps behavior consistent across the repo).
try:  # pragma: no cover - optional dependency path
    # Import names plainly; assign helper aliases below to keep isort happy.
    from physical_ai_stl.monitoring.moonlight_helper import (
        build_grid_graph,
        field_to_signal,
        get_monitor,
        load_script_from_file,
        monitor_graph_time_series,
        set_domain,
    )
    _helper_build_grid_graph = build_grid_graph
    _helper_field_to_signal = field_to_signal
    _helper_get_monitor = get_monitor
    _helper_load_script_from_file = load_script_from_file
    _helper_monitor_graph_time_series = monitor_graph_time_series
    _helper_set_domain = set_domain
except Exception:  # pragma: no cover
    build_grid_graph = None  # type: ignore[assignment]
    field_to_signal = None  # type: ignore[assignment]
    get_monitor = None  # type: ignore[assignment]
    load_script_from_file = None  # type: ignore[assignment]
    monitor_graph_time_series = None  # type: ignore[assignment]
    set_domain = None  # type: ignore[assignment]
    _helper_build_grid_graph = None  # type: ignore[assignment]
    _helper_field_to_signal = None  # type: ignore[assignment]
    _helper_get_monitor = None  # type: ignore[assignment]
    _helper_load_script_from_file = None  # type: ignore[assignment]
    _helper_monitor_graph_time_series = None  # type: ignore[assignment]
    _helper_set_domain = None  # type: ignore[assignment]

# -------------------------------------------
# Location of the demo .mls script within the repository.
_MLS_RELATIVE = ("scripts", "specs", "contain_hotspot.mls")

# Safe fallback for environments where the file is unavailable (e.g. pip installs).
_MLS_INLINE = (
    "signal { bool hot; }\n"
    "domain boolean;\n"
    "formula contain = eventually (!(somewhere (hot)));\n"
)


def _resolve_spec_file() -> Path | None:
    """Best‑effort resolution of the demo STREL script within this repo.

    Precedence:
      1) ``PHYSICAL_AI_STL_MLS_PATH`` environment variable (if valid file).
      2) Walking up from this file to find ``scripts/specs/contain_hotspot.mls``.
      3) Same search relative to the CWD (useful for ad‑hoc execution).
      4) ``None`` if not found (caller should fall back to inline script).
    """
    # Allow explicit override via env var
    env = os.environ.get("PHYSICAL_AI_STL_MLS_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent.joinpath(*_MLS_RELATIVE)
        if candidate.is_file():
            return candidate

    # Also try relative to CWD when running ad‑hoc scripts
    cwd_candidate = Path.cwd().joinpath(*_MLS_RELATIVE)
    if cwd_candidate.is_file():
        return cwd_candidate

    return None


def _build_grid_graph_local(nx: int, ny: int) -> list[list[float]]:
    """4‑neighborhood undirected grid as an adjacency matrix.

    This mirrors the helpers’ default return format. It is intentionally tiny
    (3×3 in :func:`strel_hello`), so the dense matrix is fine here.
    """
    n = nx * ny
    adj = [[0.0] * n for _ in range(n)]

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            # Von Neumann neighborhood
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    v = idx(ii, jj)
                    adj[u][v] = 1.0
                    adj[v][u] = 1.0  # ensure symmetry
    return adj


def _field_to_signal_local(u: np.ndarray, threshold: float | None) -> list[list[list[float]]]:
    """Convert a ``(nx, ny, nt)`` grid field to MoonLight’s nested list signal.

    The returned structure is a list over *time*, each entry being a list over
    *nodes*, each node carrying a single feature (hot/not hot). If a threshold
    is provided, the values are binarized to emulate boolean semantics.
    """
    if u.ndim != 3:
        raise ValueError(f"Expected a (nx, ny, nt) array; got shape {u.shape}")
    nx, ny, nt = u.shape
    n_nodes = nx * ny
    arr = u.reshape(n_nodes, nt).T
    if threshold is not None:
        arr = (arr >= threshold).astype(float)
    else:
        arr = arr.astype(float)
    # MoonLight expects a feature dimension per node.
    return arr[..., None].tolist()


def _get_monitor_local(mls: Any, formula: str) -> Any:
    # Java binding typically exposes getMonitor(name)
    try:
        return mls.getMonitor(formula)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"MoonLight script does not expose getMonitor({formula!r})") from e


def _monitor_graph_time_series(mon: Any, graph: Any, sig: Any) -> Any:
    """Version‑tolerant call into MoonLight’s STREL monitor.

    Tries (in order) the method names used across releases. If the bound
    ``monitor`` expects the older **4‑argument** signature
    ``monitor(location_times, graph_seq, signal_times, signal_seq)``, we detect
    the arity via a ``TypeError`` and automatically adapt by wrapping a static
    graph and synthesizing signal times ``[0, 1, ..., T-1]``.
    """
    # Preferred, modern names (2‑argument wrappers)
    for name in ("monitor_graph_time_series", "monitorGraphTimeSeries"):
        fn = getattr(mon, name, None)
        if callable(fn):
            return fn(graph, sig)

    # Generic 'monitor' across versions: try 2‑arg first, then 4‑arg.
    fn = getattr(mon, "monitor", None)
    if callable(fn):
        try:
            return fn(graph, sig)
        except TypeError:
            # Fall back to the 4‑argument API described in the official wiki.
            # Wrap a static graph into a one‑element sequence and build signal times.
            # If the user already passed a sequence of graphs, honor it.
            graph_seq = graph if (isinstance(graph, list) and isinstance(graph[0], list) and isinstance(graph[0][0], list)) else [graph]
            # Signal is a list over time; use integer ticks 0..T-1.
            try:
                T = len(sig)
            except Exception:
                T = 1
            times = list(float(i) for i in range(T))
            try:
                return fn([0.0], graph_seq, times, sig)  # type: ignore[misc]
            except TypeError:
                # Some very old builds flip (graph_seq, location_times). Try that last.
                return fn(times, sig, [0.0], graph_seq)  # type: ignore[misc]

    raise RuntimeError(
        "MoonLight STREL monitor: no compatible monitor method found. "
        "Tried 'monitor_graph_time_series', 'monitorGraphTimeSeries', and 'monitor' (2‑ or 4‑arg)."
    )


def _to_ndarray(out: Any) -> np.ndarray:
    """Best‑effort conversion of MoonLight output to ``np.ndarray`` of ``float``."""
    try:
        arr = np.asarray(out, dtype=float)
        return arr  # (N, 2) in the common case
    except Exception:  # pragma: no cover
        # Fallback: take first value if a mapping {node -> series}
        if isinstance(out, dict) and out:
            first = next(iter(out.values()))
            return _to_ndarray(first)
        raise


def strel_hello() -> np.ndarray:
    """Run a tiny STREL demo and return the monitor output as ``np.ndarray``.

    Implementation details:
      * Loads the STREL script from the repository if available (or falls back
        to a minimal inline script that declares the same ``contain`` formula).
      * Builds a 3×3 grid graph and a 2‑frame field with a transient hotspot at
        the center at ``t = 0`` (Boolean threshold at ``1.0``).
      * Evaluates the ``contain`` formula and returns MoonLight’s raw output.

    Notes
    -----
    - Requires the ``moonlight`` Python package (Java backend). If it is not
      installed, this function raises with a clear message.
    - The output shape varies by MoonLight version and selected domain. We cast
      to ``float`` for easy inspection and downstream use.
    """
    # If the optional dependency isn't present, fail fast with a clear error.
    if load_script_from_file is None:
        raise RuntimeError("MoonLight is not available; skipping STREL example.")

    # Load the STREL script (from file if available; otherwise use inline fallback).
    spec_path = _resolve_spec_file()
    if spec_path is not None:
        mls = load_script_from_file(str(spec_path))  # type: ignore[arg-type]
    else:  # pragma: no cover - rare when packaging w/o scripts/
        # Fall back to compiling the in‑memory script.
        from moonlight import ScriptLoader  # type: ignore
        mls = ScriptLoader.loadFromText(_MLS_INLINE)

    # Try to pin Boolean semantics explicitly (safe if script already set it).
    if _helper_set_domain is not None:  # pragma: no branch
        try:
            _helper_set_domain(mls, "boolean")
        except Exception:
            # Some releases don't expose domain setters; it is safe to ignore.
            pass

    # Build a tiny 3×3 grid graph and a 2‑frame field with a transient hotspot.
    build_graph = _helper_build_grid_graph or _build_grid_graph_local
    to_signal = _helper_field_to_signal or _field_to_signal_local
    get_mon = _helper_get_monitor or _get_monitor_local
    monitor_fn = _monitor_graph_time_series  # always use the robust local wrapper

    # Prefer the compact "triples" format when helpers are available.
    if _helper_build_grid_graph is not None:
        graph = build_graph(3, 3, return_format="triples")  # type: ignore[call-arg]
    else:
        graph = build_graph(3, 3)

    field = np.zeros((3, 3, 2), dtype=float)
    field[1, 1, 0] = 2.0  # hotspot at the center at t=0
    sig = to_signal(field, threshold=1.0)

    mon = get_mon(mls, "contain")
    out = monitor_fn(mon, graph, sig)

    return _to_ndarray(out)


__all__ = ["strel_hello"]
