from __future__ import annotations

"""Small, version‑robust helpers for the MoonLight (STREL) Python API.

This module keeps *all* MoonLight specifics in one place so the rest of the
codebase can stay framework‑agnostic and easy to test.  It deliberately avoids
importing MoonLight at module import time so environments without Java/MoonLight
can still import and run other parts of the project.

Key capabilities
----------------
- Lazy import of MoonLight's :class:`ScriptLoader` with a clear error message.
- Loading `.mls` scripts from a *file* (preferred) or *text* (fallback).
- Safe access to monitors by name, with helpful diagnostics.
- Utilities to build simple grid graphs in the shapes MoonLight accepts.
- Conversion of numpy arrays to the Python nested lists MoonLight expects.
- A thin, version‑tolerant bridge to invoke the spatio‑temporal monitoring call
  even when method names differ across MoonLight versions.

Design notes
------------
The shapes below mirror the upstream MoonLight Python wiki examples
(see GitHub wiki and paper).  In short:

- **Temporal** monitoring expects:
    ``monitor(time: list[float], signal: list[tuple[float, ...]])``

- **Spatio‑temporal** monitoring expects:
    ``monitor(location_times, graph_seq, signal_times, signal_seq)``, where
    ``graph_seq`` is a *list* of graphs (one per time in ``location_times``)
    and each graph is either an adjacency matrix or a list of ``[u, v, w]``
    triples.  ``signal_seq`` is a list over time, each entry being
    ``list[node][feature]``.

These helpers generate the latter two structures.

"""

from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as _np


# ---------------------------------------------------------------------------
# Lazy MoonLight import
# ---------------------------------------------------------------------------
def _import_moonlight() -> "ScriptLoader":  # type: ignore[name-defined]
    """Return MoonLight's :class:`ScriptLoader` or raise a helpful ImportError.

    The upstream Python interface is a thin JNI bridge to the Java engine.
    For the pip package, Java 21+ must be available on ``PATH`` and pyjnius
    must be importable.

    Raises
    ------
    ImportError
        If the Python package is missing or the Java bridge fails to load.
    """
    try:
        from moonlight import ScriptLoader  # type: ignore
    except Exception as e:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "MoonLight is not available. Install with `pip install moonlight`\n"
            "and ensure a compatible Java runtime is on PATH (Java 21+ is required).\n"
            "If you use Conda, prefer a standard Python or `pyenv` environment."
        ) from e
    return ScriptLoader


# ---------------------------------------------------------------------------
# Loading scripts and getting monitors
# ---------------------------------------------------------------------------
def load_script_from_text(script: str):
    """Load a MoonLight script from a string and return a MoonLightScript.

    This is the most portable option because it avoids file path resolution
    across the Java boundary.

    Parameters
    ----------
    script:
        The content of a ``.mls`` MoonLight script.

    Returns
    -------
    MoonLightScript
        The object exposing ``getMonitor(name)`` and (optionally) domain setters
        like ``setBooleanDomain()`` or ``setMinMaxDomain()``.
    """
    ScriptLoader = _import_moonlight()
    return ScriptLoader.loadFromText(str(script))


def load_script_from_file(path: str | Path):
    """Load a MoonLight script from a file path.

    Preference order:
      1) Use the upstream ``loadFromFile`` (lets MoonLight resolve includes).
      2) Fall back to reading the file and using ``loadFromText``.

    This keeps behavior correct across MoonLight versions while remaining
    resilient to older releases that may lack one of the two methods.

    Parameters
    ----------
    path:
        Path to the ``.mls`` file.

    Returns
    -------
    MoonLightScript

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    RuntimeError
        If both loading strategies fail.
    """
    ScriptLoader = _import_moonlight()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"MoonLight script file not found: {p}")
    # Try the native file loader first to preserve relative include resolution.
    try:
        return ScriptLoader.loadFromFile(str(p))
    except Exception as first_exc:
        # Fallback: read the file and feed text
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as io_exc:  # pragma: no cover - rare I/O issue
            raise RuntimeError(f"Failed reading script file: {p}") from io_exc
        try:
            return ScriptLoader.loadFromText(text)
        except Exception as second_exc:  # pragma: no cover - unlikely if file exists
            raise RuntimeError(
                "MoonLight failed to load the script via both loadFromFile and "
                "loadFromText. See nested exceptions for details."
            ) from second_exc


def set_domain(mls, domain: Literal["boolean", "minmax", "real", "robustness"] | None) -> None:
    """Optionally override the script's evaluation domain.

    ``domain='boolean'`` selects truth‑valued semantics; any of
    ``{'minmax', 'real', 'robustness'}`` selects robustness semantics if
    supported by the installed MoonLight version.  If the method is not
    available or the script already fixes the domain, this is a no‑op.

    Parameters
    ----------
    mls:
        MoonLightScript returned by :func:`load_script_from_*`.
    domain:
        Desired domain selector or ``None`` to leave as specified in the script.
    """
    if domain is None:
        return
    try:
        if domain == "boolean":
            getattr(mls, "setBooleanDomain")()  # type: ignore[attr-defined]
        else:
            # Upstream names have varied; try the most common ones.
            for name in ("setMinMaxDomain", "setRobustnessDomain"):
                fn = getattr(mls, name, None)
                if callable(fn):
                    fn()  # type: ignore[misc]
                    break
    except Exception:
        # Silently ignore — domain control is best effort.
        pass


def get_monitor(mls, name: str):
    """Return a monitor for the formula ``name`` from a MoonLightScript.

    Tries the standard ``getMonitor`` first, then a Pythonic ``get_monitor``
    if present.  Raises a clear ``KeyError`` on failure.
    """
    # Fast path — common case
    try:
        return mls.getMonitor(name)
    except Exception:
        pass
    # Fallback names for older/alternate bindings
    alt = getattr(mls, "get_monitor", None)
    if callable(alt):  # pragma: no cover - hit only on unusual releases
        try:
            return alt(name)
        except Exception as e:
            raise KeyError(f"MoonLight formula not found: {name!r}") from e
    # Final error
    raise KeyError(f"MoonLight formula not found: {name!r}")


# ---------------------------------------------------------------------------
# Grid graph utilities
# ---------------------------------------------------------------------------
def _grid_adjacency(nx: int, ny: int, weight: float) -> list[list[float]]:
    """Return an ``(N×N)`` adjacency matrix for a 2‑D 4‑neighborhood grid.

    Nodes are indexed in **row‑major** order: ``idx(i, j) = i*ny + j``.
    Edges are *undirected* with symmetric weights and no self‑loops.

    Parameters
    ----------
    nx, ny:
        Grid dimensions along x and y; both must be positive.
    weight:
        Edge weight (distance/cost). Must be finite and non‑negative for most
        STREL semantics.

    Returns
    -------
    list[list[float]]
        Dense adjacency matrix in the plain‑Python format MoonLight accepts.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"grid dimensions must be positive, got nx={nx}, ny={ny}")
    n = nx * ny
    adj = [[0.0] * n for _ in range(n)]

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            # 4‑neighborhood: up, down, left, right
            if i > 0:
                v = idx(i - 1, j)
                adj[u][v] = weight
                adj[v][u] = weight
            if i + 1 < nx:
                v = idx(i + 1, j)
                adj[u][v] = weight
                adj[v][u] = weight
            if j > 0:
                v = idx(i, j - 1)
                adj[u][v] = weight
                adj[v][u] = weight
            if j + 1 < ny:
                v = idx(i, j + 1)
                adj[u][v] = weight
                adj[v][u] = weight
    return adj


def _grid_triples(nx: int, ny: int, weight: float) -> list[list[float]]:
    """Return a list of ``[u, v, w]`` triples for the same grid.

    Both directions are emitted for each undirected edge to match the examples
    in the MoonLight documentation and avoid surprises across versions.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"grid dimensions must be positive, got nx={nx}, ny={ny}")
    triples: list[list[float]] = []

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            if i + 1 < nx:
                v = idx(i + 1, j)
                w = float(weight)
                triples.append([float(u), float(v), w])
                triples.append([float(v), float(u), w])
            if j + 1 < ny:
                v = idx(i, j + 1)
                w = float(weight)
                triples.append([float(u), float(v), w])
                triples.append([float(v), float(u), w])
    return triples


def build_grid_graph(
    n_x: int,
    n_y: int,
    *,
    weight: float = 1.0,
    return_format: Literal["adjacency", "triples", "nodes_edges"] = "adjacency",
) -> (
    list[list[float]]
    | list[list[float]]
    | tuple[_np.ndarray, _np.ndarray]
):
    """Construct a simple 2‑D grid graph in various formats.

    Parameters
    ----------
    n_x, n_y:
        Grid size along x and y (both positive).
    weight:
        Edge weight for cardinal neighbors.
    return_format:
        - ``'adjacency'``  → dense ``(N×N)`` matrix (MoonLight accepts this).
        - ``'triples'``    → list of ``[u, v, weight]`` triples (also accepted).
        - ``'nodes_edges'``→ convenience form: a ``(n_x×n_y)`` node grid and a
          ``(#edges×2)`` array of directed edges ``(u, v)`` for further use.

    Returns
    -------
    One of the formats above.
    """
    if return_format == "adjacency":
        return _grid_adjacency(n_x, n_y, float(weight))
    elif return_format == "triples":
        return _grid_triples(n_x, n_y, float(weight))
    elif return_format == "nodes_edges":
        # Nodes as a shaped grid and directed edges (u,v) in integer dtype.
        nodes = _np.arange(n_x * n_y, dtype=_np.int64).reshape(n_x, n_y)
        edges: list[tuple[int, int]] = []
        for i in range(n_x):
            for j in range(n_y):
                v = int(nodes[i, j])
                if i + 1 < n_x:
                    edges.append((v, int(nodes[i + 1, j])))
                    edges.append((int(nodes[i + 1, j]), v))
                if j + 1 < n_y:
                    edges.append((v, int(nodes[i, j + 1])))
                    edges.append((int(nodes[i, j + 1]), v))
        return nodes, _np.asarray(edges, dtype=_np.int64)
    else:  # pragma: no cover
        raise ValueError(f"Unknown return_format: {return_format!r}")


def as_graph_time_series(
    graph: list[list[float]] | list[list[list[float]]],
    times: Sequence[float] | None,
) -> tuple[list[float], list[list[list[float]]]]:
    """Wrap a *static* graph for the spatio‑temporal monitor call.

    MoonLight's spatio‑temporal API expects ``(location_times, graph_seq)``.
    If you pass a single graph and a single time (e.g., ``[0.0]``), this helper
    returns ``([0.0], [graph])``.  If ``graph`` is already a *sequence* of
    graphs, it is returned unchanged (with basic validation).

    Parameters
    ----------
    graph:
        Either a single graph (adjacency matrix or triples) or a list of such
        graphs.
    times:
        Sequence of time stamps at which the spatial model changes; if ``None``,
        a single time ``[0.0]`` is assumed.

    Returns
    -------
    (location_times, graph_seq)
    """
    if times is None:
        t_list = [0.0]
    else:
        t_list = [float(t) for t in times]
        if len(t_list) == 0:
            t_list = [0.0]

    # Heuristic: a graph is a list whose first element is a list. If the first
    # inner element is itself a float, we likely have *adjacency*; if it is a
    # three‑long list, we have *triples*.  Both are fine.
    if graph and isinstance(graph[0], list) and graph and isinstance(graph[0][0], list):  # type: ignore[index]
        # Already a sequence of graphs – assume correct and return.
        return t_list, graph  # type: ignore[return-value]

    # Otherwise wrap the single graph once.
    return t_list, [graph]  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _as_py_float_nested(a: _np.ndarray) -> list[list[list[float]]]:
    """Convert ``(T×N[×F])`` arrays to nested plain‑Python lists of floats.

    NumPy scalars are not Java‑serializable across pyjnius; nested Python
    lists of native ``float`` are the most robust bridge.
    """
    # Ensure we have shape (T, N, F)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape {tuple(a.shape)}")
    # Avoid materializing a copy unless dtype/contiguity requires it.
    a = _np.asarray(a, dtype=float, order="C")
    # Convert to nested lists of native floats.
    return a.tolist()  # NumPy casts to Python float during tolist()


def field_to_signal(
    u: _np.ndarray,
    threshold: float | None = None,
    *,
    layout: Literal["xy_t", "t_xy"] = "xy_t",
) -> list[list[list[float]]]:
    """Convert a 3‑D field on a grid to a MoonLight spatio‑temporal signal.

    Parameters
    ----------
    u:
        Array with layout ``(nx, ny, nt)`` if ``layout='xy_t'`` (default) or
        ``(nt, nx, ny)`` if ``layout='t_xy'``.
    threshold:
        If not ``None``, values ``>= threshold`` are cast to ``1.0`` and others
        to ``0.0`` (Boolean semantics).  If ``None``, values are passed as
        ``float`` (robustness semantics).
    layout:
        Choose between the two supported layouts.

    Returns
    -------
    list[list[list[float]]]
        Nested lists with shape ``[t][node][feature]``, where feature dimension
        is 1.  This feeds directly into ``monitor(..., signal_seq)``.
    """
    a = _np.asarray(u)
    if layout == "xy_t":
        if a.ndim != 3:
            raise ValueError(f"Expected (nx, ny, nt) for layout 'xy_t'; got {a.shape}")
        nx, ny, nt = a.shape
        flat = a.reshape(nx * ny, nt).T  # (T, N)
    elif layout == "t_xy":
        if a.ndim != 3:
            raise ValueError(f"Expected (nt, nx, ny) for layout 't_xy'; got {a.shape}")
        nt, nx, ny = a.shape
        flat = a.reshape(nt, nx * ny)  # (T, N)
    else:  # pragma: no cover
        raise ValueError(f"Unknown layout {layout!r}")

    if threshold is not None:
        flat = (flat >= threshold).astype(float, copy=False)
    else:
        flat = flat.astype(float, copy=False)

    return _as_py_float_nested(flat)  # adds feature dim


# ---------------------------------------------------------------------------
# Version‑tolerant bridge to the monitor call
# ---------------------------------------------------------------------------
def monitor_graph_time_series(mon, graph, sig):
    """Invoke the spatio‑temporal monitor with best‑effort compatibility.

    MoonLight's Python bindings have used slightly different method names
    across releases.  This helper tries the most common ones (in order):

      1) ``monitor_graph_time_series(graph, sig)``  — Pythonic snake_case
      2) ``monitorGraphTimeSeries(graph, sig)``     — original CamelCase
      3) ``monitor(graph, sig)``                    — older generic method
         (and a final swap to ``monitor(sig, graph)`` for *very* old builds)

    Parameters
    ----------
    mon:
        A monitor object returned by :func:`get_monitor`.
    graph:
        Either a single graph or the *sequence* expected by MoonLight.  If you
        pass a single graph, prefer wrapping it with :func:`as_graph_time_series`.
    sig:
        The spatio‑temporal signal as produced by :func:`field_to_signal`.

    Returns
    -------
    The raw output from MoonLight. Most commonly an ``(N×2)`` array‑like where
    the first column is node indices or times and the second the value, but
    shape depends on the monitored formula.
    """
    # Preferred, modern names
    for name in ("monitor_graph_time_series", "monitorGraphTimeSeries"):
        fn = getattr(mon, name, None)
        if callable(fn):
            return fn(graph, sig)
    # Older bindings sometimes only expose a generic 'monitor'
    fn = getattr(mon, "monitor", None)
    if callable(fn):  # pragma: no cover - exercised only on old releases
        try:
            return fn(graph, sig)
        except TypeError:
            # Some very old versions flip the order
            return fn(sig, graph)
    raise RuntimeError(
        "MoonLight monitor exposes none of the expected methods: "
        "'monitor_graph_time_series', 'monitorGraphTimeSeries', or 'monitor'."
    )


__all__ = [
    "load_script_from_text",
    "load_script_from_file",
    "set_domain",
    "get_monitor",
    "build_grid_graph",
    "field_to_signal",
    "as_graph_time_series",
    "monitor_graph_time_series",
]
