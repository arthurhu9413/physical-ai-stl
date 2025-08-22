from __future__ import annotations
"""
Evaluate a saved 2-D heat-equation rollout against a MoonLight STREL spec.

This script is intentionally *standalone* and talks directly to the official
`moonlight` Python package, following the usage documented here:
  - https://github.com/MoonLightSuite/moonlight/wiki/Python

It does **not** depend on `physical_ai_stl.monitoring.moonlight_helper`
anymore, because that layer has slightly different expectations about
the monitor signature and was causing runtime TypeError issues like:

    SpatialTemporalScriptComponent.monitor() missing 3 required
    positional arguments: 'graph', 'signalTimeArray', and 'signalValues'

Instead, we construct the spatial model and the spatio-temporal signal
directly in the format MoonLight expects and call:

    result = monitor.monitor(graph_times, graph, signal_times, signal_values)

where:
  * graph_times     = [0.0]  (static spatial model)
  * graph           = [[[src, dst, weight], ...]]  (4-neighborhood grid)
  * signal_times    = [0.0, 1.0, 2.0, ..., nt-1]   (or scaled by --dt)
  * signal_values   = shape [n_locations][nt][1]   (1-D scalar per node)

You can run this exactly as you tried before, e.g.:

    python scripts/eval_heat2d_moonlight.py ^
      --field assets\\heat2d_scalar\\field_xy_t.npy ^
      --layout xy_t ^
      --mls scripts\\specs\\contain_hotspot.mls ^
      --formula contain_hotspot ^
      --out-json results\\heat2d_contain_hotspot.json
"""

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Tuple, List

import numpy as np

# -------------------------------------------------------------------------
# MoonLight import (directly from the PyPI package)
# -------------------------------------------------------------------------
try:
    # This is the API documented in the official MoonLight wiki.
    from moonlight import ScriptLoader
    _MOONLIGHT_IMPORT_ERROR: BaseException | None = None
except Exception as exc:  # pragma: no cover
    ScriptLoader = None  # type: ignore
    _MOONLIGHT_IMPORT_ERROR = exc


# -------------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------------

def _load_field_from_npy(path: Path, layout: str = "xy_t") -> Tuple[np.ndarray, int, int, int]:
    """Load a 3-D numpy array and return `(u, nx, ny, nt)`.

    layout == "xy_t"  -> array is (nx, ny, nt)
    layout == "t_xy"  -> array is (nt, nx, ny)
    """
    a = np.load(path, mmap_mode="r")
    if a.ndim != 3:
        raise ValueError(f"Expected 3-D array in {path}, got shape {a.shape}")
    if layout == "xy_t":
        nx, ny, nt = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        return np.asarray(a), nx, ny, nt
    if layout == "t_xy":
        nt, nx, ny = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        return np.asarray(a).transpose(1, 2, 0), nx, ny, nt  # (nx, ny, nt)
    raise ValueError(f"Invalid layout '{layout}'; use 'xy_t' or 't_xy'.")


def _slice_time(u: np.ndarray, t_start: int | None, t_end: int | None) -> np.ndarray:
    """Slice a field `(nx, ny, nt)` by time indices without copying if possible."""
    if t_start is None and t_end is None:
        return u
    t0 = 0 if t_start is None else int(t_start)
    t1 = u.shape[-1] if t_end is None else int(t_end)
    if t0 < 0 or t1 < 0 or t0 > t1 or t1 > u.shape[-1]:
        raise ValueError(f"Invalid time slice [{t0}:{t1}] for nt={u.shape[-1]}")
    return u[..., t0:t1]


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _spec_declares_boolean_signal(mls_text: str) -> bool:
    """Detect whether the script declares a boolean-valued input signal.

    We look for a `bool` type inside the `signal { ... }` block.

    Note:
        `domain boolean;` controls the semantics of the output (boolean
        vs robustness) and does not by itself require a boolean-valued
        input signal, so we deliberately ignore it here.
    """
    mt_sig = re.search(
        r"signal\s*\{[^}]*\bbool\b",
        mls_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return bool(mt_sig)


def _spec_uses_boolean_domain(mls_text: str) -> bool:
    """Return True if the script explicitly sets `domain boolean;`."""
    return bool(re.search(r"domain\s+boolean\s*;", mls_text, flags=re.IGNORECASE))


def _auto_threshold(u: np.ndarray, z_k: float | None, quantile: float | None) -> float:
    flat = np.asarray(u, dtype=float).reshape(-1)
    if quantile is not None:
        if not (0.0 < quantile < 1.0):
            raise ValueError("--quantile must be in (0, 1)")
        return float(np.quantile(flat, quantile))
    m = float(flat.mean())
    s = float(flat.std(ddof=0))
    k = 0.5 if z_k is None else float(z_k)
    return float(m + k * s)


# -------------------------------------------------------------------------
# Converters: build_grid_graph / field_to_signal
# -------------------------------------------------------------------------

def build_grid_graph(nx: int, ny: int, weight: float = 1.0) -> Tuple[List[float], List[List[List[float]]]]:
    """Build a 4-neighborhood grid as MoonLight expects.

    Returns (graph_times, graph_edges) where:

      * graph_times  = [0.0]   (static spatial model)
      * graph_edges  = [edges] where `edges` is a list of triples
                       [src, dst, weight] with float indices.

    Node indexing is row-major: node = i * ny + j (0-based).
    """
    edges: List[List[float]] = []
    for i in range(nx):
        for j in range(ny):
            src = float(i * ny + j)
            # 4-neighborhood: up, down, left, right
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    dst = float(ii * ny + jj)
                    edges.append([src, dst, float(weight)])
    graph_times = [0.0]            # graph is static over time
    graph = [edges]                # one snapshot at t=0.0
    return graph_times, graph


def field_to_signal(
    u: np.ndarray,
    threshold: float | None,
    dt: float,
) -> Tuple[List[float], List[List[List[float]]]]:
    """Convert `(nx, ny, nt)` field into MoonLight's spatio-temporal signal.

    We generate:

      * signal_times  = [0, dt, 2*dt, ..., (nt-1)*dt]
      * signal_values = shape [n_locations][nt][1]

    If `threshold` is not None, values are binarized (>= thr -> 1.0 else 0.0).
    """
    if u.ndim != 3:
        raise ValueError(f"Expected (nx, ny, nt) field, got shape {u.shape}")
    nx, ny, nt = u.shape
    signal_times: List[float] = [float(t * dt) for t in range(nt)]
    signal_values: List[List[List[float]]] = []

    for i in range(nx):
        for j in range(ny):
            ts: List[List[float]] = []
            for t in range(nt):
                v = float(u[i, j, t])
                if threshold is not None:
                    v = 1.0 if v >= threshold else 0.0
                ts.append([v])  # 1-D feature
            signal_values.append(ts)

    return signal_times, signal_values


def _summarize_spatiotemporal_output(out: Any) -> dict:
    """Turn MoonLight's output into a small summary dict.

    Positive values = satisfaction; negative = violation.
    For spatio-temporal outputs, we conservatively take the minimum across
    locations at each time (i.e. require all locations to satisfy).
    """
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 1:
        per_time = arr
    elif arr.ndim == 2:
        # either (n_times, 2) [time, val] or (n_times, n_nodes).
        if arr.shape[1] == 2:
            per_time = arr[:, 1]
        else:
            per_time = arr.min(axis=1)
    else:
        arr2 = np.squeeze(arr)
        if arr2.ndim == 1:
            per_time = arr2
        elif arr2.ndim == 2:
            per_time = arr2.min(axis=1)
        else:
            raise ValueError(f"Unexpected monitor output shape {arr.shape}")

    satisfied_idx = np.flatnonzero(per_time > 0.0)
    satisfied_eventually = bool(satisfied_idx.size > 0)
    first_sat_idx = int(satisfied_idx[0]) if satisfied_eventually else -1

    return {
        "out_shape": tuple(arr.shape),
        "per_time_len": int(per_time.shape[0]),
        "satisfied_eventually": satisfied_eventually,
        "first_satisfaction_index": first_sat_idx,
        "per_time_min": float(np.min(per_time)),
        "per_time_max": float(np.max(per_time)),
    }


# -------------------------------------------------------------------------
# Suppress / capture MoonLight's very verbose stdout/stderr
# -------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_moonlight_output(to_file: Path | None = None):
    """Temporarily redirect process-level stdout/stderr while MoonLight runs.

    MoonLight's Java backend can print huge numeric arrays directly to the
    terminal. We redirect the low-level file descriptors so that even
    Java-side prints are captured.

    If `to_file` is not None, everything is written to that file.
    Otherwise it's discarded (os.devnull).
    """
    if to_file is None:
        target = open(os.devnull, "w")
    else:
        to_file = Path(to_file)
        to_file.parent.mkdir(parents=True, exist_ok=True)
        target = open(to_file, "w", encoding="utf-8")

    # Duplicate low-level file descriptors for stdout/stderr.
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(target.fileno(), 1)
        os.dup2(target.fileno(), 2)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        target.close()


# -------------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Audit a saved Heat2D run with a MoonLight STREL spec "
            "(direct call to moonlight.monitor)."
        ),
    )

    src = ap.add_argument_group("input source")
    src.add_argument(
        "--field",
        type=Path,
        required=True,
        help="Single .npy file with a 3-D array (nx, ny, nt) or (nt, nx, ny).",
    )
    src.add_argument(
        "--layout",
        type=str,
        default="xy_t",
        choices=("xy_t", "t_xy"),
        help="Axis order of --field.",
    )

    timeg = ap.add_argument_group("time")
    timeg.add_argument("--t-start", type=int, default=None, help="Start index (inclusive).")
    timeg.add_argument("--t-end", type=int, default=None, help="End index (exclusive).")
    timeg.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step used to build the signal time array.",
    )

    grid = ap.add_argument_group("grid")
    grid.add_argument("--nx", type=int, default=None, help="Grid size in x (rows).")
    grid.add_argument("--ny", type=int, default=None, help="Grid size in y (cols).")
    grid.add_argument(
        "--adj-weight",
        type=float,
        default=1.0,
        help="Edge weight for the 4-neighborhood grid.",
    )

    spec = ap.add_argument_group("MoonLight spec")
    spec.add_argument(
        "--mls",
        type=Path,
        required=True,
        help="Path to a MoonLight .mls script.",
    )
    spec.add_argument(
        "--formula",
        type=str,
        required=True,
        help="Formula name inside the .mls script.",
    )

    binz = ap.add_argument_group("binarization (if spec expects boolean-valued signal)")
    binz.add_argument(
        "--binarize",
        dest="binarize",
        action="store_true",
        help="Force binary signal (>= threshold -> 1 else 0).",
    )
    binz.add_argument(
        "--no-binarize",
        dest="binarize",
        action="store_false",
        help="Force real-valued signal (no thresholding).",
    )
    binz.set_defaults(binarize=None)
    binz.add_argument(
        "--z-k",
        type=float,
        default=0.5,
        help="Threshold = mean + k*std (ignored if --quantile is set).",
    )
    binz.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="If set, threshold = this quantile of field values (0<q<1).",
    )
    binz.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute threshold; overrides --quantile and --z-k if set.",
    )

    outg = ap.add_argument_group("output")
    outg.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write a JSON summary to this path.",
    )

    args = ap.parse_args()

    # ---- MoonLight availability guard ----------------------------------
    if ScriptLoader is None:  # pragma: no cover
        print("[MoonLight] Could not import ScriptLoader from 'moonlight':")
        print(f"  {_MOONLIGHT_IMPORT_ERROR}")
        sys.exit(1)

    # ---- Load field ----------------------------------------------------
    u, nx_auto, ny_auto, nt = _load_field_from_npy(args.field, layout=args.layout)

    if args.nx is not None and int(args.nx) != nx_auto:
        raise ValueError(
            f"--nx={args.nx} does not match field's first dimension nx_auto={nx_auto}"
        )
    if args.ny is not None and int(args.ny) != ny_auto:
        raise ValueError(
            f"--ny={args.ny} does not match field's second dimension ny_auto={ny_auto}"
        )

    nx = nx_auto if args.nx is None else int(args.nx)
    ny = ny_auto if args.ny is None else int(args.ny)

    # Optional time slice
    u = _slice_time(u, args.t_start, args.t_end)
    nt = int(u.shape[-1])

    # Basic stats on the (optionally sliced) field
    flat_u = np.asarray(u, dtype=float).reshape(-1)
    field_min = float(flat_u.min())
    field_max = float(flat_u.max())
    field_mean = float(flat_u.mean())
    field_std = float(flat_u.std(ddof=0))

    print(f"[input] field: shape=(nx={nx}, ny={ny}, nt={nt})")
    print(
        "[input] value stats: "
        f"min={field_min:.6g}, max={field_max:.6g}, "
        f"mean={field_mean:.6g}, std={field_std:.6g}"
    )
    print(f"[spec]  mls={args.mls}  formula={args.formula}")

    # ---- Load spec via MoonLight --------------------------------------
    if not args.mls.exists():
        raise FileNotFoundError(f"MoonLight spec not found: {args.mls}")
    mls_text = _read_text(args.mls)
    spec_has_bool_signal = _spec_declares_boolean_signal(mls_text)
    spec_has_boolean_domain = _spec_uses_boolean_domain(mls_text)

    domain_str = "boolean" if spec_has_boolean_domain else "min-max (robustness) / default"
    print(f"[spec]  declares boolean-valued signal: {spec_has_bool_signal}")
    print(f"[spec]  domain: {domain_str}")

    try:
        script_obj = ScriptLoader.loadFromFile(str(args.mls))
    except Exception as exc:
        print("[MoonLight] ERROR: failed to load script via ScriptLoader.loadFromFile")
        print(f"  {exc}")
        sys.exit(1)

    try:
        mon = script_obj.getMonitor(args.formula)
    except Exception as exc:
        print(f"[MoonLight] ERROR: script has no formula named '{args.formula}'")
        print(f"  {exc}")
        sys.exit(1)

    # ---- Build graph and signal ---------------------------------------
    graph_times, graph = build_grid_graph(nx, ny, weight=float(args.adj_weight))

    # Decide binarization (boolean-valued input signal -> default True)
    if args.binarize is None:
        do_binarize = spec_has_bool_signal
    else:
        do_binarize = bool(args.binarize)

    if do_binarize:
        if args.threshold is not None:
            thr = float(args.threshold)
        else:
            thr = _auto_threshold(u, z_k=args.z_k, quantile=args.quantile)
    else:
        thr = None

    # Binarization diagnostics (before building the signal)
    if do_binarize and thr is not None:
        n_total = int(flat_u.size)
        n_ge = int(np.count_nonzero(flat_u >= thr))
        frac_ge = n_ge / n_total if n_total > 0 else float("nan")
        print(
            f"[signal] binarization enabled (threshold={thr:.6g}): "
            f"{n_ge}/{n_total} ~= {frac_ge:.3%} samples >= threshold"
        )
        if n_ge == 0:
            print("[signal] WARNING: threshold is above the maximum field value -> all zeros after binarization.")
        elif n_ge == n_total:
            print("[signal] WARNING: threshold is at or below the minimum field value -> all ones after binarization.")
    else:
        n_ge = None
        frac_ge = None

    signal_times, signal_values = field_to_signal(u, threshold=thr, dt=float(args.dt))

    print(f"[graph] times={graph_times} (len={len(graph_times)})  edges_per_snapshot={len(graph[0])}")
    print(f"[signal] times[0..3]={signal_times[:3]}  (#loc={len(signal_values)}, nt={len(signal_values[0])})")
    if do_binarize and thr is not None:
        print(f"[signal] binarized with threshold={thr:.6g}")
    else:
        print("[signal] real-valued (no binarization)")

    # ---- Call MoonLight monitor ---------------------------------------
    if args.out_json is not None:
        raw_log_path = Path(args.out_json).with_suffix(".moonlight_raw.txt")
    else:
        raw_log_path = Path("moonlight_raw_output.txt")

    print(f"[MoonLight] Calling monitor; raw MoonLight stdout/stderr will be captured in {raw_log_path}")

    try:
        with _suppress_moonlight_output(raw_log_path):
            out = mon.monitor(graph_times, graph, signal_times, signal_values)
    except TypeError as exc:
        print("[MoonLight] ERROR: call to monitor(graph_times, graph, signal_times, signal_values) failed.")
        print("  This usually means MoonLight changed its Python signature again.")
        print(f"  Underlying error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print("[MoonLight] ERROR: unexpected exception from monitor(...):")
        print(f"  {exc}")
        sys.exit(1)

    # ---- Summarize results --------------------------------------------
    summary = _summarize_spatiotemporal_output(out)
    summary.update(
        {
            "nx": nx,
            "ny": ny,
            "nt": nt,
            "field": str(args.field),
            "mls": str(args.mls),
            "formula": str(args.formula),
            "binarized": bool(do_binarize),
            "threshold": None if thr is None else float(thr),
            "graph_times": graph_times,
            "field_min": field_min,
            "field_max": field_max,
            "field_mean": field_mean,
            "field_std": field_std,
            "n_samples": int(flat_u.size),
            "n_samples_ge_threshold": None if n_ge is None else int(n_ge),
            "fraction_ge_threshold": None if frac_ge is None else float(frac_ge),
            "spec_declares_bool_signal": bool(spec_has_bool_signal),
            "spec_uses_boolean_domain": bool(spec_has_boolean_domain),
            "moonlight_raw_log": str(raw_log_path),
        }
    )

    print("\n[summary]")
    print(f"  output shape: {summary['out_shape']}")
    print(f"  per-time length: {summary['per_time_len']}")
    if summary["satisfied_eventually"]:
        print("  verdict: PASS - property satisfied at least once")
        print(f"  first satisfaction index: t={summary['first_satisfaction_index']}")
    else:
        print("  verdict: FAIL - property never satisfied over the horizon")
    print(f"  per-time min/max: {summary['per_time_min']:.3g} .. {summary['per_time_max']:.3g}")
    print(f"  MoonLight raw stdout/stderr captured in: {raw_log_path}\n")

    # Optional JSON artifact
    if args.out_json is not None:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
        print(f"[MoonLight] Wrote JSON summary to {outp}")


if __name__ == "__main__":
    main()
