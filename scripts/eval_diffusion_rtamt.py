#!/usr/bin/env python3
# ruff: noqa: I001
from __future__ import annotations

"""
Evaluate STL robustness for a saved diffusion PINN field.

What this script does
---------------------
1) Loads a *_field.pt checkpoint that contains a spatio‑temporal field (e.g., `u`).
2) Reduces spatial dimensions to a single time series via a chosen aggregator.
3) Monitors the series against an STL specification using RTAMT (dense or discrete time).
   - Built‑ins: global upper/lower/range bounds under the outer `always`.
   - NEW: pass a *custom* STL formula with ``--stl`` (variable name is ``s``).
4) Supports optional evaluation over a **time window** (``--t0/--t1`` in seconds or
   ``--idx0/--idx1`` in indices). When no time vector is present, ``--dt`` enables
   absolute‑time windows.

Robustness semantics follow RTAMT (when installed). For the simple bound predicates
(upper/lower/range), an exact, fast fallback is provided if RTAMT is unavailable.

Examples
--------
Dense‑time, custom STL over 3.0s–5.0s window:
    python scripts/eval_diffusion_rtamt.py --ckpt results/diffusion1d_week2_field.pt \\
        --agg quantile --q 0.9 --semantics dense --dt 0.01 --t0 3.0 --t1 5.0 \\
        --stl "always[0,1](s <= 0.8)"

Discrete‑time, built‑in `always upper`:
    python scripts/eval_diffusion_rtamt.py --ckpt results/diffusion1d_week2_field.pt \\
        --semantics discrete --agg amax --spec upper --u-max 1.0

Notes
-----
* STL variable name is **s** (post‑spatially‑reduced scalar). For custom specs that
  refer to additional variables, use your own monitoring script.
* RTAMT supports time‑bounded operators like ``always[a,b]`` and ``eventually[a,b]``
  (syntax accepts comma/colon separators depending on version). See RTAMT docs.
* JSON summary (``--json``) captures all inputs, the computed dt, and robustness.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

from physical_ai_stl.monitoring.rtamt_monitor import (
    evaluate_series as _rtamt_evaluate_series,
    satisfied as _rtamt_satisfied,
    stl_always_upper_bound as _rtamt_stl_upper,
)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate STL robustness with RTAMT for a saved diffusion PINN field."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="results/diffusion1d_week2_field.pt",
        help="Path to *_field.pt artifact saved by training script",
    )
    ap.add_argument(
        "--var",
        type=str,
        default="u",
        help="Field variable name in the checkpoint",
    )
    ap.add_argument(
        "--semantics",
        type=str,
        choices=["dense", "discrete"],
        default="dense",
        help="Time semantics to use for STL monitor",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step (if not inferable from checkpoint 'T')",
    )
    # --- NEW: evaluation window ---
    ap.add_argument(
        "--t0",
        type=float,
        default=None,
        help="Start time (seconds) for evaluation window; requires 'T' or --dt.",
    )
    ap.add_argument(
        "--t1",
        type=float,
        default=None,
        help="End time (seconds, inclusive) for evaluation window; requires 'T' or --dt.",
    )
    ap.add_argument(
        "--idx0",
        type=int,
        default=None,
        help="Start index for evaluation window (Python slicing semantics).",
    )
    ap.add_argument(
        "--idx1",
        type=int,
        default=None,
        help="End index (exclusive) for evaluation window (Python slicing semantics).",
    )
    ap.add_argument(
        "--agg",
        type=str,
        choices=["mean", "amax", "amin", "median", "quantile", "lp", "softmax"],
        default="mean",
        help="Spatial reducer applied before monitoring",
    )
    ap.add_argument(
        "--p",
        type=float,
        default=2.0,
        help="p for --agg lp (p-norm)",
    )
    ap.add_argument(
        "--q",
        type=float,
        default=0.95,
        help="q for --agg quantile in [0,1]",
    )
    ap.add_argument(
        "--temp",
        type=float,
        default=0.1,
        help="temperature for --agg softmax (higher≈harder max)",
    )
    # --- STL preset vs custom ---
    ap.add_argument(
        "--spec",
        type=str,
        choices=["upper", "lower", "range"],
        default="upper",
        help="Which STL predicate to enforce under the outer 'always'",
    )
    ap.add_argument(
        "--stl",
        type=str,
        default=None,
        help=(
            "Custom RTAMT STL formula over variable 's', e.g. "
            "\"always[0,5](s <= 0.8)\". Overrides --spec."
        ),
    )
    ap.add_argument(
        "--u-max",
        dest="u_max",
        type=float,
        default=None,
        help="Upper bound for 'upper'/'range' (ignored if --stl is given)",
    )
    ap.add_argument(
        "--u-min",
        dest="u_min",
        type=float,
        default=None,
        help="Lower bound for 'lower'/'range' (ignored if --stl is given)",
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        type=str,
        default=None,
        help="Optional path to write a JSON summary",
    )
    ap.add_argument(
        "--dump-series",
        dest="dump_series",
        type=str,
        default=None,
        help="Optional path to write the reduced series as CSV with columns t,s",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose prints (shape guesses, dt, etc.)",
    )
    return ap


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------


def _load_ckpt(path: Path) -> Mapping[str, Any]:
    try:
        import torch  # local import to avoid hard dependency at import-time
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"[fatal] PyTorch not available: {e}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise SystemExit(f"[fatal] Unexpected checkpoint format (expected dict): {type(data)}")
    return data


def _as_tensor(x: Any):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu").float()
    except Exception:
        pass
    # Fallback: try to build a torch tensor (this keeps code uniform)
    import torch  # type: ignore
    return torch.as_tensor(x, dtype=torch.float32).detach().to("cpu")


def _infer_time_axis(u: Any, T: Any | None) -> tuple[int, int]:
    U = _as_tensor(u)
    shape = list(U.shape)
    if T is not None:
        try:
            tlen = int(_as_tensor(T).numel())
            matches = [i for i, s in enumerate(shape) if int(s) == tlen]
            if len(matches) == 1:
                return matches[0], tlen
        except Exception:
            pass
    # Default: last axis
    if not shape:
        raise SystemExit("[fatal] Empty tensor for field 'u'")
    return len(shape) - 1, int(shape[-1])


def _reduce_spatial(U, time_axis: int, *, mode: str, p: float, q: float, temp: float):
    import torch
    X = _as_tensor(U)
    # Move time to last, then flatten spatial dims
    if time_axis != X.ndim - 1:
        X = X.movedim(time_axis, -1)
    if X.ndim == 1:
        # Already [nt]
        S = X
    else:
        spatial = X.flatten(0, X.ndim - 2)  # [ns, nt]
        if mode == "mean":
            S = spatial.mean(dim=0)
        elif mode == "amax":
            S = spatial.amax(dim=0)
        elif mode == "amin":
            S = spatial.amin(dim=0)
        elif mode == "median":
            S = spatial.median(dim=0).values
        elif mode == "quantile":
            qv = float(q)
            if not (0.0 <= qv <= 1.0):
                raise SystemExit(f"[fatal] --q must be in [0,1], got {qv}")
            S = torch.quantile(spatial, qv, dim=0)
        elif mode == "lp":
            pv = float(p)
            # Lp norm converted to an "average magnitude" (scale‑aware yet monotone)
            S = spatial.abs().pow(pv).mean(dim=0).pow(1.0 / pv)
            # (Avoid exact max to keep the reducer continuous for p<∞)
        elif mode == "softmax":
            # Differentiable max via log‑sum‑exp scaling
            tau = float(temp)
            S = (spatial * tau).logsumexp(dim=0) / tau
        else:
            raise SystemExit(f"[fatal] Unknown --agg={mode!r}")
    return S.detach().to("cpu")


def _infer_dt(T: Any | None, fallback: float | None, nt: int) -> float:  # noqa: ARG001
    if T is not None:
        tt = _as_tensor(T).flatten().numpy()
        if tt.size >= 2:
            diffs = (tt[1:] - tt[:-1]).astype("float64")
            dt = float(diffs.mean())
            # If highly non‑uniform, emit a verbose note.
            if diffs.size and (abs(diffs.std()) > 1e-6 * max(1.0, abs(dt))):
                print(
                    f"[note] Non‑uniform time grid (σ={diffs.std():.3g}); using mean dt={dt:.6g}",
                    file=sys.stderr,
                )
            return dt
    if fallback is None:
        raise SystemExit("[fatal] Could not infer dt (no 'T' and no --dt provided)")
    return float(fallback)


def _build_spec(
    var: str,
    kind: str,
    u_min: float | None,
    u_max: float | None,
    semantics: str,
):
    try:
        # Try building via RTAMT; reuse helper for the common 'upper' case.
        if kind == "upper":
            return _rtamt_stl_upper(
                var=var,
                u_max=float(u_max if u_max is not None else 1.0),
                time_semantics=semantics,
            )
        # Generic builder for others
        import rtamt  # type: ignore
        SpecCls = (
            getattr(rtamt, "StlDenseTimeSpecification", None)
            or getattr(rtamt, "StlDenseTimeOfflineSpecification", None)
            if semantics == "dense"
            else None
        )
        if SpecCls is None:
            SpecCls = rtamt.StlDiscreteTimeSpecification
        spec = SpecCls()
        spec.declare_var(var, "float")
        if kind == "lower":
            if u_min is None:
                raise SystemExit("[fatal] --u-min is required for --spec lower")
            spec.spec = f"always ({var} >= {float(u_min)})"
        elif kind == "range":
            if u_min is None or u_max is None:
                raise SystemExit("[fatal] --u-min and --u-max are required for --spec range")
            spec.spec = f"always ( ({float(u_min)} <= {var}) and ({var} <= {float(u_max)}) )"
        else:
            raise SystemExit(f"[fatal] Unknown --spec={kind!r}")
        spec.parse()
        return spec
    except Exception as e:
        # RTAMT missing or misconfigured → gracefully fall back later
        if isinstance(e, SystemExit):
            raise
        return None


def _build_custom_spec(formula: str, semantics: str):
    """Build a custom RTAMT spec over variable 's' from a raw STL formula string."""
    try:
        import rtamt  # type: ignore
        SpecCls = (
            getattr(rtamt, "StlDenseTimeSpecification", None)
            or getattr(rtamt, "StlDenseTimeOfflineSpecification", None)
            if semantics == "dense"
            else None
        )
        if SpecCls is None:
            SpecCls = rtamt.StlDiscreteTimeSpecification
        spec = SpecCls()
        spec.declare_var("s", "float")
        spec.spec = str(formula)
        spec.parse()
        return spec
    except Exception as e:
        raise SystemExit(
            "[fatal] --stl requires RTAMT to be installed and the formula to be valid.\n"
            f"Reason: {e}"
        )


def _robustness_fallback(
    series: Iterable[float],
    *,
    kind: str,
    u_min: float | None,
    u_max: float | None,
) -> float:
    import numpy as np
    s = np.asarray(list(series), dtype=float)
    if s.ndim != 1 or s.size == 0:
        return float("nan")
    if kind == "upper":
        if u_max is None:
            raise SystemExit("[fatal] --u-max required for --spec upper")
        rob_t = (float(u_max) - s)
        return float(np.min(rob_t))
    if kind == "lower":
        if u_min is None:
            raise SystemExit("[fatal] --u-min required for --spec lower")
        rob_t = (s - float(u_min))
        return float(np.min(rob_t))
    if kind == "range":
        if u_min is None or u_max is None:
            raise SystemExit("[fatal] --u-min and --u-max required for --spec range")
        rob1 = s - float(u_min)
        rob2 = float(u_max) - s
        return float(np.min(np.minimum(rob1, rob2)))
    raise SystemExit(f"[fatal] Unknown spec kind {kind!r}")


def _evaluate(
    var_name: str,  # noqa: ARG001 - kept for clarity in prints when extended
    series: Sequence[float],
    dt: float,
    *,
    spec_kind: str,
    semantics: str,
    u_min: float | None,
    u_max: float | None,
    custom_formula: str | None = None,
) -> tuple[float, bool, str]:
    # If user passed a custom formula, we must use RTAMT
    if custom_formula:
        spec = _build_custom_spec(custom_formula, semantics)
        rob = float(_rtamt_evaluate_series(spec, var="s", series=series, dt=float(dt)))
        try:
            sat = bool(_rtamt_satisfied(rob))
        except Exception:
            sat = bool(rob >= 0.0)
        return rob, sat, "rtamt"

    # Try RTAMT first for presets
    spec = _build_spec("s", spec_kind, u_min, u_max, semantics)
    if spec is not None:
        rob = float(_rtamt_evaluate_series(spec, var="s", series=series, dt=float(dt)))
        try:
            sat = bool(_rtamt_satisfied(rob))
        except Exception:
            sat = bool(rob >= 0.0)
        return rob, sat, "rtamt"
    # Fallback: exact discrete‑time robustness
    rob = _robustness_fallback(series, kind=spec_kind, u_min=u_min, u_max=u_max)
    sat = bool(rob >= 0.0)
    return rob, sat, "fallback"


def _slice_by_time_and_index(
    series: Sequence[float],
    T: Any | None,
    dt: float | None,
    t0: float | None,
    t1: float | None,
    idx0: int | None,
    idx1: int | None,
) -> tuple[list[float], list[float]]:
    """Return (series_sliced, tvec_sliced). If neither time nor index window is given,
    returns the input series and an inferred time vector.

    * If T is None, dt must be provided to build a synthetic time vector starting at 0.
    * t1 is treated as inclusive.
    * Index slicing uses Python semantics [idx0:idx1].
    """
    import numpy as np

    s = np.asarray(series, dtype=float).reshape(-1)
    nt = int(s.size)

    # Build a time vector
    if T is not None:
        tt = _as_tensor(T).flatten().numpy().astype(float)
        if tt.size != nt:
            # Best effort: trim to min length
            m = int(min(tt.size, nt))
            tt = tt[:m]
            s = s[:m]
            nt = m
    else:
        if dt is None:
            raise SystemExit("[fatal] No time vector in checkpoint. Provide --dt for time windows.")
        tt = (np.arange(nt, dtype=float) * float(dt))

    # Time window
    if (t0 is not None) or (t1 is not None):
        lo = float(t0) if t0 is not None else float(tt[0])
        hi = float(t1) if t1 is not None else float(tt[-1])
        if hi < lo:
            raise SystemExit(f"[fatal] --t1 ({hi}) < --t0 ({lo})")
        mask = (tt >= lo) & (tt <= hi)
        if not mask.any():
            raise SystemExit("[fatal] Empty slice after applying --t0/--t1 window")
        tt = tt[mask]
        s = s[mask]
        nt = int(s.size)

    # Index window
    if (idx0 is not None) or (idx1 is not None):
        i0 = int(idx0) if idx0 is not None else 0
        i1 = int(idx1) if idx1 is not None else nt
        if i1 < i0:
            raise SystemExit(f"[fatal] --idx1 ({i1}) < --idx0 ({i0})")
        sl = slice(i0, i1, None)
        tt = tt[sl]
        s = s[sl]
        nt = int(s.size)

    return s.astype(float).tolist(), tt.astype(float).tolist()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


@dataclass
class Args:
    ckpt: str
    var: str
    semantics: str          # 'dense' or 'discrete'
    dt: float | None
    t0: float | None
    t1: float | None
    idx0: int | None
    idx1: int | None
    agg: str                # spatial reducer
    p: float                # lp norm (for --agg lp)
    q: float                # quantile (for --agg quantile)
    temp: float             # temperature (for --agg softmax)
    spec: str               # 'upper' | 'lower' | 'range'
    stl: str | None
    u_max: float | None
    u_min: float | None
    json_out: str | None
    dump_series: str | None
    verbose: bool


def main() -> None:
    args = Args(**vars(build_argparser().parse_args()))

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"[fatal] Checkpoint not found: {ckpt_path}")

    data = _load_ckpt(ckpt_path)
    if args.var not in data:
        # Give a helpful hint if user passed the *model* checkpoint
        keys = ", ".join(sorted(data.keys()))
        raise SystemExit(
            f"[fatal] Variable {args.var!r} not found in checkpoint. "
            f"Available keys: {keys}. Did you pass the *_field.pt artifact?"
        )
    U = data[args.var]
    # Try common time vector key names
    T = None
    for key in ("T", "t", "time", "times"):
        if key in data:
            T = data[key]
            break

    # Determine which axis is time and reduce spatially to a single series
    time_axis, nt0 = _infer_time_axis(U, T)
    series_tensor = _reduce_spatial(
        U,
        time_axis=time_axis,
        mode=str(args.agg).lower(),
        p=float(args.p),
        q=float(args.q),
        temp=float(args.temp),
    )
    # Coerce to a plain list[float] for the monitor
    full_series: list[float] = [float(x) for x in series_tensor.flatten().tolist()]
    if len(full_series) != nt0:  # sanity
        nt0 = len(full_series)

    # Infer dt (from full series) — used for windows and for dense-time semantics
    dt = _infer_dt(T, args.dt, nt0)

    # Apply optional evaluation window(s)
    series, tvec = _slice_by_time_and_index(
        full_series, T, dt, args.t0, args.t1, args.idx0, args.idx1
    )
    nt = len(series)

    # Evaluate robustness
    rob, sat, backend = _evaluate(
        args.var,
        series,
        dt,
        spec_kind=str(args.spec).lower(),
        semantics=str(args.semantics).lower(),
        u_min=args.u_min,
        u_max=args.u_max,
        custom_formula=args.stl,
    )

    # Pretty print
    if args.verbose:
        import numpy as np
        Ushape = tuple(int(s) for s in _as_tensor(U).shape)
        print(f"[info] ckpt={ckpt_path}")
        print(f"[info] var={args.var!r} shape={Ushape}  time_axis={time_axis}  nt_full={nt0}  nt_eval={nt}")
        print(f"[info] semantics={args.semantics}  dt={dt:.6g}  backend={backend}")
        print(
            f"[info] agg={args.agg}"
            + (f"(p={args.p:g})" if args.agg == "lp" else "")
            + (f"(q={args.q:g})" if args.agg == "quantile" else "")
            + (f"(temp={args.temp:g})" if args.agg == "softmax" else "")
        )
        arr = np.asarray(series, dtype=float)
        print(
            f"[info] series stats: min={arr.min():.6g}  max={arr.max():.6g}  "
            f"mean={arr.mean():.6g}  std={arr.std():.6g}"
        )
        if (args.t0 is not None) or (args.t1 is not None) or (args.idx0 is not None) or (args.idx1 is not None):
            t0v = tvec[0] if tvec else float('nan')
            t1v = tvec[-1] if tvec else float('nan')
            print(f"[info] window: t∈[{t0v:.6g}, {t1v:.6g}]  idx∈[{args.idx0 if args.idx0 is not None else 0}:{args.idx1 if args.idx1 is not None else nt0}]")

    status = "SAT" if sat else "UNSAT"
    print(f"Robustness = {rob:.6g}   [{status}]   (backend={backend})")

    # Optional outputs
    if args.dump_series:
        outp = Path(args.dump_series)
        outp.parent.mkdir(parents=True, exist_ok=True)
        try:
            with outp.open("w", encoding="utf-8") as f:
                f.write("t,s\n")
                for t, v in zip(tvec, series):
                    f.write(f"{float(t):.9g},{float(v):.9g}\n")
            print(f"[saved] reduced series → {outp}")
        except Exception as e:
            print(f"[warn] Failed to save --dump-series to {outp}: {e}", file=sys.stderr)

    # Optional JSON summary
    if args.json_out:
        summary = {
            "ckpt": str(ckpt_path),
            "var": args.var,
            "shape": tuple(int(s) for s in _as_tensor(U).shape),
            "time_axis": int(time_axis),
            "nt_full": nt0,
            "nt_eval": nt,
            "dt": float(dt),
            "semantics": str(args.semantics),
            "agg": {
                "name": str(args.agg),
                "p": float(args.p),
                "q": float(args.q),
                "temp": float(args.temp),
            },
            "window": {
                "t0": None if args.t0 is None else float(args.t0),
                "t1": None if args.t1 is None else float(args.t1),
                "idx0": None if args.idx0 is None else int(args.idx0),
                "idx1": None if args.idx1 is None else int(args.idx1),
            },
            "spec": {
                "kind": ("custom" if args.stl else str(args.spec)),
                "formula": None if not args.stl else str(args.stl),
                "u_min": None if args.u_min is None else float(args.u_min),
                "u_max": None if args.u_max is None else float(args.u_max),
            },
            "robustness": float(rob),
            "satisfied": bool(sat),
            "backend": backend,
        }
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
