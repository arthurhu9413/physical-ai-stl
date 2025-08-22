#!/usr/bin/env python3
# ruff: noqa: I001
"""
train_neuromancer_stl.py
========================

Purpose
-------
Train a tiny, dependency-light **STL-regularized** regressor and (optionally)
a Neuromancer counterpart, then save **reproducible** metrics to JSON.

What this script demonstrates (in one file):

1) A minimal *physics/ML* training loop (pure PyTorch) with a differentiable
   Signal Temporal Logic (STL) penalty (safety: `G(y <= bound)`).
2) Optional evaluation via RTAMT, if available (`pip install rtamt`), to report
   an independent offline robustness value for the same spec.
3) Optional Neuromancer run using the repository helper
   `physical_ai_stl.frameworks.neuromancer_stl_demo`—if Neuromancer is not
   present, the script will still succeed and record `neuromancer=None`.

Design notes
------------
- The STL loss uses a simple differentiable surrogate: `relu(y - bound)`
  averaged in time. This encourages satisfaction of the safety guard
  `G(y <= bound)` during training. The *offline* robustness we report is the
  classic `min_t (bound - y_hat(t))` (positive means satisfied).
- If RTAMT is installed we also compute robustness with the library’s
  monitor for the formula `always (y <= bound)` to provide an independent
  check. Otherwise we just omit the RTAMT field.
- The PyTorch and Neuromancer paths share the same synthetic dataset
  (fit a sin wave). Neuromancer training is delegated to the maintained
  helper in `src/physical_ai_stl/frameworks/neuromancer_stl_demo.py`.

References
----------
• Neuromancer docs (examples, Problem/Loss API): pnnl.github.io/neuromancer .  # noqa: E501
• RTAMT – STL monitoring library (discrete/dense time, offline/online).         # noqa: E501

Both references are linked and cited in the commit message / report.

Usage
-----
Basic run and pretty JSON to stdout:

    python scripts/train_neuromancer_stl.py --pretty

Tweak the STL bound/weight and epochs and save to a file:

    python scripts/train_neuromancer_stl.py \\
        --n 256 --epochs 200 --lr 1e-3 \\
        --bound 0.8 --stl-weight 100 \\
        --out runs/neuromancer_stl/latest.json --pretty

Ask for an independent RTAMT robustness number as well:

    python scripts/train_neuromancer_stl.py --rtamt --pretty

Reproducibility: set seed and device explicitly:

    python scripts/train_neuromancer_stl.py --seed 7 --device cpu

Output JSON schema (stable, professor‑friendly)
-----------------------------------------------
{
  "config": { ... exact CLI and defaults ... },
  "env":    { "python": "...", "torch": "...", "neuromancer": "..."},
  "pytorch": {
      "final_mse": <float>,
      "final_violation": <float>,
      "robustness_min": <float>,
      "rtamt_robustness": <float or null>
  },
  "neuromancer": {
      "final_mse": <float>,
      "final_violation": <float>,
      "robustness_min": <float>
  } | null
}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Repository import conveniences
# ---------------------------------------------------------------------------

def _ensure_repo_src_on_path() -> None:
    """
    Ensure `src/` is importable when the script is run from anywhere.

    We keep this tiny and dependency-light so that a fresh checkout "just works".
    """
    here = Path(__file__).resolve()
    # Try common repository layouts (script may be run from a sdist or wheel).
    candidates = [
        here.parent.parent / "src",        # repo/scripts -> repo/src
        here.parent.parent.parent / "src", # repo/tools/scripts -> repo/src
        here.parents[1] / "src",           # loose execution
    ]
    for c in candidates:
        if (c / "physical_ai_stl").exists():
            sys.path.insert(0, str(c))
            break

_ensure_repo_src_on_path()


# ---------------------------------------------------------------------------
# Optional imports guarded carefully (PyTorch, RTAMT, Neuromancer helpers)
# ---------------------------------------------------------------------------

def _try_import_torch() -> Any | None:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore
        return torch
    except Exception:
        return None


def _try_import_rtamt_helpers():
    try:
        from physical_ai_stl.monitoring import rtamt_monitor as rm  # type: ignore
        return rm
    except Exception:
        return None


def _try_import_nm_demo():
    """
    Import the Neuromancer demo helper from this repository.

    We don't require Neuromancer itself here; the helper already handles
    "Neuromancer not installed" gracefully by returning None.
    """
    try:
        from physical_ai_stl.frameworks.neuromancer_stl_demo import (  # type: ignore
            DemoConfig as _DemoConfig,
            _train_neuromancer as _nm_train,
        )
        return _DemoConfig, _nm_train
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PTConfig:
    n: int = 256
    epochs: int = 200
    lr: float = 1e-3
    bound: float = 0.8
    stl_weight: float = 100.0
    device: str = "cpu"
    seed: int = 7


@dataclass(frozen=True)
class RunConfig:
    pt: PTConfig
    run_neuromancer: bool = True
    use_rtamt: bool = False
    out: Optional[Path] = None
    save_model: Optional[Path] = None
    quiet: bool = False
    pretty: bool = False


# ---------------------------------------------------------------------------
# Small math / STL helpers (dependency-light)
# ---------------------------------------------------------------------------

def _stl_violation_relu(y: "torch.Tensor", bound: float) -> "torch.Tensor":
    """
    Differentiable surrogate of `G(y <= bound)`:
        violation(t) = relu(y(t) - bound), loss = mean_t violation(t)
    """
    torch = _try_import_torch()
    assert torch is not None, "PyTorch is required for training"
    return torch.relu(y - float(bound))  # type: ignore[operator]


def _offline_robustness_min(y: "torch.Tensor", bound: float) -> float:
    """
    Classic offline robustness for `always (y <= bound)` on a discrete signal:
        rho = min_t (bound - y(t))
    """
    torch = _try_import_torch()
    assert torch is not None, "PyTorch is required for evaluation"
    return float((float(bound) - y).min().item())


def _seed_everything(seed: int = 7) -> None:
    import random
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch = _try_import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Data + model (tiny, fast)
# ---------------------------------------------------------------------------

def _make_data(n: int, device: str = "cpu") -> dict[str, "torch.Tensor"]:
    """
    Synthetic demo: regress y(t) = sin(t) for t in [0, 2π].

    Returns dict with shape conventions matching the Neuromancer helper.
    """
    torch = _try_import_torch()
    assert torch is not None, "PyTorch is required for training"
    t = torch.linspace(0.0, 2.0 * math.pi, int(n), device=device).reshape(int(n), 1)
    y_true = torch.sin(t)
    return {"t": t, "y_true": y_true}


def _mlp() -> "torch.nn.Module":
    """
    Compact MLP used in tests and the Neuromancer demo helper for parity.
    """
    torch = _try_import_torch()
    assert torch is not None, "PyTorch is required for training"
    import torch.nn as nn  # type: ignore
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )


def _train_pytorch(cfg: PTConfig) -> tuple[dict[str, float], "torch.nn.Module", dict[str, "torch.Tensor"]]:
    """
    Train the MLP with MSE + STL penalty; return (metrics, model, data).
    """
    torch = _try_import_torch()
    assert torch is not None, "PyTorch is required for training"
    import torch.nn.functional as F  # type: ignore

    _seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    net = _mlp().to(device)  # type: ignore[call-arg]
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg.lr))  # type: ignore[attr-defined]

    data = _make_data(cfg.n, device=str(device))
    t = data["t"]
    y = data["y_true"]

    for _ in range(int(cfg.epochs)):
        opt.zero_grad(set_to_none=True)
        y_hat = net(t)
        fit = F.mse_loss(y_hat, y)  # type: ignore[arg-type]
        penalty = _stl_violation_relu(y_hat, cfg.bound).mean()
        loss = fit + float(cfg.stl_weight) * penalty
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_hat = net(t)
        final_mse = F.mse_loss(y_hat, y).item()  # type: ignore[arg-type]
        final_violation = _stl_violation_relu(y_hat, cfg.bound).mean().item()
        rho = _offline_robustness_min(y_hat, cfg.bound)

    metrics = {
        "final_mse": float(final_mse),
        "final_violation": float(final_violation),
        "robustness_min": float(rho),
    }
    return metrics, net, data


# ---------------------------------------------------------------------------
# Optional monitors (RTAMT) and Neuromancer
# ---------------------------------------------------------------------------

def _maybe_rtamt_robustness(y_hat: "torch.Tensor", bound: float, dt: float = 1.0) -> Optional[float]:
    """
    If RTAMT is importable, evaluate robustness of `always (y <= bound)` on y_hat.
    Returns None if RTAMT is not available.
    """
    rm = _try_import_rtamt_helpers()
    if rm is None:
        return None
    # Build spec and evaluate series. Use dense-time to accept (t, y) pairs.
    spec = rm.stl_always_upper_bound(var="y", u_max=float(bound), time_semantics="dense")
    # Build time series as list[(t, y)] to be robust to RTAMT versions.
    ts = [(float(t), float(v)) for (t, v) in zip(y_hat.new_tensor(range(len(y_hat))).tolist(), y_hat.flatten().tolist())]  # type: ignore[attr-defined]
    try:
        rho = rm.evaluate_series(spec, var="y", series=ts, dt=float(dt))
    except Exception:
        # As a fallback, RTAMT may accept just the values with implicit dt.
        rho = float("nan")
        try:
            rho = rm.evaluate_series(spec, var="y", series=[float(v) for v in y_hat.flatten().tolist()], dt=float(dt))
        except Exception:
            pass
    # Guard against NaN from exotic backends.
    return float(rho) if (rho is not None and not (isinstance(rho, float) and (math.isnan(rho) or math.isinf(rho)))) else None


def _maybe_neuromancer(cfg: PTConfig, data: dict[str, "torch.Tensor"]) -> Optional[dict[str, float]]:
    DemoConfig, nm_train = _try_import_nm_demo()
    if DemoConfig is None or nm_train is None:
        return None
    # Adapt PT config to DemoConfig (same field names in this repository).
    try:
        nm_cfg = DemoConfig(n=cfg.n, epochs=cfg.epochs, lr=cfg.lr, bound=cfg.bound, weight=cfg.stl_weight, device=cfg.device, seed=cfg.seed)  # type: ignore[call-arg]
        metrics = nm_train(nm_cfg, data)  # type: ignore[misc]
        return metrics
    except Exception:
        # Any import or API mismatch should not break the script.
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_neuromancer_stl.py",
        description=(
            "Train a tiny STL‑regularized regressor (PyTorch) and optionally run the "
            "Neuromancer counterpart. Saves metrics as compact JSON."
        ),
    )
    p.add_argument("--n", type=int, default=PTConfig.n, help="number of samples (default: %(default)s)")
    p.add_argument("--epochs", type=int, default=PTConfig.epochs, help="training epochs (default: %(default)s)")
    p.add_argument("--lr", type=float, default=PTConfig.lr, help="learning rate (default: %(default)s)")
    p.add_argument("--bound", type=float, default=PTConfig.bound, help="STL safety bound for y: `G(y <= bound)` (default: %(default)s)")
    p.add_argument("--stl-weight", type=float, default=PTConfig.stl_weight, help="weight on STL penalty (default: %(default)s)")
    p.add_argument("--device", type=str, default=PTConfig.device, help="torch device string (default: %(default)s)")
    p.add_argument("--seed", type=int, default=PTConfig.seed, help="random seed (default: %(default)s)")
    p.add_argument("--no-nm", dest="no_nm", action="store_true", help="disable Neuromancer run")
    p.add_argument("--rtamt", action="store_true", help="evaluate offline robustness with RTAMT if available")
    p.add_argument("--out", type=Path, default=None, help="path to write JSON metrics (default: runs/neuromancer_stl/<timestamp>.json)")
    p.add_argument("--save-model", type=Path, default=None, help="optional path to save final PyTorch model weights (.pt)")
    p.add_argument("--quiet", action="store_true", help="suppress progress messages")
    p.add_argument("--pretty", action="store_true", help="pretty-print JSON to stdout")
    return p


def _resolve_out_path(out: Optional[Path]) -> Path:
    if out is not None:
        out = out.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    default_dir = Path("runs") / "neuromancer_stl"
    default_dir.mkdir(parents=True, exist_ok=True)
    # use an ISO-ish compact timestamp to avoid colons on Windows
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return default_dir / f"metrics_{ts}.json"


def _gather_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    # torch
    torch = _try_import_torch()
    if torch is not None:
        info["torch"] = getattr(torch, "__version__", "unknown")
        info["device_count_cuda"] = int(torch.cuda.device_count()) if hasattr(torch, "cuda") else 0  # type: ignore[attr-defined]
    else:
        info["torch"] = None
    # Neuromancer availability/version
    try:
        import importlib
        nm = importlib.import_module("physical_ai_stl.frameworks.neuromancer_hello")
        ver = nm.neuromancer_version()  # type: ignore[attr-defined]
        info["neuromancer"] = ver
    except Exception:
        info["neuromancer"] = None
    # RTAMT availability
    info["rtamt"] = _try_import_rtamt_helpers() is not None
    return info


def _as_jsonable(x: Any) -> Any:
    """
    Coerce nested dataclasses and Paths to plain JSON types.
    """
    if dataclass_isinstance := (hasattr(x, "__dataclass_fields__")):
        return {k: _as_jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_as_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _as_jsonable(v) for k, v in x.items()}
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    # Resolve configuration
    pt_cfg = PTConfig(
        n=int(args.n),
        epochs=int(args.epochs),
        lr=float(args.lr),
        bound=float(args.bound),
        stl_weight=float(args.stl_weight),
        device=str(args.device),
        seed=int(args.seed),
    )
    run_cfg = RunConfig(
        pt=pt_cfg,
        run_neuromancer=(not bool(args.no_nm)),
        use_rtamt=bool(args.rtamt),
        out=args.out,
        save_model=args.save_model,
        quiet=bool(args.quiet),
        pretty=bool(args.pretty),
    )

    # --- Train PyTorch path (required) ---
    torch = _try_import_torch()
    if torch is None:
        raise SystemExit(
            "ERROR: PyTorch is required for training. Please install PyTorch or "
            "run in an environment with PyTorch available."
        )

    if not run_cfg.quiet:
        print(f"[PyTorch] training for {pt_cfg.epochs} epochs, bound={pt_cfg.bound}, weight={pt_cfg.stl_weight} ...")

    metrics_pt, model, data = _train_pytorch(pt_cfg)

    # Optional: RTAMT offline robustness on the learned signal.
    y_hat = model(data["t"]).detach()
    rtamt_rho: Optional[float] = None
    if run_cfg.use_rtamt:
        rtamt_rho = _maybe_rtamt_robustness(y_hat, bound=pt_cfg.bound, dt=1.0)
    if rtamt_rho is not None:
        metrics_pt["rtamt_robustness"] = float(rtamt_rho)
    else:
        metrics_pt["rtamt_robustness"] = None

    # Optional: save model weights
    if run_cfg.save_model is not None:
        run_cfg.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(run_cfg.save_model))

    # --- Neuromancer path (optional, best-effort) ---
    metrics_nm: Optional[dict[str, float]] = None
    if run_cfg.run_neuromancer:
        if not run_cfg.quiet:
            print("[Neuromancer] attempting Neuromancer demo training ...")
        metrics_nm = _maybe_neuromancer(pt_cfg, data)

    # --- Assemble payload ---
    payload: dict[str, Any] = {
        "config": _as_jsonable(run_cfg),
        "env": _gather_env(),
        "pytorch": metrics_pt,
        "neuromancer": metrics_nm,
    }

    # --- Write JSON (and echo to stdout) ---
    out_path = _resolve_out_path(run_cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_as_jsonable(payload), f, indent=2, sort_keys=True)
    if not run_cfg.quiet:
        print(f"Saved results to {out_path}")

    # Echo results to stdout for convenience.
    if run_cfg.pretty or not run_cfg.quiet:
        print(json.dumps(_as_jsonable(payload), indent=2, sort_keys=True))
    else:
        print(json.dumps(_as_jsonable(payload)))


if __name__ == "__main__":
    main()
