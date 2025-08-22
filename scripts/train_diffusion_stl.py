# ruff: noqa: I001
from __future__ import annotations

"""
Train a 1‑D diffusion PINN with differentiable STL regularization.

Overview
--------
This script trains a physics‑informed neural network (PINN) for the 1‑D heat /
diffusion equation

    u_t = α u_xx on  x ∈ [x_min, x_max],  t ∈ [t_min, t_max],

using a compact MLP and a collocation‑based residual loss. On top of the physics
and soft boundary/initial penalties, we add a *differentiable* Signal Temporal
Logic (STL) regularizer based on smooth min/max (log‑sum‑exp) semantics.  This
follows common practice in the literature (e.g., "STLnet", NeurIPS 2020) where
robustness is optimized directly via soft aggregations.

Compared to a minimal PINN training loop, this script adds:
• Flexible STL specs: upper/lower/range bounds with ALWAYS or EVENTUALLY,
  over either the *entire* horizon, a sliding time *window*, or a fixed
  *interval* [t0, t1] of interest.
• Spatial aggregation choices (mean/softmax/amax) to approximate ∀_x or ∃_x.
• Optional *Dirichlet masking* to enforce homogeneous boundary values exactly,
  which often accelerates convergence for PDEs.
• Mixed‑precision (AMP), gradient clipping, learning‑rate schedulers, and
  resumable logging & checkpoints.

The artifacts saved at the end include the dense field u(x,t) on a grid along
with the X/T axes and metadata to facilitate later evaluation and auditing.
"""

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim

from physical_ai_stl.models.mlp import MLP
from physical_ai_stl.monitoring.stl_soft import (
    STLPenalty,
    always,
    always_window,
    eventually,
    eventually_window,
    pred_geq,
    pred_leq,
    softmax as stl_softmax,
)
from physical_ai_stl.physics.diffusion1d import (
    MaskedModel,
    boundary_loss,
    make_dirichlet_mask_1d,
    residual_loss,
)
from physical_ai_stl.training.grids import grid1d, sample_interior_1d
from physical_ai_stl.utils.logger import CSVLogger
from physical_ai_stl.utils.seed import seed_everything


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _auto_device(user_choice: str | None = None) -> torch.device:
    if user_choice:
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_compile(module: nn.Module, do_compile: bool) -> nn.Module:
    if not do_compile:
        return module
    compile_fn = getattr(torch, "compile", None)
    if callable(compile_fn):  # pragma: no cover - depends on torch version
        return compile_fn(module)  # type: ignore[misc]
    return module


def _stl_spatial_reduce(u_xt: torch.Tensor, mode: str, temp: float) -> torch.Tensor:
    """
    Reduce a space×time field to a per‑time signal using a spatial aggregator.

    Args:
        u_xt: (Nx, Nt) field values.
        mode: 'mean' | 'softmax' | 'amax'
        temp: temperature for 'softmax' (larger → closer to max).
    """
    mode = str(mode).lower()
    if mode == "mean":
        return u_xt.mean(dim=0)
    if mode == "softmax":
        return stl_softmax(u_xt, temp=float(temp), dim=0, keepdim=False)  # type: ignore[arg-type]
    if mode == "amax":
        return u_xt.amax(dim=0)
    raise ValueError(f"Unknown stl_spatial mode: {mode!r}")


def _select_time_window(nt: int, t0: float, t1: float) -> slice:
    """
    Return a slice selecting a closed interval [t0,t1] ⊂ [0,1] on an Nt grid.
    """
    t0c = max(0.0, min(1.0, float(t0)))
    t1c = max(0.0, min(1.0, float(t1)))
    if t1c < t0c:
        t0c, t1c = t1c, t0c
    i0 = int(math.floor(t0c * (nt - 1)))
    i1 = int(math.ceil (t1c * (nt - 1)))
    return slice(i0, i1 + 1)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class Args:
    # model
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    out_act: str | None = None

    # grid/domain
    nx: int = 128
    nt: int = 64
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # optimization
    lr: float = 2e-3
    epochs: int = 200
    batch: int = 4096
    opt: str = "adam"  # adam | adamw
    weight_decay: float = 0.0
    sched: str = "none"  # none | onecycle | cosine
    grad_clip: float = 0.0

    # physics
    alpha: float = 0.1
    n_boundary: int = 256
    n_initial: int = 512
    sample_method: str = "sobol"  # sobol | uniform
    w_boundary: float = 1.0
    w_initial: float = 1.0
    dirichlet_mask: bool = False  # strong homogeneous Dirichlet BC via masking

    # STL (semantics)
    stl_use: bool = True
    stl_weight: float = 5.0
    stl_temp: float = 0.1
    stl_spatial: str = "softmax"  # mean | softmax | amax

    # property: 'upper' enforces u <= u_max, 'lower' enforces u >= u_min,
    #           'range' enforces u_min <= u <= u_max (both sides).
    stl_spec: Literal["upper", "lower", "range"] = "upper"
    stl_u_max: float = 0.9
    stl_u_min: float = 0.0

    # time aggregation outside the predicate
    stl_outer: Literal["always", "eventually"] = "always"
    stl_time_mode: Literal["all", "window", "interval"] = "all"
    stl_window: int = 16       # for time_mode='window': sliding window length (in coarse steps)
    stl_stride: int = 1        # for time_mode='window': sliding window stride
    stl_t0: float = 0.0        # for time_mode='interval': normalized ∈ [0,1]
    stl_t1: float = 1.0
    stl_every: int = 1         # how often (epochs) to evaluate STL during training
    stl_nx: int = 64           # coarse grid for STL (space)
    stl_nt: int = 64           # coarse grid for STL (time)
    stl_warmup: int = 0        # epochs to wait before adding STL loss

    # system / numerics
    device: str | None = None
    dtype: str = "float32"  # float32 | float64
    amp: bool = False
    compile: bool = False
    seed: int = 0
    print_every: int = 25

    # I/O
    results_dir: str = "results"
    tag: str = "run"
    save_ckpt: bool = False
    resume: str | None = None


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Train a 1‑D diffusion PINN with differentiable STL regularization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # model
    p.add_argument("--hidden", type=int, nargs="+", default=(64, 64, 64))
    p.add_argument(
        "--activation",
        type=str,
        default="tanh",
        help="MLP hidden activation (tanh/relu/sine/...)",
    )
    p.add_argument(
        "--out-act", type=str, default=None, help="optional output activation (e.g., tanh)"
    )

    # grid/domain
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--nt", type=int, default=64)
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=1.0)
    p.add_argument("--t-min", type=float, default=0.0)
    p.add_argument("--t-max", type=float, default=1.0)

    # optimization
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--opt", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--sched", type=str, default="none", choices=["none", "onecycle", "cosine"])
    p.add_argument("--grad-clip", type=float, default=0.0)

    # physics
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n-boundary", type=int, default=256)
    p.add_argument("--n-initial", type=int, default=512)
    p.add_argument("--sample-method", type=str, default="sobol", choices=["sobol", "uniform"])
    p.add_argument("--w-boundary", type=float, default=1.0)
    p.add_argument("--w-initial", type=float, default=1.0)
    p.add_argument("--dirichlet-mask", action="store_true", help="Enforce u=0 at x∈{x_min,x_max} via multiplicative mask.")

    # STL
    p.add_argument("--no-stl", dest="stl_use", action="store_false", help="Disable STL penalty.")
    p.add_argument("--stl-weight", type=float, default=5.0)
    p.add_argument("--stl-temp", type=float, default=0.1, help="Soft (log‑sum‑exp) temperature.")
    p.add_argument("--stl-spatial", type=str, default="softmax", choices=["mean", "softmax", "amax"])
    p.add_argument("--stl-spec", type=str, default="upper", choices=["upper", "lower", "range"])
    p.add_argument("--stl-u-max", type=float, default=0.9)
    p.add_argument("--stl-u-min", type=float, default=0.0)
    p.add_argument("--stl-outer", type=str, default="always", choices=["always", "eventually"])
    p.add_argument("--stl-time-mode", type=str, default="all", choices=["all", "window", "interval"])
    p.add_argument("--stl-window", type=int, default=16)
    p.add_argument("--stl-stride", type=int, default=1)
    p.add_argument("--stl-t0", type=float, default=0.0)
    p.add_argument("--stl-t1", type=float, default=1.0)
    p.add_argument("--stl-every", type=int, default=1, help="Compute STL loss every N epochs.")
    p.add_argument("--stl-nx", type=int, default=64)
    p.add_argument("--stl-nt", type=int, default=64)
    p.add_argument("--stl-warmup", type=int, default=0, help="Skip STL for the first N epochs.")

    # system / numerics
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--amp", action="store_true", help="Use AMP (mixed precision) when available.")
    p.add_argument("--compile", action="store_true", help="Use torch.compile if available.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-every", type=int, default=25)

    # I/O
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--resume", type=str, default=None)

    a = p.parse_args()
    return Args(
        hidden=tuple(a.hidden),
        activation=a.activation,
        out_act=a.out_act,
        nx=a.nx,
        nt=a.nt,
        x_min=a.x_min,
        x_max=a.x_max,
        t_min=a.t_min,
        t_max=a.t_max,
        lr=a.lr,
        epochs=a.epochs,
        batch=a.batch,
        opt=a.opt,
        weight_decay=a.weight_decay,
        sched=a.sched,
        grad_clip=a.grad_clip,
        alpha=a.alpha,
        n_boundary=a.n_boundary,
        n_initial=a.n_initial,
        sample_method=a.sample_method,
        w_boundary=a.w_boundary,
        w_initial=a.w_initial,
        dirichlet_mask=bool(a.dirichlet_mask),
        stl_use=bool(a.stl_use),
        stl_weight=a.stl_weight,
        stl_temp=a.stl_temp,
        stl_spatial=a.stl_spatial,
        stl_spec=a.stl_spec,  # type: ignore[arg-type]
        stl_u_max=a.stl_u_max,
        stl_u_min=a.stl_u_min,
        stl_outer=a.stl_outer,  # type: ignore[arg-type]
        stl_time_mode=a.stl_time_mode,  # type: ignore[arg-type]
        stl_window=a.stl_window,
        stl_stride=a.stl_stride,
        stl_t0=a.stl_t0,
        stl_t1=a.stl_t1,
        stl_every=max(1, int(a.stl_every)),
        stl_nx=a.stl_nx,
        stl_nt=a.stl_nt,
        stl_warmup=max(0, int(a.stl_warmup)),
        device=a.device,
        dtype=a.dtype,
        amp=bool(a.amp),
        compile=bool(a.compile),
        seed=a.seed,
        print_every=a.print_every,
        results_dir=a.results_dir,
        tag=a.tag,
        save_ckpt=bool(a.save_ckpt),
        resume=a.resume,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()
    seed_everything(int(cfg.seed))

    device = _auto_device(cfg.device)
    torch.set_default_dtype(getattr(torch, cfg.dtype))  # affects newly created tensors

    # Grid for training and coarse grid for STL monitoring
    X, T, XT = grid1d(
        n_x=cfg.nx,
        n_t=cfg.nt,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        device=device,
        dtype=getattr(torch, cfg.dtype),
    )
    Xs, Ts, XTs = grid1d(
        n_x=max(8, int(cfg.stl_nx)),
        n_t=max(4, int(cfg.stl_nt)),
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        device=device,
        dtype=getattr(torch, cfg.dtype),
    )

    # Model
    base = MLP(
        in_dim=2,
        out_dim=1,
        hidden=cfg.hidden,
        activation=cfg.activation,
        out_activation=cfg.out_act,
        last_layer_scale=0.01,
        device=device,
        dtype=getattr(torch, cfg.dtype),
    )
    model: nn.Module
    if cfg.dirichlet_mask:
        model = MaskedModel(base, make_dirichlet_mask_1d(cfg.x_min, cfg.x_max))
    else:
        model = base
    model = _maybe_compile(model, cfg.compile)

    # Optimizer
    opt_cls = optim.AdamW if cfg.opt.lower() == "adamw" else optim.Adam
    opt = opt_cls(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # Scheduler
    if cfg.sched == "onecycle":
        sched = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(cfg.lr),
            total_steps=int(cfg.epochs),
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=1e2,
        )
    elif cfg.sched == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.epochs))
    else:
        sched = None

    # Scalar penalty wrapper for robustness
    penalty = STLPenalty(weight=float(cfg.stl_weight), margin=0.0, kind="softplus", beta=10.0)

    # I/O
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log = CSVLogger(
        results_dir / f"diffusion1d_{cfg.tag}.csv",
        header=["epoch", "lr", "loss", "loss_pde", "loss_bcic", "loss_stl", "robustness"],
    )

    # (Optional) resume
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location="cpu")
        sd = ckpt.get("model")
        if sd:
            model.load_state_dict(sd, strict=True)
            print(f"[resume] Loaded model state from {cfg.resume}")

    use_autocast = bool(cfg.amp) and torch.cuda.is_available()
    if use_autocast:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None  # type: ignore[assignment]

    # Training loop
    model.train()
    for epoch in range(int(cfg.epochs)):
        # Interior collocation points
        coords = sample_interior_1d(
            int(cfg.batch),
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            method=cfg.sample_method,
            device=device,
            dtype=getattr(torch, cfg.dtype),
            seed=int(cfg.seed) + epoch,
        )
        coords.requires_grad_(True)

        with torch.cuda.amp.autocast(enabled=use_autocast):
            # PDE residual at collocation points (MSE)
            loss_pde = residual_loss(model, coords, alpha=cfg.alpha, reduction="mean")

            # Soft BC/IC penalties (new samples every epoch)
            loss_bcic = boundary_loss(
                model,
                x_left=cfg.x_min,
                x_right=cfg.x_max,
                t_min=cfg.t_min,
                t_max=cfg.t_max,
                device=device,
                dtype=getattr(torch, cfg.dtype),
                method=cfg.sample_method,
                n_boundary=cfg.n_boundary,
                n_initial=cfg.n_initial,
                seed=int(cfg.seed) + 13 * epoch,
                w_boundary=cfg.w_boundary,
                w_initial=cfg.w_initial,
            )

            # STL robustness (optional). We construct a per‑time signal by
            # reducing across space, then select the requested time aggregation.
            if (
                cfg.stl_use
                and float(cfg.stl_weight) > 0.0
                and (epoch % int(cfg.stl_every) == 0)
                and (epoch >= int(cfg.stl_warmup))
            ):
                # Evaluate model on a coarse grid (memory‑friendly)
                with torch.no_grad():
                    U = model(XTs).reshape(Xs.shape[0], Xs.shape[1])
                # Per‑time signal via spatial aggregator
                s_t = _stl_spatial_reduce(U, cfg.stl_spatial, cfg.stl_temp)

                # Choose predicate margins based on the spec
                margins_list: list[torch.Tensor] = []
                if cfg.stl_spec in {"upper", "range"}:
                    margins_list.append(pred_leq(s_t, cfg.stl_u_max))
                if cfg.stl_spec in {"lower", "range"}:
                    margins_list.append(pred_geq(s_t, cfg.stl_u_min))
                # Combine: conjunction via soft‑and ≈ min → softmin already
                # happens when we aggregate across the list by taking the min.
                # Implement as stack+softmin using always(eventually) on an extra axis.
                if len(margins_list) == 1:
                    margins = margins_list[0]
                else:
                    # equivalent to softmin over "predicate axis" using same temp
                    margins = torch.stack(margins_list, dim=0).amin(dim=0)

                # Time aggregation
                if cfg.stl_time_mode == "all":
                    if cfg.stl_outer == "always":
                        rob = always(margins, temp=float(cfg.stl_temp), time_dim=0)
                    else:
                        rob = eventually(margins, temp=float(cfg.stl_temp), time_dim=0)
                elif cfg.stl_time_mode == "window":
                    window = max(1, int(cfg.stl_window))
                    stride = max(1, int(cfg.stl_stride))
                    if cfg.stl_outer == "always":
                        # "always over every window": robustness is the softmin over windows
                        rob_seq = always_window(margins, window=window, stride=stride, temp=float(cfg.stl_temp), time_dim=0, keepdim=False)
                        rob = rob_seq.amin(dim=0) if rob_seq.ndim > 0 else rob_seq
                    else:
                        rob_seq = eventually_window(margins, window=window, stride=stride, temp=float(cfg.stl_temp), time_dim=0, keepdim=False)
                        rob = rob_seq.amax(dim=0) if rob_seq.ndim > 0 else rob_seq
                else:  # interval
                    # Restrict to [t0, t1] on the coarse grid
                    sl = _select_time_window(s_t.shape[0], cfg.stl_t0, cfg.stl_t1)
                    m_sub = margins[sl]
                    if cfg.stl_outer == "always":
                        rob = always(m_sub, temp=float(cfg.stl_temp), time_dim=0)
                    else:
                        rob = eventually(m_sub, temp=float(cfg.stl_temp), time_dim=0)

                loss_stl = penalty(rob)
            else:
                rob = torch.zeros((), device=device, dtype=getattr(torch, cfg.dtype))
                loss_stl = torch.zeros((), device=device, dtype=getattr(torch, cfg.dtype))

            loss = loss_pde + loss_bcic + loss_stl

        opt.zero_grad(set_to_none=True)
        if use_autocast:
            assert scaler is not None
            scaler.scale(loss).backward()
            # Gradient clipping (unscale first)
            if float(cfg.grad_clip) > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if float(cfg.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            opt.step()

        if sched is not None:
            sched.step()

        # Logging/print
        lr_now = opt.param_groups[0]["lr"]
        log.append(
            [
                epoch,
                lr_now,
                float(loss),
                float(loss_pde),
                float(loss_bcic),
                float(loss_stl),
                float(rob),
            ]
        )
        if (epoch % max(1, int(cfg.print_every)) == 0) or (epoch == int(cfg.epochs) - 1):
            print(
                "[diffusion1d] "
                f"epoch={epoch:04d} lr={lr_now:.2e} "
                f"loss={float(loss):.4e} "
                f"pde={float(loss_pde):.4e} "
                f"bcic={float(loss_bcic):.4e} "
                f"stl={float(loss_stl):.4e}"
            )

    # --- artifacts ------------------------------------------------------------
    saved: list[str] = []

    if cfg.save_ckpt:
        ckpt_path = results_dir / f"diffusion1d_{cfg.tag}.pt"
        torch.save({"model": model.state_dict(), "config": vars(cfg)}, ckpt_path)
        saved.append(str(ckpt_path))

    # Save final field on the full grid
    model.eval()
    with torch.no_grad():
        U = model(XT).reshape(cfg.nx, cfg.nt).detach().to("cpu")
        X_cpu, T_cpu = X.detach().to("cpu"), T.detach().to("cpu")
    field_path = results_dir / f"diffusion1d_{cfg.tag}_field.pt"
    torch.save(
        {
            "u": U,
            "X": X_cpu,
            "T": T_cpu,
            "u_max": float(cfg.stl_u_max),
            "u_min": float(cfg.stl_u_min),
            "alpha": float(cfg.alpha),
            "config": vars(cfg),
        },
        field_path,
    )
    saved.append(str(field_path))

    print(
        f"[diffusion1d] done → {field_path} "
        f"(and {len(saved) - 1} other artifact(s))"
    )


if __name__ == "__main__":
    main()
