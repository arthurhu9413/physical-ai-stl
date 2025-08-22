# ruff: noqa: I001
from __future__ import annotations

"""
Train 1‑D viscous Burgers' equation with TorchPhysics **plus STL regularization**.

Why this script?
----------------
Professor Johnson asked for monitoring / enforcing STL-style specs in "physical AI"
frameworks. This file upgrades the TorchPhysics Burgers' PINN with a *differentiable*
Signal Temporal Logic (STL) penalty and a robust post‑hoc monitor. It is designed to:

1) **Solve** Burgers' PDE on x∈[x_min,x_max], t∈[t_min,t_max]
     u_t + u u_x = ν u_xx
2) **Enforce** a safety property during training via a smooth hinge on |u| ≤ u_max
3) **Evaluate** robust STL satisfaction after training (ρ = u_max − max_{x,t}|u|)
4) **Be fast & reproducible**: efficient samplers, fused losses, optional AMP, deterministic seeds
5) **Be pretty**: clear structure, dataclass config, thorough docstrings, sane defaults

References (APIs & ideas)
-------------------------
- TorchPhysics docs (domains, samplers, models, conditions, solver).  We use FCN,
  RandomUniformSampler & PINNCondition to express PDE/BC/IC losses.  See API pages.  # noqa: E501
- Differentiable STL via smooth min/max (log‑sum‑exp) as used in common literature.
- RTAMT / MoonLight libraries are complementary monitors; here we implement an
  in‑house differentiable penalty and a robust post‑hoc check compatible with STL "G" (always).

Notes
-----
* STL training penalty here is *soft* (& differentiable). Post‑hoc we compute a *hard*
  robustness margin ρ on a dense grid. If desired, you can switch the penalty on/off
  with --stl_weight and adjust its smoothness with --stl_temp.
* Boundary conditions: homogeneous Dirichlet at x = {x_min,x_max}.  Initial condition:
  u(x,t_min) = -sin(π * (x - x_min)/(x_max - x_min)).
* The code degrades gracefully if TorchPhysics is not installed (prints a helpful message).

Usage
-----
python scripts/train_burgers_torchphysics.py \
  --tag exp1 --max_steps 20000 --stl_weight 1.0 --u_max 0.9 --stl_temp 0.02

Artifacts
---------
- results/torchphysics_burgers/{tag}/burgers1d_{tag}.pt        (checkpoint + config)
- results/torchphysics_burgers/{tag}/burgers1d_{tag}_field.pt  (dense field, grid, metrics)
"""

from dataclasses import dataclass
from typing import Literal, Tuple
import math
import os
import time
import json
from pathlib import Path

# Lazy imports for optional dependencies
try:
    import torch
except Exception as _e:
    torch = None  # type: ignore

# TorchPhysics is optional at edit-time; we soft-fail with instructions at runtime.
try:
    import pytorch_lightning as pl
    import torchphysics as tp  # noqa: F401
except Exception:
    pl = None  # type: ignore
    tp = None  # type: ignore


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    # Domain
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    nu: float = 0.01 / math.pi  # Viscosity

    # Model
    hidden: Tuple[int, ...] = (64, 64, 64, 64)
    activation: Literal["tanh", "gelu", "sin"] = "tanh"

    # Collocation / sampling
    n_pde: int = 6000
    n_ic: int = 512
    n_bc: int = 512

    # STL (training penalty)
    stl_weight: float = 1.0      # set 0.0 to disable STL loss
    u_max: float = 1.0           # |u| <= u_max
    stl_temp: float = 0.02       # smooth-min temperature (smaller ~ closer to hard min)
    stl_warmup: int = 0          # steps to delay STL activation

    # Grid for evaluation / artifacts
    nx: int = 256
    nt: int = 256

    # Optimization
    lr: float = 1e-3
    max_steps: int = 20000
    optimizer: Literal["adam", "adamw"] = "adam"
    scheduler: Literal["none", "cosine", "step"] = "none"
    scheduler_gamma: float = 0.5
    scheduler_step: int = 5000

    # System
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal[32, 16] = 32  # automatic mixed precision if 16
    seed: int = 1234
    deterministic: bool = True
    num_threads: int = 1

    # Logging / saving
    tag: str = "default"
    log_every: int = 100
    save_ckpt: bool = True
    results_root: str = "results/torchphysics_burgers"


def _parse_args() -> Config:
    import argparse

    p = argparse.ArgumentParser(description="TorchPhysics Burgers with STL regularization")
    # Domain
    p.add_argument("--x_min", type=float, default=Config.x_min)
    p.add_argument("--x_max", type=float, default=Config.x_max)
    p.add_argument("--t_min", type=float, default=Config.t_min)
    p.add_argument("--t_max", type=float, default=Config.t_max)
    p.add_argument("--nu", type=float, default=Config.nu)

    # Model
    p.add_argument("--hidden", type=int, nargs="+", default=list(Config.hidden))
    p.add_argument("--activation", type=str, choices=["tanh", "gelu", "sin"], default=Config.activation)

    # Collocation
    p.add_argument("--n_pde", type=int, default=Config.n_pde)
    p.add_argument("--n_ic", type=int, default=Config.n_ic)
    p.add_argument("--n_bc", type=int, default=Config.n_bc)

    # STL
    p.add_argument("--stl_weight", type=float, default=Config.stl_weight)
    p.add_argument("--u_max", type=float, default=Config.u_max)
    p.add_argument("--stl_temp", type=float, default=Config.stl_temp)
    p.add_argument("--stl_warmup", type=int, default=Config.stl_warmup)

    # Grid
    p.add_argument("--nx", type=int, default=Config.nx)
    p.add_argument("--nt", type=int, default=Config.nt)

    # Optimization
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--max_steps", type=int, default=Config.max_steps)
    p.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default=Config.optimizer)
    p.add_argument("--scheduler", type=str, choices=["none", "cosine", "step"], default=Config.scheduler)
    p.add_argument("--scheduler_gamma", type=float, default=Config.scheduler_gamma)
    p.add_argument("--scheduler_step", type=int, default=Config.scheduler_step)

    # System
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=Config.device)
    p.add_argument("--precision", type=int, choices=[16, 32], default=Config.precision)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--deterministic", action="store_true", default=Config.deterministic)
    p.add_argument("--num_threads", type=int, default=Config.num_threads)

    # IO
    p.add_argument("--tag", type=str, default=Config.tag)
    p.add_argument("--log_every", type=int, default=Config.log_every)
    p.add_argument("--save_ckpt", action="store_true", default=Config.save_ckpt)
    p.add_argument("--results_root", type=str, default=Config.results_root)

    args = p.parse_args()
    cfg = Config(**{
        "x_min": args.x_min, "x_max": args.x_max, "t_min": args.t_min, "t_max": args.t_max, "nu": args.nu,
        "hidden": tuple(args.hidden), "activation": args.activation,
        "n_pde": args.n_pde, "n_ic": args.n_ic, "n_bc": args.n_bc,
        "stl_weight": args.stl_weight, "u_max": args.u_max, "stl_temp": args.stl_temp, "stl_warmup": args.stl_warmup,
        "nx": args.nx, "nt": args.nt,
        "lr": args.lr, "max_steps": args.max_steps, "optimizer": args.optimizer,
        "scheduler": args.scheduler, "scheduler_gamma": args.scheduler_gamma, "scheduler_step": args.scheduler_step,
        "device": args.device, "precision": args.precision, "seed": args.seed,
        "deterministic": args.deterministic, "num_threads": args.num_threads,
        "tag": args.tag, "log_every": args.log_every, "save_ckpt": args.save_ckpt,
        "results_root": args.results_root,
    })
    return cfg


# -----------------------------
# Utilities
# -----------------------------

def _setup_system(cfg: Config) -> torch.device:
    assert torch is not None, "PyTorch is required."
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    try:
        import numpy as np
        np.random.seed(cfg.seed)
    except Exception:
        pass

    if cfg.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(cfg.device)
    if cfg.precision == 16:
        # AMP: we set default dtype to float32 (kept), autocast handled by PL/AMP
        pass
    return dev


def _act(name: str):
    import torch.nn as nn
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU(approximate="none")
    if name == "sin":
        class Sin(nn.Module):
            def forward(self, x):  # pragma: no cover
                return torch.sin(x)
        return Sin()
    raise ValueError(f"Unknown activation {name}")


def _maybe_placeholder(cfg: Config):
    if (tp is None) or (pl is None):
        print(
            "[train_burgers_torchphysics] TorchPhysics/PyTorch Lightning not found.\n"
            "Install with:\n"
            "  pip install torch pytorch-lightning torchphysics\n"
            "This script expects TorchPhysics API (~0.0 post1)."
        )
        # Still create an empty artifact folder to not break automation
        results_dir = Path(cfg.results_root) / cfg.tag
        results_dir.mkdir(parents=True, exist_ok=True)
        meta_path = results_dir / f"burgers1d_{cfg.tag}_MISSING.txt"
        meta_path.write_text("TorchPhysics or Lightning not installed.\n", encoding="utf-8")
        return True
    return False


# -----------------------------
# Smooth min/max (STL building blocks)
# -----------------------------

def softmax(x: torch.Tensor, temp: float) -> torch.Tensor:
    # smooth approximation of max via log-sum-exp
    return temp * torch.logsumexp(x / temp, dim=0)

def softmin(x: torch.Tensor, temp: float) -> torch.Tensor:
    # smooth approximation of min via -log-sum-exp
    return -temp * torch.logsumexp((-x) / temp, dim=0)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = _parse_args()
    if _maybe_placeholder(cfg):
        return

    # Resolve device, seeds, dtypes
    dev = _setup_system(cfg)

    # Names / spaces
    X = tp.spaces.R1("x")
    T = tp.spaces.R1("t")
    U = tp.spaces.R1("u")

    # Domains
    Omega_x = tp.domains.Interval(X, cfg.x_min, cfg.x_max)
    Omega_t = tp.domains.Interval(T, cfg.t_min, cfg.t_max)
    Omega = Omega_x * Omega_t

    # Model: FCN with normalization layer (scales inputs to (-1,1)ᵈ)
    model = tp.models.Sequential(
        tp.models.NormalizationLayer(Omega),
        tp.models.FCN(X * T, U, hidden=cfg.hidden, activations=_act(cfg.activation)),
    )

    # Samplers
    S_pde = tp.samplers.RandomUniformSampler(Omega, n_points=cfg.n_pde)
    S_ic = tp.samplers.RandomUniformSampler(Omega_x * Omega_t.boundary_left, n_points=cfg.n_ic)
    S_bc = tp.samplers.RandomUniformSampler(Omega_x.boundary * Omega_t, n_points=cfg.n_bc)

    # Differential operators
    from torchphysics.utils.differentialoperators import grad, laplacian  # type: ignore

    # PDE residual: u_t + u u_x - nu u_xx = 0
    def pde_residual(u, x, t):
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = laplacian(u, x)
        return u_t + u * u_x - cfg.nu * u_xx

    # IC: u(x, t_min) = -sin(pi * ξ), where ξ rescales x to [0, 1]
    def u0(x):
        xi = (x - cfg.x_min) / (cfg.x_max - cfg.x_min)
        return -torch.sin(math.pi * xi)

    def ic_residual(u, x, t):
        return u - u0(x)

    # BC: Dirichlet u = 0 on x-boundary
    def bc_residual(u, x, t):
        return u

    # STL training penalty (soft): encourage |u| ≤ u_max on a subsample of Ω
    # We implement a smooth min over the batch and penalize only negative margins.
    # margin_i = u_max - |u_i| ;  ρ ≈ softmin_i margin_i ;  loss = relu(-ρ)^2
    stl_sampler = tp.samplers.RandomUniformSampler(Omega, n_points=max(256, cfg.n_pde // 8))

    def stl_residual(u, x, t):
        # Per‑point margins, shape [N, 1]
        margins = cfg.u_max - torch.abs(u)
        # Smooth min over the batch
        rho = softmin(margins.squeeze(-1), temp=float(cfg.stl_temp))
        # Broadcast global penalty to match unreduced loss convention
        loss = torch.nn.functional.relu(-rho) ** 2
        return loss.expand(margins.shape[0], 1)

    # Conditions
    cond_pde = tp.problem.conditions.PINNCondition(
        module=model,
        sampler=S_pde,
        residual_fn=pde_residual,
        name="pde",
        weight=1.0,
    )
    cond_ic = tp.problem.conditions.PINNCondition(
        module=model,
        sampler=S_ic,
        residual_fn=ic_residual,
        name="ic",
        weight=1.0,
    )
    cond_bc = tp.problem.conditions.PINNCondition(
        module=model,
        sampler=S_bc,
        residual_fn=bc_residual,
        name="bc",
        weight=1.0,
    )

    # STL condition (optional via weight)
    cond_stl = tp.problem.conditions.PINNCondition(
        module=model,
        sampler=stl_sampler,
        residual_fn=stl_residual,
        name="stl_soft_always_abs_le_u_max",
        weight=float(cfg.stl_weight),
    )

    train_conditions = [cond_pde, cond_ic, cond_bc]
    if cfg.stl_weight > 0.0:
        train_conditions.append(cond_stl)

    # Optimizer settings
    opt_cls = torch.optim.Adam if cfg.optimizer == "adam" else torch.optim.AdamW
    opt_setting = tp.solver.OptimizerSetting(optimizer_class=opt_cls, lr=cfg.lr)

    # Trainer
    trainer = tp.solver.Solver(train_conditions, optimizer_setting=opt_setting)

    # Lightning trainer settings
    callbacks = []
    # Optional: periodic weight checkpoint
    callbacks.append(tp.utils.callbacks.WeightSaveCallback(
        model, path=f"{cfg.results_root}/{cfg.tag}", name=f"burgers1d_{cfg.tag}",
        check_interval=max(cfg.max_steps // 5, 2000), save_initial_model=False, save_final_model=False
    ))

    # Precision handling: PL uses "precision=16" for AMP, else 32
    pl_trainer = pl.Trainer(
        max_steps=cfg.max_steps,
        accelerator=("gpu" if dev.type == "cuda" else "cpu"),
        precision=cfg.precision,
        deterministic=cfg.deterministic,
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
        enable_progress_bar=True,
        gradient_clip_val=0.0,
        log_every_n_steps=max(1, cfg.log_every),
    )

    # Warmup: if requested, run a few steps without STL
    if cfg.stl_weight > 0.0 and cfg.stl_warmup > 0:
        old_w = cond_stl.weight
        cond_stl.weight = 0.0
        pl_trainer.fit(trainer)
        cond_stl.weight = float(old_w)
        remaining = max(0, cfg.max_steps - pl_trainer.global_step)
        if remaining > 0:
            pl_trainer.fit(trainer, ckpt_path=None)
    else:
        pl_trainer.fit(trainer)

    # -----------------------------
    # Post‑hoc evaluation & artifacts
    # -----------------------------
    results_dir = Path(cfg.results_root) / cfg.tag
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save final checkpoint + config
    if cfg.save_ckpt:
        ckpt_path = results_dir / f"burgers1d_{cfg.tag}.pt"
        torch.save({"model": model.state_dict(), "config": vars(cfg)}, ckkt_path)

    # Dense grid
    gx = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, device=dev).unsqueeze(-1)
    gt = torch.linspace(cfg.t_min, cfg.t_max, cfg.nt, device=dev).unsqueeze(-1)
    # TorchPhysics sampler: product grid (mesh)
    GX = tp.spaces.Points.from_coordinates({"x": gx.repeat(cfg.nt, 1), "t": gt.repeat_interleave(cfg.nx, dim=0)})
    model.eval()
    with torch.no_grad():
        U = model(GX).reshape(cfg.nt, cfg.nx).detach().to("cpu")  # [nt, nx]
    ux_abs_max = float(torch.max(torch.abs(U)))
    rho = float(cfg.u_max - ux_abs_max)
    stl_satisfied = bool(rho >= 0.0)

    metrics = {
        "ux_abs_max": ux_abs_max,
        "stl_rho": rho,
        "stl_satisfied": stl_satisfied,
        "nu": float(cfg.nu),
        "train_steps": int(pl_trainer.global_step),
    }

    # Save dense field & metadata
    field_path = results_dir / f"burgers1d_{cfg.tag}_field.pt"
    torch.save(
        {
            "x": torch.linspace(cfg.x_min, cfg.x_max, cfg.nx),
            "t": torch.linspace(cfg.t_min, cfg.t_max, cfg.nt),
            "u": U,              # shape [nt, nx]
            "metrics": metrics,
            "config": vars(cfg),
        },
        field_path,
    )

    # Friendly stdout summary
    print(
        f"[burgers1d] tag={cfg.tag}  steps={pl_trainer.global_step}  "
        f"max|u|={ux_abs_max:.4f}  u_max={cfg.u_max:.4f}  ρ={rho:.4f}  "
        f"satisfied={'yes' if stl_satisfied else 'no'}"
    )
    print(f"[burgers1d] artifacts: {field_path}" )


if __name__ == "__main__":
    main()
