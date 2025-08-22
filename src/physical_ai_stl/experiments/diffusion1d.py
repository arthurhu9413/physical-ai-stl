# ruff: noqa: I001
from __future__ import annotations

"""
Diffusion PINN (1‑D) with optional differentiable STL regularization.

This experiment trains a compact neural field \(u_\theta(x,t)\) to solve the 1‑D heat/diffusion
equation on a rectangular domain,
    u_t = alpha * u_xx,   (x, t) ∈ [x_min, x_max] × [t_min, t_max],

with soft penalties for Dirichlet boundary conditions and an initial condition.
Optionally, we add a *differentiable* STL regularizer using smooth min/max semantics
(log‑sum‑exp), enforcing the global‑in‑time bound

    G_t [ reduce_x u(x,t) <= u_max ],

where ``reduce_x`` is one of {mean, softmax(temp), amax}.  This follows the spirit of
smooth robustness monitoring and STL‑guided learning (e.g., RTAMT/MoonLight for monitoring,
and STLnet‑style training with end‑to‑end backprop through a differentiable robustness).

Design goals
------------
• **Professor‑friendly**: single, readable file; reproducible; clear artifacts (CSV, checkpoint,
  final field tensor).  
• **Safe default numerics**: fp32 by default (stable higher‑order autograd); mixed precision
  optional; supports ``torch.compile`` where available.  
• **Modular STL**: STL is entirely optional; if the lightweight soft semantics module is
  unavailable, training proceeds without it.

Artifacts
---------
Writes to ``{results_dir}/diffusion1d_{tag}*``:
  • CSV log with losses and (if enabled) robustness.
  • Model checkpoint (state_dict + config).
  • Final field tensor on a regular grid:  dict(u, X, T, alpha, u_max, config).

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from ..models.mlp import MLP

# Optional differentiable STL (soft semantics)
try:  # avoid heavyweight deps; this is a tiny, local module
    from ..monitoring.stl_soft import STLPenalty, always, pred_leq, softmax
    _HAS_STL = True
except Exception:  # pragma: no cover
    _HAS_STL = False

from ..physics.diffusion1d import boundary_loss, residual_loss
from ..training.grids import grid1d, sample_interior_1d
from ..utils.logger import CSVLogger
from ..utils.seed import seed_everything

__all__ = ["Diffusion1DConfig", "run_diffusion1d"]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class Diffusion1DConfig:
    # --- model ---
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"            # MLP activation: "tanh" | "relu" | "sine" | "auto" | ...
    out_act: str | None = None           # optional output activation

    # --- grid / domain (for exported snapshots; not the training batch size) ---
    n_x: int = 128
    n_t: int = 64
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # --- optimization ---
    lr: float = 2e-3
    epochs: int = 200
    batch: int = 4096                    # interior collocation points per step
    weight_decay: float = 0.0

    # --- physics ---
    alpha: float = 0.1                   # diffusion coefficient

    # --- BC/IC sampling ---
    n_boundary: int = 256                # samples along x = x_min/x_max over t ∈ [t_min,t_max]
    n_initial: int = 512                 # samples along t = t_min over x ∈ [x_min,x_max]
    sample_method: str = "sobol"         # "sobol" | "uniform"

    # --- system ---
    device: str | None = None            # "cuda" | "mps" | "cpu" | None(auto)
    dtype: str = "float32"
    amp: bool = False                    # mixed precision: off by default (2nd‑order grads stability)
    compile: bool = False                # use torch.compile if available (PyTorch >= 2.0)
    print_every: int = 25

    # --- STL penalty (optional) ---
    stl_use: bool = False
    stl_weight: float = 0.0              # λ: weight on STL penalty
    stl_u_max: float = 1.0               # safety bound: u(x,t) ≤ u_max
    stl_temp: float = 0.1                # temperature for smooth min/max
    stl_spatial: str = "mean"            # "mean" | "softmax" | "amax"  (reduce over x)
    stl_every: int = 1                   # evaluate STL loss every k steps (k ≥ 1)
    stl_n_x: int = 64                    # coarse monitor grid (decoupled from train batch)
    stl_n_t: int = 64

    # --- output / logging ---
    results_dir: str = "results"
    tag: str = "run"
    save_ckpt: bool = True

    # --- misc ---
    seed: int = 0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _auto_device(user_choice: str | None = None) -> torch.device:
    if user_choice:
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_compile(module: nn.Module, do_compile: bool) -> nn.Module:  # pragma: no cover - depends on torch version
    if do_compile:
        compile_fn = getattr(torch, "compile", None)
        if callable(compile_fn):
            try:
                module = compile_fn(module, mode="reduce-overhead", fullgraph=False)  # type: ignore[misc]
            except Exception:
                # Fall back silently if compile fails (e.g., unsupported ops on platform)
                pass
    return module


def _stl_spatial_reduce(u_xt: torch.Tensor, mode: str = "mean", temp: float = 0.1) -> torch.Tensor:
    """Reduce u(x,t) over spatial dimension x -> a per‑time signal.

    Args:
        u_xt: Tensor of shape (N_x, N_t)
        mode: 'mean' | 'softmax' | 'amax'
        temp: temperature for smooth softmax aggregator
    Returns:
        Tensor of shape (N_t,)
    """
    if mode == "mean":
        return u_xt.mean(dim=0)
    if mode == "softmax":
        # Differentiable softmax aggregator (weighted sum); from monitoring.stl_soft
        return softmax(u_xt, temp=float(temp), dim=0, keepdim=False)  # type: ignore[name-defined]
    if mode == "amax":
        return u_xt.amax(dim=0)
    raise ValueError(f"Unknown stl_spatial mode: {mode!r}")


def _parse_config(cfg_dict: dict[str, Any]) -> Diffusion1DConfig:
    model = cfg_dict.get("model", {}) or {}
    grid = cfg_dict.get("grid", {}) or {}
    optim_cfg = cfg_dict.get("optim", {}) or {}
    physics = cfg_dict.get("physics", {}) or {}
    stl = cfg_dict.get("stl", {}) or {}
    io = cfg_dict.get("io", {}) or {}

    return Diffusion1DConfig(
        # model
        hidden=tuple(model.get("hidden", (64, 64, 64))),
        activation=model.get("activation", "tanh"),
        out_act=model.get("out_activation"),
        # grid
        n_x=grid.get("n_x", 128),
        n_t=grid.get("n_t", 64),
        x_min=grid.get("x_min", 0.0),
        x_max=grid.get("x_max", 1.0),
        t_min=grid.get("t_min", 0.0),
        t_max=grid.get("t_max", 1.0),
        # optim
        lr=optim_cfg.get("lr", 2e-3),
        epochs=optim_cfg.get("epochs", 200),
        batch=optim_cfg.get("batch", 4096),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
        # physics
        alpha=physics.get("alpha", 0.1),
        # BC/IC sampling (allow under 'optim' for convenience)
        n_boundary=optim_cfg.get("n_boundary", cfg_dict.get("n_boundary", 256)),
        n_initial=optim_cfg.get("n_initial", cfg_dict.get("n_initial", 512)),
        sample_method=optim_cfg.get("sample_method", cfg_dict.get("sample_method", "sobol")),
        # system
        device=cfg_dict.get("device"),
        dtype=cfg_dict.get("dtype", "float32"),
        amp=cfg_dict.get("amp", False),
        compile=cfg_dict.get("compile", False),
        print_every=cfg_dict.get("print_every", 25),
        # STL
        stl_use=stl.get("use", cfg_dict.get("stl_use", False)),
        stl_weight=stl.get("weight", cfg_dict.get("stl_weight", 0.0)),
        stl_u_max=stl.get("u_max", cfg_dict.get("stl_u_max", 1.0)),
        stl_temp=stl.get("temp", cfg_dict.get("stl_temp", 0.1)),
        stl_spatial=stl.get("spatial", cfg_dict.get("stl_spatial", "mean")),
        stl_every=stl.get("every", cfg_dict.get("stl_every", 1)),
        stl_n_x=stl.get("n_x", grid.get("n_x", 128)),
        stl_n_t=stl.get("n_t", grid.get("n_t", 64)),
        # I/O
        results_dir=io.get("results_dir", cfg_dict.get("results_dir", "results")),
        tag=cfg_dict.get("tag", "run"),
        save_ckpt=io.get("save_ckpt", cfg_dict.get("save_ckpt", True)),
        # misc
        seed=cfg_dict.get("seed", 0),
    )


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def run_diffusion1d(cfg_dict: dict[str, Any]) -> str:
    """Run the 1‑D diffusion experiment and return the path to the saved field tensor."""
    cfg = _parse_config(cfg_dict)
    seed_everything(cfg.seed)

    device = _auto_device(cfg.device)
    torch.set_default_dtype(getattr(torch, cfg.dtype))
    dtype = torch.get_default_dtype()

    # --- model ----------------------------------------------------------------
    model = MLP(
        in_dim=2, out_dim=1, hidden=cfg.hidden, activation=cfg.activation, out_activation=cfg.out_act
    ).to(device)
    model = _maybe_compile(model, cfg.compile)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Mixed precision scaler (conservative default off because of higher‑order grads)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))  # type: ignore[attr-defined]

    # STL penalty (optional)
    penalty = None
    if cfg.stl_use and _HAS_STL:
        penalty = STLPenalty(weight=float(cfg.stl_weight), margin=0.0)
    elif cfg.stl_use and not _HAS_STL:
        print("[diffusion1d] STL requested but soft semantics module is unavailable; proceeding without STL.")

    # --- grids ----------------------------------------------------------------
    # Full grid for final export/snapshots (kept on device to amortize allocations)
    X, T, XT = grid1d(
        n_x=cfg.n_x, n_t=cfg.n_t,
        x_min=cfg.x_min, x_max=cfg.x_max,
        t_min=cfg.t_min, t_max=cfg.t_max,
        device=device, dtype=dtype,
    )

    # Coarse grid for STL monitoring (decoupled from training batch size)
    Xs, Ts, XTs = grid1d(
        n_x=max(8, int(cfg.stl_n_x)), n_t=max(4, int(cfg.stl_n_t)),
        x_min=cfg.x_min, x_max=cfg.x_max,
        t_min=cfg.t_min, t_max=cfg.t_max,
        device=device, dtype=dtype,
    )

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log = CSVLogger(
        results_dir / f"diffusion1d_{cfg.tag}.csv",
        header=["epoch", "lr", "loss", "loss_pde", "loss_bcic", "loss_stl", "robustness"],
    )

    # --- training loop --------------------------------------------------------
    model.train()
    for epoch in range(int(cfg.epochs)):
        coords = sample_interior_1d(
            int(cfg.batch),
            x_min=cfg.x_min, x_max=cfg.x_max,
            t_min=cfg.t_min, t_max=cfg.t_max,
            method=cfg.sample_method, device=device, dtype=dtype,
            seed=cfg.seed + epoch,
        )
        coords.requires_grad_(True)

        use_autocast = (cfg.amp and device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=use_autocast):  # type: ignore[attr-defined]
            # PDE residual at collocation points
            loss_pde = residual_loss(model, coords, alpha=cfg.alpha, reduction="mean")  # E[(u_t - alpha u_xx)^2]

            # Soft BC/IC penalties (new samples every epoch)
            loss_bcic = boundary_loss(
                model,
                x_left=cfg.x_min, x_right=cfg.x_max, t_min=cfg.t_min, t_max=cfg.t_max,
                device=device, dtype=dtype, method=cfg.sample_method, seed=cfg.seed + 17 * epoch,
                n_boundary=cfg.n_boundary, n_initial=cfg.n_initial,
            )

            # Optional STL penalty
            loss_stl = torch.zeros((), device=device, dtype=dtype)
            rob = torch.zeros((), device=device, dtype=dtype)
            if penalty is not None and (epoch % max(1, int(cfg.stl_every)) == 0):
                # Evaluate on coarse STL grid WITH grad so the penalty influences training.
                u = model(XTs).reshape(Xs.shape)             # (N_x_stl, N_t_stl)
                signal_t = _stl_spatial_reduce(u, mode=cfg.stl_spatial, temp=float(cfg.stl_temp))  # (N_t_stl,)
                margins = pred_leq(signal_t, float(cfg.stl_u_max))  # ≤ u_max per time
                rob = always(margins, temp=float(cfg.stl_temp), time_dim=-1)  # scalar robustness
                loss_stl = penalty(rob)

            loss = loss_pde + loss_bcic + loss_stl

        # Parameter update
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():  # type: ignore[attr-defined]
            scaler.scale(loss).backward()                    # type: ignore[attr-defined]
            scaler.step(opt)                                 # type: ignore[attr-defined]
            scaler.update()                                  # type: ignore[attr-defined]
        else:
            loss.backward()
            opt.step()

        # --- logging / progress ----------------------------------------------
        lr_now = opt.param_groups[0]["lr"]
        log.append([epoch, lr_now, float(loss), float(loss_pde), float(loss_bcic), float(loss_stl), float(rob)])
        if (epoch % max(1, int(cfg.print_every)) == 0) or (epoch == cfg.epochs - 1):
            print(
                f"[diffusion1d] epoch={epoch:04d} lr={lr_now:.2e} "
                f"loss={float(loss):.4e} pde={float(loss_pde):.4e} "
                f"bcic={float(loss_bcic):.4e} stl={float(loss_stl):.4e} rob={float(rob):.4e}"
            )

    # --- artifacts ------------------------------------------------------------
    saved: list[str] = []

    if cfg.save_ckpt:
        ckpt_path = results_dir / f"diffusion1d_{cfg.tag}.pt"
        torch.save({"model": model.state_dict(), "config": cfg.__dict__}, ckpt_path)
        saved.append(str(ckpt_path))

    # Save final field on the full grid
    model.eval()
    with torch.no_grad():
        U = model(XT).reshape(cfg.n_x, cfg.n_t).detach().to("cpu")
        X_cpu, T_cpu = X.detach().to("cpu"), T.detach().to("cpu")
    field_path = results_dir / f"diffusion1d_{cfg.tag}_field.pt"
    torch.save(
        {
            "u": U, "X": X_cpu, "T": T_cpu,
            "u_max": float(cfg.stl_u_max), "alpha": float(cfg.alpha),
            "config": cfg.__dict__,
        },
        field_path,
    )
    saved.append(str(field_path))

    # Return the main artifact path for convenience
    return str(field_path)
