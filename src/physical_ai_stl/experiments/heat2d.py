# ruff: noqa: I001
from __future__ import annotations

"""
2‑D Heat Equation PINN (+ optional STL / STREL‑style regularization).

This experiment trains a compact neural field :math:`u_\\theta(x,y,t)` that solves

    u_t = alpha * (u_xx + u_yy)      on  (x,y,t) ∈ [x_min,x_max]×[y_min,y_max]×[t_min,t_max],

with soft penalties for Dirichlet boundary conditions (u=0 on the spatial boundary) and an
initial condition (default: small Gaussian bump).  Optionally, we add a *differentiable* STL
regularizer that encourages bounds such as  ``G_t  (∀_{x,y∈Ω} u(x,y,t) ≤ u_max)`` or
windowed/eventual variants.  Spatial quantifiers are approximated by smooth min/max across
spatial samples; temporal operators use smooth min/max (“log‑sum‑exp”) semantics.

Artifacts written to ``{results_dir}/heat2d_{tag}*``:
  • CSV log with losses (PDE, BC/IC, STL) and learning rate.
  • Model checkpoint (state_dict + minimal config).
  • Optional per‑time numpy frames ``*_t{k}.npy`` and gradient‑magnitude PNGs.
  • A compact ``*_field.pt`` tensor with the final field sampled on a dense grid.

The implementation is *framework‑agnostic* beyond PyTorch.  Monitoring uses the light‑weight
``physical_ai_stl.monitoring.stl_soft`` module.  External tools like RTAMT or MoonLight can be
layered on top for offline monitoring, but are not required at train time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
from torch import nn, optim

from ..models.mlp import MLP
from ..physics.heat2d import bc_ic_heat2d, residual_heat2d
from ..training.grids import grid2d
from ..utils.logger import CSVLogger
from ..utils.seed import seed_everything

# Optional differentiable STL semantics (kept optional to avoid hard deps)
try:  # pragma: no cover - optional extra
    from ..monitoring.stl_soft import (
        STLPenalty,
        always,
        always_window,
        eventually,
        eventually_window,
        pred_leq,
        softmin,
        softmax,
    )
    _HAS_STL = True
except Exception:  # pragma: no cover
    _HAS_STL = False


__all__ = ["Heat2DConfig", "run_heat2d"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Heat2DConfig:
    # --- model ---
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"                     # "tanh" | "relu" | "gelu" | "sine" (SIREN) | ...
    out_activation: str | None = None            # optional output activation (e.g., "tanh")

    # --- grid / domain (for sampling + exports) ---
    n_x: int = 64
    n_y: int = 64
    n_t: int = 16
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # --- optimization ---
    lr: float = 2e-3
    epochs: int = 200
    batch: int = 4096
    weight_decay: float = 0.0
    amsgrad: bool = False
    scheduler: Literal["none", "cosine", "step"] = "none"
    step_size: int = 100
    gamma: float = 0.5
    grad_clip: float | None = None               # clip global grad‑norm if not None
    compile: bool = False                        # use torch.compile if available
    amp: bool = True                             # use CUDA autocast + GradScaler
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["float32", "float64"] = "float32"

    # --- physics ---
    alpha: float = 0.1                            # diffusivity
    bcic_weight: float = 1.0                      # BC/IC penalty weight
    n_boundary: int = 512                         # samples for boundary penalty per step
    n_initial: int = 512                          # samples for initial condition per step
    use_dirichlet_mask_pow: int = 0               # if >0, multiply NN output by boundary mask^pow

    # --- residual‑aware resampling (RAR) ---
    rar_pool: int = 0                             # if >0, evaluate residual on this many pool points
    rar_hard_frac: float = 0.5                    # fraction of batch to take as hardest by |residual|
    rar_every: int = 10                           # how often (epochs) to use RAR; 0 disables

    # --- STL penalty (all optional) ---
    stl_use: bool = False
    stl_weight: float = 0.0
    stl_u_min: float | None = None
    stl_u_max: float | None = None
    stl_margin: float = 0.0                       # desired positive robustness margin
    stl_beta: float = 10.0                        # smooth penalty sharpness
    stl_temp: float = 0.1                         # temperature for soft min/max along time/space
    # temporal operator: "always" (G) or "eventually" (F); window==0 => whole horizon
    stl_operator: Literal["always", "eventually"] = "always"
    stl_window: int = 0                           # temporal window length (in samples)
    stl_stride: int = 1                           # temporal stride for window ops
    # spatial quantifier: "forall" ~ softmin, "exists" ~ softmax
    stl_space_op: Literal["forall", "exists"] = "forall"
    stl_space_temp: float = 0.1                   # temperature for spatial soft (min/max)
    # STL evaluation grid (coarse to save time)
    stl_nx: int = 32
    stl_ny: int = 32
    stl_nt: int = 16
    stl_every: int = 10                           # evaluate penalty every N epochs
    # optional subregion for STL
    stl_x_min: float | None = None
    stl_x_max: float | None = None
    stl_y_min: float | None = None
    stl_y_max: float | None = None
    stl_t_min: float | None = None
    stl_t_max: float | None = None

    # --- output / logging ---
    results_dir: str = "results"
    tag: str = "run"
    save_ckpt: bool = True
    save_frames: bool = True
    frames_idx: Iterable[int] = (0, 8, 15)
    save_figs: bool = True
    print_every: int = 10
    seed: int = 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _select_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Metal
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def _maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:  # PyTorch < 2.0
        return model
    try:  # pragma: no cover
        return compile_fn(model, mode="default")
    except Exception:  # pragma: no cover - compilation is best‑effort
        return model


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _gradmag_numpy(u_2d: np.ndarray) -> np.ndarray:
    """Centered‑difference gradient magnitude for a 2‑D array (x by y)."""
    gx = np.zeros_like(u_2d)
    gy = np.zeros_like(u_2d)
    gx[1:-1, :] = 0.5 * (u_2d[2:, :] - u_2d[:-2, :])
    gy[:, 1:-1] = 0.5 * (u_2d[:, 2:] - u_2d[:, :-2])
    return np.sqrt(gx * gx + gy * gy)


def _spatial_reduce(m: torch.Tensor, *, op: Literal["forall", "exists"], temp: float) -> torch.Tensor:
    """
    Reduce robustness across space.

    Args:
        m:  Tensor of shape (Nxy, T) or (Nxy,) containing per‑location margins or robustness.
        op: "forall" ≈ smooth min (AND across space), "exists" ≈ smooth max (OR across space).
        temp: temperature for soft min/max.
    Returns:
        Tensor of shape (T,) or scalar, depending on input.
    """
    if op == "forall":
        return softmin(m, temp=temp, dim=-2 if m.ndim == 2 else -1)
    elif op == "exists":
        return softmax(m, temp=temp, dim=-2 if m.ndim == 2 else -1)
    else:  # pragma: no cover
        raise ValueError(f"Unknown spatial op: {op}")


def _stl_penalty(model: nn.Module, cfg: Heat2DConfig, device: torch.device, *, dtype: torch.dtype) -> torch.Tensor:
    """
    Compute a differentiable STL penalty based on bounds u_min/u_max on a coarse grid.

    Returns a scalar tensor (on the correct device/dtype).  If STL is disabled or misconfigured,
    returns 0.0.
    """
    if not (cfg.stl_use and _HAS_STL and cfg.stl_weight > 0.0 and (cfg.stl_u_min is not None or cfg.stl_u_max is not None)):
        return torch.zeros((), device=device, dtype=dtype)

    # Determine STL subregion and grid
    x0 = cfg.stl_x_min if cfg.stl_x_min is not None else cfg.x_min
    x1 = cfg.stl_x_max if cfg.stl_x_max is not None else cfg.x_max
    y0 = cfg.stl_y_min if cfg.stl_y_min is not None else cfg.y_min
    y1 = cfg.stl_y_max if cfg.stl_y_max is not None else cfg.y_max
    t0 = cfg.stl_t_min if cfg.stl_t_min is not None else cfg.t_min
    t1 = cfg.stl_t_max if cfg.stl_t_max is not None else cfg.t_max

    Xs, Ys, Ts, XYTs = grid2d(
        n_x=max(2, int(cfg.stl_nx)),
        n_y=max(2, int(cfg.stl_ny)),
        n_t=max(2, int(cfg.stl_nt)),
        x_min=float(x0), x_max=float(x1),
        y_min=float(y0), y_max=float(y1),
        t_min=float(t0), t_max=float(t1),
        device=device, dtype=dtype, return_cartesian=True,
    )  # XYTs: (Nxy*Nt, 3)
    u = model(XYTs).reshape(cfg.stl_nx * cfg.stl_ny, cfg.stl_nt)  # (Nxy, Nt)

    # Build per‑(x,y,t) margins from predicates
    margins: list[torch.Tensor] = []
    if cfg.stl_u_max is not None:
        margins.append(pred_leq(u, float(cfg.stl_u_max)))     # u ≤ u_max
    if cfg.stl_u_min is not None:
        margins.append(pred_leq(-u, float(-cfg.stl_u_min)))   # −u ≤ −u_min  (u ≥ u_min)

    if not margins:
        return torch.zeros((), device=device, dtype=dtype)

    # AND the predicates by smooth min across predicate axis, then reduce time/space.
    m = margins[0] if len(margins) == 1 else torch.minimum(margins[0], margins[1])

    # Temporal reduction
    if cfg.stl_window and cfg.stl_window > 0:
        # Windowed versions over time (size W, stride S)
        if cfg.stl_operator == "always":
            r_t = always_window(m, window=int(cfg.stl_window), stride=int(cfg.stl_stride), temp=float(cfg.stl_temp), time_dim=-1)
        else:
            r_t = eventually_window(m, window=int(cfg.stl_window), stride=int(cfg.stl_stride), temp=float(cfg.stl_temp), time_dim=-1)
        # r_t has shape (Nxy, n_windows).  Aggregate windows by smooth min (always) or smooth max (eventually).
        if cfg.stl_operator == "always":
            r_t = softmin(r_t, temp=float(cfg.stl_temp), dim=-1)
        else:
            r_t = softmax(r_t, temp=float(cfg.stl_temp), dim=-1)
    else:
        if cfg.stl_operator == "always":
            r_t = always(m, temp=float(cfg.stl_temp), time_dim=-1)          # (Nxy,)
        else:
            r_t = eventually(m, temp=float(cfg.stl_temp), time_dim=-1)      # (Nxy,)

    # Spatial reduction to a scalar robustness
    r = _spatial_reduce(r_t, op=cfg.stl_space_op, temp=float(cfg.stl_space_temp))
    if r.ndim != 0:
        r = r.mean()  # safety: if something left a vector, average

    penalty = STLPenalty(weight=float(cfg.stl_weight), margin=float(cfg.stl_margin), kind="softplus", beta=float(cfg.stl_beta))
    return penalty(r)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_config(cfg_dict: dict[str, Any]) -> Heat2DConfig:
    model = cfg_dict.get("model", {}) or {}
    grid = cfg_dict.get("grid", {}) or {}
    optim_cfg = cfg_dict.get("optim", {}) or {}
    physics = cfg_dict.get("physics", {}) or {}
    rar = cfg_dict.get("rar", {}) or {}
    stl = cfg_dict.get("stl", {}) or {}
    io = cfg_dict.get("io", {}) or {}

    return Heat2DConfig(
        # model
        hidden=tuple(model.get("hidden", (64, 64, 64))),
        activation=model.get("activation", "tanh"),
        out_activation=model.get("out_activation"),
        # grid
        n_x=int(grid.get("n_x", 64)),
        n_y=int(grid.get("n_y", 64)),
        n_t=int(grid.get("n_t", 16)),
        x_min=float(grid.get("x_min", 0.0)),
        x_max=float(grid.get("x_max", 1.0)),
        y_min=float(grid.get("y_min", 0.0)),
        y_max=float(grid.get("y_max", 1.0)),
        t_min=float(grid.get("t_min", 0.0)),
        t_max=float(grid.get("t_max", 1.0)),
        # optim
        lr=float(optim_cfg.get("lr", 2e-3)),
        epochs=int(optim_cfg.get("epochs", 200)),
        batch=int(optim_cfg.get("batch", 4096)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        amsgrad=bool(optim_cfg.get("amsgrad", False)),
        scheduler=str(optim_cfg.get("scheduler", "none")),
        step_size=int(optim_cfg.get("step_size", 100)),
        gamma=float(optim_cfg.get("gamma", 0.5)),
        grad_clip=optim_cfg.get("grad_clip"),
        compile=bool(optim_cfg.get("compile", False)),
        amp=bool(optim_cfg.get("amp", True)),
        device=str(optim_cfg.get("device", "auto")),
        dtype=str(optim_cfg.get("dtype", "float32")),
        # physics
        alpha=float(physics.get("alpha", 0.1)),
        bcic_weight=float(physics.get("bcic_weight", 1.0)),
        n_boundary=int(physics.get("n_boundary", 512)),
        n_initial=int(physics.get("n_initial", 512)),
        use_dirichlet_mask_pow=int(physics.get("use_dirichlet_mask_pow", 0)),
        # rar
        rar_pool=int(rar.get("pool", 0)),
        rar_hard_frac=float(rar.get("hard_frac", 0.5)),
        rar_every=int(rar.get("every", 10)),
        # stl
        stl_use=bool(stl.get("use", False)),
        stl_weight=float(stl.get("weight", 0.0)),
        stl_u_min=stl.get("u_min"),
        stl_u_max=stl.get("u_max"),
        stl_margin=float(stl.get("margin", 0.0)),
        stl_beta=float(stl.get("beta", 10.0)),
        stl_temp=float(stl.get("temp", 0.1)),
        stl_operator=str(stl.get("operator", "always")),
        stl_window=int(stl.get("window", 0)),
        stl_stride=int(stl.get("stride", 1)),
        stl_space_op=str(stl.get("space_op", "forall")),
        stl_space_temp=float(stl.get("space_temp", 0.1)),
        stl_nx=int(stl.get("n_x", 32)),
        stl_ny=int(stl.get("n_y", 32)),
        stl_nt=int(stl.get("n_t", 16)),
        stl_every=int(stl.get("every", 10)),
        stl_x_min=stl.get("x_min"),
        stl_x_max=stl.get("x_max"),
        stl_y_min=stl.get("y_min"),
        stl_y_max=stl.get("y_max"),
        stl_t_min=stl.get("t_min"),
        stl_t_max=stl.get("t_max"),
        # io
        results_dir=str(io.get("results_dir", "results")),
        tag=str(cfg_dict.get("tag", "run")),
        save_ckpt=bool(io.get("save_ckpt", True)),
        save_frames=bool(io.get("save_frames", True)),
        frames_idx=tuple(io.get("frames_idx", (0, 8, 15))),
        save_figs=bool(io.get("save_figs", True)),
        print_every=int(io.get("print_every", 10)),
        seed=int(cfg_dict.get("seed", 0)),
    )


def run_heat2d(cfg_dict: dict[str, Any]) -> list[str]:
    """
    Train the 2‑D heat PINN with optional STL regularization.

    Returns:
        List of artifact paths (CSV, checkpoint, field tensor, frames/figures if requested).
    """
    cfg = _parse_config(cfg_dict)

    # --- setup ----------------------------------------------------------------
    seed_everything(cfg.seed)
    device = _select_device(cfg.device)
    torch.set_default_dtype(getattr(torch, cfg.dtype))
    dtype = torch.get_default_dtype()

    # Precompute dense grid for sampling & for frame export
    X, Y, T, XYT = grid2d(
        n_x=cfg.n_x, n_y=cfg.n_y, n_t=cfg.n_t,
        x_min=cfg.x_min, x_max=cfg.x_max,
        y_min=cfg.y_min, y_max=cfg.y_max,
        t_min=cfg.t_min, t_max=cfg.t_max,
        device=device, dtype=dtype, return_cartesian=True,
    )  # XYT: (n_x*n_y*n_t, 3)

    # Model (optionally masked to satisfy Dirichlet BC exactly on the boundary)
    model: nn.Module = MLP(
        in_dim=3, out_dim=1, hidden=cfg.hidden, activation=cfg.activation, out_activation=cfg.out_activation
    ).to(device)
    if cfg.use_dirichlet_mask_pow and cfg.use_dirichlet_mask_pow > 0:
        # Deferred import to avoid circulars
        from ..physics.heat2d import make_dirichlet_mask, MaskedModel  # type: ignore
        mask = make_dirichlet_mask(cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max, pow=int(cfg.use_dirichlet_mask_pow))
        model = MaskedModel(model, mask)  # type: ignore[assignment]
    model = _maybe_compile(model, cfg.compile)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)

    if cfg.scheduler == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    elif cfg.scheduler == "step":
        sched = optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # Prepare logging
    _ensure_dir(cfg.results_dir)
    csv_path = Path(cfg.results_dir) / f"heat2d_{cfg.tag}.csv"
    logger = CSVLogger(
        csv_path,
        header=["epoch", "lr", "loss", "loss_pde", "loss_bcic", "loss_stl"],
        overwrite=None,
        create_dirs=True,
        float_precision=6,
    )

    saved: list[str] = []

    # --- training loop --------------------------------------------------------
    n_total = XYT.shape[0]
    batch = min(cfg.batch, n_total)
    hard_k = max(0, int(cfg.rar_hard_frac * batch)) if (cfg.rar_pool > 0 and cfg.rar_every > 0) else 0

    for epoch in range(int(cfg.epochs)):
        model.train()

        # RAR: sample a pool, select hardest by |residual|
        if hard_k > 0 and cfg.rar_pool > 0 and (epoch % max(1, cfg.rar_every) == 0):
            pool_idx = torch.randint(0, n_total, (min(int(cfg.rar_pool), n_total),), device=device)
            pool = XYT[pool_idx].requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                res_pool = residual_heat2d(model, pool, alpha=cfg.alpha).squeeze(-1)
                scores = res_pool.abs()
            hard_idx_rel = torch.topk(scores, k=min(hard_k, scores.numel())).indices
            hard_coords = pool[hard_idx_rel].detach()  # detach, will re‑enable grad below
            # Fill the rest randomly
            rand_k = batch - hard_coords.shape[0]
            rand_coords = XYT[torch.randint(0, n_total, (rand_k,), device=device)]
            coords = torch.cat([hard_coords, rand_coords], dim=0)
        else:
            idx = torch.randint(0, n_total, (batch,), device=device)
            coords = XYT[idx]

        coords = coords.requires_grad_(True)

        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
            # PDE residual loss
            res = residual_heat2d(model, coords, alpha=cfg.alpha)
            loss_pde = (res.square()).mean()

            # Boundary and initial condition penalties (soft by default)
            loss_bcic = bc_ic_heat2d(
                model,
                x_min=cfg.x_min, x_max=cfg.x_max, y_min=cfg.y_min, y_max=cfg.y_max, t_min=cfg.t_min, t_max=cfg.t_max,
                device=device, dtype=dtype, n_boundary=int(cfg.n_boundary), n_initial=int(cfg.n_initial),
            )

            # Optional STL penalty (computed every stl_every epochs to save time)
            if cfg.stl_use and _HAS_STL and cfg.stl_weight > 0.0 and (epoch % max(1, cfg.stl_every) == 0):
                loss_stl = _stl_penalty(model, cfg, device, dtype=dtype)
            else:
                loss_stl = torch.zeros((), device=device, dtype=dtype)

            loss = loss_pde + cfg.bcic_weight * loss_bcic + loss_stl

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():  # type: ignore[attr-defined]
            scaler.scale(loss).backward()                    # type: ignore[attr-defined]
            if cfg.grad_clip is not None:
                scaler.unscale_(opt)                         # type: ignore[attr-defined]
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(opt)                                 # type: ignore[attr-defined]
            scaler.update()                                  # type: ignore[attr-defined]
        else:
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()

        if sched is not None:
            sched.step()

        # Logging & progress
        lr_now = float(opt.param_groups[0]["lr"])
        logger.append([epoch, lr_now, float(loss), float(loss_pde), float(loss_bcic), float(loss_stl)])

        if (epoch % max(1, cfg.print_every) == 0) or (epoch == cfg.epochs - 1):
            print(
                f"[heat2d] epoch={epoch:04d} lr={lr_now:.2e} "
                f"loss={float(loss):.4e} pde={float(loss_pde):.4e} "
                f"bcic={float(loss_bcic):.4e} stl={float(loss_stl):.4e}"
            )

    # --- export artifacts -----------------------------------------------------
    # Checkpoint (state dict + minimalist config)
    if cfg.save_ckpt:
        ckpt_path = Path(cfg.results_dir) / f"heat2d_{cfg.tag}.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, ckpt_path)
        saved.append(str(ckpt_path))

    # Save a dense field tensor on the grid for downstream analysis/monitoring
    with torch.no_grad():
        model.eval()
        u = model(XYT).reshape(cfg.n_x, cfg.n_y, cfg.n_t).detach().cpu()
        field_path = Path(cfg.results_dir) / f"heat2d_{cfg.tag}_field.pt"
        torch.save({"u": u, "X": X.detach().cpu(), "Y": Y.detach().cpu(), "T": T.detach().cpu(), "alpha": float(cfg.alpha),
                    "config": cfg.__dict__}, field_path)
        saved.append(str(field_path))

    # Optionally export selected time frames and gradient‑magnitude figures
    if cfg.save_frames or cfg.save_figs:
        for k in list(cfg.frames_idx):
            if 0 <= int(k) < cfg.n_t:
                tval = float(T[0, 0, k].item())
                with torch.no_grad():
                    # Slice XYT for this time index
                    xt = X[:, :, k].reshape(-1, 1)
                    yt = Y[:, :, k].reshape(-1, 1)
                    tt = torch.full_like(xt, fill_value=tval)
                    inp = torch.cat([xt, yt, tt], dim=-1).to(device)
                    u2 = model(inp).reshape(cfg.n_x, cfg.n_y).detach().cpu().numpy()

                if cfg.save_frames:
                    npy = Path(cfg.results_dir) / f"heat2d_{cfg.tag}_t{k}.npy"
                    np.save(npy, u2)
                    saved.append(str(npy))

                if cfg.save_figs:
                    import matplotlib.pyplot as plt  # lazy import
                    gradmag = _gradmag_numpy(u2)
                    plt.figure()
                    plt.imshow(gradmag.T, origin="lower", aspect="auto")
                    plt.colorbar(label="|∇u|")
                    plt.xlabel("x‑index")
                    plt.ylabel("y‑index")
                    plt.title(f"2‑D Heat |∇u|, frame t[{k}]")
                    figp = Path(cfg.results_dir) / f"heat2d_{cfg.tag}_gradmag_t{k}.png"
                    plt.tight_layout()
                    plt.savefig(figp, dpi=150)
                    plt.close()
                    saved.append(str(figp))

    # Return artifact list
    return saved
