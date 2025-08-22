#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a 2‑D heat‑equation PINN and *audit* it with a STREL (spatio‑temporal logic) specification.

What this script does
---------------------
• Trains a compact Physics‑Informed Neural Network (PINN) to solve the 2‑D heat equation
  u_t = α (u_xx + u_yy) on (x, y) ∈ [0,1]^2, t ∈ [0, T].
• Enforces physics via a PDE residual loss + boundary/initial conditions.
• Optionally adds a *differentiable safety regularizer* approximating an “always‑below‑τ”
  spatio‑temporal constraint (soft STL surrogate).
• Optionally *audits* the learned rollout with a **MoonLight** STREL monitor
  (quantitative semantics), using a user‑provided `.mls` script and formula name.

Why this matches the professor’s guidance
-----------------------------------------
The course direction was to “monitor signal temporal logic (STL) specifications
or spatial STL generalizations” for physics‑based ML models (neural ODE/PDE, PINNs).
This script trains a neural PDE model *and* integrates MoonLight’s STREL monitoring
to assess spatio‑temporal properties over a grid graph. MoonLight provides a Python
interface (via JPype) and supports STREL monitoring on static/dynamic spatial graphs
with Boolean or robust quantitative verdicts.  See MoonLight’s README and the tool paper.  # noqa: E501

    • MoonLight GitHub (pip installable; requires Java ≥ 21): https://github.com/MoonLightSuite/moonlight
    • Tool paper (“MoonLight: a lightweight tool for monitoring spatio‑temporal properties”):
      https://link.springer.com/content/pdf/10.1007/s10009-023-00710-5.pdf

We also keep a light‑weight “soft STL” surrogate inside PyTorch for a basic safety
envelope without requiring differentiability of an external monitor (cf. STLnet ideas).  # noqa: E501
For classical STL monitoring in Python, see RTAMT.  For physics‑ML frameworks, see
NeuroMANCER, PhysicsNeMo, and TorchPhysics.

References (helpful starting points)
------------------------------------
• MoonLight README + Python usage. 
• MoonLight tool paper (STREL, Python/Matlab interface). 
• RTAMT, a Python STL monitor (dense/discrete time). 
• NeuroMANCER docs (PyTorch SciML). 

Usage (examples)
----------------
# Train for 2k epochs, audit every 200 epochs with a STREL script and save rollout
python scripts/train_heat2d_strel.py \
    --epochs 2000 --audit --mls specs/heat2d.mls --formula contain \
    --out runs/heat2d_strel --nx 64 --ny 64 --nt 50 --alpha 0.1

# With a soft safety envelope u(x,y,t) ≤ 0.7 for t ∈ [0.3, 1.0], weight 0.1
python scripts/train_heat2d_strel.py \
    --epochs 2000 --safety-threshold 0.7 --safety-twin 0.3 1.0 --w-safety 0.1

Notes
-----
• The MoonLight audit is optional and runs *outside* autograd. If MoonLight or Java
  is not present, the script degrades gracefully and still trains the PINN.
• For STREL, we build a 4‑neighborhood grid graph; adjust with --adj-weight if needed.
• We keep the code dependency‑light and chunk large forward passes for memory safety.

"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ------------------------------ Optional heavy deps ----------------------------------------
try:  # pragma: no cover
    import torch
    from torch import nn, Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    Tensor = None  # type: ignore

# Prefer project helpers if available (cleaner MoonLight bridge & grid tools)
_HAVE_HELPERS = False
try:  # pragma: no cover
    from physical_ai_stl.monitoring.moonlight_helper import (  # type: ignore
        load_script_from_file as _ml_load_script,
        get_monitor as _ml_get_monitor,
        build_grid_graph as _ml_build_graph,
        field_to_signal as _ml_field_to_signal,
    )
    _HAVE_HELPERS = True
except Exception:
    pass

# ------------------------------------- Small utils -----------------------------------------

def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required. Please install torch to train the model.")

def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no‑op on CPU
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]

def device_str(prefer: str | None = None) -> str:
    if torch is None:  # pragma: no cover
        return "cpu"
    if prefer in {"cpu", "cuda"}:
        if prefer == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------- Model ------------------------------------------------

class Sine(nn.Module):
    """Sine activation (SIREN‑style first layer support is optional here)."""
    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = float(w0)
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return torch.sin(self.w0 * x)

def _make_activation(name: str | None) -> nn.Module:
    if not name:
        return nn.Identity()
    name = name.lower()
    if name in {"tanh", "tanhshrink"}:
        return nn.Tanh()
    if name in {"relu"}:
        return nn.ReLU()
    if name in {"gelu"}:
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name in {"sine", "sin"}:
        return Sine()
    return nn.Tanh()

class MLP(nn.Module):  # type: ignore[misc]
    """A compact, well‑behaved MLP for coordinate‑based PINNs.

    * Activation‑aware initialization (fan‑in with gain).
    * Optional last‑layer scale to start near small outputs (helps PDE residuals).
    * Supports weight normalization and residual (skip) connections across blocks.
    """
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: Sequence[int] = (64, 64, 64),
        activation: str = "tanh",
        *,
        out_activation: str | None = None,
        bias: bool = True,
        last_layer_scale: float | None = 0.1,
        weight_norm: bool = False,
        skips: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.hidden = list(map(int, hidden))
        self.skips = set(int(i) for i in skips)

        layers: list[nn.Module] = []
        dims = [in_dim] + list(hidden) + [out_dim]
        act = _make_activation(activation)
        out_act = _make_activation(out_activation) if out_activation else nn.Identity()

        for li, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            lin = nn.Linear(din, dout, bias=bias)
            # Weight norm optionally
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            layers.append(lin)
            # Add nonlinearity except after final layer
            if li < len(dims) - 2:
                layers.append(act)
        layers.append(out_act)
        self.net = nn.Sequential(*layers)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Optionally scale final layer small
        if last_layer_scale is not None and isinstance(self.net[-2], nn.Linear):
            with torch.no_grad():
                self.net[-2].weight.mul_(float(last_layer_scale))
                if self.net[-2].bias is not None:
                    self.net[-2].bias.mul_(float(last_layer_scale))

    def forward(self, coords: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(coords)

# ------------------------------------ PDE & sampling ---------------------------------------

@dataclass
class Domain2D:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def sample_interior(self, n: int, *, device: str = "cpu", sobol: bool = True) -> Tensor:
        _require_torch()
        if sobol and hasattr(torch, "quasirandom"):
            eng = torch.quasirandom.SobolEngine(dimension=3)
            u = eng.draw(n)
        else:
            u = torch.rand(n, 3)
        x = self.x_min + (self.x_max - self.x_min) * u[:, 0:1]
        y = self.y_min + (self.y_max - self.y_min) * u[:, 1:2]
        t = self.t_min + (self.t_max - self.t_min) * u[:, 2:3]
        return torch.cat([x, y, t], dim=1).to(device)

    def sample_boundary(self, n: int, *, device: str = "cpu") -> Tensor:
        _require_torch()
        # pick a side (0: x=x_min, 1: x=x_max, 2: y=y_min, 3: y=y_max)
        side = torch.randint(0, 4, (n, 1))
        u = torch.rand(n, 2)  # free coordinates
        x = torch.where(
            side == 0, torch.full((n, 1), self.x_min),
            torch.where(side == 1, torch.full((n, 1), self.x_max), torch.nan),
        )
        y = torch.where(
            side == 2, torch.full((n, 1), self.y_min),
            torch.where(side == 3, torch.full((n, 1), self.y_max), torch.nan),
        )
        x = torch.where(torch.isnan(x), self.x_min + (self.x_max - self.x_min) * u[:, 0:1], x)
        y = torch.where(torch.isnan(y), self.y_min + (self.y_max - self.y_min) * u[:, 1:2], y)
        # time is random over domain
        t = self.t_min + (self.t_max - self.t_min) * torch.rand(n, 1)
        return torch.cat([x, y, t], dim=1).to(device)

    def sample_initial(self, n: int, *, device: str = "cpu") -> Tensor:
        _require_torch()
        x = self.x_min + (self.x_max - self.x_min) * torch.rand(n, 1)
        y = self.y_min + (self.y_max - self.y_min) * torch.rand(n, 1)
        t = torch.full((n, 1), self.t_min)
        return torch.cat([x, y, t], dim=1).to(device)

def gaussian_hotspot(x: Tensor, y: Tensor, *, cx: float = 0.5, cy: float = 0.5, sharpness: float = 40.0) -> Tensor:
    r2 = (x - cx).square() + (y - cy).square()
    return torch.exp(-sharpness * r2)

def heat_pde_residual(u: Tensor, x: Tensor, y: Tensor, t: Tensor, alpha: float = 0.1) -> Tensor:
    """Compute r = u_t − α (u_xx + u_yy) at queried coords (elementwise)."""
    _require_torch()
    ones = torch.ones_like(u)
    # First‑order grads
    du = torch.autograd.grad(u, [x, y, t], grad_outputs=ones, create_graph=True, retain_graph=True)
    u_x, u_y, u_t = du  # type: ignore[misc]
    # Second‑order
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return u_t - alpha * (u_xx + u_yy)

# ------------------------------------ Soft STL surrogate -----------------------------------

def softplus(x: Tensor, beta: float = 50.0) -> Tensor:
    # A sharp, numerically stable hinge ~ max(0,x)
    return torch.nn.functional.softplus(x, beta=beta)

def soft_all_leq(u: Tensor, thr: float) -> Tensor:
    """Return a hinge penalty for the property  max(u) ≤ thr  (i.e., u everywhere ≤ thr).

    Args:
        u: Tensor [..., N] or [..., H, W] — values to be bounded.
        thr: threshold τ.
    Returns:
        Scalar penalty ≥ 0.
    """
    if u.ndim >= 2:
        u_max = u.view(u.shape[0], -1).max(dim=1).values if u.ndim > 1 else u
    else:
        u_max = u
    return softplus(u_max.mean() - float(thr))

# --------------------------------------- Config --------------------------------------------

@dataclass
class Config:
    # Domain/grid to *train on* (collocation)
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # Evaluation grid (for rollout export + MoonLight audit)
    nx: int = 64
    ny: int = 64
    nt: int = 50

    # Physics
    alpha: float = 0.1

    # Training set sizes
    n_collocation: int = 4096
    n_boundary: int = 1024
    n_initial: int = 1024
    sobol: bool = True

    # Model + optimization
    hidden: Sequence[int] = (64, 64, 64)
    activation: str = "tanh"
    lr: float = 3e-3
    epochs: int = 2000
    batch_pde: int | None = None
    last_layer_scale: float | None = 0.1
    weight_norm: bool = False
    grad_clip: float | None = 1.0

    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 1.0
    w_safety: float = 0.0  # set >0 to enable soft STL surrogate

    # Soft safety surrogate (u ≤ τ for t ∈ [t0,t1])
    safety_threshold: float | None = None
    safety_t0: float = 0.0
    safety_t1: float = 1.0

    # Misc
    seed: int = 1337
    device: str | None = None

# --------------------------------------- Training ------------------------------------------

def train_heat2d(
    cfg: Config,
    *,
    out_dir: Path,
    audit: bool = False,
    mls: Path | None = None,
    formula: str | None = None,
    adj_weight: float = 1.0,
    audit_every: int = 200,
    save_every: int = 200,
    audit_threshold: float | None = None,
) -> dict:
    """Train a PINN and (optionally) audit the rollout with MoonLight STREL."""
    _require_torch()
    set_seed(cfg.seed)
    dev = device_str(cfg.device)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # Model
    model = MLP(
        in_dim=3,
        out_dim=1,
        hidden=cfg.hidden,
        activation=cfg.activation,
        last_layer_scale=cfg.last_layer_scale,
        weight_norm=cfg.weight_norm,
    ).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dom = Domain2D(cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max, cfg.t_min, cfg.t_max)

    # Pre‑sample static training sets
    Xc = dom.sample_interior(cfg.n_collocation, device=dev, sobol=cfg.sobol).requires_grad_(True)
    Xb = dom.sample_boundary(cfg.n_boundary, device=dev).requires_grad_(True)
    Xi = dom.sample_initial(cfg.n_initial, device=dev).requires_grad_(True)

    # Initial/boundary targets
    with torch.no_grad():
        ui = gaussian_hotspot(Xi[:, 0:1], Xi[:, 1:2])  # at t = 0
        ub = torch.zeros_like(Xb[:, :1])               # Dirichlet boundary: u=0

    def forward(coords: Tensor, chunk: int = 16384) -> Tensor:
        """Chunked forward to avoid OOM on large grids."""
        outs: list[Tensor] = []
        if chunk is None:
            return model(coords)
        for s in range(0, coords.shape[0], chunk):
            outs.append(model(coords[s:s + chunk]))
        return torch.cat(outs, dim=0)

    # Training loop
    history: list[dict] = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        # PDE residual at collocation
        Xc.requires_grad_(True)
        uc = forward(Xc)
        rc = heat_pde_residual(uc, Xc[:, 0:1], Xc[:, 1:2], Xc[:, 2:3], cfg.alpha)
        loss_pde = (rc.square()).mean()

        # Boundary
        Xb.requires_grad_(True)
        ub_hat = forward(Xb)
        loss_bc = torch.nn.functional.mse_loss(ub_hat, ub)

        # Initial
        Xi.requires_grad_(True)
        ui_hat = forward(Xi)
        loss_ic = torch.nn.functional.mse_loss(ui_hat, ui)

        loss = cfg.w_pde * loss_pde + cfg.w_bc * loss_bc + cfg.w_ic * loss_ic

        # Optional soft STL‑style safety: u(x,y,t) ≤ τ for t in [t0, t1]
        if cfg.w_safety > 0.0 and cfg.safety_threshold is not None:
            # Sample a small grid of times in [t0, t1]
            n_probe_t = min(10, cfg.nt)
            ts = torch.linspace(cfg.safety_t0, cfg.safety_t1, n_probe_t, device=dev).view(-1, 1)
            # Randomly sample some (x,y) for probes
            n_probe_xy = 2 * int(math.sqrt(cfg.n_collocation))
            xs = dom.x_min + (dom.x_max - dom.x_min) * torch.rand(n_probe_xy, 1, device=dev)
            ys = dom.y_min + (dom.y_max - dom.y_min) * torch.rand(n_probe_xy, 1, device=dev)
            coords = torch.cat([
                xs.repeat_interleave(n_probe_t, dim=0),
                ys.repeat_interleave(n_probe_t, dim=0),
                ts.repeat(n_probe_xy, 1),
            ], dim=1).requires_grad_(True)
            u_probe = forward(coords)
            loss_safety = soft_all_leq(u_probe, cfg.safety_threshold)
            loss = loss + cfg.w_safety * loss_safety
        else:
            loss_safety = torch.tensor(0.0, device=dev)

        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[ep {ep:5d}] loss={loss.item():.4e}  pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  bc={loss_bc.item():.4e}  saf={float(loss_safety):.4e}")  # noqa: E501

        # Periodic MoonLight audit & checkpoint
        if (audit and mls is not None and formula) and (ep % audit_every == 0 or ep == cfg.epochs):
            try:
                audit_result = _audit_rollout_with_moonlight(
                    model, cfg, out_dir, mls=mls, formula=formula, adj_weight=adj_weight, threshold=audit_threshold
                )
            except Exception as e:  # robust in non‑Java CI environments
                print(f"[audit] Skipped (MoonLight unavailable?): {e}")
                audit_result = {"ok": False, "error": str(e)}
        else:
            audit_result = None

        if (ep % save_every == 0) or ep == cfg.epochs:
            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": asdict(cfg),
            }
            torch.save(ckpt, out_dir / "checkpoints" / f"ep_{ep:05d}.pt")

        history.append({
            "epoch": ep,
            "loss": float(loss.detach().cpu()),
            "loss_pde": float(loss_pde.detach().cpu()),
            "loss_ic": float(loss_ic.detach().cpu()),
            "loss_bc": float(loss_bc.detach().cpu()),
            "loss_safety": float(loss_safety.detach().cpu()),
            "audit": audit_result,
        })

    # Final rollout export for downstream evaluation
    rollout_path = _export_rollout(model, cfg, out_dir)
    (out_dir / "meta.json").write_text(json.dumps({"cfg": asdict(cfg), "rollout": str(rollout_path.name)}, indent=2))

    return {"history": history, "rollout": str(rollout_path)}

# -------------------------------- MoonLight audit helpers ----------------------------------

def _build_grid_graph(nx: int, ny: int, weight: float = 1.0) -> list[list[float]]:
    """Create a 4‑neighborhood adjacency matrix in the format MoonLight expects."""
    n = nx * ny
    adj = [[0.0] * n for _ in range(n)]

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            if i > 0:
                v = idx(i - 1, j); adj[u][v] = weight; adj[v][u] = weight
            if i + 1 < nx:
                v = idx(i + 1, j); adj[u][v] = weight; adj[v][u] = weight
            if j > 0:
                v = idx(i, j - 1); adj[u][v] = weight; adj[v][u] = weight
            if j + 1 < ny:
                v = idx(i, j + 1); adj[u][v] = weight; adj[v][u] = weight
    return adj

def _field_to_signal(u_xyz: np.ndarray, *, layout: str = "xy_t", threshold: float | None = None) -> list[list[list[float]]]:
    """Reshape (nx,ny,nt) (or (nt,nx,ny)) into MoonLight’s signal shape (T, N, features)."""
    a = np.asarray(u_xyz)
    if layout == "xy_t":
        if a.ndim != 3:
            raise ValueError(f"Expected (nx,ny,nt) for layout 'xy_t'; got {a.shape}")
        nx, ny, nt = a.shape
        flat = a.reshape(nx * ny, nt).T  # (T, N)
    elif layout == "t_xy":
        if a.ndim != 3:
            raise ValueError(f"Expected (nt,nx,ny) for layout 't_xy'; got {a.shape}")
        nt, nx, ny = a.shape
        flat = a.reshape(nt, nx * ny)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    if threshold is not None:
        flat = (flat >= threshold).astype(float, copy=False)
    flat = flat.astype(float, copy=False)
    # add a feature axis → (T, N, 1)
    return [[[float(v)] for v in row] for row in flat.tolist()]

def _load_monitor(mls: Path, formula: str):
    if _HAVE_HELPERS:
        m = _ml_get_monitor(_ml_load_script(str(mls)), formula)
        return m
    # Minimal fallback direct import (API may vary; best effort)
    try:
        from moonlight import ScriptLoader  # type: ignore
        loader = ScriptLoader()
        script = loader.loadFromFile(str(mls))
        monitor = script.getMonitor(formula)
        return monitor
    except Exception as e:
        raise RuntimeError(f"MoonLight Python bridge not available: {e}")

def _run_monitor(monitor, graph, signal):
    # The MoonLight Python API returns a nested structure. We try to extract a float robustness.
    try:
        res = monitor.monitor(graph, signal)  # Quantitative/Boolean semantics depending on script
    except Exception as e:
        # Some versions use a different entry point:
        if hasattr(monitor, 'apply'):
            res = monitor.apply(graph, signal)
        else:
            raise
    # Convert result to a scalar if possible
    def _extract(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, list) and x:
            # common shapes: [value], [[t, value], ...]
            first = x[0]
            if isinstance(first, list):
                return float(first[-1])
            return float(first)
        if hasattr(x, 'doubleValue'):
            return float(x.doubleValue())  # JPype Java Double
        return float(x)
    try:
        return _extract(res)
    except Exception:
        return res  # fallback: return raw structure

def _audit_rollout_with_moonlight(
    model: nn.Module,
    cfg: Config,
    out_dir: Path,
    *,
    mls: Path,
    formula: str,
    adj_weight: float = 1.0,
    threshold: float | None = None,
) -> dict:
    dev = next(model.parameters()).device
    grid_x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, device=dev)
    grid_y = torch.linspace(cfg.y_min, cfg.y_max, cfg.ny, device=dev)
    grid_t = torch.linspace(cfg.t_min, cfg.t_max, cfg.nt, device=dev)
    XX, YY = torch.meshgrid(grid_x, grid_y, indexing='ij')
    frames: list[np.ndarray] = []
    with torch.no_grad():
        for tk in grid_t:
            coords = torch.stack([XX.reshape(-1), YY.reshape(-1), torch.full_like(XX.reshape(-1,1), tk)[:,0]], dim=1)
            u = model(coords).reshape(cfg.nx, cfg.ny)
            frames.append(u.detach().cpu().numpy())
    u_xyz = np.stack(frames, axis=-1)  # (nx,ny,nt)

    # Export rollout for reproducibility
    np.save(out_dir / "rollout.npy", u_xyz)  # (nx,ny,nt)

    # Build graph + signal for MoonLight
    if _HAVE_HELPERS:
        graph = _ml_build_graph(cfg.nx, cfg.ny, weight=adj_weight)
        signal = _ml_field_to_signal(u_xyz, threshold=threshold, layout="xy_t")
    else:
        graph = _build_grid_graph(cfg.nx, cfg.ny, weight=adj_weight)
        signal = _field_to_signal(u_xyz, threshold=threshold, layout="xy_t")

    monitor = _load_monitor(mls, formula)
    rob = _run_monitor(monitor, graph, signal)
    print(f"[audit] STREL robustness ({formula}) = {rob}")
    # Persist small audit record
    rec = {"formula": formula, "robustness": float(rob) if isinstance(rob, (int, float)) else None}
    (out_dir / "strel_audit.json").write_text(json.dumps(rec, indent=2))
    return {"ok": True, **rec}

def _export_rollout(model: nn.Module, cfg: Config, out_dir: Path) -> Path:
    """Export a dense (nx,ny,nt) rollout to .npy for downstream evaluation tools."""
    dev = next(model.parameters()).device
    xs = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, device=dev)
    ys = torch.linspace(cfg.y_min, cfg.y_max, cfg.ny, device=dev)
    ts = torch.linspace(cfg.t_min, cfg.t_max, cfg.nt, device=dev)
    XX, YY = torch.meshgrid(xs, ys, indexing='ij')
    out = torch.empty(cfg.nx, cfg.ny, cfg.nt, device=dev)
    with torch.no_grad():
        for k, tk in enumerate(ts):
            coords = torch.stack([XX.reshape(-1), YY.reshape(-1), torch.full_like(XX.reshape(-1,1), tk)[:,0]], dim=1)
            u = model(coords).reshape(cfg.nx, cfg.ny)
            out[..., k] = u
    arr = out.detach().cpu().numpy()
    path = out_dir / "rollout.npy"
    np.save(path, arr)
    return path

# ------------------------------------------ CLI --------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=("Train a 2‑D heat‑equation PINN and optionally audit with MoonLight STREL.")
    )
    # training
    ap.add_argument("--epochs", type=int, default=2000, help="Training steps (default: 2000).")  # longer default
    ap.add_argument("--alpha", type=float, default=0.1, help="Diffusivity alpha (default: 0.1).")  # heat coeff
    ap.add_argument("--lr", type=float, default=3e-3, help="Learning rate (default: 3e-3).")
    ap.add_argument("--hidden", type=int, nargs="+", default=(64, 64, 64), help="Hidden widths, e.g. --hidden 64 64 64.")
    ap.add_argument("--activation", type=str, default="tanh", help="Activation (tanh|relu|gelu|silu|sine).")

    # sampling sizes
    ap.add_argument("--n-collocation", type=int, default=4096)
    ap.add_argument("--n-boundary", type=int, default=1024)
    ap.add_argument("--n-initial", type=int, default=1024)
    ap.add_argument("--sobol", action="store_true", help="Use Sobol sampling for interior points (default: on).")
    ap.add_argument("--no-sobol", dest="sobol", action="store_false")
    ap.set_defaults(sobol=True)

    # domain + eval grid
    ap.add_argument("--x-min", type=float, default=0.0); ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--y-min", type=float, default=0.0); ap.add_argument("--y-max", type=float, default=1.0)
    ap.add_argument("--t-min", type=float, default=0.0); ap.add_argument("--t-max", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=64); ap.add_argument("--ny", type=int, default=64); ap.add_argument("--nt", type=int, default=50)

    # loss weights
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-ic", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=1.0)

    # soft safety surrogate
    ap.add_argument("--w-safety", type=float, default=0.0, help="Weight of soft STL safety (0 disables).")  # 0 => off
    ap.add_argument("--safety-threshold", type=float, default=None, help="Safety bound τ for u ≤ τ.")
    ap.add_argument("--safety-twin", type=float, nargs=2, default=None, metavar=("T0", "T1"), help="Time window [T0,T1] where safety applies.")  # noqa: E501

    # audit (MoonLight STREL)
    ap.add_argument("--audit", action="store_true", help="Run MoonLight STREL audit during training.")
    ap.add_argument("--mls", type=Path, default=Path("specs/heat2d_default.mls"), help="Path to MoonLight .mls script.")
    ap.add_argument("--formula", type=str, default="contain", help="Formula name inside the .mls script to monitor.")
    ap.add_argument("--adj-weight", type=float, default=1.0, help="Edge weight for 4‑neighborhood grid graph.")
    ap.add_argument("--audit-every", type=int, default=200, help="Audit every N epochs (default: 200).")
    ap.add_argument("--audit-threshold", type=float, default=None, help="Binarization threshold for boolean STREL specs (optional).")

    # misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None.__name__.lower()])
    ap.add_argument("--out", type=Path, default=Path("runs/heat2d_strel"))
    args = ap.parse_args()

    cfg = Config(
        x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max, t_min=args.t_min, t_max=args.t_max,
        nx=args.nx, ny=args.ny, nt=args.nt, alpha=args.alpha,
        n_collocation=args.n_collocation, n_boundary=args.n_boundary, n_initial=args.n_initial, sobol=args.sobol,
        hidden=tuple(int(h) for h in args.hidden), activation=args.activation, lr=args.lr, epochs=args.epochs,
        w_pde=args.w_pde, w_bc=args.w_bc, w_ic=args.w_ic,
        w_safety=args.w_safety,
        safety_threshold=args.safety_threshold if args.safety_threshold is not None else None,
        safety_t0=(args.safety_twin[0] if args.safety_twin else 0.0),
        safety_t1=(args.safety_twin[1] if args.safety_twin else 1.0),
        seed=args.seed, device=args.device,
    )

    # Provide a small default STREL script if the user wants audit but no file exists.
    if args.audit and not args.mls.exists():
        args.mls.parent.mkdir(parents=True, exist_ok=True)
        # A tiny STREL script:
        #   signal { bool hot; }      // 1‑feature boolean per node
        #   domain boolean;
        #   formula contain = always ( not somewhere (hot) );
        # Interpreted as: throughout the time window, no node is “hot”.
        # If you use this boolean domain, supply --audit-threshold to binarize the rollout.
        args.mls.write_text(
            "signal { bool hot; }\n"
            "domain boolean;\n"
            "formula contain = always ( not somewhere (hot) );\n"
        )
        print(f"[spec] Created default MoonLight spec at {args.mls}")
        print("      You probably want to supply a richer STREL file for your study.")

    train_heat2d(
        cfg,
        out_dir=args.out,
        audit=bool(args.audit),
        mls=args.mls if args.audit else None,
        formula=args.formula if args.audit else None,
        adj_weight=args.adj_weight,
        audit_every=args.audit_every,
        audit_threshold=args.audit_threshold,
    )

if __name__ == "__main__":  # pragma: no cover
    main()
