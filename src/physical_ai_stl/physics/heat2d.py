from __future__ import annotations

"""
Physics utilities for the 2‑D heat equation used by the experiments.

This module provides:

• ``residual_heat2d`` – Autograd‑based PDE residual for the parabolic heat equation,
  with support for **spatially/temporally varying or anisotropic diffusivity** and
  an optional **source term**.
• ``bc_ic_heat2d`` – Lightweight helpers that produce soft losses for **boundary**
  (Dirichlet or Neumann) and **initial** conditions on a rectangular space‑time slab.
• Small convenience utilities (Gaussian IC, exact Dirichlet mask, etc.).

Design goals
------------
1) **Correctness first.** All derivatives are obtained via PyTorch autograd on
   a *single* forward pass to avoid needless recomputation and retain second‑order
   gradients for meta‑optimization.
2) **Plug‑and‑play.** Keep the public signatures backward compatible with the
   original experiment code while allowing richer options via keyword arguments.
3) **Speed where it matters.** Avoid Python‑side loops, reuse intermediate
   derivatives, and keep allocations minimal.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..models.mlp import MLP

# Optional import (kept local to avoid import cycles at module load time).
# Provides Sobol/LHS boundary sampling for rectangles, if available.
try:  # pragma: no cover - convenience only
    from ..training.grids import sample_boundary_2d
except Exception:  # pragma: no cover
    sample_boundary_2d = None  # type: ignore


# =============================================================================
# Core PDE residual
# =============================================================================

def _as_broadcastable(x: float | Tensor, n: int, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    """Return a tensor of shape ``(n, 1)`` broadcastable against model outputs."""
    if isinstance(x, Tensor):
        if x.ndim == 0:
            return x.reshape(1, 1).to(device=device, dtype=dtype).expand(n, 1)
        if x.ndim == 1:
            return x.reshape(-1, 1).to(device=device, dtype=dtype)
        if x.ndim == 2:
            return x.to(device=device, dtype=dtype)
        raise ValueError("Expected scalar/1D/2D tensor for broadcast.")
    return torch.full((n, 1), float(x), device=device, dtype=dtype)


def _split_coords(coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Split ``coords`` (N,3) into (x, y, t) each shaped (N,1)."""
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    t = coords[:, 2:3]
    return x, y, t


def residual_heat2d(
    model: MLP,
    coords: Tensor,
    alpha: float | tuple[float, float] | Tensor | Callable[[Tensor], Tensor] = 0.1,
    *,
    source: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    r"""Return the strong‑form residual for the 2‑D heat equation on a rectangle.

    The PDE (with optional source term) is taken as

        u_t - ∇·(A ∇u) = s(x, y, t),

    where ``A`` is the (possibly anisropic) diffusivity.  The default reduces to
    ``u_t - α (u_xx + u_yy)`` for scalar α, which matches the experiment scripts.

    Parameters
    ----------
    model
        Neural field mapping ``(x, y, t) -> u`` (scalar).
    coords
        Tensor of shape ``(N, 3)`` with columns ``[x, y, t]``.
    alpha
        Diffusivity specification.  Supported forms:

        • **float** (default 0.1): isotropic scalar α.
        • **tuple(float, float)**: diagonal anisotropy (α_x, α_y).
        • **Tensor** with shape broadcastable to ``(N, 1)``: spatially varying scalar α(x,y,t).
        • **Tensor** with shape broadcastable to ``(N, 2)``: diagonal anisotropy (α_x, α_y) per‑point.
        • **Callable[[coords], Tensor]**: returns either shape ``(N,1)`` or ``(N,2)``.

    source
        Optional source term ``s(coords)`` (same shape as model output).  If ``None``,
        the equation is taken as homogeneous.

    Returns
    -------
    Tensor
        Residual tensor of shape ``(N, 1)``.  A perfect solution yields values near zero.

    Notes
    -----
    • Autograd tracing requires ``coords.requires_grad=True``. This is set internally.
    • We compute the diagonal of the Hessian via two first‑order ``grad`` calls,
      which is fast and memory‑efficient for scalar fields.
    """
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (N, 3) with columns [x, y, t]")

    # Enable autograd for first/second derivatives
    coords = coords.requires_grad_(True)

    # Forward
    u: Tensor = model(coords)  # (N,1) assumed

    # First derivatives (∂u/∂x, ∂u/∂y, ∂u/∂t)
    du = torch.autograd.grad(
        u, coords, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    u_t = du[:, 2:3]

    # Second derivatives (diagonal Hessian)
    u_xx = torch.autograd.grad(
        u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0][:, 1:2]

    # Diffusivity handling -----------------------------------------------------
    N = coords.shape[0]
    device = coords.device
    dtype = u.dtype

    if callable(alpha):
        alpha_t = alpha(coords)
    elif isinstance(alpha, tuple):
        alpha_t = torch.stack(
            [
                _as_broadcastable(alpha[0], N, device, dtype).squeeze(-1),
                _as_broadcastable(alpha[1], N, device, dtype).squeeze(-1),
            ],
            dim=1,
        )
    else:
        alpha_t = alpha

    # Normalize to either (N,1) [isotropic] or (N,2) [diag anisotropic]
    if isinstance(alpha_t, Tensor):
        if alpha_t.ndim == 0:
            alpha_t = _as_broadcastable(alpha_t, N, device, dtype)
        elif alpha_t.ndim == 1:
            # ambiguity: treat as (N,) -> (N,1) isotropic
            alpha_t = alpha_t.reshape(-1, 1).to(device=device, dtype=dtype)
        elif alpha_t.ndim == 2 and alpha_t.shape[1] in (1, 2):
            alpha_t = alpha_t.to(device=device, dtype=dtype)
        else:
            raise ValueError("alpha tensor must have shape (N,), (N,1) or (N,2)")
    else:
        # plain float
        alpha_t = _as_broadcastable(float(alpha_t), N, device, dtype)

    # Construct ∇·(A ∇u)
    if alpha_t.shape[1] == 1:  # isotropic
        lap_u = u_xx + u_yy
        diff_term = alpha_t * lap_u
    else:  # diagonal anisotropy: A = diag(α_x, α_y)
        ax = alpha_t[:, 0:1]
        ay = alpha_t[:, 1:2]
        diff_term = ax * u_xx + ay * u_yy

    # Optional source s(x,y,t)
    s = source(coords) if (source is not None) else 0.0

    return u_t - diff_term - s


# =============================================================================
# Boundary / initial condition helpers (soft penalties by default)
# =============================================================================

@dataclass(frozen=True)
class SquareDomain2D:
    """Closed rectangular domain in space × time."""
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    @property
    def width(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return float(self.y_max - self.y_min)


def gaussian_ic(
    x: Tensor,
    y: Tensor,
    *,
    center: tuple[float, float] = (0.5, 0.5),
    sharpness: float = 50.0,
    amplitude: float = 1.0,
) -> Tensor:
    """Smooth Gaussian bump used as a default initial condition."""
    cx, cy = center
    r2 = (x - cx).square() + (y - cy).square()
    return amplitude * torch.exp(-sharpness * r2)


def _split_by_side(coords: Tensor, dom: SquareDomain2D, eps: float = 1e-6) -> dict[str, Tensor]:
    """Split boundary samples by which rectangle face they lie on.

    Returns a dict with keys: ``left``, ``right``, ``bottom``, ``top``.
    """
    x, y, _t = _split_coords(coords)
    left = torch.isclose(x, torch.tensor(dom.x_min, device=x.device, dtype=x.dtype), atol=eps).squeeze(1)
    right = torch.isclose(x, torch.tensor(dom.x_max, device=x.device, dtype=x.dtype), atol=eps).squeeze(1)
    bottom = torch.isclose(y, torch.tensor(dom.y_min, device=y.device, dtype=y.dtype), atol=eps).squeeze(1)
    top = torch.isclose(y, torch.tensor(dom.y_max, device=y.device, dtype=y.dtype), atol=eps).squeeze(1)

    return {
        "left": coords[left],
        "right": coords[right],
        "bottom": coords[bottom],
        "top": coords[top],
    }


def bc_ic_heat2d(
    model: MLP,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    # sampling
    n_boundary: int = 512,
    n_initial: int = 512,
    sampler: str | None = None,
    seed: int | None = None,
    boundary_split: Sequence[float] | None = None,
    # boundary type/targets
    boundary: str = "dirichlet",  # "dirichlet" | "neumann"
    boundary_value: Callable[[Tensor, Tensor, Tensor], Tensor] | float | None = 0.0,
    neumann_flux: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
    # initial condition
    ic: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    r"""Return a soft penalty for spatial boundary and initial condition.

    By default this enforces **homogeneous Dirichlet** boundary conditions
    ``u(x,y,t)=0`` on all four spatial faces and sets a compact **Gaussian IC**
    at ``t=t_min`` (usually 0).  Options allow non‑zero Dirichlet values or
    **Neumann** (flux) boundaries.

    Parameters
    ----------
    sampler
        If provided, overrides the default sampler used inside
        ``training.grids.sample_boundary_2d`` (e.g., ``"sobol"``|"rand").
        When the helper is not available in this environment, we fall back to
        a simple uniform RNG sampler.

    boundary
        Type of boundary condition: ``"dirichlet"`` (default) or ``"neumann"``.

    boundary_value
        Target value for Dirichlet boundaries.  Either a scalar or a callable
        ``f(x, y, t) -> Tensor``.  Ignored if ``boundary="neumann"``.

    neumann_flux
        For Neumann boundaries provide ``g(x, y, t, u_x, u_y) -> Tensor`` which
        returns the desired **outward normal flux** value at the sampled points.
        If not provided, zero‑flux is assumed.  Only used when
        ``boundary="neumann"``.

    ic
        Initial condition at ``t=t_min``.  If ``None``, a centered Gaussian bump
        is used.

    Returns
    -------
    Tensor
        Sum of boundary and IC mean‑squared errors (scalar 0‑D tensor).
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    dom = SquareDomain2D(x_min, x_max, y_min, y_max, t_min, t_max)

    # ---------------------------- Sample boundary -----------------------------
    if sample_boundary_2d is not None:
        bc_coords = sample_boundary_2d(
            n_boundary,
            dom.x_min,
            dom.x_max,
            dom.y_min,
            dom.y_max,
            dom.t_min,
            dom.t_max,
            method=(sampler or "sobol"),
            device=device,
            dtype=dtype,
            seed=seed,
            split=boundary_split,
        )
    else:  # fallback: uniform RNG with per‑face selection
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        t = (
            torch.rand((n_boundary, 1), generator=gen, device=device, dtype=dtype)
            * (dom.t_max - dom.t_min)
            + dom.t_min
        )
        sides = torch.randint(0, 4, (n_boundary, 1), generator=gen, device=device)
        xb = (
            torch.rand((n_boundary, 1), generator=gen, device=device, dtype=dtype)
            * (dom.x_max - dom.x_min)
            + dom.x_min
        )
        yb = (
            torch.rand((n_boundary, 1), generator=gen, device=device, dtype=dtype)
            * (dom.y_max - dom.y_min)
            + dom.y_min
        )
        xb[sides == 0] = dom.x_min
        xb[sides == 1] = dom.x_max
        yb[sides == 2] = dom.y_min
        yb[sides == 3] = dom.y_max
        bc_coords = torch.cat([xb, yb, t], dim=1)

    # ---------------------------- Boundary losses -----------------------------
    if boundary.lower().startswith("dir"):
        # Dirichlet: u ≈ boundary_value
        if callable(boundary_value):
            x, y, t = _split_coords(bc_coords)
            target = boundary_value(x, y, t)
        else:
            target = _as_broadcastable(boundary_value or 0.0, bc_coords.shape[0], device, dtype)
        loss_b = (model(bc_coords) - target).square().mean()
    elif boundary.lower().startswith("neu"):
        # Neumann: ∂u/∂n ≈ g
        bc_coords = bc_coords.requires_grad_(True)
        u_b = model(bc_coords)
        du = torch.autograd.grad(
            u_b, bc_coords, grad_outputs=torch.ones_like(u_b), create_graph=True
        )[0]
        u_x = du[:, 0:1]
        u_y = du[:, 1:2]

        # Outward normal derivative per face (deduce face from coords)
        parts = _split_by_side(bc_coords, dom)
        fluxes = []
        targets = []

        def _append(side: str, n_vec: tuple[float, float]):
            pts = parts[side]
            if pts.numel() == 0:
                return
            x, y, t = _split_coords(pts)
            # select matching indices from the full gradient tensor
            if side == "left":
                sel = torch.isclose(bc_coords[:, 0:1], torch.tensor(dom.x_min, device=bc_coords.device, dtype=bc_coords.dtype), atol=1e-6)
                g = u_x[sel] * n_vec[0] + u_y[sel] * n_vec[1]
            elif side == "right":
                sel = torch.isclose(bc_coords[:, 0:1], torch.tensor(dom.x_max, device=bc_coords.device, dtype=bc_coords.dtype), atol=1e-6)
                g = u_x[sel] * n_vec[0] + u_y[sel] * n_vec[1]
            elif side == "bottom":
                sel = torch.isclose(bc_coords[:, 1:2], torch.tensor(dom.y_min, device=bc_coords.device, dtype=bc_coords.dtype), atol=1e-6)
                g = u_x[sel] * n_vec[0] + u_y[sel] * n_vec[1]
            else:  # "top"
                sel = torch.isclose(bc_coords[:, 1:2], torch.tensor(dom.y_max, device=bc_coords.device, dtype=bc_coords.dtype), atol=1e-6)
                g = u_x[sel] * n_vec[0] + u_y[sel] * n_vec[1]

            fluxes.append(g)
            if neumann_flux is None:
                targets.append(torch.zeros_like(g))
            else:
                targets.append(neumann_flux(x, y, t, u_x[sel], u_y[sel]))

        # outward normals for rectangle
        _append("left", (-1.0, 0.0))
        _append("right", (1.0, 0.0))
        _append("bottom", (0.0, -1.0))
        _append("top", (0.0, 1.0))

        if len(fluxes) == 0:
            loss_b = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            loss_b = torch.cat([(f - t).square() for f, t in zip(fluxes, targets)], dim=0).mean()
    else:
        raise ValueError("boundary must be 'dirichlet' or 'neumann'.")

    # ---------------------------- Initial condition ---------------------------
    # Sample spatial points at t = t_min
    x0 = torch.rand((n_initial, 1), device=device, dtype=dtype) * dom.width + dom.x_min
    y0 = torch.rand((n_initial, 1), device=device, dtype=dtype) * dom.height + dom.y_min
    ic_coords = torch.cat([x0, y0, torch.full_like(x0, dom.t_min)], dim=1)

    if ic is None:
        cx = 0.5 * (dom.x_min + dom.x_max)
        cy = 0.5 * (dom.y_min + dom.y_max)
        target = gaussian_ic(x0, y0, center=(cx, cy), sharpness=50.0)
    else:
        target = ic(x0, y0)

    loss_ic = (model(ic_coords) - target).square().mean()
    return loss_b + loss_ic


# =============================================================================
# Optional: exact Dirichlet satisfaction via an output‑space mask
# =============================================================================

def make_dirichlet_mask(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    pow: int = 1,
) -> Callable[[Tensor], Tensor]:
    """Return a multiplicative mask that vanishes on the spatial boundary.

    For coordinates ``(x,y,t)`` the mask is

        m(x,y,t) = [(x-x_min)(x_max-x) · (y-y_min)(y_max-y)]^pow  clipped at 0.

    Multiplying a base model output by ``m`` **exactly** enforces homogeneous
    Dirichlet boundary conditions for *all* times.  For higher‑order vanishing,
    use ``pow > 1`` to improve smoothness near the boundary.
    """
    def mask(coords: Tensor) -> Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        mx = (x - x_min) * (x_max - x)
        my = (y - y_min) * (y_max - y)
        m = (mx * my).clamp_min(0.0)
        if pow != 1:
            m = m.pow(pow)
        return m

    return mask


class MaskedModel(torch.nn.Module):
    """Thin wrapper that enforces homogeneous Dirichlet BCs exactly.

    Example
    -------
    >>> base = MLP(in_features=3, out_features=1, hidden=(64,64), activation="tanh")
    >>> mask = make_dirichlet_mask(0,1,0,1, pow=2)
    >>> model = MaskedModel(base, mask)
    >>> # model(coords) now always evaluates to zero on the rectangle boundary.
    """
    def __init__(self, base: torch.nn.Module, mask: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.base = base
        self.mask = mask

    def forward(self, coords: Tensor) -> Tensor:  # pragma: no cover - thin wrapper
        return self.mask(coords) * self.base(coords)


__all__ = [
    "residual_heat2d",
    "bc_ic_heat2d",
    # extras
    "SquareDomain2D",
    "gaussian_ic",
    "make_dirichlet_mask",
    "MaskedModel",
]
