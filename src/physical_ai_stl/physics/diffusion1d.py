# ruff: noqa: I001
from __future__ import annotations

"""
Utilities for the 1‑D diffusion (heat) equation

    u_t = α(x,t) · u_xx,

including:
  • analytic sine solution and initializer,
  • collocation residual (supports constant or spatial/temporal α),
  • soft boundary/initial penalties (Dirichlet by default; optional Neumann/Robin),
  • a simple masking wrapper to enforce homogeneous Dirichlet BCs by construction.

All functions are written to be device/dtype‑aware and autograd‑friendly.
No heavy dependencies are introduced here; higher‑level STL hooks live in the
experiments/ and monitoring/ modules.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

# Import used only for type hints; with `from __future__ import annotations`,
# this import can be omitted at runtime, but keeping it is harmless.
from ..models.mlp import MLP

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Autograd helpers
# ---------------------------------------------------------------------------

def _grad(y: Tensor, x: Tensor) -> Tensor:
    """
    Element‑wise gradient ∂y/∂x for batched computations.

    If y has shape (N,1) and x has shape (N,D) with samples independent
    across the batch (as is the case for typical coordinate‑MLPs),
    this returns an (N,D) tensor where row i is ∇_x y_i.
    """
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def _check_coords(coords: Tensor) -> Tensor:
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape (N, 2) with columns [x, t]")
    return coords


# ---------------------------------------------------------------------------
# Core: PDE residual
# ---------------------------------------------------------------------------

def _alpha_field(
    alpha: float | Tensor | Callable[[Tensor], Tensor],
    coords: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """
    Normalize the diffusivity α to a per‑sample column tensor (N,1).

    Accepts:
      • float or 0‑D tensor (broadcast),
      • (N,) or (N,1) tensor,
      • callable alpha(coords) → (N,) or (N,1).
    """
    if callable(alpha):
        a = alpha(coords)
    elif torch.is_tensor(alpha):
        a = alpha
    else:
        a = torch.tensor(alpha, device=coords.device, dtype=dtype)
    if a.ndim == 0:
        a = a.view(1, 1).expand(coords.shape[0], 1)
    elif a.ndim == 1:
        a = a.view(-1, 1)
    elif a.ndim == 2 and a.shape[1] == 1:
        # already (N,1)
        pass
    else:
        raise ValueError("alpha must be scalar, (N,), (N,1) or a callable returning one of these.")
    return a.to(device=coords.device, dtype=dtype)


def pde_residual(
    model: torch.nn.Module,
    coords: Tensor,
    alpha: float | Tensor | Callable[[Tensor], Tensor] = 0.1,
) -> Tensor:
    """
    Return the collocation residual r(x,t) = u_t − α(x,t) u_xx evaluated at ``coords``.

    Parameters
    ----------
    model:
        Neural field mapping coordinates → solution value, i.e. model([x,t]) = u(x,t).
        Must accept a tensor of shape (N,2) and return (N,1).
    coords:
        Tensor of shape (N,2) with columns [x, t].  ``requires_grad`` will be enabled
        as needed to compute the derivatives.
    alpha:
        Diffusivity.  Either a scalar (float / 0‑D tensor), a per‑sample tensor
        of shape (N,) or (N,1), or a callable ``alpha(coords)`` returning one of
        those. Defaults to 0.1.

    Returns
    -------
    Tensor
        Residual values with shape (N,1).
    """
    coords = _check_coords(coords).requires_grad_(True)

    # Forward
    u: Tensor = model(coords)  # (N,1)

    # First derivatives
    du = _grad(u, coords)          # (N,2) -> [u_x, u_t]
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]

    # Second derivative w.r.t. x
    u_xx = _grad(u_x, coords)[:, 0:1]

    # Diffusivity field (N,1)
    a = _alpha_field(alpha, coords, dtype=u.dtype)

    return u_t - a * u_xx


def residual_loss(
    model: torch.nn.Module,
    coords: Tensor,
    alpha: float | Tensor | Callable[[Tensor], Tensor] = 0.1,
    reduction: str = "mean",
) -> Tensor:
    """
    Mean‑squared (or sum/none) loss of the PDE residual over collocation points.
    """
    r = pde_residual(model, coords, alpha)
    sq = r.square()
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


# ---------------------------------------------------------------------------
# Boundary/Initial conditions
# ---------------------------------------------------------------------------

def _unit_samples(
    n: int,
    d: int,
    *,
    method: str,
    device: torch.device | str,
    dtype: torch.dtype | None,
    seed: int | None = None,
) -> Tensor:
    """
    Sample ``n`` points in the unit hypercube [0,1]^d using either Sobol or uniform RNG.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if n <= 0:
        return torch.empty(0, d, device=device, dtype=dtype)
    if method == "sobol":
        # SobolEngine draws on CPU; move after cast.
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        u = engine.draw(n).to(dtype=dtype, device="cpu")
        return u.to(device=device)
    if method == "uniform":
        return torch.rand(n, d, device=device, dtype=dtype)
    raise ValueError("method must be 'sobol' or 'uniform'")


def _as_tensor(x: float | Tensor, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    return x if torch.is_tensor(x) else torch.tensor(x, device=device, dtype=dtype)


def sine_ic(
    x: Tensor,
    *,
    x_left: float = 0.0,
    x_right: float = 1.0,
    amplitude: float | Tensor = 1.0,
) -> Tensor:
    """
    Sine initial condition respecting homogeneous Dirichlet boundaries:
        u(x,0) = A sin(π·(x − x_left)/L),   L = x_right − x_left.
    """
    L = (x_right - x_left)
    k = torch.pi / _as_tensor(L, device=x.device, dtype=x.dtype)
    A = _as_tensor(amplitude, device=x.device, dtype=x.dtype)
    return A * torch.sin(k * (x - x_left))


def bc_ic_targets(
    x: Tensor,
    t: Tensor,
    *,
    x_left: float,
    x_right: float,
    bc_left: float | Callable[[Tensor], Tensor] = 0.0,
    bc_right: float | Callable[[Tensor], Tensor] = 0.0,
    ic: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute target values for boundary/initial conditions.

    Returns
    -------
    (u_L(t), u_R(t), u_0(x))
    """
    if ic is None:
        u0 = sine_ic(x, x_left=x_left, x_right=x_right)
    else:
        u0 = ic(x)
    if callable(bc_left):
        uL = bc_left(t)
    else:
        uL = torch.full_like(t, fill_value=float(bc_left))
    if callable(bc_right):
        uR = bc_right(t)
    else:
        uR = torch.full_like(t, fill_value=float(bc_right))
    return uL, uR, u0


def boundary_loss(
    model: MLP | torch.nn.Module,
    x_left: float = 0.0,
    x_right: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device | str = "cpu",
    n_boundary: int = 256,
    n_initial: int = 512,
    *,  # keyword‑only options (backwards‑compatible defaults)
    dtype: torch.dtype | None = None,
    method: str = "sobol",
    seed: int | None = None,
    bc_left: float | Callable[[Tensor], Tensor] = 0.0,
    bc_right: float | Callable[[Tensor], Tensor] = 0.0,
    ic: Callable[[Tensor], Tensor] | None = None,
    w_boundary: float = 1.0,
    w_initial: float = 1.0,
    # --- new, optional generalizations ---
    bc_left_type: str = "dirichlet",     # "dirichlet" | "neumann" | "robin"
    bc_right_type: str = "dirichlet",
    robin_left: tuple[float, float] | None = None,   # (a,b):  a·u + b·u_x = g_L(t)
    robin_right: tuple[float, float] | None = None,  # (a,b):  a·u + b·u_x = g_R(t)
) -> Tensor:
    """
    Soft penalty for boundary and initial conditions.

    Dirichlet is the default (compatible with prior versions):
        u(x_left,t)  = bc_left(t),
        u(x_right,t) = bc_right(t),
        u(x, t_min)  = ic(x)  [defaults to sine_ic].

    If ``bc_*_type == "neumann"``, then bc_* denotes the desired flux:
        u_x(boundary, t) = bc_*(t).

    If ``bc_*_type == "robin"``, supply Robin coefficients via ``robin_* = (a,b)``
    and set bc_* to the target function g(t):
        a·u(boundary,t) + b·u_x(boundary,t) = g(t).

    All targets can be floats (constants) or callables of time ``t → (N,1)``.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    # ---- Boundary samples (split across left/right) ----
    if n_boundary > 0:
        u = _unit_samples(n_boundary, 1, method=method, device=device, dtype=dtype, seed=seed)
        t = t_min + u * (t_max - t_min)               # (Nb,1)
        xL = torch.full_like(t, fill_value=x_left)
        xR = torch.full_like(t, fill_value=x_right)
        left = torch.cat([xL, t], dim=1)              # (Nb,2)
        right = torch.cat([xR, t], dim=1)             # (Nb,2)

        # Enable gradients on coords if we need boundary derivatives.
        need_grad = (bc_left_type != "dirichlet") or (bc_right_type != "dirichlet")
        if need_grad:
            left = left.requires_grad_(True)
            right = right.requires_grad_(True)

        # Targets
        target_L, target_R, _ = bc_ic_targets(
            x=torch.empty(0, 1, device=device, dtype=dtype),  # unused for BCs
            t=t, x_left=x_left, x_right=x_right,
            bc_left=bc_left, bc_right=bc_right, ic=None
        )

        # Predictions
        pred_L = model(left)
        pred_R = model(right)

        def _bc_residual(
            side: str,
            bc_type: str,
            coords: Tensor,
            pred: Tensor,
            target: Tensor,
            robin_ab: tuple[float, float] | None,
        ) -> Tensor:
            if bc_type == "dirichlet":
                return pred - target
            if bc_type == "neumann":
                # one‑sided normal derivative equals interior x-derivative here
                dud = _grad(pred, coords)[:, 0:1]  # u_x
                return dud - target
            if bc_type == "robin":
                a, b = (robin_ab if robin_ab is not None else (1.0, 1.0))
                a_t = _as_tensor(a, device=coords.device, dtype=pred.dtype)
                b_t = _as_tensor(b, device=coords.device, dtype=pred.dtype)
                dud = _grad(pred, coords)[:, 0:1]  # u_x
                return a_t * pred + b_t * dud - target
            raise ValueError(f"Unknown BC type {bc_type!r} for {side} boundary.")

        rL = _bc_residual("left",  bc_left_type.lower(),  left,  pred_L, target_L, robin_left)
        rR = _bc_residual("right", bc_right_type.lower(), right, pred_R, target_R, robin_right)
        loss_bc = rL.square().mean() + rR.square().mean()
    else:
        loss_bc = torch.zeros((), device=device, dtype=dtype)

    # ---- Initial condition samples at t = t_min ----
    if n_initial > 0:
        u = _unit_samples(
            n_initial, 1, method=method, device=device, dtype=dtype,
            seed=None if seed is None else seed + 7
        )
        x = x_left + u * (x_right - x_left)           # (Ni,1)
        ic_coords = torch.cat([x, torch.full_like(x, fill_value=t_min)], dim=1)
        _, _, target_ic = bc_ic_targets(
            x=x, t=torch.zeros_like(x), x_left=x_left, x_right=x_right,
            bc_left=bc_left, bc_right=bc_right, ic=ic
        )
        loss_ic = (model(ic_coords) - target_ic).square().mean()
    else:
        loss_ic = torch.zeros((), device=device, dtype=dtype)

    return _as_tensor(w_boundary, device=device, dtype=dtype) * loss_bc \
         + _as_tensor(w_initial, device=device, dtype=dtype) * loss_ic


# ---------------------------------------------------------------------------
# Extras (optional but useful)
# ---------------------------------------------------------------------------

@dataclass
class Interval1D:
    """
    Convenience container for the rectangular space‑time domain.
    """
    x_left: float = 0.0
    x_right: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    @property
    def length(self) -> float:
        return float(self.x_right - self.x_left)


def sine_solution(
    x: Tensor,
    t: Tensor,
    alpha: float | Tensor | Callable[[Tensor], Tensor] = 0.1,
    *,
    x_left: float = 0.0,
    x_right: float = 1.0,
    amplitude: float | Tensor = 1.0,
) -> Tensor:
    """
    Analytic solution for the homogeneous Dirichlet / sine initial condition case:
        u(x,t) = A · exp(−α k² t) · sin(k (x − x_left)),
    with k = π / (x_right − x_left).
    """
    L = (x_right - x_left)
    k = torch.pi / _as_tensor(L, device=x.device, dtype=x.dtype)  # spatial wavenumber
    A = _as_tensor(amplitude, device=x.device, dtype=x.dtype)
    if callable(alpha):
        a = _alpha_field(alpha, torch.stack([x, t], dim=1), dtype=x.dtype)  # (N,1)
        # If α varies in time/space, this closed form is no longer exact;
        # we use the local α for an informative baseline (explicitly documented).
        a = a
    else:
        a = _as_tensor(alpha, device=x.device, dtype=x.dtype)
    return A * torch.exp(-a * (k ** 2) * t) * torch.sin(k * (x - x_left))


def make_dirichlet_mask_1d(x_left: float = 0.0, x_right: float = 1.0) -> Callable[[Tensor], Tensor]:
    """
    Return a smooth mask m(x,t) = (x−x_left)·(x_right−x) that vanishes at the boundaries.

    Using ``û(x,t) = m(x,t) · v(x,t)`` ensures û satisfies homogeneous Dirichlet BCs.
    """
    def mask(coords: Tensor) -> Tensor:
        coords = _check_coords(coords)
        x = coords[:, 0:1]
        return (x - x_left) * (x_right - x)
    return mask


class MaskedModel(torch.nn.Module):
    """
    Thin wrapper that enforces homogeneous Dirichlet BCs by construction:

        û(x,t) = m(x,t) · base([x,t])

    where m is a mask from ``make_dirichlet_mask_1d``.
    """
    def __init__(self, base: torch.nn.Module, mask: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.base = base
        self.mask = mask

    def forward(self, coords: Tensor) -> Tensor:  # pragma: no cover - thin wrapper
        return self.mask(coords) * self.base(coords)


__all__ = [
    # core
    "pde_residual", "residual_loss", "boundary_loss",
    # extras
    "Interval1D", "sine_ic", "sine_solution", "make_dirichlet_mask_1d", "MaskedModel",
]
