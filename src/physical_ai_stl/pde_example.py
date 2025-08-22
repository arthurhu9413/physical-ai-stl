from __future__ import annotations

"""
Minimal 1‑D diffusion (heat) equation simulator + STL-style robustness utilities.

This module is intentionally lightweight and dependency‑free (NumPy only) so it can
be used as a tiny sand‑box for spatio‑temporal logic (STL / STREL‑like) monitoring
over PDE trajectories.  The defaults are chosen for clarity and numerical safety.

What’s inside
-------------
- `simulate_diffusion`: explicit FTCS(= forward‑Euler in time, centered‑difference
  in space) solver for the 1‑D heat equation with *copy‑Neumann* boundaries.
- `simulate_diffusion_with_clipping`: same, but constrains every frame to
  `lower ≤ u ≤ upper` (a common “safety envelope” post‑processing).
- Robustness helpers (`compute_robustness`, `compute_spatiotemporal_robustness`)
  implementing the min‑margin semantics used in STL monitoring
  (Donzé & Maler 2010).
- Efficient sliding‑window primitives for *Globally* (G) and *Eventually* (F)
  operators in time and space using a monotone deque (amortized O(1) per step).
- Convenience monitors for *rectangular* spatio‑temporal scopes, e.g.
      ρ_{G_t G_x (lower ≤ u ≤ upper)}
  which is often the first STL specification one writes for PDE demos.

Numerics
--------
The explicit FTCS scheme applied to diffusion is conditionally stable with the
Courant number r = α Δt / Δx² ≤ 1/2 in 1‑D.  We warn if you choose parameters
that violate this criterion to help avoid exploding solutions.  (See e.g.
Recktenwald, “FTCS Solution to the Heat Equation”, and the FTCS overview.)
"""

import warnings
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Utilities: validation and CFL / stability guidance
# ---------------------------------------------------------------------------


def _validate_length_steps(length: int, steps: int) -> None:
    if length <= 0:
        raise ValueError("length must be positive")
    if steps < 0:
        raise ValueError("steps must be non‑negative")


def cfl_number(alpha: float, dt: float, dx: float) -> float:
    """
    Return the explicit diffusion Courant number r = alpha * dt / dx^2.
    """
    return float(alpha) * float(dt) / float(dx) ** 2


def _warn_if_unstable(alpha: float, dt: float, dx: float) -> None:
    r = cfl_number(alpha, dt, dx)
    if r > 0.5 + 1e-15:  # explicit FTCS stability in 1‑D heat eq.
        warnings.warn(
            f"Explicit diffusion step likely unstable: r=alpha*dt/dx^2={r:.3g} > 0.5; "
            "consider decreasing dt or increasing dx (or use an implicit scheme).",
            RuntimeWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# PDE simulators
# ---------------------------------------------------------------------------


def simulate_diffusion(
    length: int = 5,
    steps: int = 20,
    dt: float = 0.1,
    alpha: float = 0.1,
    initial: np.ndarray | None = None,
    *,
    dx: float = 1.0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """
    Simulate the 1‑D diffusion equation u_t = α u_xx on a uniform grid.

    Scheme
    ------
    FTCS (forward Euler in time, centered differences in space), with
    *copy‑Neumann* boundaries: we reflect the edge value each step to enforce
    ∂u/∂x ≈ 0.  This is a simple and common discrete realization of insulating/
    reflecting boundaries.

    Parameters
    ----------
    length : int
        Number of spatial cells (≥ 1).
    steps : int
        Number of time steps (≥ 0).  The result contains `steps+1` frames
        including the initial condition.
    dt : float
        Time step Δt.
    alpha : float
        Diffusivity α.
    initial : (length,) array or None
        Initial condition. If None, a unit “hot spot” is placed at the left end.
    dx : float, keyword‑only
        Grid spacing Δx.
    dtype : numpy dtype, keyword‑only
        Floating dtype for the state.

    Returns
    -------
    u : (steps+1, length) ndarray[dtype]
        Trajectory, one row per time step.

    Notes
    -----
    The explicit FTCS method for 1‑D diffusion is conditionally stable for
    r = α Δt / Δx² ≤ 1/2; we emit a warning if r > 1/2.
    """
    _validate_length_steps(length, steps)
    _warn_if_unstable(alpha, dt, dx)

    r = cfl_number(alpha, dt, dx)
    u = np.zeros((steps + 1, length), dtype=dtype)

    if initial is not None:
        init = np.asarray(initial, dtype=dtype)
        if init.shape != (length,):
            raise ValueError("initial must have shape (length,)")
        u[0] = init
    else:
        # simple default: single hot spot at the left boundary
        u[0, 0] = dtype(1.0)

    if steps == 0:
        return u

    if length == 1:
        # Degenerate domain: nothing to diffuse; copy the single value forward.
        # (This keeps semantics consistent for edge cases.)
        for n in range(steps):
            u[n + 1, 0] = u[n, 0]
        return u

    # Vectorized interior update
    for n in range(steps):
        cur = u[n]
        nxt = u[n + 1]

        # u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 0:-2] - 2*u[n, 1:-1] + u[n, 2:])
        nxt[1:-1] = cur[1:-1] + r * (cur[:-2] - 2.0 * cur[1:-1] + cur[2:])

        # copy‑Neumann boundaries (enforce ∂u/∂x = 0 at n+1 by reflection)
        nxt[0] = nxt[1]
        nxt[-1] = nxt[-2]

    return u


def simulate_diffusion_with_clipping(
    length: int = 5,
    steps: int = 20,
    dt: float = 0.1,
    alpha: float = 0.1,
    lower: float = 0.0,
    upper: float = 1.0,
    initial: np.ndarray | None = None,
    *,
    dx: float = 1.0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """
    Same as `simulate_diffusion`, but clip every frame into [lower, upper].

    This can be used to emulate a simple STL “safety envelope” where the
    predicate is `lower ≤ u ≤ upper`.  Clipping is done *in place* to avoid
    allocations.
    """
    if lower > upper:
        raise ValueError("lower must be <= upper")

    u0 = simulate_diffusion(length, 0, dt, alpha, initial, dx=dx, dtype=dtype)
    # Clip the very first frame
    np.clip(u0[0], lower, upper, out=u0[0])

    if steps == 0:
        return u0

    # Efficient step‑by‑step update with clipping (no extra simulator calls)
    r = cfl_number(alpha, dt, dx)
    out = np.zeros((steps + 1, length), dtype=dtype)
    out[0] = u0[0]

    if length == 1:
        for n in range(steps):
            out[n + 1, 0] = out[n, 0]
        return out

    for n in range(steps):
        cur = out[n]
        nxt = out[n + 1]
        nxt[1:-1] = cur[1:-1] + r * (cur[:-2] - 2.0 * cur[1:-1] + cur[2:])
        # copy‑Neumann
        nxt[0] = nxt[1]
        nxt[-1] = nxt[-2]
        # clip in‑place
        np.clip(nxt, lower, upper, out=nxt)

    return out


# ---------------------------------------------------------------------------
# Robust semantics for simple STL bounds and sliding‑window operators
# ---------------------------------------------------------------------------


def compute_robustness(signal: np.ndarray, lower: float, upper: float) -> float:
    """
    Scalar robustness for a 1‑D signal w.r.t. the predicate lower ≤ x ≤ upper.

    Defined as min_i min(x_i − lower, upper − x_i).
    """
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1D")
    if sig.size == 0:
        raise ValueError("signal must not be empty")
    margins = np.minimum(sig - lower, upper - sig)
    return float(margins.min())


def compute_spatiotemporal_robustness(
    signal_matrix: np.ndarray, lower: float, upper: float
) -> float:
    """
    Scalar robustness for a 2‑D matrix (time × space) w.r.t. lower ≤ x ≤ upper.

    Defined as min_{t,x} min(u[t,x] − lower, upper − u[t,x]).
    """
    mat = np.asarray(signal_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("signal_matrix must be two‑dimensional")
    if mat.size == 0:
        raise ValueError("signal_matrix must not be empty")
    margins = np.minimum(mat - lower, upper - mat)
    return float(margins.min())


# --- Optional helpers (kept tiny; handy in notebooks/tests) -----------------


def pointwise_bounds_margin(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Pointwise margin ρ(x) = min(x − lower, upper − x).

    Positive ⇒ inside the band; negative ⇒ violation magnitude.
    """
    arr = np.asarray(x, dtype=float)
    return np.minimum(arr - lower, upper - arr)


def _sliding_extreme(x: np.ndarray, window: int, extreme: Literal["min", "max"]) -> np.ndarray:
    """
    Monotone‑deque based sliding min/max over a *1‑D* array.

    The `i`‑th output contains the extreme over the trailing window
    `[max(0, i‑window+1), i]`.  This matches standard on‑line robustness
    implementations for bounded‑future STL fragments.

    Complexity: amortized O(1) per element.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    x = np.asarray(x, dtype=float)
    from collections import deque

    is_min = (extreme == "min")
    cmp = (lambda a, b: a <= b) if is_min else (lambda a, b: a >= b)

    dq: "deque[tuple[int, float]]" = deque()
    out = np.empty_like(x, dtype=float)
    for i, val in enumerate(x):
        # pop dominated values from the right
        while dq and cmp(val, dq[-1][1]):
            dq.pop()
        dq.append((i, val))
        # drop from the left if window exceeded
        left = i - window + 1
        while dq and dq[0][0] < left:
            dq.popleft()
        out[i] = dq[0][1]
    return out


def stl_globally_robustness(rho_phi: np.ndarray, window: int) -> np.ndarray:
    """Temporal G‑operator robustness over a trailing window (1‑D)."""
    return _sliding_extreme(rho_phi, window, "min")


def stl_eventually_robustness(rho_phi: np.ndarray, window: int) -> np.ndarray:
    """Temporal F‑operator robustness over a trailing window (1‑D)."""
    return _sliding_extreme(rho_phi, window, "max")


# --- Spatial operators (1‑D neighborhoods) ----------------------------------


def _sliding_extreme_along_axis_1d(
    mat: np.ndarray, window: int, axis: int, extreme: Literal["min", "max"]
) -> np.ndarray:
    """
    Apply 1‑D sliding min/max along `axis` of a 2‑D array using a monotone deque.

    This provides the robust semantics for spatial *Globally* / *Eventually*
    over a 1‑D neighborhood of width `window`.  The window is trailing in the
    chosen axis to mirror the temporal variant (use a flipped view to obtain
    symmetric windows if needed).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if mat.ndim != 2:
        raise ValueError("mat must be 2‑D")
    mat = np.asarray(mat, dtype=float)
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    if axis == 1:
        # operate row‑wise
        out = np.empty_like(mat, dtype=float)
        for i in range(mat.shape[0]):
            out[i] = _sliding_extreme(mat[i], window, extreme)
        return out
    else:
        # operate column‑wise (apply to transposed view)
        return _sliding_extreme_along_axis_1d(mat.T, window, 1, extreme).T


def stl_spatial_globally_robustness(rho_phi_xt: np.ndarray, window: int) -> np.ndarray:
    """
    Spatial G‑operator robustness (1‑D neighborhoods) applied to each time row.

    Equivalent to morphological erosion with a flat structuring element of size
    `window`.  The operation is *trailing* in space, consistent with the temporal
    helpers in this module.
    """
    return _sliding_extreme_along_axis_1d(rho_phi_xt, window, axis=1, extreme="min")


def stl_spatial_eventually_robustness(rho_phi_xt: np.ndarray, window: int) -> np.ndarray:
    """Spatial F‑operator robustness (1‑D neighborhoods) applied to each time row."""
    return _sliding_extreme_along_axis_1d(rho_phi_xt, window, axis=1, extreme="max")


# --- Spatio‑temporal rectangles (compose G/F across time×space) -------------


def stl_rect_globally_bounds(
    u_xt: np.ndarray,
    lower: float,
    upper: float,
    t_window: int,
    x_window: int,
) -> np.ndarray:
    """
    Robustness of  G_[t_window] G_[x_window]  (lower ≤ u ≤ upper).

    Parameters
    ----------
    u_xt : (T, X) array
        Trajectory matrix (time × space).
    lower, upper : floats
        Bounds of the safety band.
    t_window, x_window : int
        Trailing window sizes in time and space.

    Returns
    -------
    rho : (T, X) array
        Robustness of the rectangular G/G property at each (t, x).

    Notes
    -----
    Because both operators use min, the order of applying time/space windows
    does not matter (min over a rectangle).  We apply space first then time.
    """
    rho = pointwise_bounds_margin(u_xt, lower, upper)
    rho = stl_spatial_globally_robustness(rho, x_window)
    rho = _sliding_extreme_along_axis_1d(rho, t_window, axis=0, extreme="min")
    return rho


def stl_rect_eventually_bounds(
    u_xt: np.ndarray,
    lower: float,
    upper: float,
    t_window: int,
    x_window: int,
) -> np.ndarray:
    """
    Robustness of  F_[t_window] F_[x_window]  (lower ≤ u ≤ upper).

    Uses max in both dimensions (apply space then time).
    """
    rho = pointwise_bounds_margin(u_xt, lower, upper)
    rho = stl_spatial_eventually_robustness(rho, x_window)
    rho = _sliding_extreme_along_axis_1d(rho, t_window, axis=0, extreme="max")
    return rho


__all__ = [
    # simulators
    "simulate_diffusion",
    "simulate_diffusion_with_clipping",
    # scalar robustness
    "compute_robustness",
    "compute_spatiotemporal_robustness",
    # simple helpers
    "pointwise_bounds_margin",
    "stl_globally_robustness",
    "stl_eventually_robustness",
    # spatial + rectangular spatio‑temporal helpers
    "stl_spatial_globally_robustness",
    "stl_spatial_eventually_robustness",
    "stl_rect_globally_bounds",
    "stl_rect_eventually_bounds",
    # guidance
    "cfl_number",
]
