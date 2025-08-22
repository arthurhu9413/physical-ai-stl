from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

Tensor = torch.Tensor


# ===========================================================================
# Utilities
# ===========================================================================

def _as_tensor(x: float | Tensor, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    """Return ``x`` as a 0‑D tensor on ``device``/``dtype`` (no copy if already a Tensor)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _linspace(min_v: float, max_v: float, n: int, *, device, dtype) -> Tensor:
    """Wrapper around :func:`torch.linspace` with validation."""
    if n <= 0:
        raise ValueError("n must be positive.")
    # NOTE: torch.linspace is inclusive by default; we rely on this.
    return torch.linspace(min_v, max_v, n, device=device, dtype=dtype)


def _stack_flat(*meshes: Tensor) -> Tensor:
    """Flatten grid *meshes* (broadcasted via meshgrid) to an (N,d) table."""
    if not meshes:
        raise ValueError("No meshes provided.")
    flat_cols = [m.reshape(-1) for m in meshes]
    return torch.stack(flat_cols, dim=-1)


# ===========================================================================
# Public API – Regular grids
# ===========================================================================

def grid1d(
    n_x: int = 128,
    n_t: int = 100,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Construct a 1‑D space × time grid.

    Returns:
        ``(X, T, XT)`` where ``X, T`` are of shape ``(n_x, n_t)`` and ``XT`` is
        an ``(n_x*n_t, 2)`` table of coordinates. If ``return_cartesian`` is True,
        ``XT`` is built with :func:`torch.cartesian_prod`, otherwise it is the
        flattened stack of the mesh tensors (identical values, different memory layout).
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)
    X, T = torch.meshgrid(x, t, indexing="ij")
    if return_cartesian:
        XT = torch.cartesian_prod(x, t)
    else:
        XT = _stack_flat(X, T)
    return X, T, XT


def grid2d(
    n_x: int = 64,
    n_y: int = 64,
    n_t: int = 50,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Construct a 2‑D (x,y) × time grid.

    Returns:
        ``(X, Y, T, XYT)`` where ``X, Y, T`` have shape ``(n_x, n_y, n_t)`` and
        ``XYT`` is an ``(n_x*n_y*n_t, 3)`` table.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    y = _linspace(y_min, y_max, n_y, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)
    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    if return_cartesian:
        XYT = torch.cartesian_prod(x, y, t)
    else:
        XYT = _stack_flat(X, Y, T)
    return X, Y, T, XYT


def grid3d(
    n_x: int = 32,
    n_y: int = 32,
    n_z: int = 32,
    n_t: int = 20,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Construct a 3‑D (x,y,z) × time grid.

    Returns:
        ``(X, Y, Z, T, XYZT)`` where ``X, Y, Z, T`` have shape ``(n_x, n_y, n_z, n_t)``
        and ``XYZT`` is an ``(n_x*n_y*n_z*n_t, 4)`` table.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    y = _linspace(y_min, y_max, n_y, device=device, dtype=dtype)
    z = _linspace(z_min, z_max, n_z, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)
    X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing="ij")
    if return_cartesian:
        XYZT = torch.cartesian_prod(x, y, z, t)
    else:
        XYZT = _stack_flat(X, Y, Z, T)
    return X, Y, Z, T, XYZT


# ===========================================================================
# Spacing
# ===========================================================================

def spacing1d(
    n_x: int,
    n_t: int,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Return grid spacings ``(dx, dt)`` for a 1‑D (x,t) tensor grid."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    dx = _as_tensor((x_max - x_min) / (max(n_x - 1, 1)), device=device, dtype=dtype)
    dt = _as_tensor((t_max - t_min) / (max(n_t - 1, 1)), device=device, dtype=dtype)
    return dx, dt


def spacing2d(
    n_x: int,
    n_y: int,
    n_t: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    t_min: float,
    t_max: float,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return grid spacings ``(dx, dy, dt)`` for a 2‑D (x,y,t) tensor grid."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    dx = _as_tensor((x_max - x_min) / (max(n_x - 1, 1)), device=device, dtype=dtype)
    dy = _as_tensor((y_max - y_min) / (max(n_y - 1, 1)), device=device, dtype=dtype)
    dt = _as_tensor((t_max - t_min) / (max(n_t - 1, 1)), device=device, dtype=dtype)
    return dx, dy, dt


def spacing3d(
    n_x: int,
    n_y: int,
    n_z: int,
    n_t: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return grid spacings ``(dx, dy, dz, dt)`` for a 3‑D (x,y,z,t) tensor grid."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    dx = _as_tensor((x_max - x_min) / (max(n_x - 1, 1)), device=device, dtype=dtype)
    dy = _as_tensor((y_max - y_min) / (max(n_y - 1, 1)), device=device, dtype=dtype)
    dz = _as_tensor((z_max - z_min) / (max(n_z - 1, 1)), device=device, dtype=dtype)
    dt = _as_tensor((t_max - t_min) / (max(n_t - 1, 1)), device=device, dtype=dtype)
    return dx, dy, dz, dt


# ===========================================================================
# Random / quasi-random samplers
# ===========================================================================

_SampleMethod = Literal["sobol", "uniform", "rand", "random", "lhs"]

def _lhs_unit_samples(num: int, dim: int, *, seed: int | None) -> Tensor:
    """Simple Latin Hypercube samples in [0,1)^dim as a CPU tensor.

    For each dimension independently, draws one sample in each of ``num``
    equiprobable strata and randomly permutes their order.  This implementation
    is *vanilla* LHS (no space-filling optimization).
    """
    if num <= 0 or dim < 0:
        raise ValueError("num must be > 0 and dim must be >= 0.")
    if dim == 0:
        return torch.empty(num, 0)
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    eps = torch.rand(num, dim, generator=g)  # CPU
    cols = []
    for j in range(dim):
        perm = torch.randperm(num, generator=g).to(dtype=torch.float32)
        cols.append((perm + eps[:, j]) / float(num))
    return torch.stack(cols, dim=1)


def _unit_samples(
    num: int,
    dim: int,
    *,
    method: _SampleMethod,
    device,
    dtype,
    seed: int | None,
) -> Tensor:
    """Samples in the unit hypercube ``[0,1)^dim`` using the given *method*.

    Supported ``method`` values:
        - ``"sobol"``: low-discrepancy Sobol sequence (scrambled if a seed is given).
        - ``"uniform"`` / ``"rand"`` / ``"random"``: i.i.d. uniform.
        - ``"lhs"``: Latin Hypercube sampling (basic stratified design).
    """
    m = method.lower()
    if m not in {"sobol", "uniform", "rand", "random", "lhs"}:
        raise ValueError("method must be one of {'sobol','uniform','rand','random','lhs'}")
    if m == "sobol" and dim > 0:
        # SobolEngine requires a concrete integer seed; default to 0 when None.
        sobol_seed = 0 if seed is None else int(seed)
        engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=sobol_seed)
        u = engine.draw(num)  # CPU tensor in [0,1)
    elif m in {"uniform", "rand", "random"}:
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            u = torch.rand(num, dim, generator=gen)  # CPU
        else:
            u = torch.rand(num, dim)  # CPU
    else:  # LHS
        u = _lhs_unit_samples(num, dim, seed=seed)  # CPU
    return u.to(device=device, dtype=dtype)


# ---- interior ---------------------------------------------------------------

def sample_interior_1d(
    n: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    """Sample ``n`` interior points uniformly/quasi-uniformly in a 1‑D (x,t) box.

    Returns an ``(n, 2)`` tensor of ``[x, t]`` pairs.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    u = _unit_samples(n, 2, method=method, device=device, dtype=dtype, seed=seed)
    mins = torch.tensor([x_min, t_min], device=device, dtype=dtype)
    maxs = torch.tensor([x_max, t_max], device=device, dtype=dtype)
    return mins + u * (maxs - mins)


def sample_interior_2d(
    n: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    """Sample ``n`` interior points in a 2‑D (x,y,t) box.  Returns shape ``(n,3)``."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    u = _unit_samples(n, 3, method=method, device=device, dtype=dtype, seed=seed)
    mins = torch.tensor([x_min, y_min, t_min], device=device, dtype=dtype)
    maxs = torch.tensor([x_max, y_max, t_max], device=device, dtype=dtype)
    return mins + u * (maxs - mins)


def sample_interior_3d(
    n: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    """Sample ``n`` interior points in a 3‑D (x,y,z,t) box.  Returns shape ``(n,4)``."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    u = _unit_samples(n, 4, method=method, device=device, dtype=dtype, seed=seed)
    mins = torch.tensor([x_min, y_min, z_min, t_min], device=device, dtype=dtype)
    maxs = torch.tensor([x_max, y_max, z_max, t_max], device=device, dtype=dtype)
    return mins + u * (maxs - mins)


# ---- boundary ---------------------------------------------------------------

def _largest_remainder_counts(n_total: int, weights: Sequence[float]) -> list[int]:
    """Return integer counts that sum to ``n_total`` using the *largest remainder* rule."""
    if n_total < 0:
        raise ValueError("n_total must be non‑negative")
    if len(weights) == 0:
        return []
    w = torch.tensor(weights, dtype=torch.float64)
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be non‑negative finite numbers")
    s = float(w.sum().item())
    if s <= 0.0:
        raise ValueError("sum of weights must be positive")
    w = w / s
    exact = w * float(n_total)
    base = exact.floor().to(dtype=torch.int64)
    counts = base.tolist()
    # Distribute remaining points to faces with largest fractional remainder.
    rem = int(n_total - int(base.sum().item()))
    if rem > 0:
        frac = (exact - base).tolist()
        order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
        for i in range(rem):
            counts[order[i % len(order)]] += 1
    return counts


def sample_boundary_1d(
    n_total: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    """Sample boundary points on the 1‑D spatial boundary × time faces.

    Splits ``n_total`` between the left (x=x_min) and right (x=x_max) faces.
    Returns an ``(n_total, 2)`` tensor of ``[x, t]`` pairs.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    n_left = n_total // 2
    n_right = n_total - n_left
    # low-discrepancy in time along each face (1‑D)
    u_t_left = _unit_samples(n_left, 1, method=method, device=device, dtype=dtype, seed=seed)
    u_t_right = _unit_samples(n_right, 1, method=method, device=device, dtype=dtype,
                              seed=None if seed is None else seed + 1)
    t_left = t_min + u_t_left[:, 0:1] * (t_max - t_min)
    t_right = t_min + u_t_right[:, 0:1] * (t_max - t_min)
    left = torch.cat([torch.full_like(t_left, fill_value=x_min), t_left], dim=1)
    right = torch.cat([torch.full_like(t_right, fill_value=x_max), t_right], dim=1)
    return torch.cat([left, right], dim=0)


def sample_boundary_2d(
    n_total: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
    split: Sequence[float] | None = None,
) -> Tensor:
    """Sample boundary points on the rectangular 2‑D spatial boundary × time.

    Faces (in order): left ``x=x_min``, right ``x=x_max``, bottom ``y=y_min``,
    top ``y=y_max``.  ``split`` distributes ``n_total`` across these four faces.

    Args:
        n_total: total number of samples over all faces.
        method: 'sobol' | 'uniform'/'rand'/'random' | 'lhs'
        split: optional 4‑tuple of non‑negative weights.  If omitted, uses equal weights.
    Returns:
        Tensor of shape ``(n_total, 3)`` with rows ``[x, y, t]``.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if split is None:
        split = (0.25, 0.25, 0.25, 0.25)
    counts = _largest_remainder_counts(n_total, split)

    # Sample times and tangential coordinates independently on each face.
    def _sample_time(n: int, seed_shift: int = 0) -> Tensor:
        if n == 0:
            return torch.empty(0, 1, device=device, dtype=dtype)
        u = _unit_samples(n, 1, method=method, device=device, dtype=dtype,
                          seed=None if seed is None else seed + seed_shift)
        return t_min + u[:, 0:1] * (t_max - t_min)

    def _sample_coord(n: int, lo: float, hi: float, seed_shift: int) -> Tensor:
        if n == 0:
            return torch.empty(0, 1, device=device, dtype=dtype)
        u = _unit_samples(n, 1, method=method, device=device, dtype=dtype,
                          seed=None if seed is None else seed + seed_shift)
        return lo + u[:, 0:1] * (hi - lo)

    # Faces: left (x=x_min), right (x=x_max), bottom (y=y_min), top (y=y_max)
    t_left = _sample_time(counts[0], 0)
    y_left = _sample_coord(counts[0], y_min, y_max, 10)
    t_right = _sample_time(counts[1], 1)
    y_right = _sample_coord(counts[1], y_min, y_max, 11)
    t_bottom = _sample_time(counts[2], 2)
    x_bottom = _sample_coord(counts[2], x_min, x_max, 12)
    t_top = _sample_time(counts[3], 3)
    x_top = _sample_coord(counts[3], x_min, x_max, 13)

    left = torch.cat([torch.full_like(t_left, x_min), y_left, t_left], dim=1)
    right = torch.cat([torch.full_like(t_right, x_max), y_right, t_right], dim=1)
    bottom = torch.cat([x_bottom, torch.full_like(t_bottom, y_min), t_bottom], dim=1)
    top = torch.cat([x_top, torch.full_like(t_top, y_max), t_top], dim=1)
    return torch.cat([left, right, bottom, top], dim=0)


def sample_boundary_3d(
    n_total: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    method: _SampleMethod = "sobol",
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    seed: int | None = None,
    split: Sequence[float] | None = None,
) -> Tensor:
    """Sample boundary points on a 3‑D box boundary × time (6 faces).

    Faces (order): x‑min, x‑max, y‑min, y‑max, z‑min, z‑max.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if split is None:
        split = (1/6,) * 6
    counts = _largest_remainder_counts(n_total, split)

    def _sample_time(n: int, seed_shift: int = 0) -> Tensor:
        if n == 0:
            return torch.empty(0, 1, device=device, dtype=dtype)
        u = _unit_samples(n, 1, method=method, device=device, dtype=dtype,
                          seed=None if seed is None else seed + seed_shift)
        return t_min + u[:, 0:1] * (t_max - t_min)

    def _sample_coord(n: int, lo: float, hi: float, seed_shift: int) -> Tensor:
        if n == 0:
            return torch.empty(0, 1, device=device, dtype=dtype)
        u = _unit_samples(n, 1, method=method, device=device, dtype=dtype,
                          seed=None if seed is None else seed + seed_shift)
        return lo + u[:, 0:1] * (hi - lo)

    # x faces
    t_xmin = _sample_time(counts[0], 0)
    y_xmin = _sample_coord(counts[0], y_min, y_max, 10)
    z_xmin = _sample_coord(counts[0], z_min, z_max, 20)

    t_xmax = _sample_time(counts[1], 1)
    y_xmax = _sample_coord(counts[1], y_min, y_max, 11)
    z_xmax = _sample_coord(counts[1], z_min, z_max, 21)

    # y faces
    t_ymin = _sample_time(counts[2], 2)
    x_ymin = _sample_coord(counts[2], x_min, x_max, 12)
    z_ymin = _sample_coord(counts[2], z_min, z_max, 22)

    t_ymax = _sample_time(counts[3], 3)
    x_ymax = _sample_coord(counts[3], x_min, x_max, 13)
    z_ymax = _sample_coord(counts[3], z_min, z_max, 23)

    # z faces
    t_zmin = _sample_time(counts[4], 4)
    x_zmin = _sample_coord(counts[4], x_min, x_max, 14)
    y_zmin = _sample_coord(counts[4], y_min, y_max, 24)

    t_zmax = _sample_time(counts[5], 5)
    x_zmax = _sample_coord(counts[5], x_min, x_max, 15)
    y_zmax = _sample_coord(counts[5], y_min, y_max, 25)

    xmin = torch.cat([torch.full_like(t_xmin, x_min), y_xmin, z_xmin, t_xmin], dim=1)
    xmax = torch.cat([torch.full_like(t_xmax, x_max), y_xmax, z_xmax, t_xmax], dim=1)
    ymin = torch.cat([x_ymin, torch.full_like(t_ymin, y_min), z_ymin, t_ymin], dim=1)
    ymax = torch.cat([x_ymax, torch.full_like(t_ymax, y_max), z_ymax, t_ymax], dim=1)
    zmin = torch.cat([x_zmin, y_zmin, torch.full_like(t_zmin, z_min), t_zmin], dim=1)
    zmax = torch.cat([x_zmax, y_zmax, torch.full_like(t_zmax, z_max), t_zmax], dim=1)

    return torch.cat([xmin, xmax, ymin, ymax, zmin, zmax], dim=0)


# ===========================================================================
# Axis-aligned box domains
# ===========================================================================

@dataclass(frozen=True)
class Box1D:
    """Axis-aligned (x,t) rectangle."""
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def grid(
        self,
        n_x: int,
        n_t: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        return_cartesian: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return grid1d(
            n_x,
            n_t,
            self.x_min,
            self.x_max,
            self.t_min,
            self.t_max,
            device=device,
            dtype=dtype,
            return_cartesian=return_cartesian,
        )

    def sample_interior(
        self,
        n: int,
        *,
        method: _SampleMethod = "sobol",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_interior_1d(
            n,
            self.x_min,
            self.x_max,
            self.t_min,
            self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def sample_boundary(
        self,
        n_total: int,
        *,
        method: _SampleMethod = "sobol",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_boundary_1d(
            n_total,
            self.x_min,
            self.x_max,
            self.t_min,
            self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
        )


@dataclass(frozen=True)
class Box2D:
    """Axis-aligned (x,y,t) rectangular slab."""
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def grid(
        self,
        n_x: int,
        n_y: int,
        n_t: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        return_cartesian: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return grid2d(
            n_x,
            n_y,
            n_t,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.t_min,
            self.t_max,
            device=device,
            dtype=dtype,
            return_cartesian=return_cartesian,
        )

    def sample_interior(
        self,
        n: int,
        *,
        method: _SampleMethod = "sobol",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_interior_2d(
            n,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.t_min,
            self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def sample_boundary(
        self,
        n_total: int,
        *,
        method: _SampleMethod = "sobol",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
        split: Sequence[float] | None = None,
    ) -> Tensor:
        return sample_boundary_2d(
            n_total,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.t_min,
            self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
            split=split,
        )


@dataclass(frozen=True)
class Box3D:
    """Axis-aligned (x,y,z,t) rectangular slab."""
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def grid(
        self,
        n_x: int,
        n_y: int,
        n_z: int,
        n_t: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        return_cartesian: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return grid3d(
            n_x,
            n_y,
            n_z,
            n_t,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
            self.t_min,
            self.t_max,
            device=device,
            dtype=dtype,
            return_cartesian=return_cartesian,
        )

    def sample_interior(
        self,
        n: int,
        *,
        method: _SampleMethod = "sobol",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_interior_3d(
            n,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
            self.t_min,
            self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def sample_boundary(
        self,
        n_total: int,
        *,
        method: _SampleMethod = "sobol",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        seed: int | None = None,
        split: Sequence[float] | None = None,
    ) -> Tensor:
        return sample_boundary_3d(
            n_total,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
            self.t_min,
            self.t_max,
            method=method,
            device=device,
            dtype=dtype,
            seed=seed,
            split=split,
        )


__all__ = [
    # original API
    "grid1d",
    "grid2d",
    # new generators
    "grid3d",
    # spacing
    "spacing1d",
    "spacing2d",
    "spacing3d",
    # samplers
    "sample_interior_1d",
    "sample_interior_2d",
    "sample_interior_3d",
    "sample_boundary_1d",
    "sample_boundary_2d",
    "sample_boundary_3d",
    # domains
    "Box1D",
    "Box2D",
    "Box3D",
]
