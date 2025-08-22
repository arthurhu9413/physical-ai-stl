from __future__ import annotations

"""
Lightweight, dependency‑minimal plotting helpers used across the repo.

Design goals
------------
- **Robust I/O**: Accept NumPy arrays, PyTorch tensors, Python sequences.
- **CPS‑friendly**: Handle typical shapes that arise in 1‑D/2‑D PDE demos
  and time–series traces from monitors (e.g., STL robustness).
- **Headless‑safe**: Never call `plt.show()`; always save to disk and close.
- **Fast**: Optionally downsample very large heatmaps to avoid rendering
  bottlenecks while preserving qualitative structure.
- **Pretty by default**: Sensible labels, titles, grids and colorbars.

All functions return the **Path** to the saved figure to make them easy to
use in scripts and tests.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Optional torch dependency (commonly available in this repo)
try:  # pragma: no cover - don't require torch in minimal envs
    import torch  # type: ignore[import-not-found]

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

# A permissive "array like" union used throughout this module
if _HAS_TORCH:
    ArrayLike = torch.Tensor | np.ndarray | Sequence[float]
else:
    ArrayLike = np.ndarray | Sequence[float]


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------
def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Best‑effort conversion to a NumPy array on CPU without gradients."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _ensure_dir(path: str | Path) -> None:
    """Create parent directory for *path* if needed (no‑op otherwise)."""
    p = Path(path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _maybe_downsample(img: np.ndarray, max_elems: int = 2_000_000) -> tuple[np.ndarray, int, int]:
    """Downsample a 2‑D image if it has more than *max_elems* pixels.

    Returns the (possibly downsampled) image and the strides used along y and x.
    """
    if img.ndim != 2:
        raise ValueError(f"expected 2‑D array, got shape {img.shape}")
    rows, cols = img.shape
    elems = rows * cols
    if elems <= max_elems:
        return img, 1, 1
    # choose integer, identical stride factors to get close to max_elems
    stride = max(1, int(np.ceil(np.sqrt(elems / max_elems))))
    return img[::stride, ::stride], stride, stride


def _extent_from_coords(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) extent tuple for imshow/pcolormesh."""
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    return (x_min, x_max, y_min, y_max)


def _reshape_to_grid(u: np.ndarray, n_y: int, n_x: int) -> np.ndarray:
    """Arrange *u* into a (n_y, n_x) grid, accepting common alternatives.

    Acceptable inputs:
      * u.shape == (n_y, n_x)
      * u.shape == (n_x, n_y)  -> will be transposed
      * u is 1‑D with size n_y*n_x -> will be reshaped row‑major
    """
    if u.ndim == 2 and u.shape == (n_y, n_x):
        return u
    if u.ndim == 2 and u.shape == (n_x, n_y):
        return u.T
    if u.ndim == 1 and u.size == n_y * n_x:
        return u.reshape(n_y, n_x)
    raise ValueError(f"u with shape {u.shape} cannot be arranged to ({n_y}, {n_x}).")


def _sorted_with_index(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted array and permutation indices that sort it ascending."""
    idx = np.argsort(a, kind="stable")
    return a[idx], idx


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def plot_u_xt(
    u: ArrayLike,
    x: ArrayLike,
    t: ArrayLike,
    *,
    out: str | Path = "figs/diffusion_heatmap.png",
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = True,
    title: str = "1‑D Diffusion PINN (u)",
    max_elems: int = 2_000_000,
    interpolation: str = "nearest",
    dpi: int = 150,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    cbar_label: str = "u(x,t)",
) -> Path:
    """Heatmap for a scalar field :math:`u(x,t)`.

    Parameters
    ----------
    u : array‑like, shape (n_x, n_t) or (n_t, n_x) or (n_x*n_t,)
        Field values on a Cartesian product of space *x* and time *t*.
    x : array‑like, shape (n_x,)
        Spatial coordinates (need not be uniform; will be **sorted**).
    t : array‑like, shape (n_t,)
        Time coordinates (need not be uniform; will be **sorted**).
    out : path‑like
        Output image path (parent directory will be created on demand).
    vmin, vmax : float or None
        Color limits. If both None, Matplotlib's autoscale is used.
    add_colorbar : bool
        Whether to draw a colorbar.
    title : str
        Figure title.
    max_elems : int
        If the product n_x * n_t exceeds this threshold, the image is
        downsampled with a uniform stride for responsiveness.
    interpolation : str
        Interpolation mode passed to :func:`matplotlib.pyplot.imshow`.
    dpi : int
        Output DPI for :func:`matplotlib.pyplot.savefig`.
    cmap : str or None
        Colormap name (e.g. ``'viridis'``). If None, Matplotlib default is used.
    figsize : (float, float) or None
        Figure size in inches.
    cbar_label : str
        Label for the colorbar.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    u_np = _to_numpy(u)
    x_np = _to_numpy(x).reshape(-1)
    t_np = _to_numpy(t).reshape(-1)

    # Sort coordinates (robust to user passing non‑monotonic arrays)
    x_sorted, ix = _sorted_with_index(x_np)
    t_sorted, it = _sorted_with_index(t_np)

    # Arrange u to (n_x, n_t) then permute to the sorted order
    grid = _reshape_to_grid(u_np, n_y=x_np.size, n_x=t_np.size)  # rows=y:=x, cols=x:=t
    grid = grid[ix, :][:, it]
    grid = np.asarray(grid, dtype=float)
    grid = np.ma.masked_invalid(grid)  # NaNs appear hollow instead of crashing

    # Optional downsampling (preserve axes by also slicing coordinates)
    grid_ds, sy, sx = _maybe_downsample(grid, max_elems=max_elems)
    extent = _extent_from_coords(y=x_sorted[::sy], x=t_sorted[::sx])

    _ensure_dir(out)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        grid_ds,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        cmap=cmap,
    )
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
    fig.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def plot_u_xy_frame(
    u_xy: ArrayLike,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    out: str | Path = "figs/heat2d_t0.png",
    add_colorbar: bool = True,
    title: str = "2‑D Heat (single frame)",
    dpi: int = 150,
    interpolation: str = "nearest",
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    cbar_label: str = "u(x,y)",
) -> Path:
    """Single 2‑D scalar frame :math:`u(x,y)`.

    If *x* and *y* are provided, we use their sizes to infer the (n_y, n_x)
    shape and we build a proper extent so axes reflect physical units. If they
    are omitted, the input is interpreted as a 2‑D grid or reshaped to a
    square grid for visualization.

    Returns the output **Path**.
    """
    u_np = _to_numpy(u_xy)

    if x is not None and y is not None:
        x_np = _to_numpy(x).reshape(-1)
        y_np = _to_numpy(y).reshape(-1)

        # Sort coordinates for robustness and align grid accordingly
        x_sorted, ix = _sorted_with_index(x_np)
        y_sorted, iy = _sorted_with_index(y_np)

        grid = _reshape_to_grid(u_np, n_y=y_np.size, n_x=x_np.size)
        grid = grid[iy, :][:, ix]
        extent = _extent_from_coords(y=y_sorted, x=x_sorted)
    else:
        # Try to keep as‑is if already 2‑D, otherwise attempt a square reshape
        if u_np.ndim == 2:
            grid = u_np
        else:
            side = int(round(np.sqrt(u_np.size)))
            if side * side != u_np.size:
                raise ValueError(
                    "u_xy must be 2‑D or have perfect‑square size when x/y are omitted."
                )
            grid = np.reshape(u_np, (side, side))
        extent = None  # pixel extent

    grid = np.ma.masked_invalid(np.asarray(grid, dtype=float))

    _ensure_dir(out)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=extent,  # type: ignore[arg-type]
        interpolation=interpolation,
        cmap=cmap,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
    fig.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def plot_time_slices(
    u: ArrayLike,
    x: ArrayLike,
    t: ArrayLike,
    *,
    times: Iterable[float] | None = None,
    num_slices: int = 4,
    out: str | Path = "figs/diffusion_slices.png",
    u_bounds: tuple[float | None, float | None] | None = None,
    title: str = "u(x,t) at selected times",
    dpi: int = 150,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Plot several spatial slices :math:`u(x, t_k)` at chosen time instants.

    If *times* is None we pick *num_slices* evenly spaced indices. Otherwise we
    select the nearest index for each requested time (after sorting *t*).

    Parameters
    ----------
    u : array‑like, shape compatible with (len(x), len(t))
    x : array‑like, shape (len(x),)
    t : array‑like, shape (len(t),)
    u_bounds : (u_min, u_max) to optionally draw dashed reference bounds.

    Returns
    -------
    Path to the saved figure.
    """
    u_np = _to_numpy(u)
    x_np = _to_numpy(x).reshape(-1)
    t_np = _to_numpy(t).reshape(-1)

    x_sorted, ix = _sorted_with_index(x_np)
    t_sorted, it = _sorted_with_index(t_np)

    U = _reshape_to_grid(u_np, n_y=x_np.size, n_x=t_np.size)
    U = U[ix, :][:, it]  # align with sorted coordinates

    if times is None:
        idx = np.linspace(0, t_sorted.size - 1, num=num_slices, dtype=int)
    else:
        times_arr = np.asarray(list(times), dtype=float)
        idx = np.clip(np.searchsorted(t_sorted, times_arr), 0, t_sorted.size - 1)

    _ensure_dir(out)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for k in np.unique(idx):
        ax.plot(x_sorted, U[:, int(k)], label=f"t={t_sorted[int(k)]:.3g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(title)
    if u_bounds is not None:
        umin, umax = u_bounds
        if umin is not None:
            ax.axhline(float(umin), linestyle="--", linewidth=1.0, alpha=0.6, label="u_min")
        if umax is not None:
            ax.axhline(float(umax), linestyle="--", linewidth=1.0, alpha=0.6, label="u_max")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", framealpha=0.8, fontsize="small")
    fig.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def plot_spatial_mean_over_time(
    u: ArrayLike,
    t: ArrayLike,
    *,
    mean_dims: tuple[int, ...] | None = None,
    out: str | Path = "figs/mean_over_time.png",
    u_max: float | None = None,
    var_name: str = "u",
    title: str | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Plot the spatial mean of *u* against time.

    By default we average over all axes **except the last** which is assumed
    to be time. You can override this by specifying *mean_dims*.

    Parameters
    ----------
    u : array‑like
        Data whose mean is taken.
    t : array‑like, shape (n_t,)
        Time vector.
    mean_dims : tuple[int, ...] or None
        Axes to average over. If None, uses ``tuple(range(u.ndim - 1))``.
    u_max : float or None
        Optional horizontal line indicating a max allowed value.
    var_name : str
        Variable name used in axis label and legend.
    """
    u_np = _to_numpy(u)
    t_np = _to_numpy(t).reshape(-1)
    if mean_dims is None:
        mean_axes = tuple(range(u_np.ndim - 1))
    else:
        mean_axes = tuple(mean_dims)
    series = u_np.mean(axis=mean_axes)
    if series.ndim != 1 or series.size != t_np.size:
        series = series.reshape(-1)
        if series.size != t_np.size:
            raise ValueError("Cannot align mean series with provided time vector.")

    _ensure_dir(out)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(t_np, series)
    ax.set_xlabel("t")
    ax.set_ylabel(f"mean_x {var_name}")
    if title is None:
        title = f"Temporal evolution of mean_x {var_name}"
    ax.set_title(title)
    if u_max is not None:
        ax.axhline(float(u_max), linestyle="--", linewidth=1.0, alpha=0.6, label=f"{var_name}_max")
        ax.legend(loc="best", framealpha=0.8, fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


# -----------------------------------------------------------------------------
# Backwards‑compatible aliases (if other scripts import the old names)
# -----------------------------------------------------------------------------
def plot_u_1d(u: ArrayLike, X: ArrayLike, T: ArrayLike, out: str | Path = "figs/diffusion_heatmap.png") -> Path:
    """Alias kept for backward compatibility: calls :func:`plot_u_xt`."""
    return plot_u_xt(u=u, x=X, t=T, out=out)


def plot_u_2d_frame(u_frame: ArrayLike, out: str | Path = "figs/heat2d_t0.png") -> Path:
    """Alias kept for backward compatibility: calls :func:`plot_u_xy_frame`."""
    return plot_u_xy_frame(u_xy=u_frame, out=out)
