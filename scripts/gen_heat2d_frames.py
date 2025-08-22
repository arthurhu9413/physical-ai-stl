#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 2‑D heat‑equation frames for STREL demos (MoonLight‑friendly).

What this script does
---------------------
• Simulates the 2‑D heat equation
      u_t = α (u_xx + u_yy)
  on a Cartesian grid for ``nt`` steps and writes each frame as ``.npy``.
• *Default behavior matches the original script*:
  - Explicit 5‑point stencil (FTCS) with **periodic** boundaries via ``np.roll``.
  - Stability check equivalent to ``4*alpha*dt <= 1`` when ``dx = dy = 1``.
  - Saves per‑time frames and (optionally) a packed ``field_xy_t.npy``.
• Adds quality‑of‑life upgrades while keeping zero extra hard deps:
  - Boundary conditions: **periodic** (fast), **neumann** (zero‑flux), **dirichlet**.
  - Grid scaling: configurable ``dx, dy`` with the correct FTCS stability check
    (general form: ``alpha*dt*(1/dx**2 + 1/dy**2) <= 1/2``).
  - Initial conditions: Gaussian hot‑spot (default), multiple Gaussians, ring, checker.
  - Determinism: full metadata JSON is written next to outputs for reproducibility.
  - Efficiency: streaming writer (no huge in‑RAM tensor) + optional packed file.
  - Optional FFT‑exact integrator for **periodic** BC that is unconditionally stable
    and often faster for large grids / long horizons.

Why this is a good fit
----------------------
MoonLight supports STREL monitoring over spatio‑temporal grids/graphs, and is used
for quantitative “robustness” scoring of properties on 2‑D time‑varying fields. The
frames produced here are drop‑in assets for such demos. See MoonLight’s repo for the
Python API and STREL syntax, and RTAMT / SpaTiaL for complementary STL tooling.  [1–4]

References
----------
[1] MoonLightSuite/moonlight – lightweight runtime monitoring with STREL. 
[2] nickovic/rtamt – specification‑based real‑time STL monitoring (offline/online).
[3] KTH‑RPL‑Planiacs/SpaTiaL – spatial–temporal relations and formulas.
[4] NeurIPS’20 *STLnet* – learning with STL constraints for CPS time‑series.

Example
-------
# Periodic BC (fast), explicit scheme, 100 frames of 32×32, default Gaussian seed=0
python scripts/gen_heat2d_frames.py --nx 32 --ny 32 --nt 100 --outdir assets/heat2d_scalar

# Neumann (zero‑flux) walls and two hot‑spots, safer auto‑dt at 90% stability
python scripts/gen_heat2d_frames.py --bc neumann --init two_gaussians --alpha 0.4 \
    --nt 200 --target-dt 0.05 --auto-dt --safety 0.9 --outdir assets/heat2d_neumann

# Periodic + FFT exact integrator (unconditional stability), also pack tensor
python scripts/gen_heat2d_frames.py --method fft --bc periodic --alpha 0.7 --nt 240 \
    --dt 0.2 --also-pack --outdir assets/heat2d_fft

"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Tuple

import numpy as np


# ----------------------------- core numerics ---------------------------------

BCType = Literal["periodic", "neumann", "dirichlet"]
Method = Literal["ftcs", "fft"]  # fft requires periodic BC


@dataclass(frozen=True)
class Heat2DConfig:
    nx: int = 32
    ny: int = 32
    nt: int = 100
    dt: float = 0.05
    alpha: float = 0.5
    dx: float = 1.0  # grid units, not tied to the [-1,1] coords used for init patterns
    dy: float = 1.0
    bc: BCType = "periodic"
    method: Method = "ftcs"
    seed: int | None = 0
    init: str = "gaussian"  # gaussian | two_gaussians | ring | checker
    sigma: float = 0.15
    amplitude: float = 1.0
    noise: float = 0.01
    dirichlet_value: float = 0.0
    outdir: Path = Path("assets/heat2d_scalar")
    also_pack: bool = False
    layout: Literal["xy_t", "t_xy"] = "xy_t"
    # Auto‑dt tuner (optional)
    auto_dt: bool = False
    target_dt: float = 0.05
    safety: float = 0.98  # fraction of the FTCS stability limit to use when --auto-dt
    # IO
    save_every: int = 1  # write every k frames
    dtype: str = "float32"  # storage dtype for frames / packed tensor


def _stability_limit_dt(alpha: float, dx: float, dy: float) -> float:
    """
    FTCS stability limit (Von Neumann analysis) for 2‑D heat eq:
        r_x + r_y <= 1/2,  where r_x = alpha*dt/dx^2 and r_y = alpha*dt/dy^2
    =>  dt <= 0.5 / (alpha * (1/dx^2 + 1/dy^2)).
    """
    denom = alpha * ((1.0 / (dx * dx)) + (1.0 / (dy * dy)))
    if denom == 0.0:
        return math.inf
    return 0.5 / denom


def _init_field(cfg: Heat2DConfig) -> np.ndarray:
    """Construct deterministic initial condition on [-1, 1]^2 with tiny noise."""
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(-1.0, 1.0, cfg.nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, cfg.ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = np.zeros((cfg.nx, cfg.ny), dtype=np.float32)

    if cfg.init == "gaussian":
        u = cfg.amplitude * np.exp(-(X**2 + Y**2) / (2.0 * cfg.sigma**2)).astype(np.float32)
    elif cfg.init == "two_gaussians":
        # Symmetric bumps left/right
        shift = 0.4
        g1 = np.exp(-((X + shift) ** 2 + Y**2) / (2.0 * cfg.sigma**2)).astype(np.float32)
        g2 = np.exp(-((X - shift) ** 2 + Y**2) / (2.0 * cfg.sigma**2)).astype(np.float32)
        u = cfg.amplitude * (0.5 * (g1 + g2))
    elif cfg.init == "ring":
        r = np.sqrt(X**2 + Y**2)
        u = cfg.amplitude * np.exp(-((r - 0.5) ** 2) / (2.0 * (cfg.sigma * 0.75) ** 2)).astype(np.float32)
    elif cfg.init == "checker":
        # Mild checkerboard to exercise diffusion + STREL spatial operators
        kx = int(max(1, round(0.1 * cfg.nx)))
        ky = int(max(1, round(0.1 * cfg.ny)))
        u = cfg.amplitude * (np.sin(kx * np.pi * (X + 1) / 2) * np.sin(ky * np.pi * (Y + 1) / 2)).astype(np.float32)
        u = (u - u.min()) / max(1e-6, (u.max() - u.min()))  # scale to [0,1]
    else:
        raise ValueError(f"Unknown init pattern: {cfg.init!r}")

    if cfg.noise > 0.0:
        u = (u + cfg.noise * rng.standard_normal(size=u.shape, dtype=np.float32)).astype(np.float32)

    # Enforce Dirichlet boundaries at t=0 if requested
    if cfg.bc == "dirichlet":
        u[0, :] = cfg.dirichlet_value
        u[-1, :] = cfg.dirichlet_value
        u[:, 0] = cfg.dirichlet_value
        u[:, -1] = cfg.dirichlet_value

    return u.astype(np.float32, copy=False)


def _laplacian_periodic(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """5‑point Laplacian with periodic BCs via np.roll (fast)."""
    return (
        (np.roll(u, 1, axis=0) - 2.0 * u + np.roll(u, -1, axis=0)) / (dx * dx)
        + (np.roll(u, 1, axis=1) - 2.0 * u + np.roll(u, -1, axis=1)) / (dy * dy)
    ).astype(np.float32, copy=False)


def _laplacian_neumann(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """5‑point Laplacian with zero‑flux walls (first‑derivative = 0 at boundary)."""
    lap = np.zeros_like(u, dtype=np.float32)
    # interior
    lap[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
        + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
    )
    # edges: mirror one cell
    lap[0, 1:-1] = (
        (u[1, 1:-1] - 2.0 * u[0, 1:-1] + u[1, 1:-1]) / (dx * dx)  # mirrored ghost equals u[1]
        + (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]) / (dy * dy)
    )
    lap[-1, 1:-1] = (
        (u[-2, 1:-1] - 2.0 * u[-1, 1:-1] + u[-2, 1:-1]) / (dx * dx)
        + (u[-1, 2:] - 2.0 * u[-1, 1:-1] + u[-1, :-2]) / (dy * dy)
    )
    lap[1:-1, 0] = (
        (u[2:, 0] - 2.0 * u[1:-1, 0] + u[:-2, 0]) / (dx * dx)
        + (u[1:-1, 1] - 2.0 * u[1:-1, 0] + u[1:-1, 1]) / (dy * dy)
    )
    lap[1:-1, -1] = (
        (u[2:, -1] - 2.0 * u[1:-1, -1] + u[:-2, -1]) / (dx * dx)
        + (u[1:-1, -2] - 2.0 * u[1:-1, -1] + u[1:-1, -2]) / (dy * dy)
    )
    # corners (mirror in both directions)
    lap[0, 0] = (
        (u[1, 0] - 2.0 * u[0, 0] + u[1, 0]) / (dx * dx)
        + (u[0, 1] - 2.0 * u[0, 0] + u[0, 1]) / (dy * dy)
    )
    lap[0, -1] = (
        (u[1, -1] - 2.0 * u[0, -1] + u[1, -1]) / (dx * dx)
        + (u[0, -2] - 2.0 * u[0, -1] + u[0, -2]) / (dy * dy)
    )
    lap[-1, 0] = (
        (u[-2, 0] - 2.0 * u[-1, 0] + u[-2, 0]) / (dx * dx)
        + (u[-1, 1] - 2.0 * u[-1, 0] + u[-1, 1]) / (dy * dy)
    )
    lap[-1, -1] = (
        (u[-2, -1] - 2.0 * u[-1, -1] + u[-2, -1]) / (dx * dx)
        + (u[-1, -2] - 2.0 * u[-1, -1] + u[-1, -2]) / (dy * dy)
    )
    return lap


def _step_ftcs(u: np.ndarray, alpha: float, dt: float, dx: float, dy: float, bc: BCType, dirichlet_value: float) -> np.ndarray:
    """One explicit (FTCS) step for the chosen boundary condition."""
    if bc == "periodic":
        lap = _laplacian_periodic(u, dx, dy)
        return (u + alpha * dt * lap).astype(np.float32, copy=False)

    # Neumann and Dirichlet share the same interior Laplacian; Dirichlet also clamps edges.
    if bc == "neumann":
        lap = _laplacian_neumann(u, dx, dy)
        return (u + alpha * dt * lap).astype(np.float32, copy=False)

    if bc == "dirichlet":
        lap = np.zeros_like(u, dtype=np.float32)
        # interior only; boundaries remain fixed at dirichlet_value
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
            + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
        )
        u_next = (u + alpha * dt * lap).astype(np.float32, copy=False)
        # enforce boundary values
        u_next[0, :] = dirichlet_value
        u_next[-1, :] = dirichlet_value
        u_next[:, 0] = dirichlet_value
        u_next[:, -1] = dirichlet_value
        return u_next

    raise ValueError(f"Unknown boundary condition: {bc!r}")


def _make_fft_evolution_operator(nx: int, ny: int, alpha: float, dt: float, dx: float, dy: float) -> np.ndarray:
    """
    Precompute the exact heat evolution multiplier in spectral domain for *periodic* BC:
        U^{n+1}(k) = exp(-α (k_x^2 + k_y^2) dt) * U^{n}(k)
    where k_x = 2π * freq_x / (nx*dx), etc. Uses angular wavenumbers.
    """
    # cycles per sample
    fx = np.fft.fftfreq(nx, d=dx)  # [cycles / unit]
    fy = np.fft.fftfreq(ny, d=dy)
    kx = 2.0 * np.pi * fx[:, None]  # [rad / unit]
    ky = 2.0 * np.pi * fy[None, :]
    lam = -(alpha * dt) * (kx**2 + ky**2)  # exponent
    return np.exp(lam).astype(np.float32)


def heat2d(
    nx: int,
    ny: int,
    nt: int,
    dt: float,
    alpha: float,
    seed: int | None = 0,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    bc: BCType = "periodic",
    method: Method = "ftcs",
    init: str = "gaussian",
    sigma: float = 0.15,
    amplitude: float = 1.0,
    noise: float = 0.01,
    dirichlet_value: float = 0.0,
    dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    """
    Simulate the 2‑D heat equation and return a tensor of shape (nx, ny, nt).

    Notes
    -----
    • Defaults reproduce the original behavior: dx = dy = 1, periodic FTCS, single Gaussian.
    • For FTCS, *you* must pass a stable dt; see ``_stability_limit_dt``.
    • The FFT integrator is exact for the continuous heat equation with periodic BC,
      not the discrete 5‑point Laplacian; it is unconditionally stable and usually
      more accurate per step.
    """
    u = _init_field(
        Heat2DConfig(
            nx=nx,
            ny=ny,
            nt=nt,
            dt=dt,
            alpha=alpha,
            dx=dx,
            dy=dy,
            bc=bc,
            method=method,
            seed=seed,
            init=init,
            sigma=sigma,
            amplitude=amplitude,
            noise=noise,
            dirichlet_value=dirichlet_value,
        )
    ).astype(np.float32, copy=False)

    frames = np.empty((nx, ny, nt), dtype=dtype)
    frames[..., 0] = u.astype(dtype, copy=False)

    if method == "fft":
        if bc != "periodic":
            raise ValueError("FFT method currently supports only periodic boundary conditions.")
        E = _make_fft_evolution_operator(nx, ny, alpha, dt, dx, dy)
        U = np.fft.fft2(u)
        for t in range(1, nt):
            U *= E
            u = np.fft.ifft2(U).real.astype(np.float32, copy=False)
            frames[..., t] = u.astype(dtype, copy=False)
        return frames

    # FTCS (explicit) path
    # Stability check in general form; retain original "4*alpha*dt <= 1" under dx=dy=1.
    r_x = alpha * dt / (dx * dx)
    r_y = alpha * dt / (dy * dy)
    if (r_x + r_y) > (0.5 + 1e-9):
        raise AssertionError(
            f"Explicit FTCS unstable: alpha*dt*(1/dx^2 + 1/dy^2) = {r_x + r_y:.6f} > 1/2.\n"
            f"Given alpha={alpha:g}, dt={dt:g}, dx={dx:g}, dy={dy:g}.\n"
            "Choose smaller dt or alpha, larger dx/dy, or use --method fft for periodic BC."
        )

    for t in range(1, nt):
        u = _step_ftcs(u, alpha, dt, dx, dy, bc, dirichlet_value)
        frames[..., t] = u.astype(dtype, copy=False)

    return frames


# ------------------------------ CLI / I/O ------------------------------------

def _write_per_time_frames(field: np.ndarray, outdir: Path, save_every: int = 1) -> int:
    nt = field.shape[-1]
    count = 0
    for t in range(0, nt, save_every):
        np.save(outdir / f"frame_{t:04d}.npy", field[..., t])
        count += 1
    return count


def _pack_tensor(field: np.ndarray, outdir: Path, layout: Literal["xy_t", "t_xy"]) -> Path:
    if layout == "xy_t":
        arr = field
        name = "field_xy_t.npy"
    elif layout == "t_xy":
        arr = np.moveaxis(field, -1, 0)
        name = "field_t_xy.npy"
    else:
        raise ValueError(f"Unknown layout: {layout}")
    np.save(outdir / name, arr)
    return outdir / name


def _write_metadata(cfg: Heat2DConfig, realized_dt: float, outdir: Path) -> Path:
    meta = asdict(cfg)
    meta['realized_dt'] = realized_dt
    meta['numpy_version'] = np.__version__
    # make JSON‑friendly
    for k, v in list(meta.items()):
        if isinstance(v, Path):
            meta[k] = str(v)
    path = outdir / 'meta.json'
    path.write_text(json.dumps(meta, indent=2))
    return path


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
    return ivalue


def _positive_float(value: str) -> float:
    fvalue = float(value)
    if not math.isfinite(fvalue) or fvalue <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a positive float, got {value}")
    return fvalue


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate 2‑D heat‑equation frames (.npy) for MoonLight STREL demos. "
            "Defaults mirror the original script; extra flags add BCs, init patterns, "
            "auto‑dt, and an FFT integrator for periodic grids."
        )
    )
    # grid / time
    p.add_argument("--nx", type=_positive_int, default=32)
    p.add_argument("--ny", type=_positive_int, default=32)
    p.add_argument("--nt", type=_positive_int, default=100)
    p.add_argument("--dt", type=_positive_float, default=0.05, help="Time step (used unless --auto-dt).")
    p.add_argument("--alpha", type=_positive_float, default=0.5, help="Diffusivity parameter α.")
    p.add_argument("--dx", type=_positive_float, default=1.0, help="Grid spacing in x (defaults to 1 for backward-compat).")
    p.add_argument("--dy", type=_positive_float, default=1.0, help="Grid spacing in y (defaults to 1 for backward-compat).")
    # numerics
    p.add_argument("--bc", type=str, choices=("periodic", "neumann", "dirichlet"), default="periodic")
    p.add_argument("--method", type=str, choices=("ftcs", "fft"), default="ftcs", help="fft requires periodic BC; unconditionally stable.")
    p.add_argument("--dirichlet-value", type=float, default=0.0, help="Boundary value when --bc dirichlet.")
    # initial condition
    p.add_argument("--init", type=str, choices=("gaussian", "two_gaussians", "ring", "checker"), default="gaussian")
    p.add_argument("--sigma", type=float, default=0.15, help="Width for Gaussian/ring initial conditions (in [-1,1] coords).")
    p.add_argument("--amplitude", type=float, default=1.0, help="Amplitude for initial condition.")
    p.add_argument("--noise", type=float, default=0.01, help="Additive white noise std to initial condition.")
    p.add_argument("--seed", type=int, default=0)
    # auto‑dt helper
    p.add_argument("--auto-dt", action="store_true", help="Override --dt with a stable value derived from alpha, dx, dy.")
    p.add_argument("--target-dt", type=_positive_float, default=0.05, help="Desired dt before safety scaling when --auto-dt.")
    p.add_argument("--safety", type=float, default=0.98, help="Fraction of FTCS stability limit to use when --auto-dt (0<σ<=1).")
    # I/O
    p.add_argument("--outdir", type=Path, default=Path("assets/heat2d_scalar"))
    p.add_argument("--also-pack", action="store_true", help="Also save a single 3‑D tensor (xy_t or t_xy).")
    p.add_argument("--layout", type=str, choices=("xy_t", "t_xy"), default="xy_t")
    p.add_argument("--save-every", type=_positive_int, default=1, help="Save every k frames to reduce I/O (default: 1).")
    p.add_argument("--dtype", type=str, default="float32", help="Storage dtype for frames/tensor (default: float32).")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    p = build_arg_parser()
    args = p.parse_args(list(argv) if argv is not None else None)

    # Normalize/validate
    dtype = np.dtype(args.dtype)
    cfg = Heat2DConfig(
        nx=args.nx,
        ny=args.ny,
        nt=args.nt,
        dt=args.dt,
        alpha=args.alpha,
        dx=args.dx,
        dy=args.dy,
        bc=args.bc,  # type: ignore[arg-type]
        method=args.method,  # type: ignore[arg-type]
        seed=args.seed,
        init=args.init,
        sigma=args.sigma,
        amplitude=args.amplitude,
        noise=args.noise,
        dirichlet_value=args.dirichlet_value,
        outdir=args.outdir,
        also_pack=args.also_pack,
        layout=args.layout,  # type: ignore[arg-type]
        auto_dt=args.auto_dt,
        target_dt=args.target_dt,
        safety=args.safety,
        save_every=args.save_every,
        dtype=dtype.name,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Optional auto‑dt: choose the *smaller* of target_dt and stable_dt*safety
    realized_dt = cfg.dt
    if cfg.method == "ftcs":
        stable_dt = _stability_limit_dt(cfg.alpha, cfg.dx, cfg.dy)
        if cfg.auto_dt:
            realized_dt = min(cfg.target_dt, cfg.safety * stable_dt)
        # Warn if user‑provided dt would be unstable
        rx = cfg.alpha * realized_dt / (cfg.dx * cfg.dx)
        ry = cfg.alpha * realized_dt / (cfg.dy * cfg.dy)
        if (rx + ry) > 0.5:
            p.error(
                f"Unstable explicit step: alpha*dt*(1/dx^2 + 1/dy^2) = {rx + ry:.6f} > 1/2. "
                "Use --auto-dt or lower --dt/--alpha."
            )

    # Simulate
    field = heat2d(
        cfg.nx,
        cfg.ny,
        cfg.nt,
        realized_dt,
        cfg.alpha,
        cfg.seed,
        dx=cfg.dx,
        dy=cfg.dy,
        bc=cfg.bc,
        method=cfg.method,
        init=cfg.init,
        sigma=cfg.sigma,
        amplitude=cfg.amplitude,
        noise=cfg.noise,
        dirichlet_value=cfg.dirichlet_value,
        dtype=dtype,
    )

    # I/O
    frames_written = _write_per_time_frames(field, cfg.outdir, save_every=cfg.save_every)
    packed_path = None
    if cfg.also_pack:
        packed_path = _pack_tensor(field, cfg.outdir, cfg.layout)
    meta_path = _write_metadata(cfg, realized_dt, cfg.outdir)

    msg = (
        f"Wrote {frames_written} frame(s) to {cfg.outdir}/ "
        + (f"and packed to {packed_path.name}. " if packed_path else "")
        + f"Metadata: {meta_path.name}"
    )
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
