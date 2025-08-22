# tests/test_pde_example.py
from __future__ import annotations

import numpy as np
import pytest

from physical_ai_stl.pde_example import (
    simulate_diffusion,
    simulate_diffusion_with_clipping,
    compute_robustness,
    compute_spatiotemporal_robustness,
)

TOL = 1e-12


# ---------------------------------------------------------------------------
# Finite‑difference diffusion simulator
# ---------------------------------------------------------------------------

def test_simulate_diffusion_shape_dtype_and_finiteness() -> None:
    u = simulate_diffusion(length=3, steps=5, dt=0.1, alpha=0.1)
    assert u.shape == (6, 3)
    assert u.dtype == np.float64
    assert np.isfinite(u).all()


def test_simulate_diffusion_zero_steps_returns_initial() -> None:
    # Default initial condition: a single hot spot at the left boundary.
    u0 = simulate_diffusion(length=4, steps=0)
    assert u0.shape == (1, 4)
    assert np.allclose(u0[0], np.array([1.0, 0.0, 0.0, 0.0], dtype=float), atol=TOL)

    # Custom initial condition is respected exactly when steps == 0.
    init = np.array([0.2, -0.1, 0.4, 0.0])
    u0b = simulate_diffusion(length=4, steps=0, initial=init)
    assert np.allclose(u0b[0], init, atol=TOL)


def test_simulate_diffusion_length_one_is_constant() -> None:
    u = simulate_diffusion(length=1, steps=5, dt=0.1, alpha=0.1)
    assert u.shape == (6, 1)
    # With a single cell, the state never changes.
    assert np.all(u == u[0])


def test_simulate_diffusion_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        simulate_diffusion(length=0, steps=1)
    with pytest.raises(ValueError):
        simulate_diffusion(length=3, steps=-1)
    with pytest.raises(ValueError):
        simulate_diffusion(length=3, steps=1, initial=np.zeros((3, 3)))


def test_interior_update_is_convex_combination_when_diff_small() -> None:
    rng = np.random.default_rng(42)
    length = 9
    init = rng.normal(size=length)
    dt, alpha = 0.1, 0.1          # alpha*dt = 0.01 <= 0.5
    u = simulate_diffusion(length=length, steps=1, dt=dt, alpha=alpha, initial=init)

    for i in range(1, length - 1):
        mn = float(min(init[i - 1], init[i], init[i + 1]))
        mx = float(max(init[i - 1], init[i], init[i + 1]))
        assert (u[1, i] >= mn - TOL) and (u[1, i] <= mx + TOL)


def test_spatial_peak_to_peak_never_increases_over_time() -> None:
    rng = np.random.default_rng(0)
    init = rng.normal(size=8)
    u = simulate_diffusion(length=8, steps=6, dt=0.1, alpha=0.1, initial=init)
    prev = np.ptp(u[0])  # max - min
    for t in range(6):
        cur = np.ptp(u[t + 1])
        assert cur <= prev + TOL
        prev = cur


# Additional short, exact check for a tiny grid
def test_length3_first_step_is_uniform_equal_to_alpha_dt() -> None:
    dt, alpha = 0.2, 0.2  # diff = alpha*dt = 0.04
    u = simulate_diffusion(length=3, steps=1, dt=dt, alpha=alpha)
    assert np.allclose(u[1], np.full(3, alpha * dt), atol=TOL)


def test_simulate_diffusion_emits_runtime_warning_when_unstable() -> None:
    # r = alpha * dt / dx^2; choose r > 0.5 to trigger the warning
    with pytest.warns(RuntimeWarning):
        u = simulate_diffusion(length=5, steps=1, dt=1.1, alpha=0.6)  # r=0.66
        assert u.shape == (2, 5)


def test_simulate_diffusion_respects_dtype_override() -> None:
    u = simulate_diffusion(length=4, steps=2, dt=0.1, alpha=0.1, dtype=np.float32)
    assert u.dtype == np.float32
    assert np.isfinite(u).all()


# ---------------------------------------------------------------------------
# Simulator with per‑step clipping
# ---------------------------------------------------------------------------

def test_diffusion_with_clipping_shape_and_bounds() -> None:
    u = simulate_diffusion_with_clipping(
        length=4, steps=3, dt=0.1, alpha=0.1, lower=-0.25, upper=0.25
    )
    assert u.shape == (4, 4)
    assert (u >= -0.25 - 1e-12).all() and (u <= 0.25 + 1e-12).all()


def test_first_frame_is_also_clipped() -> None:
    u = simulate_diffusion_with_clipping(length=3, steps=0, lower=0.0, upper=0.2)
    assert u.shape == (1, 3)
    assert u[0, 0] == pytest.approx(0.2, abs=TOL)
    assert np.all(u[0, 1:] == 0.0)


def test_per_step_clipping_never_worsens_robustness() -> None:
    base_u = simulate_diffusion(length=5, steps=5, dt=0.1, alpha=0.1)
    base_rob = compute_spatiotemporal_robustness(base_u, lower=0.0, upper=0.5)
    u_clip = simulate_diffusion_with_clipping(
        length=5, steps=5, dt=0.1, alpha=0.1, lower=0.0, upper=0.5
    )
    clip_rob = compute_spatiotemporal_robustness(u_clip, lower=0.0, upper=0.5)
    assert clip_rob >= base_rob - TOL
    # clipped signal is always within bounds, so robustness is non‑negative
    assert clip_rob >= -TOL


# ---------------------------------------------------------------------------
# Robustness helpers
# ---------------------------------------------------------------------------

def test_compute_robustness_on_simple_signal() -> None:
    sig = np.array([0.1, 0.2, 0.3, 0.4])
    rob = compute_robustness(sig, lower=0.0, upper=0.5)
    # min(min(sig - lower), min(upper - sig)) = min(0.1, 0.1) = 0.1
    assert isinstance(rob, float)
    assert rob == pytest.approx(0.1, abs=TOL)


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0,)),
        np.array([[1.0, 2.0], [3.0, 4.0]]),   # not 1‑D
    ],
)
def test_compute_robustness_invalid_inputs(bad) -> None:
    with pytest.raises(ValueError):
        compute_robustness(bad, lower=-1.0, upper=1.0)


def test_robustness_translation_and_scale_invariance_1d() -> None:
    sig = np.array([-0.3, 0.2, 0.0, 0.5], dtype=float)
    low, up = -0.4, 0.6
    c = 1.7
    k = 3.2
    base = compute_robustness(sig, low, up)
    # Shift both the signal and bounds by +c -> robustness unchanged.
    shifted = compute_robustness(sig + c, low + c, up + c)
    assert shifted == pytest.approx(base, abs=TOL)
    # Scale by k>0 -> robustness scales by k.
    scaled = compute_robustness(k * sig, k * low, k * up)
    assert scaled == pytest.approx(k * base, abs=TOL)


def test_compute_spatiotemporal_robustness_matches_flattened_1d() -> None:
    mat = np.array([[0.1, -0.2, 0.3], [0.7, 0.0, 0.4]], dtype=float)
    rob2d = compute_spatiotemporal_robustness(mat, lower=-0.5, upper=0.5)
    rob1d = compute_robustness(mat.ravel(), lower=-0.5, upper=0.5)
    assert isinstance(rob2d, float)
    assert rob2d == pytest.approx(rob1d, abs=TOL)


def test_spatiotemporal_negative_if_out_of_bounds() -> None:
    mat = np.array([[0.6, 0.2], [0.0, 0.4]], dtype=float)
    rob = compute_spatiotemporal_robustness(mat, lower=0.0, upper=0.5)
    assert rob == pytest.approx(0.5 - 0.6, abs=TOL)  # -0.1


@pytest.mark.parametrize("bad", [np.array([]), np.array([1.0, 2.0, 3.0])])
def test_compute_spatiotemporal_invalid_inputs(bad) -> None:
    with pytest.raises(ValueError):
        compute_spatiotemporal_robustness(bad, lower=0.0, upper=1.0)


def test_spatiotemporal_monotonic_in_bounds() -> None:
    mat = np.array([[0.2, 0.4, 0.6],
                    [0.1, 0.3, 0.5]], dtype=float)
    base = compute_spatiotemporal_robustness(mat, 0.0, 1.0)
    tighter = compute_spatiotemporal_robustness(mat, 0.1, 0.7)
    wider = compute_spatiotemporal_robustness(mat, -1.0, 2.0)
    assert tighter <= base + TOL
    assert wider >= base - TOL
