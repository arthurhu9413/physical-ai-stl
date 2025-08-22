# tests/test_pde_robustness.py
from __future__ import annotations

import numpy as np
import pytest

# Import the light-weight helpers under test
from physical_ai_stl import pde_example as pe


# Global numeric tolerance for comparisons
TOL = 1e-12


# ----------------------------- 1D robustness ---------------------------------
def test_compute_robustness_typical_case() -> None:
    sig = np.array([0.2, 0.4, 0.6])
    rob = pe.compute_robustness(sig, lower=0.0, upper=1.0)
    # Elementwise margins are min(sig-lower, upper-sig) = [0.2, 0.4, 0.4]
    assert isinstance(rob, float)
    assert np.isclose(rob, 0.2)


@pytest.mark.parametrize(
    "sig, lower, upper, expected",
    [
        ([0.0, 1.0], 0.0, 1.0, 0.0),     # exactly on the bounds -> zero robustness
        ([0.5, 0.5], 0.0, 1.0, 0.5),     # centered in interval -> margin = 0.5
        ([-0.1, 0.2], 0.0, 1.0, -0.1),   # below lower -> negative margin
        ([0.2, 1.2], 0.0, 1.0, -0.2),    # above upper -> negative margin
    ],
)
def test_compute_robustness_boundaries_and_out_of_range(
    sig, lower, upper, expected
) -> None:
    rob = pe.compute_robustness(np.array(sig, dtype=float), lower, upper)
    assert np.isclose(rob, expected)


def test_compute_robustness_matches_definition_random() -> None:
    # Randomized spot-checks against the literal definition
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = int(rng.integers(low=1, high=12))
        sig = rng.normal(size=n)
        lo, hi = np.sort(rng.uniform(-1.0, 1.0, size=2))
        expected = np.minimum(sig - lo, hi - sig).min().item()
        got = pe.compute_robustness(sig, lo, hi)
        assert got == pytest.approx(expected, abs=TOL)


def test_compute_robustness_order_invariant() -> None:
    a = np.array([0.25, 0.75, 0.4, 0.6])
    b = a[::-1].copy()
    la, ua = 0.0, 1.0
    assert np.isclose(pe.compute_robustness(a, la, ua), pe.compute_robustness(b, la, ua))


@pytest.mark.parametrize("shift", [-1.3, -0.5, 0.0, 0.42, 2.0])
def test_compute_robustness_translation_invariance(shift: float) -> None:
    sig = np.array([-0.2, 0.1, 0.9, 1.3])
    lower, u = 0.0, 1.0
    base = pe.compute_robustness(sig, lower, u)
    shifted = pe.compute_robustness(sig + shift, lower + shift, u + shift)
    assert np.isclose(base, shifted)


@pytest.mark.parametrize("scale", [0.2, 0.5, 1.0, 2.0, 10.0])
def test_compute_robustness_positive_homogeneity(scale: float) -> None:
    sig = np.array([0.2, 0.4, 0.6])
    lower, u = 0.0, 1.0
    base = pe.compute_robustness(sig, lower, u)
    scaled = pe.compute_robustness(scale * sig, scale * lower, scale * u)
    assert np.isclose(scaled, scale * base)


def test_compute_robustness_monotonic_in_bounds() -> None:
    sig = np.array([0.2, 0.4, 0.6])
    lower, u = 0.0, 1.0
    base = pe.compute_robustness(sig, lower, u)

    # Tighten: raise lower and lower upper (but keep signal within the new interval)
    tighter = pe.compute_robustness(sig, lower + 0.1, u - 0.3)  # new [0.1, 0.7]
    assert tighter <= base + 1e-12

    # Widen: extend both bounds
    wider = pe.compute_robustness(sig, lower - 1.0, u + 1.0)
    assert wider >= base - 1e-12


def test_compute_robustness_no_mutation_and_dtype_agnostic() -> None:
    sig_f = np.array([0.2, 0.4, 0.6], dtype=float)
    sig_i = np.array([0, 1, 2], dtype=int)  # integer inputs should be accepted
    sig_copy = sig_f.copy()
    r1 = pe.compute_robustness(sig_f, 0.0, 1.0)
    r2 = pe.compute_robustness(sig_i, 0, 3)  # map to same spacing
    assert np.array_equal(sig_f, sig_copy)  # input preserved
    # Explicit formula for the second case: min(min(sig)-lower, upper - max(sig))
    expected = min(sig_i.min() - 0, 3 - sig_i.max())
    assert r2 == pytest.approx(expected, abs=TOL)
    assert isinstance(r1, float) and isinstance(r2, float)


def test_compute_robustness_one_sided_bounds() -> None:
    sig = np.array([-1.0, 0.0, 1.0])
    # Upper-only constraint: robustness is upper - max(sig)
    r_upper_only = pe.compute_robustness(sig, lower=-np.inf, upper=1.0)
    assert r_upper_only == pytest.approx(1.0 - sig.max(), abs=TOL)
    # Lower-only constraint: robustness is min(sig) - lower
    r_lower_only = pe.compute_robustness(sig, lower=-1.0, upper=np.inf)
    assert r_lower_only == pytest.approx(sig.min() - (-1.0), abs=TOL)


def test_compute_robustness_propagates_nan() -> None:
    sig = np.array([0.0, np.nan, 1.0])
    r = pe.compute_robustness(sig, lower=0.0, upper=1.0)
    assert np.isnan(r)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([], dtype=float),          # empty
        np.zeros((0,), dtype=float),        # empty 1d
        np.zeros((1, 1), dtype=float),      # not 1d
    ],
)
def test_compute_robustness_invalid_inputs(bad) -> None:
    with pytest.raises(ValueError):
        pe.compute_robustness(bad, lower=0.0, upper=1.0)


def test_compute_robustness_degenerate_interval() -> None:
    sig = np.array([0.2, 0.3, 0.6])
    r = pe.compute_robustness(sig, lower=0.3, upper=0.3)
    # elementwise min margins: [-0.1, 0.0, -0.3] -> global min -0.3
    assert np.isclose(r, -0.3)


# --------------------------- 2D (spatiotemporal) ------------------------------
def test_compute_spatiotemporal_agrees_with_flatten() -> None:
    mat = np.array([[0.5, 0.6, 0.7], [0.2, 0.4, 0.9]])
    lower, u = 0.0, 1.0
    r2d = pe.compute_spatiotemporal_robustness(mat, lower, u)
    r1d = pe.compute_robustness(mat.ravel(), lower, u)
    assert np.isclose(r2d, r1d)


def test_compute_spatiotemporal_typical_and_constant_cases() -> None:
    mat = np.array([[0.5, 0.6], [0.7, 0.8]])
    rob = pe.compute_spatiotemporal_robustness(mat, 0.0, 1.0)
    assert np.isclose(rob, 0.2)

    const = np.full((3, 4), 0.5, dtype=float)
    assert np.isclose(
        pe.compute_spatiotemporal_robustness(const, 0.0, 1.0), 0.5
    )


def test_spatiotemporal_transpose_invariance_and_no_mutation() -> None:
    rng = np.random.default_rng(11)
    mat = rng.normal(size=(3, 5))
    lo, hi = -0.5, 0.75
    mat_copy = mat.copy()
    r = pe.compute_spatiotemporal_robustness(mat, lo, hi)
    r_T = pe.compute_spatiotemporal_robustness(mat.T, lo, hi)
    assert r == pytest.approx(r_T, abs=TOL)
    assert np.array_equal(mat, mat_copy)


def test_spatiotemporal_degenerate_interval() -> None:
    mat = np.array([[0.2, 0.3, 0.6],
                    [0.1, 0.4, 0.9]], dtype=float)
    r = pe.compute_spatiotemporal_robustness(mat, lower=0.3, upper=0.3)
    # elementwise margins -> min([-0.1, 0.0, -0.3, -0.2, 0.1, -0.6]) = -0.6
    assert r == pytest.approx(-0.6, abs=TOL)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([], dtype=float),              # empty
        np.zeros((0, 3), dtype=float),         # one empty dimension
        np.zeros((3, 0), dtype=float),         # the other empty dimension
        np.zeros((2,), dtype=float),           # not 2d
        np.zeros((1, 1, 1), dtype=float),      # not 2d
    ],
)
def test_compute_spatiotemporal_invalid_inputs(bad) -> None:
    with pytest.raises(ValueError):
        pe.compute_spatiotemporal_robustness(bad, lower=0.0, upper=1.0)


def test_spatiotemporal_monotonic_in_bounds() -> None:
    mat = np.array([[0.2, 0.4, 0.6],
                    [0.1, 0.3, 0.5]], dtype=float)
    base = pe.compute_spatiotemporal_robustness(mat, 0.0, 1.0)
    tighter = pe.compute_spatiotemporal_robustness(mat, 0.1, 0.7)
    wider = pe.compute_spatiotemporal_robustness(mat, -1.0, 2.0)
    assert tighter <= base + 1e-12
    assert wider >= base - 1e-12
