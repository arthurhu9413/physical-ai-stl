from __future__ import annotations

"""Unit tests for the STLnet-inspired synthetic dataset utilities.

The tests in this file deliberately stay NumPy-only (no torch) so they can run
quickly on CPU and serve as executable documentation.

Covered:
- ``SyntheticSTLNetDataset`` time grid + noiseless landmarks
- Noise handling and deterministic RNG behavior (Generator / RandomState / global)
- Sliding-window semantics and bounded robust STL semantics via ``BoundedAtomicSpec``

These tests are written to be deterministic, lightweight, and network-free.
"""

# Ensure the in-repo package is importable without installing the wheel.
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import copy
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import physical_ai_stl.datasets as dshub
from physical_ai_stl.datasets import (
    BoundedAtomicSpec,
    DatasetInfo,
    SyntheticSTLNetDataset,
    create_dataset,
    get_dataset_cls,
    register_dataset,
)

# Tight but robust absolute tolerance for analytical landmark checks.
EPS = 1e-12


def _isclose(a: float, b: float, tol: float = EPS) -> bool:
    """Absolute-only float comparison (stable across platforms)."""
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _clean_value(t: float) -> float:
    """Noiseless reference signal used by ``SyntheticSTLNetDataset``."""
    return 0.5 * (math.sin(2.0 * math.pi * t) + 1.0)


def _manual_robustness(
    v: np.ndarray,
    *,
    temporal: str,
    op: str,
    threshold: float,
    horizon: int,
    stride: int = 1,
) -> np.ndarray:
    """Reference implementation of the robust semantics used for testing.

    The logic subset matches ``BoundedAtomicSpec``:

    - temporal in {"always", "eventually"}
    - op in {"<=", ">="}
    - horizon H is measured in *samples*, so window length is H+1

    Returns
    -------
    np.ndarray
        Robustness per window, after applying stride.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    if int(horizon) != horizon or horizon < 0:
        raise ValueError("horizon must be a non-negative int")
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    window = int(horizon) + 1
    if window <= 0:
        raise ValueError("window must be >= 1")
    if v.size < window:
        return np.empty((0,), dtype=float)

    wins = np.stack([v[i : i + window] for i in range(v.size - window + 1)], axis=0)
    wins = wins[::stride]

    if op == "<=":
        r = threshold - wins
    elif op == ">=":
        r = wins - threshold
    else:
        raise ValueError("op must be '<=' or '>='")

    if temporal == "always":
        return np.min(r, axis=1)
    if temporal == "eventually":
        return np.max(r, axis=1)
    raise ValueError("temporal must be 'always' or 'eventually'")


@pytest.mark.parametrize("n", [0, 1, 2, 5, 33])
def test_dataset_len_types_and_vector_views(n: int) -> None:
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    assert len(ds) == n

    # ``t`` / ``v`` are 1-D float arrays of length n.
    assert isinstance(ds.t, np.ndarray)
    assert isinstance(ds.v, np.ndarray)
    assert ds.t.shape == (n,)
    assert ds.v.shape == (n,)
    assert np.issubdtype(ds.t.dtype, np.floating)
    assert np.issubdtype(ds.v.dtype, np.floating)

    if n == 0:
        with pytest.raises(IndexError):
            _ = ds[0]
        return

    # __getitem__ returns a 2-tuple of Python floats.
    t0, v0 = ds[0]
    assert isinstance(t0, float)
    assert isinstance(v0, float)

    # Vector views match scalar indexing.
    assert np.allclose(ds.t, [ds[i][0] for i in range(n)])
    assert np.allclose(ds.v, [ds[i][1] for i in range(n)])


@pytest.mark.parametrize("n", [1, 2, 5, 33])
def test_time_grid_endpoints_monotonic_and_uniform(n: int) -> None:
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)

    if n == 1:
        assert _isclose(ds.t[0], 0.0)
        return

    assert _isclose(ds.t[0], 0.0)
    assert _isclose(ds.t[-1], 1.0)

    diffs = np.diff(ds.t)
    assert np.all(diffs > 0.0)
    step = 1.0 / (n - 1)
    assert np.allclose(diffs, step, atol=EPS, rtol=0.0)


def test_sequence_indexing_semantics_and_bounds() -> None:
    n = 5
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)

    assert ds[-1] == ds[n - 1]
    assert ds[-n] == ds[0]

    with pytest.raises(IndexError):
        _ = ds[n]
    with pytest.raises(IndexError):
        _ = ds[-(n + 1)]


def test_length_one_semantics() -> None:
    ds = SyntheticSTLNetDataset(length=1, noise=0.0)
    t, v = ds[0]
    assert _isclose(t, 0.0)
    assert _isclose(v, 0.5)


def test_noiseless_quarter_point_landmarks_and_bounds_n33() -> None:
    # n=33 includes t=0.25, 0.50, 0.75 exactly (i/(n-1) with n-1=32)
    ds = SyntheticSTLNetDataset(length=33, noise=0.0)

    idxs = [0, 8, 16, 24, 32]
    expected = [0.5, 1.0, 0.5, 0.0, 0.5]
    for i, e in zip(idxs, expected):
        assert _isclose(ds.v[i], e), (i, ds.t[i], ds.v[i], e)

    assert float(ds.v.min()) >= -EPS
    assert float(ds.v.max()) <= 1.0 + EPS


@pytest.mark.parametrize("n", [2, 5, 33])
def test_noiseless_values_match_closed_form(n: int) -> None:
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    for i in range(n):
        t, v = ds[i]
        assert _isclose(v, _clean_value(t))


def test_invalid_dataset_parameters_raise() -> None:
    with pytest.raises((TypeError, ValueError)):
        _ = SyntheticSTLNetDataset(length=-1, noise=0.0)
    with pytest.raises((TypeError, ValueError)):
        _ = SyntheticSTLNetDataset(length=3.7, noise=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ = SyntheticSTLNetDataset(length=8, noise=-0.1)


def test_rng_type_validation() -> None:
    class BadRNG:
        pass

    with pytest.raises(TypeError):
        _ = SyntheticSTLNetDataset(length=8, noise=0.1, rng=BadRNG())  # type: ignore[arg-type]


def test_no_nan_or_inf_even_with_noise() -> None:
    ds = SyntheticSTLNetDataset(length=33, noise=0.3, rng=np.random.default_rng(7))
    assert np.isfinite(ds.v).all()


def test_rng_reproducibility_and_linear_noise_scaling_generator() -> None:
    length = 16
    noise_a = 0.1
    noise_b = 0.2

    g = np.random.default_rng(2024)
    state = copy.deepcopy(g.bit_generator.state)

    a = SyntheticSTLNetDataset(length=length, noise=noise_a, rng=g)
    # Reset a new generator to *the same* state so it draws identical eps.
    g2 = np.random.default_rng()
    g2.bit_generator.state = state
    b = SyntheticSTLNetDataset(length=length, noise=noise_b, rng=g2)

    # Same times, residuals scale ~linearly with the noise amplitude.
    for i in range(length):
        t, va = a[i]
        _, vb = b[i]
        clean = _clean_value(t)
        ra = va - clean
        rb = vb - clean
        if abs(ra) > 1e-15:  # avoid 0/0 on exact zeros
            assert _isclose(rb, (noise_b / noise_a) * ra, tol=1e-11)


def test_rng_reproducibility_randomstate() -> None:
    L = 10
    r1 = np.random.RandomState(12345)
    r2 = np.random.RandomState(12345)

    d1 = SyntheticSTLNetDataset(length=L, noise=0.5, rng=r1)
    d2 = SyntheticSTLNetDataset(length=L, noise=0.5, rng=r2)

    assert [d1[i] for i in range(L)] == [d2[i] for i in range(L)]


def test_global_numpy_seed_reproducibility_is_restored() -> None:
    # rng=None uses the global NumPy RNG. We test it, but we MUST restore
    # state to avoid cross-test interference.
    state = np.random.get_state()
    try:
        np.random.seed(2024)
        a = SyntheticSTLNetDataset(length=12, noise=0.2)
        np.random.seed(2024)
        b = SyntheticSTLNetDataset(length=12, noise=0.2)
        assert [a[i] for i in range(len(a))] == [b[i] for i in range(len(b))]
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize(
    "n,win,stride,expected_windows",
    [
        (9, 5, 2, 3),   # raw windows=5 -> keep 0,2,4
        (10, 3, 3, 3),  # raw windows=8 -> keep 0,3,6
        (5, 5, 1, 1),
        (4, 5, 1, 0),   # window longer than trace -> empty
    ],
)
def test_windows_shape_stride_and_alignment(n: int, win: int, stride: int, expected_windows: int) -> None:
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    t_win, v_win = ds.windows(length=win, stride=stride)

    assert t_win.shape == v_win.shape == (expected_windows, win)
    assert np.issubdtype(t_win.dtype, np.floating)
    assert np.issubdtype(v_win.dtype, np.floating)

    if expected_windows:
        # The first window is always the first `win` samples.
        assert np.allclose(t_win[0], ds.t[:win])
        assert np.allclose(v_win[0], ds.v[:win])


def test_windows_invalid_parameters_raise() -> None:
    ds = SyntheticSTLNetDataset(length=8, noise=0.0)
    with pytest.raises(ValueError):
        _ = ds.windows(length=0)
    with pytest.raises(ValueError):
        _ = ds.windows(length=-1)
    with pytest.raises(ValueError):
        _ = ds.windows(length=3.7)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ = ds.windows(length=3, stride=0)


def test_bounded_atomic_spec_validation_and_frozenness() -> None:
    spec = BoundedAtomicSpec(temporal="always", op="<=", threshold=0.0, horizon=0)

    # Frozen dataclass (immutability is helpful for configs and reproducibility).
    with pytest.raises(FrozenInstanceError):
        spec.threshold = 1.0  # type: ignore[misc]

    # Basic validation.
    with pytest.raises(ValueError):
        _ = BoundedAtomicSpec(temporal="G", op="<=", threshold=0.0, horizon=0)
    with pytest.raises(ValueError):
        _ = BoundedAtomicSpec(temporal="always", op="<", threshold=0.0, horizon=0)
    with pytest.raises(ValueError):
        _ = BoundedAtomicSpec(temporal="always", op="<=", threshold=0.0, horizon=-1)


def test_bounded_atomic_spec_robustness_matches_manual_reference() -> None:
    v = np.array([0.0, 0.5, 1.0, 0.5], dtype=float)

    for temporal in ("always", "eventually"):
        for op in ("<=", ">="):
            spec = BoundedAtomicSpec(temporal=temporal, op=op, threshold=0.6, horizon=2)
            rho = spec.robustness(v, stride=1)
            ref = _manual_robustness(v, temporal=temporal, op=op, threshold=0.6, horizon=2, stride=1)
            assert rho.shape == ref.shape
            assert np.allclose(rho, ref, atol=EPS, rtol=0.0)

    # Stride should simply sub-sample the windowed robustness.
    spec2 = BoundedAtomicSpec(temporal="always", op="<=", threshold=0.6, horizon=1)
    rho2 = spec2.robustness(v, stride=2)
    ref2 = _manual_robustness(v, temporal="always", op="<=", threshold=0.6, horizon=1, stride=2)
    assert np.allclose(rho2, ref2, atol=EPS, rtol=0.0)


def test_bounded_atomic_spec_empty_when_horizon_exceeds_trace() -> None:
    v = np.array([0.0, 1.0], dtype=float)
    spec = BoundedAtomicSpec(temporal="always", op="<=", threshold=1.0, horizon=10)
    rho = spec.robustness(v)
    assert rho.shape == (0,)


def test_satisfied_is_strictly_positive() -> None:
    # Important (and easy to miss): ``satisfied`` uses > 0, not >= 0.
    # This makes equality-with-threshold return False.
    v = np.array([0.0, 1.0], dtype=float)

    spec_eq = BoundedAtomicSpec(temporal="always", op="<=", threshold=1.0, horizon=1)
    rho_eq = spec_eq.robustness(v)
    assert np.all(rho_eq >= -EPS)
    assert not np.any(spec_eq.satisfied(v))

    spec_loose = BoundedAtomicSpec(temporal="always", op="<=", threshold=1.0 + 1e-3, horizon=1)
    assert np.all(spec_loose.satisfied(v))


def test_windowed_robustness_matches_windows_and_manual_reference() -> None:
    ds = SyntheticSTLNetDataset(length=9, noise=0.0)
    spec = BoundedAtomicSpec(temporal="always", op="<=", threshold=1.0, horizon=4)

    t_win, v_win = ds.windows(length=spec.horizon + 1, stride=2)
    t2, v2, rho = ds.windowed_robustness(spec, stride=2)

    assert np.allclose(t2, t_win)
    assert np.allclose(v2, v_win)

    ref = _manual_robustness(ds.v, temporal="always", op="<=", threshold=1.0, horizon=4, stride=2)
    assert np.allclose(rho, ref, atol=EPS, rtol=0.0)


def test_dataset_registry_round_trip_and_canonicalization() -> None:
    # The dataset hub supports name-based lookup with case/underscore/dash
    # insensitivity via a tiny canonicalization function.
    info = DatasetInfo(
        name="UNIT-STLNET_SYNTH",
        target=".stlnet_synthetic:SyntheticSTLNetDataset",
        summary="Unit-test registration for canonicalization checks.",
        tags=("unit-tests",),
    )

    # Snapshot/restore to avoid leaking state across a larger test suite.
    before = dict(dshub._REGISTRY)  # type: ignore[attr-defined]
    try:
        register_dataset(info)

        for key in ("unit_stlnet_synth", "UNIT-STLNET-SYNTH", "UnitStlNetSynth"):
            cls = get_dataset_cls(key)
            assert cls is SyntheticSTLNetDataset

            ds = create_dataset(key, length=3, noise=0.0)
            assert isinstance(ds, SyntheticSTLNetDataset)
            assert len(ds) == 3
    finally:
        dshub._REGISTRY.clear()  # type: ignore[attr-defined]
        dshub._REGISTRY.update(before)  # type: ignore[attr-defined]
