# src/physical_ai_stl/datasets/stlnet_synthetic.py
"""
Synthetic 1‑D trace used to mirror the tiny “synthetic” example from STLnet.

Design goals
------------
- **Deterministic** by default: if the user seeds NumPy's global RNG
  (``np.random.seed(s)``), two instances created with the same seed and
  parameters produce *bit‑exactly identical* tuples for list equality tests.
- **Numerically stable**: the time grid includes exact 0 and 1 endpoints
  for ``length > 1`` and uses rounding to 15 decimal places on the noisy
  value to avoid 1‑ulp drift across platforms.
- **Tiny and fast**: everything is vectorized, pure NumPy, and CPU‑only.
- **Self‑contained STL hooks**: provide a minimal, dependency‑free
  robustness evaluator for bounded‑future atomic STL formulas so unit tests
  (and simple demos) do not need RTAMT.

Public API
----------
- :class:`SyntheticSTLNetDataset`
- :class:`BoundedAtomicSpec`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Array helpers
# --------------------------------------------------------------------------- #
def _sliding_window(x: np.ndarray, window: int) -> np.ndarray:
    """
    Return a view of 1‑D array ``x`` as overlapping windows of length ``window``.

    Parameters
    ----------
    x:
        1‑D NumPy array.
    window:
        Strictly positive integer window size.

    Returns
    -------
    np.ndarray
        A 2‑D array of shape ``(n - window + 1, window)`` when ``window <= n``.
        If ``window > n`` an empty array of shape ``(0, window)`` is returned.

    Notes
    -----
    Uses ``numpy.lib.stride_tricks.sliding_window_view`` when available and
    falls back to a safe advanced‑indexing implementation otherwise.
    """
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}")
    x = np.asarray(x)
    n = int(x.shape[0])
    if window > n:
        # Empty view (no windows fit); keep a consistent 2‑D shape.
        return np.empty((0, window), dtype=x.dtype)

    # Prefer stride_tricks when present (NumPy >= 1.20).
    try:  # pragma: no cover - feature detection branch
        from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
        view = sliding_window_view(x, window_shape=window)
        # sliding_window_view returns shape (n - window + 1, window) for 1‑D x
        return view
    except Exception:
        # Fallback: stack whole windows; fast enough for our tiny default sizes.
        starts = np.arange(0, n - window + 1, dtype=int)
        idx = starts[:, None] + np.arange(window, dtype=int)[None, :]
        return x[idx]


# --------------------------------------------------------------------------- #
# Minimal STL machinery
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoundedAtomicSpec:
    """
    A tiny subset of STL sufficient for unit tests and demos:

    ``G_[0,H] (v <= c)`` or ``F_[0,H] (v >= c)`` for scalar signal ``v(t)``.

    Attributes
    ----------
    temporal:
        Either ``"always"`` for :math:`G` or ``"eventually"`` for :math:`F`.
    op:
        Either ``"<="`` or ``">="``.
    threshold:
        The scalar threshold ``c`` appearing in the atomic predicate.
    horizon:
        Non‑negative integer time horizon ``H`` measured in *samples*.
    """

    temporal: str  # "always" | "eventually"
    op: str        # "<=" | ">="
    threshold: float
    horizon: int = 0

    def __post_init__(self) -> None:
        temporal_ok = self.temporal in {"always", "eventually"}
        op_ok = self.op in {"<=", ">="}
        if not temporal_ok:
            raise ValueError(f"temporal must be 'always' or 'eventually'; got {self.temporal!r}")
        if not op_ok:
            raise ValueError(f"op must be '<=' or '>='; got {self.op!r}")
        if int(self.horizon) != self.horizon or self.horizon < 0:
            raise ValueError(f"horizon must be a non‑negative integer; got {self.horizon!r}")

    # Public API ---------------------------------------------------------------

    def robustness(self, v: np.ndarray, stride: int = 1) -> np.ndarray:
        """
        Evaluate robust semantics of the bounded formula on a discrete signal.

        Parameters
        ----------
        v:
            1‑D array of scalar samples ``v[0], …, v[n-1]``.
        stride:
            Keep one robustness value every ``stride`` windows (>= 1).

        Returns
        -------
        np.ndarray
            1‑D array of robustness values for each window.
        """
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.ndim != 1:
            raise ValueError("v must be 1‑D (scalar signal).")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")

        H = int(self.horizon)
        window = H + 1
        Wins = _sliding_window(v, window)
        if Wins.size == 0:
            return np.empty((0,), dtype=float)
        Wins = Wins[::stride, :]

        # Atomic robustness
        if self.op == "<=":
            r = self.threshold - Wins
        else:  # self.op == ">="
            r = Wins - self.threshold

        # Temporal reduction
        if self.temporal == "always":
            rho = np.min(r, axis=1)
        else:  # "eventually"
            rho = np.max(r, axis=1)
        return rho

    def satisfied(self, v: np.ndarray, stride: int = 1) -> np.ndarray:
        """Boolean satisfaction from robustness (strictly positive means True)."""
        return self.robustness(v, stride=stride) > 0.0


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #
class SyntheticSTLNetDataset:
    """
    A compact, NumPy‑only synthetic trace:

    .. math:: v(t) = 0.5(\\sin(2\\pi t) + 1) + \\sigma\\,\\varepsilon(t)

    where ``t`` is a uniform grid on ``[0, 1]`` of length ``n`` and
    ``\\varepsilon ~ N(0, 1)``. With ``noise=0`` the signal is exactly
    in ``[0, 1]`` and hits :math:`0.5, 1, 0.5, 0, 0.5` at the expected
    quarter points for ``n = 33`` (mirroring STLnet tests).

    Parameters
    ----------
    length:
        Number of samples ``n`` (>= 0). When ``n = 1`` the single time stamp
        is exactly ``0.0``.
    noise:
        Standard deviation :math:`\\sigma` of i.i.d. Gaussian noise. Must be
        non‑negative. Default ``0.05``.
    rng:
        Optional NumPy‐style RNG (``Generator`` or ``RandomState``). When
        ``None`` (default), draws come from the global NumPy RNG and therefore
        respect ``np.random.seed(s)``.

    Notes
    -----
    To avoid spurious equality failures in tests on different platforms, the
    noisy value is rounded to 15 decimal places (unit‑in‑last‑place stability)
    using ``np.round``. This does *not* affect analytical checks which use
    ``math.isclose`` elsewhere.
    """

    __slots__ = ("_data",)

    def __init__(self, length: int = 100, noise: float = 0.05, rng: object | None = None) -> None:
        if not isinstance(length, (int, np.integer)):
            raise TypeError(f"length must be an integer; got {type(length).__name__}")
        if length < 0:
            raise ValueError(f"length must be non‑negative; got {length}")
        if noise < 0:
            raise ValueError(f"noise must be non‑negative; got {noise}")

        n = int(length)

        # Time axis (linspace) – exact 0 and 1 endpoints when n>1.
        if n == 0:
            t = np.empty((0,), dtype=float)
        elif n == 1:
            t = np.array([0.0], dtype=float)
        else:
            # linspace chosen for uniform spacing and exact endpoints
            t = np.linspace(0.0, 1.0, num=n, dtype=float)

        # Clean, exactly bounded sinusoid on [0, 1].
        clean = 0.5 * (np.sin(2.0 * np.pi * t) + 1.0)

        if n == 0:
            v = clean  # empty
        elif noise == 0.0:
            v = clean
        else:
            # Draw noise from the requested RNG without disturbing global state unless asked.
            if rng is None:
                eps = np.random.randn(n)  # respects external np.random.seed
            else:
                # Support both Generator.standard_normal and RandomState.randn
                if hasattr(rng, "standard_normal"):
                    eps = rng.standard_normal(n)  # type: ignore[attr-defined]
                elif hasattr(rng, "randn"):
                    eps = rng.randn(n)  # type: ignore[attr-defined]
                else:
                    raise TypeError("rng must be a NumPy Generator or RandomState‑like object.")
            v = clean + float(noise) * eps
            # Stabilize bit‑exact reproducibility across platforms.
            v = np.round(v, decimals=15)

        self._data = np.stack((t, v), axis=1) if n > 0 else np.empty((0, 2), dtype=float)

    # Sequence protocol --------------------------------------------------------

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self._data.shape[0])

    def __getitem__(self, idx: int) -> tuple[float, float]:
        # NumPy handles negative/overflow checks.
        t, v = self._data[idx]
        # Cast to Python floats for clean repr and deterministic equality semantics.
        return float(t), float(v)

    # Convenience accessors ----------------------------------------------------

    @property
    def t(self) -> np.ndarray:
        """Time stamps as a **view** (1‑D float array)."""
        return self._data[:, 0]

    @property
    def v(self) -> np.ndarray:
        """Values as a **view** (1‑D float array)."""
        return self._data[:, 1]

    # Window helpers -----------------------------------------------------------

    def windows(self, length: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Return overlapping windows of time and value with the same slicing.

        Parameters
        ----------
        length:
            Number of consecutive samples per window (>= 1).
        stride:
            Keep one out of every ``stride`` windows (>= 1).

        Returns
        -------
        (t_win, v_win):
            Two 2‑D arrays of shape ``(num_windows, length)``.
        """
        if int(length) != length or length <= 0:
            raise ValueError(f"length must be a positive integer; got {length!r}")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")
        t_win = _sliding_window(self.t, int(length))
        v_win = _sliding_window(self.v, int(length))
        return t_win[::stride, :], v_win[::stride, :]

    def windowed_robustness(
        self,
        spec: BoundedAtomicSpec,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience wrapper that pairs sliding windows with robustness values.

        Returns
        -------
        (t_win, v_win, rho):
            Time/value windows and the robustness of ``spec`` on each window.
        """
        H = int(spec.horizon)
        t_win, v_win = self.windows(H + 1, stride=stride)
        rho = spec.robustness(self.v, stride=stride)
        return t_win, v_win, rho


__all__ = ["SyntheticSTLNetDataset", "BoundedAtomicSpec"]
