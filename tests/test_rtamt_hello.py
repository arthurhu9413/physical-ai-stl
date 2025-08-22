# tests/test_rtamt_hello.py
from __future__ import annotations

"""Smoke tests for the optional **RTAMT** dependency.
These focus on a tiny, deterministic example so the project’s CI can
quickly verify that RTAMT is importable and that our thin helpers behave
as expected across RTAMT versions.
"""

import pathlib
import sys
import pytest

# Make the package importable whether or not it has been installed yet.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Skip all tests here immediately if RTAMT isn't available (it's an optional dep).
pytest.importorskip("rtamt", reason="RTAMT not installed; skipping RTAMT tests")


def test_rtamt_hello_returns_expected_scalar() -> None:
    """The canned hello-world monitor returns a scalar robustness of -0.1.

    It encodes the STL formula **G (u ≤ 1.0)** and evaluates it on the
    series ``u = [0.2, 0.4, 1.1]`` with unit sampling. By STL robustness
    semantics this equals ``min_t (1.0 - u(t)) = -0.1``.
    """
    from physical_ai_stl.monitors.rtamt_hello import stl_hello_offline

    rob = stl_hello_offline()
    assert isinstance(rob, (int, float))
    # Numerical correctness for G(u <= 1) on [0.2, 0.4, 1.1].
    assert rob == pytest.approx(-0.1, abs=1e-9)


def test_rtamt_monitor_helpers_match_hello() -> None:
    """Our higher-level helpers reproduce the same robustness value.

    This exercises :func:`stl_always_upper_bound` and :func:`evaluate_series`
    which provide a stable facade over RTAMT’s API across versions.
    """
    from physical_ai_stl.monitoring.rtamt_monitor import (
        stl_always_upper_bound,
        evaluate_series,
    )

    spec = stl_always_upper_bound("u", u_max=1.0)
    rob = evaluate_series(spec, "u", [0.2, 0.4, 1.1], dt=1.0)
    assert isinstance(rob, (int, float))
    assert rob == pytest.approx(-0.1, abs=1e-9)
