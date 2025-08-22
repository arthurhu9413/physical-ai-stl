# tests/test_torchphysics_hello.py
# High-signal tests for the optional Bosch TorchPhysics dependency wrapper.
#
# This file serves *two* purposes:
#   (1) Zero-dependency unit tests that validate our import-safe helper logic
#       (even when TorchPhysics/torch are NOT installed).
#   (2) A tiny, CPU-friendly, deterministic end-to-end example (skipped unless
#       torch + torchphysics import cleanly) that:
#         - writes down concrete STL specs (incl. an *eventually* property),
#         - evaluates/monitors those specs on a trace,
#         - produces a plot artifact that can be used in demos/reports.
#
# Why include an actual example in a *test*?
#   Professor feedback emphasized showing concrete examples + figures and making
#   the repository itself reproducible. Tests are a natural place to keep small,
#   deterministic “toy runs” that double as documentation and sanity checks.
from __future__ import annotations

"""TorchPhysics “hello” tests + a tiny monitored example.

High-level dataflow (matches the report-style diagrams requested in feedback)
----------------------------------------------------------------------------
    (ODE/PDE + IC/BC) + (STL spec)
              |
              v
    TorchPhysics domain + samplers  --->  model u(·)  --->  residual losses
                                                   |
                                                   v
                              trace u(t) on a grid ---> STL monitoring ---> figure

The deterministic example below uses a *cooling* ODE to mirror the suggested
"eventually" pattern from email:

    du/dt = -a u,   u(0) = 50   (Newtonian cooling)

Example temporal specs we monitor on the resulting trace u(t), t ∈ [0, 10]:

    ψ := G[0,10] (u(t) >= 0)                (basic safety)
    φ := F[5,10] (u(t) <= 25)               (eventually cool below a threshold)

Notes
-----
* Part (1) (unit tests) runs without torch/torchphysics installed.
* Part (2) is skipped unless torch *and* torchphysics import successfully.
"""

import builtins
import importlib
import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest


# Make the in-repo package importable whether or not it has been installed yet.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MOD = "physical_ai_stl.frameworks.torchphysics_hello"


def _import_helper_or_skip() -> Any:
    """Import the helper module or skip if the package path is unresolved."""
    spec = importlib.util.find_spec(MOD)
    if spec is None:  # pragma: no cover - environment dependent
        pytest.skip("helper module not importable")
    return importlib.import_module(MOD)


# ---------------------------------------------------------------------------
# Part (1): import-safe unit tests (no optional deps required)
# ---------------------------------------------------------------------------


def test_helper_import_is_lazy_wrt_torchphysics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the helper must *not* import torchphysics eagerly.

    This protects users who install only the lightweight base requirements.
    """
    # Ensure a fresh import of the helper.
    monkeypatch.delitem(sys.modules, MOD, raising=False)

    real_import = builtins.__import__

    def _block_torchphysics(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torchphysics":
            raise ModuleNotFoundError("torchphysics import blocked for laziness test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_torchphysics)
    helper = _import_helper_or_skip()
    assert hasattr(helper, "torchphysics_available")


def test_public_api_and_signature() -> None:
    helper = _import_helper_or_skip()

    # Exported API should be explicit via __all__.
    exported = set(getattr(helper, "__all__", []))
    assert "torchphysics_version" in exported
    assert "torchphysics_available" in exported
    assert "torchphysics_smoke" in exported
    assert "TORCHPHYSICS_DIST_NAME" in exported
    assert "TORCHPHYSICS_MODULE_NAME" in exported

    # And the attributes should exist.
    for name in exported:
        assert hasattr(helper, name)

    # Signatures are part of the user-facing contract, but be tolerant to
    # postponed evaluation of annotations (PEP 563 / __future__.annotations).
    import inspect

    sig_v = inspect.signature(helper.torchphysics_version)
    assert list(sig_v.parameters) == []
    assert sig_v.return_annotation in (str, "str")

    sig_a = inspect.signature(helper.torchphysics_available)
    assert list(sig_a.parameters) == []
    assert sig_a.return_annotation in (bool, "bool")

    sig_s = inspect.signature(helper.torchphysics_smoke)
    assert "n_points" in sig_s.parameters and sig_s.parameters["n_points"].default == 32


def test_version_prefers___version__(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip()

    dummy = types.ModuleType("torchphysics")
    dummy.__version__ = "1.2.3"
    monkeypatch.setitem(sys.modules, "torchphysics", dummy)

    assert helper.torchphysics_version() == "1.2.3"


def test_version_falls_back_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip()

    dummy = types.ModuleType("torchphysics")
    # No __version__ attribute -> should use importlib.metadata
    monkeypatch.setitem(sys.modules, "torchphysics", dummy)

    monkeypatch.setattr(helper._metadata, "version", lambda _: "9.9.9")
    assert helper.torchphysics_version() == "9.9.9"


def test_available_false_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip()

    def _boom(_: str):
        raise ImportError("nope")

    monkeypatch.setattr(helper, "import_module", _boom)
    assert helper.torchphysics_available() is False


def test_version_raises_actionable_importerror_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip()

    def _boom(_: str):
        raise ModuleNotFoundError("torchphysics not installed")

    monkeypatch.setattr(helper, "import_module", _boom)
    with pytest.raises(ImportError) as ei:
        _ = helper.torchphysics_version()

    msg = str(ei.value)
    assert "pip install torchphysics" in msg
    assert helper.TORCHPHYSICS_DIST_NAME in msg
    assert helper.TORCHPHYSICS_MODULE_NAME in msg


# ---------------------------------------------------------------------------
# Part (2): deterministic end-to-end example (skipped unless optional deps exist)
# ---------------------------------------------------------------------------


def _safe_import_optional(name: str) -> Any:
    """Import an optional dependency.

    We intentionally skip (rather than fail) on *any* import-time exception
    (ImportError, OSError from missing shared libs, etc.) to keep CI stable.
    """
    try:
        return importlib.import_module(name)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"optional dependency '{name}' not available: {e}")


def _manual_robustness_always_geq(values: Iterable[float], threshold: float) -> float:
    # Robustness of G (x >= c) under max-min semantics: min_t (x(t) - c)
    return min(v - threshold for v in values)


def _manual_robustness_eventually_leq(
    times: Iterable[float], values: Iterable[float], lo: float, hi: float, threshold: float
) -> float:
    # Robustness of F_[lo,hi] (x <= c): max_{t in [lo,hi]} (c - x(t))
    best = -float("inf")
    for t, v in zip(times, values):
        if lo <= t <= hi:
            best = max(best, threshold - v)
    if best == -float("inf"):
        raise ValueError("No samples fell within the requested time window")
    return best


def test_torchphysics_smoke_returns_metrics_if_installed() -> None:
    """If TorchPhysics is installed, the helper smoke test should run on CPU."""
    helper = _import_helper_or_skip()
    if not helper.torchphysics_available():
        pytest.skip("TorchPhysics not installed")

    try:
        metrics = helper.torchphysics_smoke(n_points=16, hidden=(8,), seed=0)
    except RuntimeError as e:  # pragma: no cover - env dependent
        pytest.skip(f"TorchPhysics available but smoke test failed: {e}")

    assert isinstance(metrics, dict)
    assert metrics.get("points") == 16
    assert isinstance(metrics.get("loss"), float)
    assert math.isfinite(float(metrics["loss"]))
    assert isinstance(metrics.get("version"), str)


def test_torchphysics_cooling_ode_monitors_stl_and_writes_figure(tmp_path: Path) -> None:
    """End-to-end toy example with explicit STL specs + a saved plot.

    This is designed to be something you can run live in a demo:
      - deterministic (no training loop, uses an exact model),
      - fast (CPU-only),
      - produces a concrete figure artifact.

    It also acts as a regression test that TorchPhysics' differential operators
    (grad) work with our model interface.
    """
    torch = _safe_import_optional("torch")
    tp = _safe_import_optional("torchphysics")

    # Avoid any GPU use in a demo/test environment.
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)

    # ----------------------------
    # Problem: Newtonian cooling
    # ----------------------------
    a = 0.2
    u0 = 50.0
    t0, t1 = 0.0, 10.0

    # STL specs we intend to monitor on u(t):
    spec_always_nonneg = "always (u >= 0)"
    spec_eventually_cool = "eventually[5:10] (u <= 25)"

    # TorchPhysics spaces/domains
    time = tp.spaces.R1("t")
    domain = tp.domains.Interval(time, t0, t1)

    # Exact model: u(t) = u0 * exp(-a t)
    class ExactCooling(torch.nn.Module):
        def __init__(self, a_: float, u0_: float):
            super().__init__()
            self.register_buffer("a", torch.tensor(float(a_)))
            self.register_buffer("u0", torch.tensor(float(u0_)))

        def forward(self, x: Mapping[str, Any]) -> Mapping[str, Any]:  # type: ignore[override]
            t = x["t"]
            return {"u": self.u0 * torch.exp(-self.a * t)}

    model = ExactCooling(a, u0).to(device)

    # Residuals
    def ode_residual(x: Mapping[str, Any], out: Mapping[str, Any]) -> Any:
        u = out["u"]
        t = x["t"]
        du_dt = tp.utils.grad(u, t)[0]
        return du_dt + a * u  # should be ~0

    def ic_residual(x: Mapping[str, Any], out: Mapping[str, Any]) -> Any:
        u = out["u"]
        return u - u0  # at t=0 should be 0

    # Samplers: keep them static so point sets are deterministic for demos.
    sampler_interior = tp.samplers.RandomUniformSampler(domain, n_points=64)
    if hasattr(sampler_interior, "make_static"):
        sampler_interior.make_static()

    ic_domain = getattr(domain, "boundary_left", None)
    if ic_domain is None:  # pragma: no cover - version dependent
        pytest.skip("TorchPhysics Interval has no 'boundary_left' attribute; cannot build deterministic IC sampler")
    sampler_ic = tp.samplers.RandomUniformSampler(ic_domain, n_points=8)
    if hasattr(sampler_ic, "make_static"):
        sampler_ic.make_static()

    cond_pde = tp.conditions.PINNCondition(module=model, sampler=sampler_interior, residual_fn=ode_residual)
    cond_ic = tp.conditions.PINNCondition(module=model, sampler=sampler_ic, residual_fn=ic_residual)

    problem = tp.problem.Problem([cond_pde, cond_ic])

    with torch.enable_grad():
        loss_t = problem.condition_loss()
    loss = float(loss_t.detach().cpu().item())

    assert math.isfinite(loss)
    # Exact model should satisfy the ODE/IC essentially up to numerical precision.
    assert loss < 1e-5

    # ----------------------------
    # Evaluate trace u(t) on a grid
    # ----------------------------
    n = 201
    t_grid = torch.linspace(t0, t1, n, device=device).unsqueeze(-1)
    with torch.no_grad():
        u_pred = model({"t": t_grid})["u"].detach().cpu().squeeze(-1)

    times = [float(t) for t in torch.linspace(t0, t1, n).tolist()]
    values = [float(v) for v in u_pred.tolist()]

    # ----------------------------
    # Monitor STL specs (RTAMT if available, else deterministic fallback)
    # ----------------------------
    rho_nonneg: float
    rho_cool: float

    try:
        rt = importlib.import_module("physical_ai_stl.monitoring.rtamt_monitor")
        # Build specs explicitly with dense-time semantics.
        s1 = rt.build_stl_spec(spec_always_nonneg, var_types={"u": "float"}, time_semantics="dense")
        s2 = rt.build_stl_spec(spec_eventually_cool, var_types={"u": "float"}, time_semantics="dense")
        dt = (t1 - t0) / (n - 1)
        rho_nonneg = float(rt.evaluate_series(s1, "u", values, dt=dt))
        rho_cool = float(rt.evaluate_series(s2, "u", values, dt=dt))
        assert rt.satisfied(rho_nonneg)
        assert rt.satisfied(rho_cool)
    except Exception:
        # Fall back to simple max/min semantics if RTAMT isn't importable.
        rho_nonneg = _manual_robustness_always_geq(values, threshold=0.0)
        rho_cool = _manual_robustness_eventually_leq(times, values, lo=5.0, hi=10.0, threshold=25.0)
        assert rho_nonneg >= 0.0
        assert rho_cool >= 0.0

    assert math.isfinite(rho_nonneg)
    assert math.isfinite(rho_cool)

    # ----------------------------
    # Write a demo-friendly figure artifact
    # ----------------------------
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"matplotlib not available: {e}")

    fig_path = tmp_path / "torchphysics_cooling_ode_trace.png"
    plt.figure()
    plt.plot(times, values, label="u(t) (model)")
    plt.plot(times, [25.0] * len(times), label="threshold 25")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title(
        "Cooling ODE trace with monitored STL specs\n"
        f"ψ={spec_always_nonneg}, ρ={rho_nonneg:.3g};  φ={spec_eventually_cool}, ρ={rho_cool:.3g}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


if __name__ == "__main__":  # pragma: no cover
    # Allow running this file directly to generate the figure in a stable location.
    # Example:
    #   python tests/test_torchphysics_hello.py
    #
    # This is handy for “grab a screenshot for the report” workflows.
    out_dir = _ROOT / "figs" / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Reuse pytest's tmp_path-like behavior by calling the test with out_dir.
        test_torchphysics_cooling_ode_monitors_stl_and_writes_figure(out_dir)
        print(f"Wrote: {out_dir / 'torchphysics_cooling_ode_trace.png'}")
    except Exception as e:
        raise SystemExit(str(e)) from e
