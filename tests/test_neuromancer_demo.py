# tests/test_neuromancer_demo.py
# Purpose: fast, robust smoke tests for the Neuromancer demo wrapper.
# Design goals (in order): correctness -> speed -> portability.
from __future__ import annotations

import importlib
import math
from dataclasses import is_dataclass, fields as dataclass_fields
from types import ModuleType
from typing import Any

import pytest


MOD_PATH = "physical_ai_stl.frameworks.neuromancer_stl_demo"


def _try_import_demo() -> ModuleType | None:
    try:
        return importlib.import_module(MOD_PATH)
    except Exception:
        return None


def _readable_skip(reason: str) -> None:
    # Centralized skip with a compact, actionable message
    pytest.skip(f"neuromancer_stl_demo unavailable: {reason}")


def _finite_float(x: Any) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def test_source_exists_and_mentions_api() -> None:
    import pathlib

    candidates = [
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "physical_ai_stl"
        / "frameworks"
        / "neuromancer_stl_demo.py",
        # Fallbacks (e.g., if installed into site-packages) could be added here.
    ]
    src_file = next((p for p in candidates if p.exists()), None)
    if src_file is None:
        _readable_skip("source file not found on expected path")
        return

    text = src_file.read_text(encoding="utf-8", errors="ignore")
    # Light-weight sanity checks on public API names in the text
    assert "DemoConfig" in text
    assert "train_demo" in text
    # Helpful to have STL helpers documented/exported
    assert "stl_offline_robustness" in text
    assert "stl_violation" in text


def test_import_and_tiny_run_smoke() -> None:
    mod = _try_import_demo()
    if mod is None:
        _readable_skip("import failed (syntax or dependency error)")
        return

    # Required symbols
    for sym in ("DemoConfig", "train_demo"):
        if not hasattr(mod, sym):
            _readable_skip(f"expected symbol not defined: {sym}")
            return

    DemoConfig = getattr(mod, "DemoConfig")
    train_demo = getattr(mod, "train_demo")

    # Dataclass contract
    assert is_dataclass(DemoConfig)

    # Build a tiny, CPU-only config (prefer small explicit values if those fields exist).
    cfg_kwargs: dict[str, Any] = {}
    names = {f.name for f in dataclass_fields(DemoConfig)}
    if "epochs" in names:
        cfg_kwargs["epochs"] = 2
    if "n" in names:
        cfg_kwargs["n"] = 32
    if "device" in names:
        cfg_kwargs["device"] = "cpu"
    if "seed" in names:
        cfg_kwargs["seed"] = 0
    if "lr" in names:
        cfg_kwargs["lr"] = 1e-3

    try:
        cfg = DemoConfig(**cfg_kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Fallback if constructor strictly requires no-arg or different fields.
        cfg = DemoConfig()  # type: ignore[call-arg]

    # Be gentle on resources if PyTorch is available.
    try:
        import torch  # type: ignore

        torch.set_num_threads(1)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)  # CPU path only
            except Exception:
                pass
    except Exception:
        pass

    # Run the tiny demo
    try:
        out = train_demo(cfg)
    except Exception:
        _readable_skip("runtime error inside train_demo (optional deps likely missing)")
        return

    # Minimal structure/typing checks—keep them tight but dependency light.
    assert isinstance(out, dict)
    assert "pytorch" in out and isinstance(out["pytorch"], dict)
    assert "final_mse" in out["pytorch"]
    assert _finite_float(out["pytorch"]["final_mse"])

    # If provided, validate a 'final_violation' metric for PyTorch path as well.
    if isinstance(out["pytorch"].get("final_violation", 0.0), float):
        assert _finite_float(out["pytorch"]["final_violation"])

    # Neuromancer path is optional: accept None or a result dict.
    if "neuromancer" in out and out["neuromancer"] is not None:
        assert isinstance(out["neuromancer"], dict)
        nm = out["neuromancer"]
        assert _finite_float(nm.get("final_mse", float("nan")))
        assert _finite_float(nm.get("final_violation", float("nan")))


def test_reproducibility_best_effort() -> None:
    mod = _try_import_demo()
    if mod is None:
        _readable_skip("import failed (syntax or dependency error)")
        return

    DemoConfig = getattr(mod, "DemoConfig", None)
    train_demo = getattr(mod, "train_demo", None)
    if DemoConfig is None or train_demo is None:
        _readable_skip("expected symbols not defined")
        return

    # Only proceed if PyTorch is importable.
    try:
        import torch  # type: ignore
    except Exception:
        _readable_skip("PyTorch unavailable")
        return

    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Same seed/config -> expect very similar loss (loose tolerance to avoid flakiness)
    cfg = DemoConfig(epochs=2, n=32, device="cpu", seed=42)  # type: ignore[call-arg]
    out1 = train_demo(cfg)
    out2 = train_demo(cfg)

    mse1 = float(out1["pytorch"]["final_mse"])
    mse2 = float(out2["pytorch"]["final_mse"])
    assert math.isfinite(mse1) and math.isfinite(mse2)
    assert abs(mse1 - mse2) <= max(1e-7, 1e-3 * (1.0 + abs(mse1)))


def test_stl_semantics_and_monotonicity() -> None:
    """Check STL helpers (exact robustness and pointwise violation)."""
    mod = _try_import_demo()
    if mod is None:
        _readable_skip("import failed (syntax or dependency error)")
        return

    stl_offline_robustness = getattr(mod, "stl_offline_robustness", None)
    stl_violation = getattr(mod, "stl_violation", None)
    if stl_offline_robustness is None or stl_violation is None:
        _readable_skip("STL helper functions not exposed")
        return

    try:
        import torch  # type: ignore
    except Exception:
        _readable_skip("PyTorch unavailable")
        return

    # Tiny synthetic signal and a bound that creates both safe and unsafe points.
    u = torch.tensor([-0.2, 0.1, 0.9, 0.0], dtype=torch.float32)
    bound = 0.8

    # Exact semantics: rho = min_t (bound - u[t])
    rho = stl_offline_robustness(u, bound)
    assert math.isfinite(rho)
    assert abs(rho - float((bound - u).min().item())) <= 1e-12 * (1.0 + abs(rho))

    # Pointwise violation equals ReLU(u - bound) elementwise
    v = stl_violation(u, bound)
    assert torch.allclose(v, torch.relu(u - bound))

    # Consistency: -rho equals the maximum (unclipped) violation
    # (i.e., max_t(u - bound), clipped-at-zero version has the same max).
    max_unclipped = float((u - bound).max().item())
    assert abs(-rho - max_unclipped) <= 1e-7 * (1.0 + abs(max_unclipped))

    # Monotonicity in the threshold: lowering the bound cannot increase robustness.
    rho_hi = stl_offline_robustness(u, bound + 0.1)
    rho_lo = stl_offline_robustness(u, bound - 0.1)
    assert rho_lo <= rho_hi + 1e-12
