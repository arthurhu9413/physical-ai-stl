from __future__ import annotations
"""
physical_ai_stl.frameworks.physicsnemo_hello
============================================

A tiny, **zero‑training** helper around **NVIDIA PhysicsNeMo** that is:

- **Version‑tolerant**: resolves core API entry points across PhysicsNeMo
  releases without importing heavy subpackages at module import time.
- **Fast and CPU‑only**: constructs a micro MLP and performs a single
  forward pass under ``torch.no_grad()``.
- **Optional‑dependency safe**: never imports PhysicsNeMo (or PyTorch) at
  module import time; functions raise clear, actionable errors instead.

This file is deliberately self‑contained so it can be exercised in CI even when
PhysicsNeMo (or GPUs) are unavailable.

References
----------
• PhysicsNeMo repository and *Hello World* usage (see README): GitHub, 2025.  

  https://github.com/NVIDIA/physicsnemo – example uses

  ``from physicsnemo.models.mlp.fully_connected import FullyConnected`` and a

  single forward pass, plus a PDE object such as ``NavierStokes``.  

  (The library was renamed from *Modulus*; imports are now under ``physicsnemo``.)

"""  # See repo README & install instructions.
from importlib import import_module
from typing import Any, Dict, Tuple

# Public constants for clarity in messages and for downstream tooling.
PHYSICSNEMO_DIST_NAME: str = "nvidia-physicsnemo"
PHYSICSNEMO_MODULE_NAME: str = "physicsnemo"


# -----------------------------------------------------------------------------
# Lightweight import helpers
# -----------------------------------------------------------------------------
def _require_physicsnemo() -> Any:
    """Import and return the top‑level :mod:`physicsnemo` module.

    Raises a concise, actionable :class:`ImportError` if the package is not
    available.  This keeps module import cheap and deterministic in environments
    where optional stacks are not installed.
    """
    try:
        return import_module(PHYSICSNEMO_MODULE_NAME)
    except Exception as e:  # pragma: no cover - error path exercised in tests
        raise ImportError(
            "PhysicsNeMo is not installed. Install with: "
            f"pip install {PHYSICSNEMO_DIST_NAME}"
        ) from e


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def physicsnemo_version() -> str:
    """Return PhysicsNeMo version, or ``"unknown"`` if missing.

    Notes
    -----
    This mirrors *hello* helpers for other frameworks so scripts and tests can
    quickly check availability without importing heavy submodules.
    """
    mod = _require_physicsnemo()
    ver = getattr(mod, "__version__", None)
    return ver if isinstance(ver, str) and ver else "unknown"


def physicsnemo_available() -> bool:
    """Return ``True`` if :mod:`physicsnemo` can be imported, ``False`` otherwise."""
    try:
        import_module(PHYSICSNEMO_MODULE_NAME)
        return True
    except Exception:  # pragma: no cover - tiny negative path
        return False


def physicsnemo_smoke(
    batch: int = 128,
    in_features: int = 32,
    out_features: int = 64,
    seed: int = 0,
) -> Dict[str, float | str | Tuple[int, int]]:
    """Run a **minimal, CPU‑only** PhysicsNeMo forward pass.

    Constructs a tiny fully‑connected network using the public API from the
    PhysicsNeMo README and performs a single forward pass on random input.
    Returns a compact metrics dictionary suitable for simple assertions.

    Parameters
    ----------
    batch : int
        Batch size for the dummy input.
    in_features : int
        Input dimensionality of the MLP.
    out_features : int
        Output dimensionality of the MLP.
    seed : int
        Torch seed for reproducibility.

    Returns
    -------
    dict
        ``{"version": str, "out_shape": (batch, out_features)}`` along with
        scalar convenience fields ``"out_batch"`` and ``"out_dim"`` cast to
        ``float`` for JSON serialization.

    Examples
    --------
    >>> metrics = physicsnemo_smoke(batch=4, in_features=3, out_features=2)
    >>> tuple(metrics["out_shape"])  # doctest: +SKIP
    (4, 2)
    """
    # Lazy imports keep module import fast and optional‑dep safe.
    _require_physicsnemo()
    try:
        import torch  # local import (CPU‑only path in tests)
        from physicsnemo.models.mlp.fully_connected import FullyConnected  # type: ignore
    except Exception as e:  # pragma: no cover
        # Helpful error if PyTorch or submodule layout is missing.
        raise ImportError(
            "PhysicsNeMo smoke test requires PyTorch and "
            "`physicsnemo.models.mlp.fully_connected`.\n"
            "Try: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from e

    # Determinism and CPU‑only execution.
    torch.manual_seed(int(seed))
    device = torch.device("cpu")

    # Tiny model + single forward pass.
    model = FullyConnected(in_features=in_features, out_features=out_features).to(device)  # type: ignore[call-arg]
    x = torch.randn(int(batch), int(in_features), device=device)
    with torch.no_grad():
        y = model(x)

    out_shape: Tuple[int, int] = (int(y.shape[0]), int(y.shape[1]))
    metrics: Dict[str, float | str | Tuple[int, int]] = {
        "version": physicsnemo_version(),
        "out_shape": out_shape,
        # Redundant scalars (handy for JSON/metrics dashboards)
        "out_batch": float(out_shape[0]),
        "out_dim": float(out_shape[1]),
    }
    return metrics


def physicsnemo_pde_summary() -> list[str] | None:
    """
    Instantiate a small PDE object (2D Navier–Stokes) from PhysicsNeMo’s
    symbolic PDE module and return a compact textual summary.  This mirrors
    the README’s example `NavierStokes(...).pprint()` usage and is useful for
    quick sanity checks in notebooks/CI.  Returns ``None`` if the optional
    submodule is unavailable.

    Examples
    --------
    >>> info = physicsnemo_pde_summary()  # doctest: +SKIP
    >>> isinstance(info, list)            # doctest: +SKIP
    True
    """
    # Keep this optional and lightweight; not all wheels ship every submodule.
    try:  # pragma: no cover - exercised only when PhysicsNeMo is installed
        from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes  # type: ignore
    except Exception:
        return None

    try:
        ns = NavierStokes(nu=0.01, rho=1, dim=2)
    except Exception:
        return None

    # Prefer using the public `pprint()` to avoid relying on internals;
    # capture its output if available, else fall back to best-effort repr.
    try:
        import io
        from contextlib import redirect_stdout  # local import
        buf = io.StringIO()
        with redirect_stdout(buf):
            ns.pprint()  # prints a few named equations
        lines = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
        return lines[:3] if lines else [ns.__class__.__name__]
    except Exception:
        return [ns.__class__.__name__]

__all__ = [
    "PHYSICSNEMO_DIST_NAME",
    "PHYSICSNEMO_MODULE_NAME",
    "physicsnemo_version",
    "physicsnemo_available",
    "physicsnemo_smoke",
    "physicsnemo_pde_summary",
]
