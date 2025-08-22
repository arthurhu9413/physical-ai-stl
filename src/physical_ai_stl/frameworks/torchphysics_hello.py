# ruff: noqa: I001
from __future__ import annotations
"""
physical_ai_stl.frameworks.torchphysics_hello
=============================================

Tiny, import‑safe helpers around **Bosch TorchPhysics** — a PyTorch
library for mesh‑free physics‑ML (PINNs, DeepRitz, FNO, …).

What this file provides
-----------------------
- **Distribution and module constants** so downstream code can
  construct consistent user messages.
- ``torchphysics_version()`` – robustly report the installed version or
  raise a clear ``ImportError`` with an install hint.
- ``torchphysics_available()`` – quick boolean check, *no heavy import* at
  module import time.
- ``torchphysics_smoke()`` – a **zero‑training**, CPU‑only micro demo:
  builds a 1D domain, a tiny FCN and evaluates a single PINN residual
  (\\(\\partial_x u\\)) via TorchPhysics’ ``PINNCondition``. This keeps CI fast
  while exercising real APIs (spaces → domains → sampler → model → condition).

Design notes
------------
- All heavy imports are **lazy** inside functions (mirrors our Neuromancer
  and PhysicsNeMo helpers).
- The smoke test purposefully avoids PyTorch Lightning / Solvers; it just
  runs one forward call on a small sampler.

References
----------
• TorchPhysics docs (overview, tutorials, API): boschresearch.github.io/torchphysics
  – see *PINNs tutorial* for residuals and conditions, and the Spaces/Domains
    API for ``R1('x')``, ``Interval``, and ``RandomUniformSampler``.
"""

from importlib import import_module, metadata as _metadata
from typing import Any, Dict, Tuple

# Public constants for clarity in messages and for downstream tooling.
TORCHPHYSICS_DIST_NAME: str = "torchphysics"
TORCHPHYSICS_MODULE_NAME: str = "torchphysics"


# ----- Lightweight import/availability helpers --------------------------------
def _require_torchphysics() -> Any:
    """Import TorchPhysics or raise an actionable error.

    We do **not** import TorchPhysics at module import time so that this file
    stays usable in environments without the optional dependency installed.
    """
    try:
        return import_module(TORCHPHYSICS_MODULE_NAME)
    except Exception as e:  # pragma: no cover - exercised in tests
        raise ImportError(
            "TorchPhysics is not installed. Install the PyPI package "
            f"`{TORCHPHYSICS_DIST_NAME}` (module `{TORCHPHYSICS_MODULE_NAME}`).\n"
            "Example:  pip install torchphysics"
        ) from e


def torchphysics_version() -> str:
    """Return the installed TorchPhysics version string.

    Preference order:
    1) ``torchphysics.__version__`` if present and truthy
    2) ``importlib.metadata.version('torchphysics')``
    3) Fallback to ``'unknown'`` (should not happen on normal installs)

    Raises
    ------
    ImportError
        If TorchPhysics is not importable.
    """
    mod = _require_torchphysics()
    # Prefer module-local __version__ when available.
    ver = getattr(mod, "__version__", None)
    if isinstance(ver, str) and ver.strip():
        return ver
    # Fallback: distribution metadata via PyPI name.
    try:
        return _metadata.version(TORCHPHYSICS_DIST_NAME)
    except Exception:
        return "unknown"


def torchphysics_available() -> bool:
    """Return ``True`` if the TorchPhysics module can be imported, else ``False``."""
    try:
        import_module(TORCHPHYSICS_MODULE_NAME)
        return True
    except Exception:
        return False


# ----- Minimal CPU‑only smoke demo --------------------------------------------
def torchphysics_smoke(n_points: int = 32, hidden: Tuple[int, ...] = (8, 8), seed: int = 0
                       ) -> Dict[str, float | str | Tuple[int, ...]]:
    """Run a **tiny, zero‑training** TorchPhysics pipeline on CPU.

    The demo constructs:
      • a 1D input space ``R1('x')`` and scalar output space ``R1('u')``,
      • an interval domain ``[0, 1]`` with a random‑uniform sampler,
      • a small fully‑connected network (``tp.models.FCN``),
      • a single **PINN condition** whose residual is the spatial gradient
        :math:`\\partial_x u`.  We then evaluate the condition **once** to
        obtain a scalar loss (no optimization).

    Parameters
    ----------
    n_points : int
        Number of collocation points sampled from the interval.
    hidden : tuple[int, ...]
        Hidden layer sizes for the FCN (kept deliberately small).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Minimal metrics useful for sanity checks:
        ``{"version": str, "loss": float, "points": int, "hidden": tuple}``

    Examples
    --------
    >>> # doctest: +SKIP
    >>> metrics = torchphysics_smoke(n_points=16, hidden=(8,), seed=1)
    >>> isinstance(metrics["loss"], float)
    True
    """
    _require_torchphysics()
    try:
        # Local imports (CPU‑only path in tests/CI)
        import torch  # type: ignore
        import torchphysics as tp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "TorchPhysics smoke test requires PyTorch and the `torchphysics` package.\n"
            "Try: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "     pip install torchphysics"
        ) from e

    # Determinism and CPU‑only execution.
    torch.manual_seed(int(seed))
    device = torch.device("cpu")

    # 1) Define spaces and a 1D interval domain.
    X = tp.spaces.R1("x")
    U = tp.spaces.R1("u")
    interval = tp.domains.Interval(space=X, lower_bound=0.0, upper_bound=1.0)

    # 2) Tiny sampler in the domain (kept small for speed).
    sampler = tp.samplers.RandomUniformSampler(domain=interval, n_points=int(n_points))

    # 3) Small FCN model mapping x -> u.
    model = tp.models.FCN(input_space=X, output_space=U, hidden=tuple(int(h) for h in hidden))

    # 4) Residual for a simple PDE-like condition: enforce du/dx ≈ 0 (constant field).
    #    Using TorchPhysics' differentiable operator utilities.
    def residual_du_dx(u: Any, x: Any) -> Any:
        return tp.utils.grad(u, x)

    condition = tp.conditions.PINNCondition(
        module=model,
        sampler=sampler,
        residual_fn=residual_du_dx,
    )

    # Single forward evaluation on CPU (enable input gradients for operators).
    # The Condition API returns a scalar tensor loss; take a plain float.
    with torch.enable_grad():
        loss_t = condition.forward(device=str(device))  # type: ignore[call-arg]
    loss = float(loss_t.detach().cpu().item())

    metrics: Dict[str, float | str | Tuple[int, ...]] = {
        "version": torchphysics_version(),
        "loss": loss,
        "points": int(n_points),
        "hidden": tuple(int(h) for h in hidden),
    }
    return metrics


__all__ = [
    "TORCHPHYSICS_DIST_NAME",
    "TORCHPHYSICS_MODULE_NAME",
    "torchphysics_version",
    "torchphysics_available",
    "torchphysics_smoke",
]
