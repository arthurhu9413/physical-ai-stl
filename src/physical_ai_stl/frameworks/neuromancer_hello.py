from __future__ import annotations
"""
physical_ai_stl.frameworks.neuromancer_hello
============================================

A tiny, **zero‑training** "hello" helper for NeuroMANCER that is:

- **Version‑tolerant**: resolves core API entry points across multiple
  NeuroMANCER releases without importing heavy subpackages.
- **Fast and CPU‑only**: constructs a one‑node symbolic problem and evaluates
  a single convex objective under ``torch.no_grad()``.
- **Optional‑dependency safe**: never imports NeuroMANCER (or PyTorch) at
  module import time; functions raise clear, actionable errors instead.

This file is deliberately self‑contained so it can be exercised in CI even when
NeuroMANCER (or GPUs) are unavailable.
"""

from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec as _find_spec
from typing import Any


def _import_neuromancer() -> Any:
    """Import and return the ``neuromancer`` top‑level module.

    Notes
    -----
    We keep this import *inside* a function so importing this helper module
    is cheap and does not drag heavy optional dependencies into memory.
    """
    try:
        return import_module("neuromancer")
    except Exception as e:  # pragma: no cover
        # Keep the message compact and actionable for users running the helpers
        # outside of the full extra requirements.
        raise ImportError("neuromancer not installed. Try: pip install neuromancer") from e


def _resolve(nm: Any, dotted_alternatives: tuple[tuple[str, ...], ...]) -> Any:
    """Resolve an attribute from the NeuroMANCER module given *dotted* candidates.

    This utility allows us to tolerate minor API moves across NeuroMANCER
    versions. We try each alternative path in order and return the first match.

    Parameters
    ----------
    nm:
        The already‑imported ``neuromancer`` module (or a stub in tests).
    dotted_alternatives:
        A tuple of dotted attribute paths, e.g. ``(("system", "Node"), ("Node",))``.

    Returns
    -------
    Any
        The resolved attribute.

    Raises
    ------
    AttributeError
        If none of the alternatives can be found.
    """
    for parts in dotted_alternatives:
        obj: Any = nm
        ok = True
        for name in parts:
            obj = getattr(obj, name, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    # Not found: craft a helpful error message.
    alts = " | ".join(".".join(("nm",) + p) for p in dotted_alternatives)
    raise AttributeError(f"Could not resolve any of: {alts}")


def neuromancer_version() -> str:
    """Return ``neuromancer.__version__`` or ``"unknown"`` if absent.

    The function raises :class:`ImportError` with an actionable message
    if NeuroMANCER is not installed.
    """
    nm = _import_neuromancer()
    ver = getattr(nm, "__version__", None)
    return ver if isinstance(ver, str) and ver else "unknown"


def neuromancer_available() -> bool:
    """Lightweight availability check that does **not** import the package."""
    spec = _find_spec("neuromancer")
    return spec is not None


def neuromancer_smoke(batch_size: int = 4) -> dict[str, float | str]:
    """Construct and evaluate a minimal NeuroMANCER problem on CPU.

    The toy problem is intentionally tiny and deterministic:
        - A single :class:`Node` implements the identity map ``x := p``.
        - We declare a symbolic variable ``x`` and minimize ``(x - 0.5)^2``.
        - Inputs are an all‑ones batch, evaluated under ``no_grad``.

    Parameters
    ----------
    batch_size:
        Number of samples in the dummy batch (kept small by default).

    Returns
    -------
    dict
        A JSON‑friendly summary with the NeuroMANCER version, scalar loss,
        and the number of samples used.
    """
    nm = _import_neuromancer()
    try:
        import torch  # defer heavy import
    except Exception as e:  # pragma: no cover
        raise ImportError("torch is required for neuromancer_smoke. Try: pip install torch") from e

    # Resolve core entry points with version tolerance.
    variable: Callable[[str], Any] = _resolve(
        nm, (("constraint", "variable"), ("constraints", "variable"), ("variable",))
    )
    Node = _resolve(nm, (("system", "Node"), ("Node",),))
    PenaltyLoss = _resolve(nm, (("loss", "PenaltyLoss"), ("PenaltyLoss",),))
    Problem = _resolve(nm, (("problem", "Problem"), ("Problem",),))

    # Identity map p -> x, wrapped as a Node
    class _Id(torch.nn.Module):
        def forward(self, p: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return p

    node = Node(_Id(), ["p"], ["x"], name="id_map")

    # Symbolic variable and simple convex objective: min (x - 0.5)^2
    x = variable("x")
    obj = ((x - 0.5) ** 2).minimize(weight=1.0)

    # Assemble loss and problem
    loss = PenaltyLoss(objectives=[obj], constraints=[])
    problem = Problem(nodes=[node], loss=loss)

    # Dummy batch on CPU
    p = torch.ones(batch_size, 1, dtype=torch.float32)
    batch = {"p": p}

    # Evaluate loss in a version-robust way
    with torch.no_grad():
        out = problem(batch)
    if isinstance(out, dict) and "loss" in out:
        loss_tensor = out["loss"]
    elif isinstance(out, torch.Tensor):
        loss_tensor = out
    else:  # fall back to explicit API
        loss_tensor = problem.compute_loss(batch)  # type: ignore[attr-defined]

    loss_value = float(loss_tensor.detach().cpu().item())
    return {"version": neuromancer_version(), "loss": loss_value, "samples": float(batch_size)}


__all__ = ["neuromancer_version", "neuromancer_available", "neuromancer_smoke"]
