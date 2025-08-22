# src/physical_ai_stl/datasets/__init__.py
#
# Lightweight dataset hub for Physical‑AI STL experiments.
#
# Design goals:
# - Keep imports fast by lazily importing submodules on first access.
# - Offer a tiny registry so experiments can refer to datasets by name.
# - Provide helpful typing stubs for IDEs without paying import cost at runtime.
#
# Public surface kept intentionally small; do not import private names from here.

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, MutableMapping, Protocol, runtime_checkable

__all__ = [
    # Primary re‑exports (lazily loaded from submodules)
    "SyntheticSTLNetDataset",
    "BoundedAtomicSpec",
    # Submodule access
    "stlnet_synthetic",
    # Convenience helpers
    "available_datasets",
    "get_dataset_cls",
    "create_dataset",
    "register_dataset",
    "DatasetInfo",
]

# --------------------------------------------------------------------------- #
# Lazy import shim
# --------------------------------------------------------------------------- #

# Map public attribute -> "relative.module[:object]".
# Objects are resolved only when first accessed to keep imports snappy.
_LAZY: dict[str, str] = {
    # Submodule
    "stlnet_synthetic": ".stlnet_synthetic",
    # Objects re‑exported at package level
    "SyntheticSTLNetDataset": ".stlnet_synthetic:SyntheticSTLNetDataset",
    "BoundedAtomicSpec": ".stlnet_synthetic:BoundedAtomicSpec",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny import shim
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'physical_ai_stl.datasets' has no attribute {name!r}")
    if ":" in target:
        mod_name, qual = target.split(":", 1)
        obj = getattr(import_module(mod_name, __name__), qual)
        globals()[name] = obj  # cache for future lookups
        return obj
    # Return the submodule itself (e.g., `stlnet_synthetic`)
    module = import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    # Expose both already‑bound globals and the lazy attributes
    return sorted(list(globals().keys()) + list(_LAZY.keys()))

# --------------------------------------------------------------------------- #
# Minimal dataset registry
# --------------------------------------------------------------------------- #

@runtime_checkable
class TimeSeriesDataset(Protocol):
    """Protocol for simple time‑series datasets used in this project.

    Implementations typically expose:
      • ``__len__`` and ``__getitem__`` returning ``(t: float, x: float)`` tuples
      • ``array`` (``N×2`` NumPy array), and convenience ``t`` / ``v`` properties

    The protocol is intentionally small so that toy datasets and torch.utils.data
    datasets can both conform without extra dependencies.
    """
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[float, float]: ...


# Registry maps a short name to an import target and a brief description.
@dataclass(frozen=True)
class DatasetInfo:
    name: str
    target: str  # "module[:qualname]"
    summary: str
    tags: tuple[str, ...] = ()
    homepage: str | None = None


_REGISTRY: MutableMapping[str, DatasetInfo] = {}


def _register_defaults() -> None:
    # Core synthetic dataset inspired by STLnet experiments.
    _REGISTRY["stlnet_synth"] = DatasetInfo(
        name="stlnet_synth",
        target=".stlnet_synthetic:SyntheticSTLNetDataset",
        summary="Clean sinusoid on [0,1] with optional Gaussian noise; handy for STL demos and windowed robustness.",
        tags=("timeseries", "synthetic", "stl", "unit-tests"),
        homepage="https://proceedings.neurips.cc/paper/2020/hash/a7da6ba0505a41b98bd85907244c4c30-Abstract.html",
    )


def _canonical(key: str) -> str:
    """Normalize dataset lookup keys (case/underscore/dash insensitive)."""
    return "".join(ch for ch in key.lower() if ch.isalnum())


def register_dataset(info: DatasetInfo) -> None:
    """Register a dataset for name‑based lookup.

    Parameters
    ----------
    info:
        ``DatasetInfo`` describing the dataset.  The ``name`` should be a short,
        stable identifier (e.g., ``"stlnet_synth"``). Re‑registering the same
        canonical name replaces the entry.

    Notes
    -----
    This is a very small registry intended for convenience in scripts and tests.
    It avoids importing heavyweight frameworks until the user actually requests
    a dataset.
    """
    if not isinstance(info, DatasetInfo):
        raise TypeError("info must be a DatasetInfo instance")
    if not info.name:
        raise ValueError("info.name must be a non‑empty string")
    key = _canonical(info.name)
    _REGISTRY[key] = info


def available_datasets() -> Mapping[str, DatasetInfo]:
    """Return a read‑only view of the dataset registry."""
    if not _REGISTRY:
        _register_defaults()
    # Return a shallow copy to discourage accidental mutation.
    return dict(_REGISTRY)


def _resolve_target(target: str) -> Any:
    """Import an object given an import target ``'module[:qualname]'``."""
    if ":" in target:
        mod_name, qual = target.split(":", 1)
        return getattr(import_module(mod_name, __name__), qual)
    return import_module(target, __name__)


def get_dataset_cls(name_or_target: str) -> type[TimeSeriesDataset]:
    """Resolve a dataset class from a registry name or import target.

    Examples
    --------
    >>> cls = get_dataset_cls("stlnet_synth")
    >>> ds = cls(length=128, noise=0.1)
    """
    if not _REGISTRY:
        _register_defaults()
    # Try by registry name first
    info = _REGISTRY.get(_canonical(name_or_target))
    target = info.target if info is not None else name_or_target
    obj = _resolve_target(target)
    if isinstance(obj, ModuleType):
        raise TypeError(f"Target {target!r} resolved to a module, not a class.")
    return obj  # type: ignore[return-value]


def create_dataset(name_or_target: str, /, *args: Any, **kwargs: Any) -> TimeSeriesDataset:
    """Instantiate a dataset via ``name`` or fully‑qualified import target.

    Parameters
    ----------
    name_or_target:
        Either a registered dataset name (e.g., ``'stlnet_synth'``) or a
        string of the form ``'relative.module:ClassName'``.
    *args, **kwargs:
        Forwarded to the dataset class constructor.

    Returns
    -------
    TimeSeriesDataset
        An instance of the requested dataset.

    Raises
    ------
    TypeError
        If the import target resolves to a module rather than a class.
    """
    cls = get_dataset_cls(name_or_target)
    return cls(*args, **kwargs)  # type: ignore[misc]

# --------------------------------------------------------------------------- #
# Type‑checking shims (no runtime import cost)
# --------------------------------------------------------------------------- #

# Help IDEs and type checkers with concrete symbols without runtime cost.
if TYPE_CHECKING:  # pragma: no cover
    from . import stlnet_synthetic as stlnet_synthetic  # noqa: F401
    from .stlnet_synthetic import BoundedAtomicSpec as BoundedAtomicSpec  # noqa: F401
    from .stlnet_synthetic import SyntheticSTLNetDataset as SyntheticSTLNetDataset  # noqa: F401
