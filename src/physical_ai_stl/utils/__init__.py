# ruff: noqa: I001
# isort: skip_file
from __future__ import annotations

"""
Tiny, fast lazy-import shim for `physical_ai_stl.utils`.

Design goals
------------
- Keep import time small by deferring heavy dependencies until first use.
- Preserve excellent IDE / type-checker support (PEP 562: module __getattr__/__dir__).
- Keep behavior correct under all packaging layouts by anchoring relative imports.
- Be thread-safe and cache results in `globals()` after first resolution.

References:
- PEP 562: module-level ``__getattr__`` / ``__dir__``.  (Python ≥ 3.7)
- Official docs on ``importlib.import_module`` and the import system.
"""

from importlib import import_module
from threading import RLock
from typing import TYPE_CHECKING, Any, Final

# Map public attribute name -> "relative.module[:qualified_object]"
# Submodules and re-exports are imported on first access and then cached.
_LAZY: dict[str, str] = {
    # Submodules (kept import-light)
    "seed": ".seed",
    "logger": ".logger",
    # Re-exports from the submodules
    "seed_everything": ".seed:seed_everything",
    "seed_worker": ".seed:seed_worker",
    "torch_generator": ".seed:torch_generator",
    "CSVLogger": ".logger:CSVLogger",
}

# Export exactly what the package intends to expose, in insertion order.
# Using the mapping as the single source of truth avoids drift.
__all__: tuple[str, ...] = tuple(_LAZY.keys())

# A tiny lock prevents rare races on first access under concurrency.
_LOCK: Final[RLock] = RLock()


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny import shim
    """
    Lazily resolve attributes defined in ``_LAZY`` on first access.

    On success:
        - Import the requested submodule/object (anchored to this package).
        - Cache the resolved value in ``globals()`` for zero-cost future lookups.

    On failure:
        - Raise AttributeError with a helpful "did you mean …?" hint.
    """
    target = _LAZY.get(name)
    if target is None:
        # Avoid importing difflib on the hot path — only on errors.
        try:
            from difflib import get_close_matches  # local import: error path only
        except Exception:
            get_close_matches = None  # type: ignore[assignment]
        hint = ""
        if get_close_matches is not None:
            matches = get_close_matches(name, _LAZY.keys(), n=1, cutoff=0.8)
            if matches:
                hint = f" Did you mean {matches[0]!r}?"
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}.{hint}")

    # Double-checked locking: cheap fast-path if another thread already cached it.
    if name in globals():
        return globals()[name]

    with _LOCK:
        if name in globals():  # recheck under the lock
            return globals()[name]

        if ":" in target:
            mod_name, qual = target.split(":", 1)
            value = getattr(import_module(mod_name, __name__), qual)
        else:
            value = import_module(target, __name__)  # submodule itself

        globals()[name] = value  # cache for subsequent lookups
        return value


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    # Expose both already-bound globals and the lazy attributes (sorted for stability).
    # set(...) avoids duplicates if some names have already been imported/cached.
    return sorted(set(globals().keys()) | set(_LAZY.keys()))


# Help IDEs and type checkers with concrete symbols without runtime cost.
if TYPE_CHECKING:  # pragma: no cover
    from . import logger, seed  # noqa: F401
    from .logger import CSVLogger  # noqa: F401
    from .seed import (  # noqa: F401
        seed_everything,
        seed_worker,
        torch_generator,
    )
