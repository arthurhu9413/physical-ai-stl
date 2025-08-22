# ruff: noqa: I001
from __future__ import annotations
"""
physical_ai_stl.frameworks.spatial_spec_hello
=============================================

Tiny, import-safe helpers around **SpaTiaL Specifications** — the Python
package published on PyPI as ``spatial-spec`` with importable module
name ``spatial_spec``.  These helpers are deliberately minimal so they can
be exercised in CI or on bare environments without pulling heavy optional
dependencies at module import time.

What this file provides
-----------------------
- **Distribution and module constants** so other parts of the project
  can build consistent user messages.
- **``spatial_spec_version()``** — robust version discovery that:
    1) *imports the runtime module first* (to mirror real usage and to
       raise a clean ``ImportError`` if SpaTiaL is absent), then
    2) prefers ``spatial_spec.__version__`` when available, and
    3) falls back to ``importlib.metadata.version('spatial-spec')``,
       finally returning ``'unknown'`` if all else fails.
- **``spatial_spec_available()``** — lightweight availability probe that
  never raises and never imports subpackages at module import time.

Design notes
------------
- We purposely avoid importing ``spatial_spec`` at *module import* time.
  This keeps startup fast and prevents hard failures on machines where
  SpaTiaL is optional.
- Error messages include both the **distribution** (``spatial-spec``) and
  **module** (``spatial_spec``) names to make installation hints precise.

References
----------
• SpaTiaL PyPI (dist: ``spatial-spec``) — https://pypi.org/project/spatial-spec/
• SpaTiaL (research / code umbrella) — https://github.com/KTH-RPL-Planiacs/SpaTiaL
"""

from importlib import import_module, metadata as _metadata

# Public constants for clarity in messages and for downstream tooling.
SPATIAL_SPEC_DIST_NAME: str = "spatial-spec"
SPATIAL_SPEC_MODULE_NAME: str = "spatial_spec"


def spatial_spec_version() -> str:
    """
    Return the installed SpaTiaL (``spatial-spec``) version string.

    Behavior
    --------
    1) First attempts to import the runtime module (``spatial_spec``).
       If this fails for *any* reason, re-raise as a clean ImportError so
       callers/tests can rely on that signal.
    2) On success, prefers the conventional ``__version__`` attribute.
    3) If unavailable, falls back to the distribution metadata version
       for the PyPI name (``spatial-spec``).
    4) If neither path yields a version string, returns ``'unknown'``.

    Returns
    -------
    str
        The version string, e.g. ``'0.1.1'`` or ``'unknown'``.

    Raises
    ------
    ImportError
        If the ``spatial_spec`` module cannot be imported.  The error
        message includes concrete installation guidance.
    """
    # 1) Import the module — this mirrors real usage and ensures tests
    #    can explicitly assert ImportError semantics when absent.
    try:
        mod = import_module(SPATIAL_SPEC_MODULE_NAME)
    except Exception as e:  # pragma: no cover - error path exercised in tests
        raise ImportError(
            "SpaTiaL is not installed. Install the PyPI package "
            f"`{SPATIAL_SPEC_DIST_NAME}` (module `{SPATIAL_SPEC_MODULE_NAME}`).\n"
            "Example:  pip install spatial-spec"
        ) from e

    # 2) Prefer module-local __version__ when available.
    ver = getattr(mod, "__version__", None)
    if isinstance(ver, str) and ver.strip():
        return ver

    # 3) Fallback: query distribution metadata by PyPI name.
    try:
        return _metadata.version(SPATIAL_SPEC_DIST_NAME)
    except Exception:
        # 4) Last resort — stay resilient across editable/dev installs.
        return "unknown"


def spatial_spec_available() -> bool:
    """
    Fast, exception-swallowing probe for SpaTiaL availability.

    Returns
    -------
    bool
        ``True`` iff importing ``spatial_spec`` succeeds, ``False`` otherwise.
    """
    try:
        import_module(SPATIAL_SPEC_MODULE_NAME)
        return True
    except Exception:
        return False


__all__ = [
    "SPATIAL_SPEC_DIST_NAME",
    "SPATIAL_SPEC_MODULE_NAME",
    "spatial_spec_version",
    "spatial_spec_available",
]
