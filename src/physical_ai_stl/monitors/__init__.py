# ruff: noqa: I001
"""Convenience accessors and optional‑backend helpers for :mod:`physical_ai_stl.monitors`.

This package intentionally *lazy‑loads* its example monitor modules so that the top‑level
package stays importable even when optional third‑party toolkits (RTAMT, MoonLight, SpaTiaL)
are not installed.  In addition, we expose small utilities to *inspect* what is available and
to *require* a given backend with a clear, actionable error message.

The design goals are (in order):
1) **Correctness & usability** – imports fail with helpful guidance rather than stack traces.
2) **Performance** – probing is cached and can be disabled via ``PAI_STL_NO_PROBE=1``.
3) **Clarity** – minimal, well‑typed surface area and stable names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module, metadata as _metadata, util as _import_util
from os import getenv
from typing import TYPE_CHECKING, Any

__all__ = [
    # Submodules (lazy)
    "rtamt_hello",
    "moonlight_hello",
    "moonlight_strel_hello",
    "spatial_demo",
    # Convenience re‑exports (lazy)
    "stl_hello_offline",
    "temporal_hello",
    "strel_hello",
    "spatial_run_demo",
    # Environment helpers
    "available_backends",
    "probe_backend",
    "require_backend",
]

# ----- Lazy import shims -----------------------------------------------------

_SUBMODULES: Mapping[str, str] = {
    "rtamt_hello": "physical_ai_stl.monitors.rtamt_hello",
    "moonlight_hello": "physical_ai_stl.monitors.moonlight_hello",
    "moonlight_strel_hello": "physical_ai_stl.monitors.moonlight_strel_hello",
    "spatial_demo": "physical_ai_stl.monitors.spatial_demo",
}

# function_name -> "module_path:object"
_HELPERS: Mapping[str, str] = {
    "stl_hello_offline": "physical_ai_stl.monitors.rtamt_hello:stl_hello_offline",
    "temporal_hello": "physical_ai_stl.monitors.moonlight_hello:temporal_hello",
    "strel_hello": "physical_ai_stl.monitors.moonlight_strel_hello:strel_hello",
    "spatial_run_demo": "physical_ai_stl.monitors.spatial_demo:run_demo",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    if name in _SUBMODULES:
        try:
            mod = import_module(_SUBMODULES[name])
        except Exception as e:  # Add friendlier guidance for optional deps
            raise ModuleNotFoundError(
                f"Failed to import '{name}'. This example may require an optional monitoring backend.\n"
                f"→ Try: pip install 'physical-ai-stl[monitoring]' (see README).\n"
                f"Original error: {e.__class__.__name__}: {e}"
            ) from e
        globals()[name] = mod
        return mod
    if name in _HELPERS:
        mod_name, obj_name = _HELPERS[name].split(":")
        try:
            obj = getattr(import_module(mod_name), obj_name)
        except Exception as e:
            raise ModuleNotFoundError(
                f"Failed to import helper '{name}' from {mod_name}.\n"
                f"→ Try: pip install 'physical-ai-stl[monitoring]'.\n"
                f"Original error: {e.__class__.__name__}: {e}"
            ) from e
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'physical_ai_stl.monitors' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()) + list(_HELPERS.keys()))


# ----- Optional backend inspection ------------------------------------------

# Map *import* name -> candidate *distribution* names on PyPI
# Notes:
#  • RTAMT is published as 'rtamt' (import 'rtamt').
#  • MoonLight Python wrapper is 'moonlight' (import 'moonlight').
#  • SpaTiaL exposes module 'spatial'; wheels are available either as 'spatial'
#    (via Git) or 'spatial-spec' (PyPI helper metapackage).
_DIST_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "rtamt": ("rtamt",),
    "moonlight": ("moonlight",),
    "spatial": ("spatial", "spatial-spec"),
}


@dataclass(slots=True, frozen=True)
class BackendInfo:
    """Status of an optional monitoring backend.

    Attributes
    ----------
    name:
        The Python *import* name (e.g., ``'rtamt'``).
    available:
        Whether the module can be imported (found on ``sys.meta_path``).
    version:
        Best‑effort semantic version string, if discoverable via distribution
        metadata or ``module.__version__``.
    distribution:
        The *distribution* name that provided the version, if any.
    error:
        Last error string encountered while probing (useful diagnostics).
    """

    name: str
    available: bool
    version: str | None
    distribution: str | None
    error: str | None = None


def _find_spec(mod_name: str) -> bool:
    return _import_util.find_spec(mod_name) is not None


def _get_version(mod_name: str) -> tuple[str | None, str | None, str | None]:
    """Return (version, distribution, error) for *mod_name*.

    We try candidate distribution names first (fast path). If that fails,
    we attempt to import the module and read ``__version__``. All failures
    are converted to a short diagnostic string.
    """
    last_err: str | None = None
    for dist in _DIST_CANDIDATES.get(mod_name, (mod_name,)):
        try:
            ver = _metadata.version(dist)  # type: ignore[arg-type]
            return ver, dist, None
        except Exception as e:
            last_err = f"{e.__class__.__name__}: {e}"

    # Fallback: try module.__version__ (may be absent for Git installs)
    try:
        m = import_module(mod_name)
        ver = getattr(m, "__version__", None)
        return (str(ver) if ver is not None else None), None, None
    except Exception as e:  # importing solely for version shouldn't raise if spec exists
        last_err = f"{e.__class__.__name__}: {e}"
        return None, None, last_err


def _probe(mod_name: str) -> BackendInfo:
    if not _find_spec(mod_name):
        return BackendInfo(name=mod_name, available=False, version=None, distribution=None, error=None)
    ver, dist, err = _get_version(mod_name)
    return BackendInfo(name=mod_name, available=True, version=ver, distribution=dist, error=err)


@lru_cache(maxsize=1)
def available_backends() -> dict[str, dict[str, bool | str | None]]:
    """Detect optional monitoring toolkits.

    The result is cached for the lifetime of the process. Set the environment
    variable ``PAI_STL_NO_PROBE=1`` to return an empty mapping instantly
    (useful for very constrained environments).
    """
    if getenv("PAI_STL_NO_PROBE") in {"1", "true", "True"}:
        return {}
    report: dict[str, dict[str, bool | str | None]] = {}
    for name in _DIST_CANDIDATES.keys():
        info = _probe(name)
        report[name] = {
            "available": info.available,
            "version": info.version,
            "distribution": info.distribution,
            "error": info.error,
        }
    return report


def probe_backend(name: str) -> BackendInfo:
    """Probe a single backend and return a rich :class:`BackendInfo` record."""
    return _probe(name)


def require_backend(name: str, *, min_version: str | None = None) -> None:
    """Ensure an optional backend is present, raising a helpful error otherwise.

    Examples
    --------
    >>> require_backend('rtamt')
    >>> require_backend('spatial', min_version='0.1.1')
    """
    info = _probe(name)
    if not info.available:
        candidates = ", ".join(_DIST_CANDIDATES.get(name, (name,)))
        raise ModuleNotFoundError(
            f"Backend '{name}' is not installed.\n"
            f"→ Install via: pip install 'physical-ai-stl[monitoring]'\n"
            f"  or: pip install {candidates}\n"
            f"See project README for OS‑specific notes (e.g., MONA for SpaTiaL)."
        )
    if min_version and info.version is not None:
        try:
            from packaging.version import Version

            if Version(info.version) < Version(min_version):
                raise ModuleNotFoundError(
                    f"Backend '{name}' version {info.version} < required {min_version}.\n"
                    f"→ Upgrade via: pip install -U {info.distribution or name}"
                )
        except Exception:
            # If packaging is unavailable or version unparsable, skip strict check.
            pass


# ----- Static imports for type checkers only --------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from . import moonlight_hello as moonlight_hello  # noqa: F401
    from . import moonlight_strel_hello as moonlight_strel_hello  # noqa: F401
    from . import rtamt_hello as rtamt_hello  # noqa: F401
    from . import spatial_demo as spatial_demo  # noqa: F401
    from .moonlight_hello import temporal_hello as temporal_hello  # noqa: F401
    from .moonlight_strel_hello import strel_hello as strel_hello  # noqa: F401
    from .rtamt_hello import stl_hello_offline as stl_hello_offline  # noqa: F401
    from .spatial_demo import run_demo as spatial_run_demo  # noqa: F401
