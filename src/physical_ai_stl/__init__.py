# ruff: noqa: I001
# isort: skip_file
from __future__ import annotations

"""Top-level package for :mod:`physical_ai_stl`.

This initializer keeps **imports fast** and the **public surface tidy**:

- Subpackages (e.g. :mod:`datasets`, :mod:`frameworks`) are **lazy‑loaded**
  following :pep:`562` so simply importing :mod:`physical_ai_stl` does not
  pull heavy optional dependencies (PyTorch, PhysicsNeMo, …).
- Selected tiny utilities (``seed_everything``, ``CSVLogger``) are re‑exported
  for convenience.
- A small optional‑dependency inspector (:func:`optional_dependencies` and
  :func:`about`) helps users verify that *Physical‑AI* frameworks (Neuromancer,
  PhysicsNeMo, TorchPhysics) and STL tooling (RTAMT, MoonLight, SpaTiaL) are
  visible in the current environment.
"""

from collections.abc import Mapping
from importlib import import_module
from importlib import metadata as _metadata
from importlib import util as _import_util
from typing import Any, TYPE_CHECKING

__all__ = [
    "__version__",
    # Lazy subpackages/modules
    "datasets",
    "experiments",
    "frameworks",
    "models",
    "monitoring",
    "monitors",
    "physics",
    "training",
    "utils",
    "pde_example",
    # Small re‑exports
    "seed_everything",
    "CSVLogger",
    # Helpers
    "about",
    "optional_dependencies",
    "require_optional",
]

# NOTE: Keep this as a literal string for Hatch (pyproject.toml -> [tool.hatch.version])
# to source the package version directly from this file.
__version__ = "0.1.0"

# ----- Lazy access to subpackages (PEP 562) ---------------------------------

# Map attribute name -> fully qualified module path
_SUBMODULES: Mapping[str, str] = {
    "datasets": "physical_ai_stl.datasets",
    "experiments": "physical_ai_stl.experiments",
    "frameworks": "physical_ai_stl.frameworks",  # namespace package (no heavy import)
    "models": "physical_ai_stl.models",
    "monitoring": "physical_ai_stl.monitoring",
    "monitors": "physical_ai_stl.monitors",
    "physics": "physical_ai_stl.physics",
    "training": "physical_ai_stl.training",
    "utils": "physical_ai_stl.utils",
    "pde_example": "physical_ai_stl.pde_example",
}

# Lightweight helpers (attribute -> "module:object")
_HELPERS: Mapping[str, str] = {
    "seed_everything": "physical_ai_stl.utils.seed:seed_everything",
    "CSVLogger": "physical_ai_stl.utils.logger:CSVLogger",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    if name in _HELPERS:
        mod_name, obj_name = _HELPERS[name].split(":")
        obj = getattr(import_module(mod_name), obj_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'physical_ai_stl' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()) + list(_HELPERS.keys()))


# ----- Optional dependency inspection ---------------------------------------

# Map "import name" -> distribution name on PyPI (for version lookups). When the
# names match, we can use a single entry. If unknown, leave the dist empty.
# Notes:
#   • NVIDIA renamed **Modulus** -> **PhysicsNeMo** in 2025; some projects still
#     import ``modulus`` while newer code imports ``physicsnemo``. We probe both.
#   • SpaTiaL publishes the PyPI distribution as ``spatial-spec`` but the import
#     name is ``spatial_spec``.
_OPT_DEPS: Mapping[str, str] = {
    # Core scientific stack (often present)
    "numpy": "numpy",
    "torch": "torch",
    # Physics‑ML frameworks
    "neuromancer": "neuromancer",             # PNNL NeuroMANCER
    "physicsnemo": "nvidia-physicsnemo",      # import name -> PyPI dist
    "modulus": "nvidia-modulus",              # legacy import for older examples
    "torchphysics": "torchphysics",           # Bosch TorchPhysics
    # STL / spatio‑temporal monitoring
    "rtamt": "rtamt",
    "moonlight": "moonlight",
    "spatial_spec": "spatial-spec",
}

# Human‑readable ``pip install`` hints for nicer error messages.
_INSTALL_HINT: Mapping[str, str] = {
    "torch": "pip install torch",
    "neuromancer": "pip install neuromancer",
    "physicsnemo": "pip install nvidia-physicsnemo",
    "modulus": "pip install nvidia-modulus",
    "torchphysics": "pip install torchphysics",
    "rtamt": "pip install rtamt",
    "moonlight": "pip install moonlight",
    "spatial_spec": "pip install spatial-spec",
    "numpy": "pip install numpy",
}

# Small cache so repeated environment reports are O(1) after the first call.
_OPT_CACHE: dict[str, tuple[bool, str | None]] = {}


def _probe_module(mod_name: str) -> tuple[bool, str | None]:
    """Return (available, version) for *mod_name* without importing heavy deps.

    Uses :mod:`importlib.util.find_spec` to *detect* a module and only imports
    it when necessary to resolve its version. Results are cached to keep
    repeated calls fast.
    """
    cached = _OPT_CACHE.get(mod_name)
    if cached is not None:
        return cached

    spec = _import_util.find_spec(mod_name)
    if spec is None:
        result = (False, None)
        _OPT_CACHE[mod_name] = result
        return result

    dist_name = _OPT_DEPS.get(mod_name) or mod_name
    version: str | None = None
    try:
        version = _metadata.version(dist_name)  # type: ignore[arg-type]
    except Exception:
        # Fall back to asking the module for __version__ without importing
        # large subpackages – this import should be light for well-behaved pkgs.
        try:
            mod = import_module(mod_name)
            version = getattr(mod, "__version__", None)
        except Exception:
            version = None

    result = (True, version)
    _OPT_CACHE[mod_name] = result
    return result


def optional_dependencies(
    refresh: bool = False,
    include_pip_hints: bool = True,
) -> dict[str, dict[str, str | bool | None]]:
    """Inspect availability/versions of optional frameworks and monitors.

    Parameters
    ----------
    refresh:
        If ``True``, clear the cached probe results and re-scan the
        environment.
    include_pip_hints:
        If ``True`` (default), include a concise ``"pip"`` hint for packages
        that are not available to guide installation.

    Returns
    -------
    dict
        Mapping from import name to a small record:

        ``{"available": bool, "version": Optional[str], "pip": Optional[str]}``

        The ``"pip"`` key is present only when ``include_pip_hints=True``.
    """
    if refresh:
        _OPT_CACHE.clear()

    report: dict[str, dict[str, str | bool | None]] = {}
    for mod in _OPT_DEPS.keys():
        ok, ver = _probe_module(mod)
        item: dict[str, str | bool | None] = {"available": ok, "version": ver}
        if include_pip_hints and not ok:
            item["pip"] = _INSTALL_HINT.get(mod)
        report[mod] = item
    return report


def require_optional(mod_name: str, min_version: str | None = None) -> None:
    """Assert that an *optional* dependency is importable (and optionally recent).

    This helper keeps import sites clean and produces **actionable** errors.
    It does **not** import the target at module import time.

    Examples
    --------
    >>> require_optional("rtamt")
    >>> require_optional("physicsnemo", "1.1.0")  # doctest: +SKIP

    Parameters
    ----------
    mod_name:
        The **import name** (e.g., ``"moonlight"`` or ``"spatial_spec"``).
    min_version:
        Optional minimum version string. If supplied and resolvable, an
        informative ``ImportError`` is raised when the installed version is
        older.

    Raises
    ------
    ImportError
        If the module is not importable or does not satisfy ``min_version``.
    """
    ok, found_version = _probe_module(mod_name)
    if not ok:
        hint = _INSTALL_HINT.get(mod_name)
        dist = _OPT_DEPS.get(mod_name) or mod_name
        msg = (
            f"Optional dependency '{mod_name}' is required but not installed.\n"
            f"→ Install via: {hint or f'pip install {dist}'}"
        )
        # Special case: PhysicsNeMo rename – provide a friendly alias note
        if mod_name == "physicsnemo":
            msg += (
                "\nNOTE: NVIDIA renamed 'Modulus' to 'PhysicsNeMo'. If you have an "
                "older environment with 'modulus' installed, either migrate imports "
                "to 'physicsnemo' or install the new package."
            )
        raise ImportError(msg)

    if min_version and found_version:
        # Compare versions when packaging is available; otherwise do a safe best‑effort.
        try:
            from packaging.version import Version  # local import
        except Exception:
            Version = None  # type: ignore
        if Version is not None:  # type: ignore[truthy-bool]
            if Version(found_version) < Version(min_version):  # type: ignore[arg-type]
                raise ImportError(
                    f"'{mod_name}' version >= {min_version} required; found {found_version}."
                )
        # If we cannot parse versions, prefer to be permissive and continue.



def about() -> str:
    """Return a compact, human-readable environment summary."""
    lines = [f"physical_ai_stl {__version__}", "Optional deps:"]
    report = optional_dependencies(include_pip_hints=False)
    width = max((len(k) for k in report.keys()), default=0)
    for name in sorted(report.keys()):
        avail = report[name]["available"]
        ver = report[name]["version"]
        lines.append(f"  {name.ljust(width)}  {'yes' if avail else 'no ':<3} ({ver or '-'})")

    # Give a clear one-liner about the Modulus → PhysicsNeMo rename if relevant.
    phys_ok = bool(report.get("physicsnemo", {}).get("available"))
    mod_ok = bool(report.get("modulus", {}).get("available"))
    if mod_ok and not phys_ok:
        lines.append("  note: 'modulus' is installed; consider migrating to 'physicsnemo'.")
    return "\n".join(lines)


# ----- Static imports for type checkers only --------------------------------

if TYPE_CHECKING:  # pragma: no cover
    # These imports help IDEs and type checkers without paying runtime import cost.
    from . import (  # noqa: F401
        datasets,
        experiments,
        frameworks,
        models,
        monitoring,
        monitors,
        pde_example,
        physics,
        training,
        utils,
    )
    from .utils.logger import CSVLogger as CSVLogger  # noqa: F401
    from .utils.seed import seed_everything as seed_everything  # noqa: F401
