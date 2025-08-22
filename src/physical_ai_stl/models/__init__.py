# ruff: noqa: I001
# isort: skip_file
from __future__ import annotations

"""
physical_ai_stl.models
======================

Tiny **model hub** for this repository.

Design goals
------------
- Keep imports *fast* by lazily importing heavy dependencies (e.g., PyTorch).
- Provide a *stable*, typed surface for scripts:
  ``available()``, ``get_builder()``, ``build()``, ``from_spec()``, ``register()``.
- Make configs ergonomic: accept strings (``"mlp"``), qualified targets
  (``"physical_ai_stl.models.mlp:MLP"``), or dict specs with ``name/type/target``.
- Be helpful: error messages suggest the right optional extra to install.

This module intentionally stays minimal: it does **not** attempt to be a full
plugin system. If you need a bespoke model, register a new builder at runtime.
"""

import importlib
from dataclasses import dataclass
from importlib import import_module
from importlib import util as _import_util
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

__all__ = [
    # Lazily forwarded classes
    "MLP",
    # Registry & helpers
    "ModelBuilder",
    "register",
    "register_model",
    "available",
    "get_builder",
    "build",
    "from_spec",
    "about",
]

# --------------------------------------------------------------------------- #
# Lazy export of submodules/objects
# --------------------------------------------------------------------------- #

# Map attribute name -> "relative.module[:object]".
# Objects are imported only on first access to keep imports snappy.
_LAZY: dict[str, str] = {
    # Submodule
    "mlp": ".mlp",
    # Objects re‑exported at package level
    "MLP": ".mlp:MLP",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny import shim
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'physical_ai_stl.models' has no attribute {name!r}")
    if ":" in target:
        mod_name, qual = target.split(":", 1)
        obj = getattr(import_module(mod_name, __name__), qual)
        globals()[name] = obj  # cache for future lookups
        return obj
    # Return the submodule itself (e.g., `mlp`)
    module = import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    # Expose both already-bound globals and the lazy attributes
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


# --------------------------------------------------------------------------- #
# Simple registry
# --------------------------------------------------------------------------- #

# A builder takes (*args, **kwargs) and returns an initialized model instance.
ModelBuilder = Callable[..., Any]

# Canonical name -> builder
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}

# Optional metadata to improve `about()` output
@dataclass(frozen=True)
class _ModelInfo:
    name: str
    target: str | None
    summary: str | None = None
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    homepage: str | None = None


_MODEL_INFO: dict[str, _ModelInfo] = {}


def _canonical(name: str) -> str:
    """Normalize keys (case/underscore/dash insensitive)."""
    return "".join(ch for ch in name.lower().replace("-", "_") if ch.isalnum() or ch == "_")


def _resolve_target(target: str) -> Any:
    """
    Import an object from a ``'pkg.mod:QualName'`` or ``'.rel:Name'`` string.
    """
    if ":" in target:
        mod_name, qual = target.split(":", 1)
    else:
        # If the user passed a dotted path, assume final token is the object.
        # e.g., "physical_ai_stl.models.mlp.MLP"
        parts = target.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(f"Cannot resolve target {target!r}; expected 'module:object' or dotted path.")
        mod_name, qual = parts
    mod = import_module(mod_name, __name__)
    return getattr(mod, qual)


def register(name: str, builder: ModelBuilder, *, aliases: Iterable[str] = (), summary: str | None = None) -> ModelBuilder:
    """
    Register a model *builder* under a short, canonical name.

    Examples
    --------
    >>> def build_my_model(dim: int): ...
    >>> register("my_model", build_my_model, aliases=("mymodel",))
    """
    key = _canonical(name)
    if key in _MODEL_REGISTRY and _MODEL_REGISTRY[key] is not builder:
        raise KeyError(f"Model name '{name}' is already registered to a different builder.")
    _MODEL_REGISTRY[key] = builder
    # record primary info
    if key not in _MODEL_INFO:
        _MODEL_INFO[key] = _ModelInfo(name=key, target=None, summary=summary, aliases=tuple(aliases))
    # also register aliases
    for alias in aliases:
        akey = _canonical(alias)
        if akey in _MODEL_REGISTRY and _MODEL_REGISTRY[akey] is not builder:
            raise KeyError(f"Alias '{alias}' is already registered to a different builder.")
        _MODEL_REGISTRY[akey] = builder
        if akey not in _MODEL_INFO:
            _MODEL_INFO[akey] = _ModelInfo(name=akey, target=None, summary=summary, aliases=())
    return builder


def register_model(name: str, target: str, *, aliases: Iterable[str] = (), summary: str | None = None) -> None:
    """
    Register by import *target* (``'pkg.mod:Obj'``) instead of a builder function.
    The import happens lazily the first time the model is built.
    """
    def _builder(*args: Any, **kwargs: Any) -> Any:
        cls_or_fn = _resolve_target(target)
        return cls_or_fn(*args, **kwargs)
    register(name, _builder, aliases=aliases, summary=summary)
    # keep richer metadata when available
    _MODEL_INFO[_canonical(name)] = _ModelInfo(name=_canonical(name), target=target, summary=summary, aliases=tuple(aliases))


def available() -> list[str]:
    """Return all registered names (including aliases), sorted."""
    return sorted(_MODEL_REGISTRY.keys())


def get_builder(name_or_target: str) -> ModelBuilder:
    """
    Retrieve a builder by *name* or import *target* if a qualified path is given.

    Accepts:
    - Short registry names/aliases (e.g., ``'mlp'``).
    - Qualified import strings (e.g., ``'physical_ai_stl.models.mlp:MLP'``).
    - Dotted paths (e.g., ``'physical_ai_stl.models.mlp.MLP'``).
    """
    key = _canonical(name_or_target)
    builder = _MODEL_REGISTRY.get(key)
    if builder is not None:
        return builder
    # If not in the registry, interpret as an import target and return a thin builder.
    if ":" in name_or_target or "." in name_or_target:
        def _builder(*args: Any, **kwargs: Any) -> Any:
            obj = _resolve_target(name_or_target)
            return obj(*args, **kwargs)
        return _builder
    raise KeyError(f"Unknown model {name_or_target!r}. Known: {', '.join(available()) or '—'}")


def _maybe_rewrite_import_error(e: ImportError) -> ImportError:
    """
    Rewrite common dependency errors into actionable messages.
    """
    msg = str(e).lower()
    # Heuristic: if torch is missing, suggest installing the optional extra.
    if ("torch" in msg or "pytorch" in msg) and (_import_util.find_spec("torch") is None):
        return ImportError(
            "This model requires PyTorch. Install the optional extra:\n"
            '  pip install "physical-ai-stl[torch]"\n'
            "or install PyTorch directly from https://pytorch.org ."
        )
    return e


def build(name_or_target: str, /, *args: Any, **kwargs: Any) -> Any:
    """
    Instantiate a model by *name* or *import target*.

    Parameters
    ----------
    name_or_target:
        Registry name/alias (``'mlp'``) or import target (``'pkg.mod:Class'``).
    *args, **kwargs:
        Passed through to the builder/class constructor.

    Returns
    -------
    Any
        The constructed model instance.

    Raises
    ------
    KeyError
        If the name is unknown.
    ImportError
        If an optional dependency is missing.
    """
    builder = get_builder(name_or_target)
    try:
        return builder(*args, **kwargs)
    except ImportError as e:  # pragma: no cover - tiny UX shim
        raise _maybe_rewrite_import_error(e) from e


def from_spec(spec: str | Mapping[str, Any] | Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """
    Instantiate a model from a flexible specification.

    Accepted forms
    --------------
    1) ``"mlp"``                                 -> uses the registry
    2) ``"pkg.mod:Obj"`` or ``"pkg.mod.Obj"``    -> imports that symbol
    3) ``callable``                               -> treated as a builder/constructor
    4) mapping with keys:
        - ``name``/``type``/``target``: identifier
        - ``args`` (optional list/tuple): extra positional args
        - ``kwargs`` (optional mapping): keyword args
      Any remaining keys are merged into kwargs (handy for concise YAML).

    Example
    -------
    >>> from_spec({"name": "mlp", "in_dim": 4, "out_dim": 2, "hidden": [64, 64]})
    """
    if callable(spec):
        return spec(*args, **kwargs)
    if isinstance(spec, str):
        return build(spec, *args, **kwargs)
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a string, mapping, or callable.")

    data = dict(spec)  # shallow copy
    ident = data.pop("name", None) or data.pop("type", None) or data.pop("target", None)
    extra_args = list(data.pop("args", ()))
    extra_kwargs = dict(data.pop("kwargs", {}))
    # The remaining keys are treated as kwargs to keep configs concise.
    extra_kwargs.update(data)
    # Allow user kwargs to override spec-provided kwargs
    extra_kwargs.update(kwargs)
    return build(str(ident), *(list(args) + extra_args), **extra_kwargs)


def about() -> str:
    """
    Human‑readable one‑pager describing what's available.
    """
    lines: list[str] = []
    lines.append("Models available:")
    names = sorted({info.name for info in _MODEL_INFO.values()} or _MODEL_REGISTRY.keys())
    for nm in names:
        info = _MODEL_INFO.get(nm)
        summary = (info.summary if info else None) or ""
        aliases = ", ".join(sorted(a for a in (_MODEL_INFO.get(nm).aliases if _MODEL_INFO.get(nm) else ()) if a != nm))
        ali = f" (aliases: {aliases})" if aliases else ""
        tgt = f" -> {info.target}" if info and info.target else ""
        lines.append(f"  - {nm}{ali}{tgt}{(': ' + summary) if summary else ''}")
    if not names:
        lines.append("  (none registered)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Default registrations (kept lazy and dependency‑light)
# --------------------------------------------------------------------------- #

# Avoid importing torch/MLP at import time. The builder imports on demand.
def _build_mlp(*args: Any, **kwargs: Any) -> Any:
    from .mlp import MLP  # local import to keep top‑level import fast
    return MLP(*args, **kwargs)


# Common aliases for convenience in configs.
register("mlp", _build_mlp, aliases=("dense", "fc", "fully_connected"), summary="Plain multilayer perceptron (PyTorch)")

# Also record a target for richer `about()` output.
_MODEL_INFO[_canonical("mlp")] = _ModelInfo(
    name=_canonical("mlp"),
    target="physical_ai_stl.models.mlp:MLP",
    summary="Plain multilayer perceptron (PyTorch)",
    aliases=("dense", "fc", "fully_connected"),
    tags=("torch", "feedforward"),
    homepage=None,
)

# --------------------------------------------------------------------------- #
# Type‑checking shims (no runtime import cost)
# --------------------------------------------------------------------------- #

# Help IDEs and type checkers with concrete symbols without runtime cost.
if TYPE_CHECKING:  # pragma: no cover
    from . import mlp as mlp  # noqa: F401
    from .mlp import MLP as MLP  # re‑export for IDEs / static analysis
