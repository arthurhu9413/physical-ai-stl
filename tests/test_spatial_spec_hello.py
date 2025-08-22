from __future__ import annotations

"""Unit tests for :mod:`physical_ai_stl.frameworks.spatial_spec_hello`.

Design goals
------------
- **Robust to missing optional deps.** If the SpaTiaL package (PyPI
  distribution ``spatial-spec``; import name ``spatial_spec``) is not
  available in the environment, tests **skip** instead of failing.
- **Deterministic and fast.** We rely only on monkeypatching the import
  system and never import heavy stacks.
- **Behavior-focused.** We verify (1) version resolution policy,
  (2) graceful ImportError surface, and (3) the convenience
  availability helper when exposed by the module.
"""

import builtins
import importlib
import sys
from types import ModuleType

import pytest

MOD = "physical_ai_stl.frameworks.spatial_spec_hello"


def _import_or_skip():
    """Import the helper module or *skip* if the package isn't available."""
    try:
        return importlib.import_module(MOD)
    except Exception:
        pytest.skip("SpaTiaL helper missing")
        raise  # pragma: no cover


def _make_dummy_spatial_spec(*, version: str | None) -> ModuleType:
    """Create a light-weight dummy ``spatial_spec`` module.

    Parameters
    ----------
    version:
        If not ``None``, the dummy exposes ``__version__ = version``.
        If ``None``, the attribute is omitted to trigger fallback code paths.
    """
    mod = ModuleType("spatial_spec")
    if version is not None:
        mod.__version__ = version  # type: ignore[attr-defined]
    return mod


def test_spatial_spec_version_real_or_skip() -> None:
    """Smoke test: real call returns a non-empty string or skips gracefully."""
    mod = _import_or_skip()
    spatial_spec_version = getattr(mod, "spatial_spec_version", None)
    if spatial_spec_version is None:  # pragma: no cover - defensive
        pytest.skip("spatial_spec_version not found; skipping")
        return

    try:
        v = spatial_spec_version()
    except ImportError:
        # The underlying SpaTiaL package isn't installed in this environment.
        pytest.skip("spatial-spec not installed")
        return

    assert isinstance(v, str)
    assert v.strip() != ""


def test_spatial_spec_version_prefers_module_dunder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``spatial_spec.__version__`` exists, it is returned verbatim."""
    mod = _import_or_skip()
    spatial_spec_version = getattr(mod, "spatial_spec_version", None)
    if spatial_spec_version is None:  # pragma: no cover - defensive
        pytest.skip("spatial_spec_version not found; skipping")
        return

    dummy = _make_dummy_spatial_spec(version="9.9.9-test")
    # Ensure our dummy is used regardless of whether the real package exists.
    monkeypatch.setitem(sys.modules, "spatial_spec", dummy)
    assert spatial_spec_version() == "9.9.9-test"


def test_spatial_spec_version_falls_back_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``__version__`` is absent, we fall back to the literal ``"unknown"``."""
    mod = _import_or_skip()
    spatial_spec_version = getattr(mod, "spatial_spec_version", None)
    if spatial_spec_version is None:  # pragma: no cover - defensive
        pytest.skip("spatial_spec_version not found; skipping")
        return

    dummy = _make_dummy_spatial_spec(version=None)
    monkeypatch.setitem(sys.modules, "spatial_spec", dummy)
    assert spatial_spec_version() == "unknown"


def test_spatial_spec_version_raises_importerror_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``spatial_spec`` cannot be imported *at all*, ImportError surfaces."""
    mod = _import_or_skip()
    spatial_spec_version = getattr(mod, "spatial_spec_version", None)
    if spatial_spec_version is None:  # pragma: no cover - defensive
        pytest.skip("spatial_spec_version not found; skipping")
        return

    # Ensure we're not reusing a pre-imported real module.
    monkeypatch.delitem(sys.modules, "spatial_spec", raising=False)

    real_import = builtins.__import__

    def _block_spatial_spec(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "spatial_spec":
            raise ModuleNotFoundError("No module named 'spatial_spec'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_spatial_spec)
    with pytest.raises(ImportError):
        spatial_spec_version()


def test_spatial_spec_available_truth_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the convenience availability helper if the module exposes it."""
    mod = _import_or_skip()
    spatial_spec_available = getattr(mod, "spatial_spec_available", None)
    if spatial_spec_available is None:
        pytest.skip("spatial_spec_available not exported; skipping")
        return

    # Case 1: present in sys.modules -> True
    monkeypatch.setitem(sys.modules, "spatial_spec", _make_dummy_spatial_spec(version="0.0"))
    assert spatial_spec_available() is True

    # Case 2: import blocked -> False
    monkeypatch.delitem(sys.modules, "spatial_spec", raising=False)
    real_import = builtins.__import__

    def _block(name, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "spatial_spec":
            raise ModuleNotFoundError("blocked")  # the message is irrelevant
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _block)
    assert spatial_spec_available() is False


def test_spatial_spec_constant_names_if_present() -> None:
    """Optional: constants match public names if exported by the helper module."""
    mod = _import_or_skip()
    dist = getattr(mod, "SPATIAL_SPEC_DIST_NAME", None)
    module = getattr(mod, "SPATIAL_SPEC_MODULE_NAME", None)
    if dist is None or module is None:
        pytest.skip("constants not exported; skipping")
        return
    assert isinstance(dist, str) and isinstance(module, str)
    assert dist == "spatial-spec"
    assert module == "spatial_spec"
