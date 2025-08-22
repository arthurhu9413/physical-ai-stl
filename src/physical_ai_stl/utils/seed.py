from __future__ import annotations

import os
import random
import warnings
from typing import Any

__all__ = ["seed_everything", "seed_worker", "torch_generator"]


def _set_if_unset(env: str, value: str) -> None:
    """Set an environment variable iff it is not already set.

    Many CUDA/cuBLAS determinism switches are *read once* when the CUDA
    context or library is initialized. Respecting a user's pre-set
    value avoids surprising overrides.
    """
    if os.environ.get(env) is None:
        os.environ[env] = value


def _to_uint32(x: int) -> int:
    """Return ``x`` mapped into the unsigned 32‑bit range ``[0, 2**32 - 1]``.

    NumPy's ``np.random.seed`` requires seeds in this range.
    """
    # Using modulo keeps behavior stable even if negative values are passed.
    return int(x) % 2**32


def seed_everything(
    seed: int = 0,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
    set_pythonhashseed: bool = True,
    configure_cuda_env: bool = True,
    disable_tf32_when_deterministic: bool = True,
    verbose: bool = False,
) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch for reproducibility.

    This function follows PyTorch's reproducibility guidance for seeding
    and determinism. It intentionally **does not hard‑fail** if optional
    dependencies are missing, keeping it safe to call in lightweight
    environments (unit tests, CPU‑only CI, etc.).

    Parameters
    ----------
    seed:
        Base seed used for Python and (if available) NumPy and PyTorch.
    deterministic:
        If ``True``, request deterministic algorithms and disable cuDNN
        benchmarking. If ``False``, prefer performance‑oriented defaults.
    warn_only:
        Forwarded to ``torch.use_deterministic_algorithms(..., warn_only=...)``
        when available. When ``True`` PyTorch will emit warnings instead of
        raising if a strictly deterministic implementation is unavailable.
    set_pythonhashseed:
        If ``True``, ensure ``PYTHONHASHSEED`` is set (as a string). Note that
        changing this *after* interpreter start will not retroactively rehash
        existing hash‑based objects, but it is still helpful for child
        processes spawned later (e.g., ``DataLoader`` workers).
    configure_cuda_env:
        If ``True`` and determinism is requested, set CUDA/cuBLAS env knobs
        (e.g., ``CUBLAS_WORKSPACE_CONFIG``) **only if not already set**.
    disable_tf32_when_deterministic:
        If ``True`` and determinism is requested, attempt to disable TF32 on
        Ampere+ GPUs for full FP32 reproducibility.
    verbose:
        If ``True``, emit ``warnings.warn`` diagnostics instead of failing hard.
    """
    # --- Python stdlib RNG ----------------------------------------------------
    if set_pythonhashseed:
        # Respect an existing value so users can override from the shell.
        _set_if_unset("PYTHONHASHSEED", str(_to_uint32(seed)))
    random.seed(seed)

    # --- NumPy ----------------------------------------------------------------
    try:
        import numpy as np  # type: ignore

        np.random.seed(_to_uint32(seed))
    except Exception:
        # Keep going even if NumPy is unavailable.
        pass

    # --- PyTorch (optional) ---------------------------------------------------
    try:
        import torch  # type: ignore
    except Exception:
        return

    try:
        torch.manual_seed(seed)
        # If CUDA is built/available, also seed all CUDA devices.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # If something very old/broken, don't hard fail.
        if verbose:
            warnings.warn("Failed to set PyTorch seeds.", RuntimeWarning)

    # Determinism/performance toggles
    try:
        if deterministic:
            # Set environment knobs before CUDA context is created if possible.
            if configure_cuda_env and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                # Values recommended by NVIDIA/cuBLAS for reproducibility.
                # ":4096:8" generally gives best perf among deterministic choices.
                _set_if_unset("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                try:
                    # If CUDA context was already created, cuBLAS may ignore this.
                    if torch.cuda.is_available() and torch.cuda.is_initialized() and verbose:
                        warnings.warn(
                            "CUBLAS_WORKSPACE_CONFIG was set after CUDA initialization; "
                            "full determinism may not be guaranteed. Set it early in "
                            "your program for stricter guarantees.",
                            RuntimeWarning,
                        )
                except Exception:
                    pass

            # Prefer official switch when available.
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)  # type: ignore[call-arg]
            except TypeError:
                # For older PyTorch that lacks warn_only.
                torch.use_deterministic_algorithms(True)  # type: ignore[misc]
            except Exception:
                # Ignore if not supported.
                pass

            # cuDNN knobs
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass

            if disable_tf32_when_deterministic:
                # Disable TF32 on Ampere+ for fully consistent FP32 matmuls/convs.
                # Prefer the legacy flags for broad compatibility.
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Newer PyTorch releases expose per‑backend precision toggles; try them too.
                try:
                    # These attributes exist in recent versions (PyTorch ≥ 2.9).
                    getattr(torch.backends, "fp32_precision", None)
                    torch.backends.fp32_precision = "ieee"  # type: ignore[attr-defined]
                    torch.backends.cuda.matmul.fp32_precision = "ieee"  # type: ignore[attr-defined]
                    torch.backends.cudnn.fp32_precision = "ieee"  # type: ignore[attr-defined]
                except Exception:
                    # Older versions won't have these; silently ignore.
                    pass
        else:
            # Performance‑oriented path.
            try:
                torch.use_deterministic_algorithms(False)  # type: ignore[misc]
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            except Exception:
                pass
            # Allow TF32 where supported for speed.
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        if verbose:
            warnings.warn("Failed to configure PyTorch determinism/performance flags.", RuntimeWarning)


def seed_worker(worker_id: int) -> None:  # pragma: no cover - utility for DataLoader
    """Deterministically (re)seed NumPy & Python in a PyTorch worker process.

    This mirrors the official PyTorch recipe, deriving a 32‑bit seed from
    ``torch.initial_seed()`` so each worker has a different, reproducible
    seed even across epochs when a ``torch.Generator`` is passed to the
    ``DataLoader``.
    """
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except Exception:
        # Nothing to do if torch/numpy are unavailable.
        return

    # Derive a 32-bit seed from PyTorch's worker seed.
    worker_seed = int(torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Propagate for libraries that might key off PYTHONHASHSEED in child processes.
    _set_if_unset("PYTHONHASHSEED", str(worker_seed))


def torch_generator(seed: int, device: Any | None = None):
    """Create a ``torch.Generator`` seeded for deterministic DataLoader usage.

    Pass the returned generator to ``DataLoader(generator=...)`` to make
    shuffling, sampling, and augmentation repeatable across runs. Optionally
    place the generator on a specific device to avoid implicit device moves.
    """
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover - only used when torch is installed
        raise RuntimeError("torch is required for torch_generator") from e

    gen = torch.Generator(device=device) if device is not None else torch.Generator()
    gen.manual_seed(seed)
    return gen
