# tests/conftest.py — hermetic test bootstrap
# Goal order: (1) determinism & stability on laptops/CI  (2) sane performance  (3) readability.
# - Single-thread math kernels to prevent BLAS/OpenMP storms on small machines.
# - Reproducible RNG across Python/NumPy/(optional) Torch/CuPy/TensorFlow.
# - Conservative PyTorch cudnn/cublas knobs (no-ops on CPU) to prefer determinism.
# - Helpful header to make runs auditable.
from __future__ import annotations

import os
import random
import time
from typing import Callable

# -------------------------
# A. Environment defaults
# -------------------------
def _setdefault_env(var: str, value: str) -> None:
    # Respect user/CI overrides when present.
    os.environ.setdefault(var, value)

# Keep timestamps consistent in assertions/logs (POSIX only).
_setdefault_env("TZ", "UTC")
try:  # Windows lacks time.tzset
    time.tzset()
except Exception:
    pass

# Avoid thread explosions across common math stacks.
for _var in (
    "OMP_NUM_THREADS",        # OpenMP consumers (NumPy/SciPy, PyTorch, etc.)
    "OPENBLAS_NUM_THREADS",   # OpenBLAS
    "MKL_NUM_THREADS",        # Intel/oneMKL
    "NUMEXPR_NUM_THREADS",    # numexpr
    "VECLIB_MAXIMUM_THREADS", # Apple Accelerate
    "BLIS_NUM_THREADS",       # BLIS / AOCL
    "RAYON_NUM_THREADS",      # Rust rayon (used by some tokenizers/other deps)
):
    _setdefault_env(_var, "1")

# Quiet & de-parallelize HuggingFace tokenizers (prevents deadlocks after fork).
_setdefault_env("TOKENIZERS_PARALLELISM", "false")

# Encourage deterministic cuBLAS behavior when CUDA is present.
# (Safe on CPU-only machines; PyTorch checks this when initializing cuBLAS)
_setdefault_env("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# Hint OpenMP runtime not to busy-spin.
_setdefault_env("OMP_WAIT_POLICY", "PASSIVE")

# NOTE: For fully deterministic hashing across *process restarts* set this
# *before* Python starts, e.g. run:  PYTHONHASHSEED=0 pytest
# os.environ.setdefault("PYTHONHASHSEED", "0")  # (Setting it here has no effect.)

# ------------------------------------
# B. Seed everything we might import
# ------------------------------------
DEFAULT_SEED = int(os.environ.get("TEST_SEED", "0"))

def _seed_core(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass

def _try(module: str, fn: Callable) -> None:
    try:
        mod = __import__(module)
        fn(mod)  # type: ignore[misc]
    except Exception:
        pass

def _seed_torch(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)  # no-op if CUDA unavailable
        except Exception:
            pass

        # Keep thread usage predictable.
        for setter in (getattr(torch, "set_num_threads", None),
                       getattr(torch, "set_num_interop_threads", None)):
            try:
                if setter is not None:
                    setter(1)
            except Exception:
                pass

        # Prefer deterministic behavior on GPUs.
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # Disable TF32 to reduce subtle bitwise differences across GPUs.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

        # Ban known nondeterministic ops where possible (warn-only to avoid brittle failures).
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass  # Torch is optional for parts of the suite

def _seed_optional_backends(seed: int) -> None:
    _try("cupy", lambda cp: cp.random.seed(seed))          # GPU arrays (optional)
    _try("tensorflow", lambda tf: tf.random.set_seed(seed)) # TF (optional)

# Initial seeding at import time:
_seed_core(DEFAULT_SEED)
_seed_torch(DEFAULT_SEED)
_seed_optional_backends(DEFAULT_SEED)

# ------------------------------------
# C. Nice-to-have PyTest integration
# ------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        action="store",
        default=str(DEFAULT_SEED),
        help="Override global RNG seed (default: env TEST_SEED or 0).",
    )

def pytest_configure(config):
    # Re-seed if user provided --seed on the CLI.
    try:
        cli_seed = int(config.getoption("--seed"))
    except Exception:
        cli_seed = DEFAULT_SEED
    if cli_seed != DEFAULT_SEED:
        _seed_core(cli_seed)
        _seed_torch(cli_seed)
        _seed_optional_backends(cli_seed)

def pytest_report_header(config):
    # Helpful for diagnostics and reproducibility.
    keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "CUBLAS_WORKSPACE_CONFIG",
        "TZ",
    ]
    knobs = ", ".join(f"{k}={os.environ.get(k)}" for k in keys if os.environ.get(k) is not None)
    return f"Determinism seed={os.environ.get('TEST_SEED', '0')} | {knobs}"
