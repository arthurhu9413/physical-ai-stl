#!/usr/bin/env python3
"""
run_experiment.py
-----------------
A small, dependency‑light **experiment launcher** for the Physical‑AI‑STL
repository.

Goals
-----
• Professor‑friendly: one command runs a well‑named, reproducible experiment.
• Framework‑agnostic: delegates to `src/physical_ai_stl/experiments/*`.
• Robust: helpful errors, seed control, dry‑run, Cartesian sweeps, resume‑safe.
• Efficient: optional multi‑process execution for small local sweeps.

Example
-------
# list experiments (with availability if the package exposes it)
python scripts/run_experiment.py --list

# run from YAML (see configs/*.yaml)
python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml

# override values (YAML typed) and sweep two parameters (2×3 = 6 runs)
python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml \
    --set seed=1 --set model.hidden=[64,64] \
    --set sweep.model.activation=['tanh','gelu'] \
    --set sweep.training.epochs=[400,800,1200] -j 2

Design notes
------------
• Uses PyYAML *only*; everything else is from the stdlib.
• Performs minimal config validation; experiments do the heavy lifting.
• Discovers experiments dynamically. If the package provides a registry
  (``physical_ai_stl.experiments``), we prefer that; otherwise we fall back
  to module discovery.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import socket
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable
from datetime import datetime

# ---------------------------------------------------------------------------
# YAML loader (with crisp errors) + tiny `include` + deep-merge
# ---------------------------------------------------------------------------

def _require_yaml():
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover - friendly fatal
        raise SystemExit(
            "Missing dependency: pyyaml. Install it with:\n"
            "  pip install pyyaml\n"
            "or via extras:\n"
            "  pip install -r requirements-extra.txt"
        ) from e
    return yaml

def _read_text(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    with open(path, encoding="utf-8") as f:
        return f.read()

def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Return *a merged with b*, recursively (b overrides a)."""
    out = dict(a)
    for k, bv in b.items():
        av = out.get(k)
        if isinstance(av, dict) and isinstance(bv, dict):
            out[k] = _deep_merge(av, bv)
        else:
            out[k] = bv
    return out

def load_yaml(path: str) -> dict[str, Any]:
    yaml = _require_yaml()
    # Resolve working directory for relative includes
    path = os.path.expanduser(os.path.expandvars(path))
    base_dir = os.path.abspath(os.path.dirname(path))
    raw = _read_text(path)

    # Parse once to check for `include` keys
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Top‑level YAML must be a mapping (got {type(data)})")

    # Handle lightweight includes: either a string or list of strings.
    inc_val = data.pop("include", None)
    merged: dict[str, Any] = {}
    if inc_val:
        inc_paths = inc_val if isinstance(inc_val, list) else [inc_val]
        for p in inc_paths:
            if not isinstance(p, str):
                raise SystemExit("Each 'include' entry must be a string path")
            inc_path = p
            if not os.path.isabs(inc_path):
                inc_path = os.path.join(base_dir, inc_path)
            merged = _deep_merge(merged, load_yaml(inc_path))

    # Now merge *this file* over any included base(s).
    merged = _deep_merge(merged, data)

    # Expand env vars and ~ inside strings *post‑merge* to avoid YAML surprises.
    def _expand(obj: Any) -> Any:
        if isinstance(obj, str):
            return os.path.expandvars(os.path.expanduser(obj))
        if isinstance(obj, (list, tuple)):
            return type(obj)(_expand(v) for v in obj)
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        return obj

    return _expand(merged)

# ---------------------------------------------------------------------------
# Small, dependency‑free experiment registry (prefers package registry)
# ---------------------------------------------------------------------------

def _ensure_src_on_path() -> None:
    try:
        import physical_ai_stl  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    src = os.path.join(repo_root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)

@dataclass(frozen=True)
class ExpInfo:
    name: str                      # e.g. 'diffusion1d'
    module: str                    # 'physical_ai_stl.experiments.diffusion1d'
    run_candidates: tuple[str, ...]  # ordered possible run function names

def _discover_via_registry() -> list[ExpInfo] | None:
    """If the package exposes a registry, use it for names and availability."""
    _ensure_src_on_path()
    try:
        import physical_ai_stl.experiments as exps  # type: ignore
    except Exception:
        return None

    infos: list[ExpInfo] = []
    # Prefer registry introspection if provided
    names_fn: Callable[[], list[str]] | None = getattr(exps, "names", None)  # type: ignore[assignment]
    if names_fn is None:
        return None
    names = names_fn()
    for n in names:
        module = f"physical_ai_stl.experiments.{n}"
        candidates = (f"run_{n}", "run")
        infos.append(ExpInfo(name=n, module=module, run_candidates=candidates))
    return sorted(infos, key=lambda i: i.name)

def discover_experiments() -> list[ExpInfo]:
    reg = _discover_via_registry()
    if reg is not None:
        return reg

    _ensure_src_on_path()
    try:
        pkg = importlib.import_module("physical_ai_stl.experiments")
    except Exception as e:
        raise SystemExit(
            "Cannot import 'physical_ai_stl.experiments'. If running from a clone,\n"
            "ensure the repository root's 'src/' is on PYTHONPATH or install the\n"
            "package (e.g., 'pip install -e .').\n\n"
            f"Original error: {e}"
        ) from e

    infos: list[ExpInfo] = []
    for modinfo in pkgutil.iter_modules(pkg.__path__):  # type: ignore[attr-defined]
        name = modinfo.name
        module = f"physical_ai_stl.experiments.{name}"  # consider only modules for now
        candidates = (f"run_{name}", "run")
        infos.append(ExpInfo(name=name, module=module, run_candidates=candidates))
    return sorted(infos, key=lambda i: i.name)

def get_runner(exp_name: str):
    infos = {i.name: i for i in discover_experiments()}
    if exp_name not in infos:
        available = ", ".join(sorted(infos))
        raise SystemExit(f"Unknown experiment '{exp_name}'. Available: [{available}]")
    info = infos[exp_name]
    mod = importlib.import_module(info.module)
    for fn in info.run_candidates:
        if hasattr(mod, fn):
            return getattr(mod, fn)
    raise SystemExit(
        f"No runnable function found in {info.module}. Tried: {info.run_candidates}"
    )

# ---------------------------------------------------------------------------
# Config utilities: dotted overrides and tiny sweep helper
# ---------------------------------------------------------------------------

def _parse_override(s: str) -> tuple[list[str], Any]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("--set expects KEY=VALUE (use quotes for lists)")
    key, val = s.split("=", 1)
    key_parts = [k for k in key.split(".") if k]
    if not key_parts:
        raise argparse.ArgumentTypeError(f"Invalid key in override: {s}")
    # Reuse YAML parser to get numbers/bools/lists right
    yaml = _require_yaml()
    value = yaml.safe_load(val)
    return key_parts, value

def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value

def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    for o in overrides:
        keys, value = _parse_override(o)
        _set_nested(cfg, keys, value)
    return cfg

def iter_sweep_cfgs(base: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    sweep = base.get("sweep")
    if not sweep:
        yield "", base
        return

    # Normalize: keys (dotted) -> lists of values
    items: list[tuple[list[str], list[Any]]] = []
    for k, v in sweep.items():
        keys = [p for p in str(k).split(".") if p]
        if not isinstance(v, list) or len(v) == 0:
            raise SystemExit(f"Each sweep entry must be a non-empty list: {k}")
        items.append((keys, v))

    # Cartesian product
    from itertools import product

    for combo in product(*[vals for _, vals in items]):
        cfg = deepcopy(base)
        parts: list[str] = []
        for (keys, _), v in zip(items, combo, strict=True):
            _set_nested(cfg, keys, v)
            # Build a compact, file-system-safe suffix
            sval = repr(v).replace(" ", "")
            sval = sval.replace("/", "-").replace(os.sep, "-")
            parts.append(f"{'.'.join(keys)}={sval}")
        yield "__".join(parts), cfg

# ---------------------------------------------------------------------------
# Seeding and run directory handling
# ---------------------------------------------------------------------------

def try_set_seed(seed: int | None) -> None:
    if seed is None:
        return
    try:
        import random
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no-op if CUDA absent
        if hasattr(torch, "use_deterministic_algorithms"):
            # default to fast/compatible; experiments can override if needed
            torch.use_deterministic_algorithms(False)
    except Exception:
        pass

def _unique_timestamp() -> str:
    # Include microseconds to avoid collisions in parallel runs
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")

def make_run_dir(cfg: dict[str, Any]) -> str:
    io_cfg = cfg.setdefault("io", {})
    results_dir = str(io_cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    exp = str(cfg.get("experiment", "")).strip() or "exp"
    tag = str(cfg.get("tag", "run")).strip() or "run"
    ts = _unique_timestamp()
    base = os.path.join(results_dir, f"{exp}--{tag}--{ts}")
    run_dir = base
    # Be robust if multiple procs hit the same microsecond
    i = 1
    while True:
        try:
            os.makedirs(run_dir)
            break
        except FileExistsError:
            i += 1
            run_dir = f"{base}-{i}"
    io_cfg["run_dir"] = run_dir
    return run_dir

def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def dump_effective_config(run_dir: str, cfg: dict[str, Any]) -> None:
    yaml = _require_yaml()
    path = os.path.join(run_dir, "config.effective.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def _env_summary() -> dict[str, Any]:
    """Collect a tiny, dependency‑tolerant environment summary."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "hostname": socket.gethostname(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Optional: torch / CUDA
    try:
        import torch  # type: ignore
        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception:
        pass
    # Best-effort git hash (if available)
    try:
        import subprocess
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
            .decode()
            .strip()
        )
        info["git"] = {"rev": sha}
    except Exception:
        pass
    return info

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generic runner for Physical‑AI–STL experiments (YAML‑driven).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", required=False, help="Path to YAML config.")
    p.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit.",
    )
    p.add_argument(
        "--describe",
        metavar="EXP",
        help="Describe one experiment (if supported by the package) and exit.",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help=("Override config with KEY=VALUE (YAML values). Can be repeated."),
    )
    p.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved config (after --set and includes) and exit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the resolved experiment without running.",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Parallel processes for sweeps (use 1 to run sequentially).",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="If a run fails, continue the sweep instead of aborting.",
    )
    return p

# ---------------------------------------------------------------------------
# Worker (separate function so it is picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _run_worker(exp: str, subcfg: dict[str, Any], suffix: str | None) -> dict[str, Any]:
    """Run a single sub‑experiment. Returns a small result dict."""
    try:
        try_set_seed(subcfg.get("seed"))
        run_dir = make_run_dir(subcfg)
        if suffix:
            subcfg.setdefault("io", {})["run_dir"] = os.path.join(run_dir, suffix)
            os.makedirs(subcfg["io"]["run_dir"], exist_ok=True)
            run_dir = subcfg["io"]["run_dir"]
        # Persist inputs early for provenance
        dump_effective_config(run_dir, subcfg)
        _write_json(os.path.join(run_dir, "env.json"), _env_summary())

        # Import runner lazily inside the worker
        runner = get_runner(exp)
        t0 = time.perf_counter()
        out = runner(subcfg)  # type: ignore[misc]
        dt = time.perf_counter() - t0

        # If the runner returns a mapping, persist it as metrics.json
        if isinstance(out, dict):
            _write_json(os.path.join(run_dir, "metrics.json"), out)

        return {"ok": True, "run_dir": run_dir, "elapsed_s": dt, "out": out}
    except Exception as e:
        # Best‑effort report; keep the sweep going if requested
        try:
            # If run_dir exists, write an error file
            rd = subcfg.get("io", {}).get("run_dir")
            if isinstance(rd, str):
                with open(os.path.join(rd, "error.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{e!r}\n\n{traceback.format_exc()}")
        except Exception:
            pass
        return {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    # --list and --describe are independent of specific configs
    if args.list:
        # Prefer package's built‑in summary if available
        try:
            _ensure_src_on_path()
            import physical_ai_stl.experiments as exps  # type: ignore
            print(exps.about())  # type: ignore[attr-defined]
        except Exception:
            infos = discover_experiments()
            print("Available experiments:")
            for i in infos:
                print(f"  - {i.name}  (module: {i.module})")
        return

    if args.describe:
        try:
            _ensure_src_on_path()
            import physical_ai_stl.experiments as exps  # type: ignore
            if hasattr(exps, "describe"):
                print(exps.describe(args.describe))  # type: ignore[attr-defined]
            else:
                # Fallback: show module docstring if any
                info = {i.name: i for i in discover_experiments()}.get(args.describe)
                if not info:
                    raise SystemExit(f"Unknown experiment: {args.describe}")
                mod = importlib.import_module(info.module)
                print((mod.__doc__ or "").strip() or f"No description for {args.describe}")
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"Failed to describe '{args.describe}': {e}") from e
        return

    if not args.config:
        raise SystemExit("--config is required unless using --list/--describe")

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    # Experiment name (allow inference from filename)
    exp = str(cfg.get("experiment", "")).strip().lower()
    if not exp:
        base = os.path.basename(os.path.splitext(args.config)[0])
        exp = base.split("_")[0].lower()
        cfg["experiment"] = exp

    # Optional: print the fully resolved config and exit
    if args.show_config:
        yaml = _require_yaml()
        print("# --- Resolved config ---")
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    # Materialize sweep list up‑front so we can show progress and parallelize
    sweep_items = list(iter_sweep_cfgs(cfg))
    total = len(sweep_items)
    if args.dry_run:
        if total == 0:
            print("[DRY‑RUN] No runs were scheduled (empty sweep?).")
            return
        print(f"[DRY‑RUN] {total} run(s) planned:")
        for suffix, subcfg in sweep_items:
            tag = subcfg.get("tag", "")
            print(f"  • {exp}  tag='{tag}'  sweep='{suffix or '-'}'")
        return

    # Sequential (default) or parallel execution
    ran_any = False
    if max(1, int(args.jobs)) == 1 or total <= 1:
        for idx, (suffix, subcfg) in enumerate(sweep_items, start=1):
            print(f"[{idx}/{total}] {exp}  tag='{subcfg.get('tag','')}'  sweep='{suffix or '-'}'")
            res = _run_worker(exp, subcfg, suffix or None)
            if res.get("ok"):
                print(f"  ↳ done in {res['elapsed_s']:.1f}s → {res['run_dir']}")
                ran_any = True
            else:
                print(f"  ↳ FAILED: {res.get('error')}")
                if not args.keep_going:
                    raise SystemExit("Aborting on failure. Use --keep-going to continue.")
    else:
        jobs = max(1, int(args.jobs))
        print(f"Launching {total} run(s) with {jobs} worker process(es)…")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=None) as ex:
            fut2desc = {
                ex.submit(_run_worker, exp, subcfg, suffix or None): (suffix, subcfg)
                for suffix, subcfg in sweep_items
            }
            for i, fut in enumerate(as_completed(fut2desc), start=1):
                suffix, subcfg = fut2desc[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"[{i}/{total}] {exp}  tag='{subcfg.get('tag','')}'  sweep='{suffix or '-'}'")
                    print(f"  ↳ FAILED (executor): {e!r}")
                    if not args.keep_going:
                        raise
                    continue

                print(f"[{i}/{total}] {exp}  tag='{subcfg.get('tag','')}'  sweep='{suffix or '-'}'")
                if res.get("ok"):
                    print(f"  ↳ done in {res['elapsed_s']:.1f}s → {res['run_dir']}")
                    ran_any = True
                else:
                    print(f"  ↳ FAILED: {res.get('error')}")
                    if not args.keep_going:
                        raise SystemExit("Aborting on failure. Use --keep-going to continue.")

    if not ran_any:
        print("No runs were executed.")

if __name__ == "__main__":  # pragma: no cover
    main()
