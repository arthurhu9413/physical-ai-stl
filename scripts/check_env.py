#!/usr/bin/env python3
"""physical-ai-stl — environment quick check

Purpose
-------
Give a crisp, fast, and actionable summary of the local environment for the
CS‑3860 "physical AI + STL" repo. It focuses on three physics‑ML stacks
and three STL/STREL monitoring toolkits discussed with Prof. Johnson:

  • Frameworks:    NeuroMANCER, NVIDIA PhysicsNeMo, Bosch TorchPhysics
  • STL tooling:   RTAMT (STL), MoonLight (STREL, requires Java≥21), SpaTiaL (object‑centric specs; needs MONA+ltlf2dfa)

Design goals
------------
1) **Robust**: never crash; import lazily; degrade gracefully.
2) **Fast**: no heavy computation; at most a few subprocess calls.
3) **Actionable**: if something is missing, print the exact install hint.
4) **Portable**: Linux/macOS/Windows/WSL; CPU‑only friendly.
5) **Machine‑readable**: `--json` and `--md` outputs for CI/reporting.

Exit code
---------
• 0  → all *core* items are present (Python≥3.10, NumPy).
• 1  → some *core* items are missing (useful for CI).

Usage
-----
  python scripts/check_env.py            # human summary
  python scripts/check_env.py --md       # Markdown table
  python scripts/check_env.py --json     # JSON diagnostic blob
  python -m scripts.check_env --quick    # skip slow external probes

"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

# ------------------------------- utilities ---------------------------------

def _supports_color(stream) -> bool:
    try:
        import curses  # noqa: F401
        return stream.isatty()
    except Exception:
        return False

class _Color:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _c(self, code: str) -> str:
        return code if self.enabled else ""

    @property
    def reset(self) -> str: return self._c("\033[0m")
    @property
    def dim(self) -> str: return self._c("\033[2m")
    @property
    def bold(self) -> str: return self._c("\033[1m")
    @property
    def green(self) -> str: return self._c("\033[32m")
    @property
    def red(self) -> str: return self._c("\033[31m")
    @property
    def yellow(self) -> str: return self._c("\033[33m")
    @property
    def blue(self) -> str: return self._c("\033[34m")
    @property
    def magenta(self) -> str: return self._c("\033[35m")

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def _run(cmd: List[str], timeout: float = 2.0) -> Tuple[int, str, str]:
    """Run a small command; never raise; short timeout."""
    try:
        p = subprocess.run(
            cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"

# ------------------------------- versions ----------------------------------

def _version_of_distribution(names: Iterable[str]) -> Optional[str]:
    """Try importlib.metadata first; fall back to module __version__ on import.
    Special-case Python itself to report the interpreter version.
    """
    # Handle Python interpreter explicitly
    lowered = tuple(n.lower() for n in names)
    if "python" in lowered:
        try:
            return platform.python_version()
        except Exception:
            pass
    try:
        from importlib import metadata as _md  # Py3.8+
    except Exception:  # pragma: no cover
        try:
            import importlib_metadata as _md  # type: ignore
        except Exception:
            _md = None  # type: ignore

    # 1) try package metadata
    if _md is not None:
        for n in names:
            try:
                return _md.version(n)
            except Exception:
                pass
    # 2) try module import
    for n in names:
        try:
            mod = importlib.import_module(n)
            v = getattr(mod, "__version__", None)
            if isinstance(v, str) and v:
                return v
        except Exception:
            continue
    return None

def _parse_version(v: str) -> Tuple:
    """Lenient version parser (digits + text)."""
    try:
        from packaging.version import Version  # type: ignore
        return (Version(v),)
    except Exception:
        import re as _re
        parts = _re.split(r"[^0-9A-Za-z]+", v)
        norm: List[Any] = []
        for p in parts:
            if p == "":
                continue
            if p.isdigit():
                norm.append(int(p))
            else:
                norm.append(p.lower())
        return tuple(norm)

def _meets(v: Optional[str], min_v: Optional[str]) -> bool:
    if min_v is None:
        return True
    if v is None:
        return False
    return _parse_version(v) >= _parse_version(min_v)

# ----------------------------- probe schema --------------------------------

@dataclass(frozen=True)
class Dependency:
    display: str                    # human‑readable name
    import_names: Tuple[str, ...]   # module import path(s)
    pip_names: Tuple[str, ...]      # distribution names for pip/metadata
    required: bool = False
    min_version: Optional[str] = None
    notes: str = ""               # short hint
    extra_probe: Optional[Callable[[Optional[ModuleType]], Mapping[str, Any]]] = None

@dataclass
class ProbeResult:
    present: bool
    imported: bool
    version: Optional[str]
    message: str
    extra: Dict[str, Any]

def _probe(dep: Dependency, *, quick: bool = False) -> Tuple[Dependency, ProbeResult]:
    # presence via find_spec on any alias
    present = False
    spec_name: Optional[str] = None
    for name in dep.import_names:
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is not None:
            present = True
            spec_name = name
            break
    imported = False
    version = _version_of_distribution(dep.pip_names + dep.import_names)
    module: Optional[ModuleType] = None
    message = ""
    extra: Dict[str, Any] = {}
    if present and spec_name is not None:
        try:
            module = importlib.import_module(spec_name)
            imported = True
        except Exception as e:
            message = f"import error: {type(e).__name__}: {e}"
    else:
        message = "module not found"
    # min version check
    if dep.min_version and (version is not None) and not _meets(version, dep.min_version):
        message = f"needs >= {dep.min_version}; found {version}"
    # run lightweight extra probe (even if module failed to import)
    if dep.extra_probe is not None:
        try:
            if not quick:
                extra.update(dict(dep.extra_probe(module)))
        except Exception as e:
            extra.setdefault("warning", f"extra probe failed: {type(e).__name__}: {e}")
    # final message
    if not message:
        message = "OK" if imported else ("present (import failed)" if present else "missing")
    return dep, ProbeResult(present, imported, version, message, extra)

# ------------------------------ extra probes --------------------------------

def _probe_torch(mod: ModuleType) -> Mapping[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        return {"error": f"torch import failed: {type(e).__name__}: {e}"}
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    info["build_cuda"] = getattr(torch.version, "cuda", None)
    cudnn_v = None
    try:
        cudnn_v = torch.backends.cudnn.version()
    except Exception:
        pass
    info["cudnn"] = cudnn_v
    if torch.cuda.is_available():
        try:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name0"] = torch.cuda.get_device_name(0)
            info["capability0"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        except Exception:
            pass
    # Optional: nvidia-smi for driver (fast)
    if _which("nvidia-smi") and torch.cuda.is_available():
        code, out, err = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout=1.5)
        if code == 0 and out:
            line0 = out.splitlines()[0].strip()
            info["nvidia_smi"] = line0
        elif err:
            info["nvidia_smi"] = f"(error) {err.splitlines()[:1][0]}"
    return info

def _probe_java(_: ModuleType | None = None) -> Mapping[str, Any]:
    # Plain external probe; MoonLight requires Java 21+
    info: Dict[str, Any] = {}
    jpath = _which("java")
    if not jpath:
        info["java"] = "not found"
        return info
    code, out, err = _run([jpath, "-version"], timeout=1.5)
    raw = (out + "\n" + err).strip()
    info["java_raw"] = raw.splitlines()[0] if raw else ""
    # parse: 'openjdk version "21.0.2" ...'
    import re as _re
    m = _re.search(r"(\d+)(?:\.(\d+))?", raw)
    if m:
        major = int(m.group(1))
        info["java_major"] = major
        info["java_ok_for_moonlight"] = bool(major >= 21)
    return info


def _probe_mona(_: ModuleType | None = None) -> Mapping[str, Any]:
    """External probe for SpaTiaL: check MONA/ltlf2dfa availability."""
    info: Dict[str, Any] = {}
    mpath = _which("mona")
    if mpath:
        info["mona_available"] = True
        info["mona_path"] = mpath
        code, out, err = _run([mpath, "-v"], timeout=1.5)
        raw = (out or err or "").strip()
        if raw:
            info["mona_raw"] = raw.splitlines()[0]
    else:
        info["mona_available"] = False
    ltlf = _which("ltlf2dfa")
    if ltlf:
        info["ltlf2dfa_available"] = True
        info["ltlf2dfa_path"] = ltlf
    else:
        info["ltlf2dfa_available"] = False
    return info

# ------------------------------- inventory ---------------------------------

# Core: keep lean; the repo's base requirement is numpy
CORE: Tuple[Dependency, ...] = (
    Dependency(
        display="Python",
        import_names=("sys",),
        pip_names=("python",),
        required=True,
        min_version="3.10",
        notes="Project requires Python ≥ 3.10.",
    ),
    Dependency(
        display="NumPy",
        import_names=("numpy",),
        pip_names=("numpy",),
        required=True,
        min_version="1.26.4",
        notes="Vector operations & array API.",
    ),
)

# Physics‑ML frameworks (optional; highly recommended)
FRAMEWORKS: Tuple[Dependency, ...] = (
    Dependency(
        display="PyTorch",
        import_names=("torch",),
        pip_names=("torch",),
        min_version=None,  # we don't force a version here
        notes="Deep learning backend; GPU optional.",
        extra_probe=_probe_torch,
    ),
    Dependency(
        display="NeuroMANCER",
        import_names=("neuromancer",),
        pip_names=("neuromancer",),
        min_version=None,
        notes="PyTorch SciML + DPC; pip install neuromancer.",
    ),
    Dependency(
        display="PhysicsNeMo",
        import_names=("physicsnemo",),
        pip_names=("nvidia-physicsnemo",),
        min_version=None,
        notes="NVIDIA Physics‑ML; python -m pip install nvidia-physicsnemo[all].",
    ),
    Dependency(
        display="PhysicsNeMo‑Sym",
        import_names=("physicsnemo.sym",),
        pip_names=("nvidia-physicsnemo.sym","nvidia-physicsnemo-sym"),
        min_version=None,
        notes="Symbolic PDE utils; may require Cython; see NVIDIA docs; install via: python -m pip install \"Cython\" && python -m pip install nvidia-physicsnemo.sym --no-build-isolation.",
    ),
    Dependency(
        display="TorchPhysics",
        import_names=("torchphysics",),
        pip_names=("torchphysics",),
        min_version=None,
        notes="PINNs/DeepRitz/DeepONet/FNO; pip install torchphysics.",
    ),
)

# STL / STREL toolkits
STL_TOOLS: Tuple[Dependency, ...] = (
    Dependency(
        display="RTAMT (STL)",
        import_names=("rtamt",),
        pip_names=("rtamt",),
        min_version=None,
        notes="Signal Temporal Logic monitors; pip install rtamt.",
    ),
    Dependency(
        display="MoonLight (STREL)",
        import_names=("moonlight",),
        pip_names=("moonlight",),
        min_version=None,
        notes="STREL monitors; requires Java ≥ 21 (java -version).",
        extra_probe=lambda _m: _probe_java(None),
    ),
    Dependency(
        display="SpaTiaL / spatial‑spec",
        import_names=("spatial_spec",),
        pip_names=("spatial-spec",),
        min_version=None,
        notes="Spatio‑temporal relations; pip install spatial-spec (Linux/macOS).",
        extra_probe=lambda _m: _probe_mona(None),
    ),
)

EVERYTHING = CORE + FRAMEWORKS + STL_TOOLS

# ------------------------------- rendering ---------------------------------

def _status_icon(ok: bool, c: _Color, ascii_only: bool = False) -> str:
    if ascii_only:
        return "OK" if ok else "!!"
    return (c.green + "✔" + c.reset) if ok else (c.red + "✖" + c.reset)

def _render_row(dep: Dependency, res: ProbeResult, c: _Color, *, ascii_only: bool) -> str:
    parts: List[str] = []
    good = res.imported and (dep.min_version is None or _meets(res.version, dep.min_version))
    parts.append(_status_icon(good, c, ascii_only))
    parts.append(f"{dep.display}")
    ver = res.version or "—"
    parts.append(ver)
    msg = res.message
    if dep.notes and msg == "OK":
        msg = dep.notes
    parts.append(msg)
    # extra one‑liners
    if res.extra:
        extras = []
        for k in ("gpu_name0", "cuda_available", "mps_available", "build_cuda", "java_ok_for_moonlight", "mona_available", "ltlf2dfa_available"):
            if k in res.extra:
                extras.append(f"{k}={res.extra[k]}")
        if extras:
            parts.append("[" + ", ".join(extras) + "]")
    return "  ".join(parts)

def _print_human(results: Mapping[str, Tuple[Dependency, ProbeResult]], *, ascii_only: bool, extended: bool) -> None:
    c = _Color(enabled=_supports_color(sys.stdout) and not ascii_only)
    sep = c.dim + ("-" * 79) + c.reset
    print(c.bold + "physical-ai-stl • environment check" + c.reset)
    print(sep)
    # Host information
    sys_line = f"{platform.platform()} • Python {results.get('Python', (None, ProbeResult(False, False, None, '', {})))[1].version or platform.python_version()}"
    print(c.dim + sys_line + c.reset)
    # Detect environment manager hints (conda/venv) for user context
    try:
        env_mgr = []
        if os.environ.get("CONDA_DEFAULT_ENV"):
            env_mgr.append(f"conda:{os.environ.get('CONDA_DEFAULT_ENV')}")
        if os.environ.get("VIRTUAL_ENV"):
            env_mgr.append("venv")
        if env_mgr:
            print(c.dim + "env: " + ", ".join(env_mgr) + c.reset)
    except Exception:
        pass

    def block(title: str, names: Iterable[str]) -> None:
        print(c.blue + title + c.reset)
        for name in names:
            dep, res = results[name]
            print("  " + _render_row(dep, res, c, ascii_only=ascii_only))
        print(sep)

    block("Core", [d.display for d in CORE])
    block("Physics‑ML frameworks", [d.display for d in FRAMEWORKS])
    block("STL / STREL", [d.display for d in STL_TOOLS])

    # Helpful hints for missing items
    hints: List[str] = []
    # generic pip hints for missing imports
    for dep, res in results.values():
        if not res.imported and dep.pip_names:
            # prefer first pip name for hint
            # use python -m pip for robustness across environments
            hints.append(f"python -m pip install {dep.pip_names[0]}")
    # targeted OS-specific hints for external tools
    sysname = platform.system()
    # MoonLight: Java 21+
    if "MoonLight (STREL)" in results:
        dep_ml, res_ml = results["MoonLight (STREL)"]
        jr = res_ml.extra or {}
        need_java = (jr.get("java") == "not found") or (jr.get("java_major") and int(jr.get("java_major")) < 21)
        if need_java:
            if sysname == "Darwin":
                hints.append("brew install openjdk@21    # ensure JAVA_HOME and PATH are set.")  # see Homebrew docs
            elif sysname == "Windows":
                hints.append("winget install --id=Microsoft.OpenJDK.21 -e    # or: choco install temurin --version=21")  # Windows package managers
            else:
                hints.append("sudo apt-get update && sudo apt-get install -y openjdk-21-jre    # or openjdk-21-jdk")
    # SpaTiaL: MONA + ltlf2dfa
    if "SpaTiaL / spatial‑spec" in results:
        dep_sp, res_sp = results["SpaTiaL / spatial‑spec"]
        er = res_sp.extra or {}
        if not er.get("mona_available", False):
            if sysname == "Linux":
                hints.append("sudo apt-get install -y mona    # MONA automata tool required by spatial-spec")
            elif sysname == "Darwin":
                hints.append("Install MONA from https://www.brics.dk/mona/ (build from source)    # Homebrew formula may not exist")
            elif sysname == "Windows":
                hints.append("Download mona.exe from https://www.brics.dk/mona/ and ensure it is on PATH")
        if not er.get("ltlf2dfa_available", False):
            hints.append("python -m pip install ltlf2dfa    # required by spatial-spec for automata conversion")
    if hints:
        print(c.magenta + "Install hints:" + c.reset)
        # de‑duplicate while preserving order
        seen = set()
        unique = [h for h in hints if not (h in seen or seen.add(h))]
        for h in unique:
            print("  • " + h)

    # Extended details
    if extended:
        print(sep)
        print(c.yellow + "Details:" + c.reset)
        for name, (dep, res) in results.items():
            if res.extra:
                print(f"  {dep.display}:")
                for k, v in res.extra.items():
                    print(f"    - {k}: {v}")
        print(sep)

def _print_markdown(results: Mapping[str, Tuple[Dependency, ProbeResult]], *, extended: bool) -> None:
    # Markdown table for README/issue pasting
    headers = ["Status", "Package", "Version", "Notes"]
    print("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |" )
    for dep, res in results.values():
        ok = res.imported and (dep.min_version is None or _meets(res.version, dep.min_version))
        status = "OK" if ok else "Missing"
        ver = res.version or "—"
        note = res.message if res.message != "OK" else dep.notes
        print(f"| {status} | {dep.display} | {ver} | {note} |")

def _print_json(results: Mapping[str, Tuple[Dependency, ProbeResult]]) -> None:
    out = {
        name: {
            "present": res.present,
            "imported": res.imported,
            "version": res.version,
            "message": res.message,
            "extra": res.extra,
            "required": dep.required,
            "min_version": dep.min_version,
        }
        for name, (dep, res) in results.items()
    }
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()

# ---------------------------------- main -----------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check environment for physical-ai-stl.")
    p.add_argument("--json", action="store_true", help="machine‑readable JSON output")
    p.add_argument("--md", action="store_true", help="markdown table output")
    p.add_argument("--plain", action="store_true", help="ASCII only (no colors/emoji)")
    p.add_argument("--quick", action="store_true", help="skip external probes (java, nvidia‑smi)")
    p.add_argument("--extended", action="store_true", help="reserved for future use (no‑op)")
    args = p.parse_args(argv)

    # Prepare color
    c = _Color(enabled=_supports_color(sys.stdout) and not args.plain)

    # Probe everything deterministically
    results: Dict[str, Tuple[Dependency, ProbeResult]] = {}
    for dep in EVERYTHING:
        d, r = _probe(dep, quick=args.quick)
        results[dep.display] = (d, r)

    if args.json:
        _print_json(results)
    elif args.md:
        _print_markdown(results, extended=args.extended)
    else:
        _print_human(results, ascii_only=args.plain, extended=args.extended)

    # Exit 0 on all required present/importable with version OK
    missing_core = [
        d.display
        for d in CORE
        if not (
            results[d.display][1].imported and _meets(results[d.display][1].version, d.min_version)
        )
    ]
    return 0 if not missing_core else 1

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
