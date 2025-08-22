#!/usr/bin/env python3
"""
framework_survey.py
-------------------

Quick, robust survey of the **physics‑AI frameworks** and **STL/STREL toolkits**
discussed with Prof. Johnson for CS‑3860‑01.

What this prints
----------------
A compact table (text, Markdown, or JSON) with, for each library:

- human name
- whether it is installed
- detected version
- `import` module name (may differ from the PyPI distribution)
- primary PyPI name (for `pip install ...`)
- notes (e.g., Java requirement for MoonLight)

Design goals
------------
1) **Correct & actionable**: reflect current package naming (e.g., PhysicsNeMo),
   and call out runtime prerequisites (Java for MoonLight; MONA for SpaTiaL planning).
2) **Fast**: use only importlib.metadata for distribution versions; *optionally*
   import modules with `--deep`.
3) **Safe**: never crash; all probes are wrapped with timeouts and try/except.
4) **Pretty**: readable fixed‑width table in `--format text`, or Markdown table
   with `--format md`.

Examples
--------
    # Plain text overview
    python scripts/framework_survey.py

    # Markdown for a README
    python scripts/framework_survey.py --format md > docs/framework-survey.md

    # JSON for tooling
    python scripts/framework_survey.py --format json

    # Include runtime probes (imports / Java / CUDA)
    python scripts/framework_survey.py --deep

    # Filter to frameworks or STL tools only
    python scripts/framework_survey.py --only framework
    python scripts/framework_survey.py --only stl
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata as metadata
from typing import Any


# ------------------------------ spec -----------------------------------------


@dataclass(frozen=True)
class Pkg:
    name: str                   # Human-friendly name for display
    pip_names: tuple[str, ...]  # One or more possible distribution names on PyPI
    import_name: str | None     # Python import name (may differ from pip name)
    desc: str                   # One-line description
    category: str               # 'framework' | 'stl'


PKGS: list[Pkg] = [
    # --- Frameworks ---
    Pkg(
        name="Neuromancer",
        pip_names=("neuromancer",),
        import_name="neuromancer",
        desc=(
            "PyTorch-based differentiable programming for "
            "physics-informed optimization & control (SciML)."
        ),
        category="framework",
    ),
    Pkg(
        name="PhysicsNeMo",
        pip_names=("nvidia-physicsnemo",),
        import_name="physicsnemo",
        desc="NVIDIA framework for AI-driven multi-physics models (ex‑Modulus).",
        category="framework",
    ),
    Pkg(
        name="TorchPhysics",
        pip_names=("torchphysics",),
        import_name="torchphysics",
        desc="Bosch research library for PINNs/DeepRitz to solve ODE/PDE.",
        category="framework",
    ),
    # --- STL / STREL toolkits ---
    Pkg(
        name="RTAMT",
        pip_names=("rtamt",),
        import_name="rtamt",
        desc="Runtime STL monitoring (discrete & dense time, C++ backend).",
        category="stl",
    ),
    Pkg(
        name="MoonLight",
        pip_names=("moonlight",),
        import_name="moonlight",
        desc="STREL/STL monitoring (Java engine with Python bindings).",
        category="stl",
    ),
    Pkg(
        name="SpaTiaL",
        pip_names=("spatial-spec",),
        import_name="spatial_spec",  # PyPI name != import name
        desc="Object‑centric spatio‑temporal specifications (planning via MONA).",
        category="stl",
    ),
]


# ---------------------------- helpers ----------------------------------------


def _dist_version_for_names(names: tuple[str, ...]) -> str | None:
    """Return the first importlib.metadata version found for the given names.

    Handles PEP 503 normalization (dash vs underscore).
    """
    for n in names:
        try:
            return metadata.version(n)
        except Exception:
            try:
                return metadata.version(n.replace("-", "_"))
            except Exception:
                continue
    return None


def _import_version(import_name: str | None) -> str | None:
    """Import a module and try to extract a version string.

    Returns None if import fails or no version attribute found.
    """
    if not import_name:
        return None
    try:
        mod = importlib.import_module(import_name)
    except Exception:
        return None
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str) and v:
            return v
        # Some modules expose a tuple or object – stringify carefully.
        try:
            if v is not None:
                s = str(v)
                if s and s != repr(v):
                    return s
        except Exception:
            pass
    return None


def _check_java_version(timeout: float = 3.0) -> str | None:
    """Return the Java runtime version string or None if not found.

    Uses a short timeout and scrapes stderr/stdout from `java -version`.
    """
    try:
        proc = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r'\\version\\s+"([^"]+)"', blob)
        return m.group(1) if m else (blob.strip() or None)
    except Exception:
        return None


def _check_torch() -> dict[str, Any]:
    """Return a small dict summarizing torch + CUDA if available."""
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_available": None,
        "cuda_version": None,
        "gpus": None,
    }
    try:
        import torch  # type: ignore

        info["installed"] = True
        info["version"] = getattr(torch, "__version__", None)

        try:
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
            if info["cuda_available"]:
                try:
                    cnt = torch.cuda.device_count()
                    names = [torch.cuda.get_device_name(i) for i in range(cnt)]
                    info["gpus"] = names
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    return info


def _survey(*, deep: bool = False) -> dict[str, Any]:
    """Build the rows + system info dictionary.

    If `deep` is True, attempt to import packages (for runtime version) and
    query Java (for MoonLight) and CUDA (for torch).
    """
    rows: list[dict[str, Any]] = []

    for pkg in PKGS:
        # Distribution version from installed wheel/egg metadata (no import).
        dist_version = _dist_version_for_names(pkg.pip_names)
        # Runtime version from actual module import (optional).
        imp_version = _import_version(pkg.import_name) if deep else None
        installed = (dist_version is not None) or (imp_version is not None)

        # Contextual notes per package.
        notes = ""
        if pkg.name == "PhysicsNeMo":
            notes = "Import: physicsnemo · pip: nvidia-physicsnemo (optionally: [all])"
        elif pkg.name == "MoonLight":
            if deep:
                jv = _check_java_version()
                if jv:
                    # Java 21+ is the current requirement upstream.
                    major = None
                    m = re.match(r"(\\d+)", jv)
                    if m:
                        try:
                            major = int(m.group(1))
                        except Exception:
                            major = None
                    warn = "" if (major is not None and major >= 21) else " (need Java ≥21)"
                    notes = f"Java {jv} detected{warn}"
                else:
                    notes = "Requires Java ≥21 runtime (not detected)"
            else:
                notes = "Requires Java ≥21 runtime"
        elif pkg.name == "SpaTiaL":
            notes = "Planning uses MONA via ltlf2dfa; Windows unsupported for MONA"
        # else keep empty

        rows.append(
            {
                "name": pkg.name,
                "pip": pkg.pip_names[0],
                "import": pkg.import_name or "",
                "installed": bool(installed),
                "version": dist_version or imp_version or "not installed",
                "desc": pkg.desc,
                "category": pkg.category,
                "notes": notes,
            }
        )

    sysinfo: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _check_torch(),
    }
    if deep:
        sysinfo["java"] = _check_java_version() or "not detected"

    return {"rows": rows, "sys": sysinfo}


# ---------------------------- formatting -------------------------------------


def _format_text_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Package", "Installed", "Version", "Import", "PyPI", "Notes"]
    # Column target widths; values are soft limits (strings are truncated if needed).
    col_targets = [18, 9, 18, 18, 26, 56]

    # Compute actual widths (respecting soft caps while accommodating headers).
    def _cap(w_header: int, w_target: int, data: list[str]) -> int:
        mx = max([w_header] + [len(s) for s in data])
        return min(mx, w_target)

    cols: list[list[str]] = [
        [r["name"] for r in rows],
        [("yes" if r["installed"] else "no") for r in rows],
        [str(r["version"]) for r in rows],
        [r["import"] for r in rows],
        [r["pip"] for r in rows],
        [r["notes"] for r in rows],
    ]
    col_w = [
        _cap(len(headers[i]), col_targets[i], cols[i]) for i in range(len(headers))
    ]

    def _trunc(s: str, w: int) -> str:
        return s if len(s) <= w else (s[: max(0, w - 1)] + "…")

    def _line(values: list[str]) -> str:
        return "  ".join(_trunc(values[i], col_w[i]).ljust(col_w[i]) for i in range(len(headers)))

    out: list[str] = []
    out.append(_line(headers))
    out.append(_line(["-" * w for w in col_w]))
    for r in rows:
        out.append(
            _line(
                [
                    r["name"],
                    "yes" if r["installed"] else "no",
                    str(r["version"]),
                    r["import"],
                    r["pip"],
                    r["notes"],
                ]
            )
        )
    return "\n".join(out)


def _format_md_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Package", "Installed", "Version", "Import", "PyPI", "Notes"]
    md = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    r["name"],
                    "✅" if r["installed"] else "❌",
                    str(r["version"]),
                    f"`{r['import']}`" if r["import"] else "",
                    f"`{r['pip']}`",
                    r["notes"].replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(md)


# ------------------------------ CLI ------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Summarize versions & availability of physical‑AI frameworks and STL/STREL toolkits."
    )
    ap.add_argument(
        "--format",
        choices=["text", "md", "json"],
        default="text",
        help="Output format (default: text).",
    )
    ap.add_argument(
        "--deep",
        action="store_true",
        help="Import packages and probe runtime deps (Java/CUDA).",
    )
    ap.add_argument(
        "--only",
        choices=["all", "framework", "stl"],
        default="all",
        help="Filter rows by category.",
    )
    ap.add_argument(
        "--show-install",
        action="store_true",
        help="Print pip install lines for any missing packages.",
    )
    args = ap.parse_args(argv)

    result = _survey(deep=args.deep)
    rows = result["rows"]
    sysinfo = result["sys"]

    if args.only != "all":
        rows = [r for r in rows if r["category"] == args.only]

    # Render
    if args.format == "json":
        print(json.dumps({"rows": rows, "sys": sysinfo}, indent=2, sort_keys=True))
    elif args.format == "md":
        print(_format_md_table(rows))
        # Append a tiny system footer
        print("\n> Python", sysinfo["python"], "on", sysinfo["platform"])
        if args.deep:
            torch = sysinfo.get("torch", {})
            if torch and torch.get("installed"):
                cuda = "CUDA " + str(torch.get("cuda_version")) if torch.get("cuda_version") else "CPU"
                print(f"> torch {torch.get('version')} · {cuda}")
            if "java" in sysinfo:
                print(f"> Java: {sysinfo['java']}")
    else:
        print(_format_text_table(rows))
        print("\nSystem:")
        print(f"  Python:   {sysinfo['python']}")
        print(f"  Platform: {sysinfo['platform']}")
        if args.deep:
            torch = sysinfo.get("torch", {})
            if torch and torch.get("installed"):
                cuda = (
                    f"CUDA {torch.get('cuda_version')}, GPUs: {', '.join(torch.get('gpus') or [])}"
                    if torch.get("cuda_available")
                    else "CPU"
                )
                print(f"  torch:    {torch.get('version')} · {cuda}")
            if "java" in sysinfo:
                print(f"  Java:     {sysinfo['java']}")

    if args.show_install:
        missing = [r for r in rows if not r["installed"]]
        if missing:
            print("\nInstall commands for missing packages:")
            for r in missing:
                # Prefer the primary pip name; PhysicsNeMo benefits from extras by default.
                pip_str = "nvidia-physicsnemo[all]" if r["name"] == "PhysicsNeMo" else r["pip"]
                print(f"  pip install {pip_str}")
        else:
            print("\nAll listed packages are installed.")


if __name__ == "__main__":  # pragma: no cover
    main()
