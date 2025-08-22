from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import textwrap
from dataclasses import dataclass
from typing import Iterable, Mapping

# Import only the *lightweight* utilities from the package.
# These do not pull heavy optional dependencies on import.
from . import __version__, about as _about_text, optional_dependencies

# ------------------------------- helpers ----------------------------------

def _table(rows: list[list[str]], headers: list[str] | None = None) -> str:
    """Return a compact monospace table without external deps.
    Each cell is left-aligned; column width is computed from content.
    """
    data = rows[:]  # shallow copy
    if headers:
        data = [list(headers)] + data
    widths = [max(len(str(cell)) for cell in col) for col in zip(*data)] if data else []
    def fmt_row(row: Iterable[str]) -> str:
        return "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
    out = []
    for i, row in enumerate(data):
        out.append(fmt_row(row))
        if headers and i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)

def _env_summary() -> dict[str, str]:
    """Small, dependency-free environment summary."""
    impl = platform.python_implementation()
    pyver = platform.python_version()
    machine = platform.machine()
    system = platform.system()
    proc = platform.processor() or ""
    return {
        "python": f"{impl} {pyver}",
        "platform": f"{system} ({machine})",
        "executable": sys.executable,
        "processor": proc,
        "cwd": os.getcwd(),
    }

def _emit_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))

# ------------------------------- commands ---------------------------------

def cmd_about(args: argparse.Namespace) -> int:
    info = _about_text()
    if args.brief:
        print(info.splitlines()[0])
        return 0
    if args.json:
        report = optional_dependencies(include_pip_hints=True)
        payload = {"version": __version__, "about": info, "optional_deps": report, "env": _env_summary()}
        _emit_json(payload)
        return 0
    print(info)
    return 0

def cmd_doctor(args: argparse.Namespace) -> int:
    report = optional_dependencies(refresh=args.refresh, include_pip_hints=not args.no_pip_hints)
    # table: name | available | version | pip (if present)
    names = sorted(report.keys())
    rows: list[list[str]] = []
    missing = set()
    for name in names:
        item = report[name]
        avail = bool(item.get("available"))
        ver = item.get("version") or "-"
        pip = (item.get("pip") or "") if not avail else ""
        rows.append([name, "yes" if avail else "no", ver, pip])
        if not avail:
            missing.add(name)

    if args.json:
        payload = {
            "version": __version__,
            "env": _env_summary(),
            "optional_deps": report,
        }
        _emit_json(payload)
    else:
        print(f"physical_ai_stl {__version__}")
        print("Environment:")
        print(_table([[k, v] for k, v in _env_summary().items()], headers=["Key", "Value"]))  # noqa: E501
        print()
        print("Optional dependencies:")
        print(_table(rows, headers=["name", "ok", "version", "pip hint if missing"]))  # noqa: E501
        # Friendly note about NVIDIA's rename if only legacy 'modulus' is present
        if ("modulus" in report and report.get("modulus", {}).get("available")) and not report.get("physicsnemo", {}).get("available"):  # noqa: E501
            print("\nNOTE: 'modulus' is installed; NVIDIA renamed it to 'PhysicsNeMo'. Consider installing 'nvidia-physicsnemo' and migrating imports to 'physicsnemo'.")  # noqa: E501

    # Evaluate requirement policy, if any
    # Policies: any, all.  Groups: core, physics, stl
    rc = 0
    if args.require:
        req = set(args.require)
        groups: Mapping[str, set[str]] = {
            "core": {"numpy", "torch"},
            "physics": {"neuromancer", "physicsnemo", "torchphysics"},
            "stl": {"rtamt", "moonlight", "spatial_spec"},
            "all": set(names),
        }
        unmet: dict[str, set[str]] = {}
        for spec in req:
            if ":" in spec:
                group, policy = spec.split(":", 1)
            else:
                group, policy = spec, args.policy
            group = group.lower()
            policy = policy.lower()
            if group not in groups:
                print(f"warning: unknown group '{group}' (known: {', '.join(groups)})")
                rc = rc or 2
                continue
            want = groups[group]
            have = {n for n in want if report.get(n, {}).get("available")}
            if policy == "any":
                if not have:
                    unmet[group] = want
            elif policy == "all":
                lacking = want - have
                if lacking:
                    unmet[group] = lacking
            else:
                print(f"warning: unknown policy '{policy}', expected 'any' or 'all'")
                rc = rc or 2
        if unmet:
            rc = rc or 1
            if not args.json:
                print("\nMissing requirements:")
                for g, names_missing in unmet.items():
                    print(f"  {g}: {', '.join(sorted(names_missing))}")
    return rc

def cmd_pip(args: argparse.Namespace) -> int:
    report = optional_dependencies(include_pip_hints=True)
    cmds = [item["pip"] for item in report.values() if not item.get("available") and item.get("pip")]
    if args.json:
        _emit_json({"pip_install": cmds, "count": len(cmds)})
    else:
        if not cmds:
            print("All optional dependencies appear to be installed.")
        else:
            print("Run the following to install missing optional dependencies:")
            print("\n".join(cmds))
    return 0

def cmd_version(args: argparse.Namespace) -> int:  # noqa: ARG001
    print(__version__)
    return 0

# ------------------------------- parser -----------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="physical_ai_stl",
        description=(
            "Physical-AI STL utility CLI — fast environment checks,\n"
            "installation hints, and a concise 'about' report.\n\n"
            "Examples:\n"
            "  python -m physical_ai_stl                 # print about()\n"
            "  python -m physical_ai_stl --brief        # one-line summary\n"
            "  python -m physical_ai_stl doctor         # environment table\n"
            "  python -m physical_ai_stl doctor --require physics stl\n"
            "  python -m physical_ai_stl doctor --json  # machine-readable\n"
        ),
    )
    p.add_argument("--brief", action="store_true", help="Print a one-line summary (alias for 'about --brief')")  # noqa: E501
    p.add_argument("--json", action="store_true", help="Emit JSON payload when applicable")  # noqa: E501

    sub = p.add_subparsers(dest="cmd")

    # about (default) --------------------------------------------------------
    sp = sub.add_parser("about", help="Show package about() and optional-deps summary")  # noqa: E501
    sp.add_argument("--brief", action="store_true", help="Print a one-line summary")  # noqa: E501
    sp.add_argument("--json", action="store_true", help="Emit JSON payload (includes env + optional-deps)")  # noqa: E501
    sp.set_defaults(_fn=cmd_about)

    # doctor -----------------------------------------------------------------
    sp = sub.add_parser("doctor", help="Inspect environment and validate optional deps")  # noqa: E501
    sp.add_argument("--refresh", action="store_true", help="Rescan environment (ignore cached probe)")  # noqa: E501
    sp.add_argument("--no-pip-hints", action="store_true", help="Hide pip install suggestions")  # noqa: E501
    sp.add_argument("--require", nargs="*", default=[], help=(
        "Require groups to be satisfied; space-separated, e.g. 'physics stl'.\n"
        "You can override policy per-group via 'group:POLICY'.\n"
        "Groups: core, physics, stl, all. Policies: any (default), all."
    ))
    sp.add_argument("--policy", choices=["any", "all"], default="any", help="Default requirement policy for groups")  # noqa: E501
    sp.add_argument("--json", action="store_true", help="Emit JSON payload")  # noqa: E501
    sp.set_defaults(_fn=cmd_doctor)

    # pip --------------------------------------------------------------------
    sp = sub.add_parser("pip", help="Print pip install commands for missing optional deps")  # noqa: E501
    sp.add_argument("--json", action="store_true", help="Emit JSON payload")  # noqa: E501
    sp.set_defaults(_fn=cmd_pip)

    # version ----------------------------------------------------------------
    sp = sub.add_parser("version", help="Print package version")  # noqa: E501
    sp.set_defaults(_fn=cmd_version)

    return p

def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    # Support top-level --brief/--json flags without an explicit subcommand.
    args, rest = parser.parse_known_args(argv)
    if args.cmd is None:
        # No subcommand provided → behave like 'about'
        if args.json or args.brief:
            return cmd_about(args)
        # Reparse with default subcommand so help works consistently.
        # Inject 'about' while preserving other flags.
        return cmd_about(argparse.Namespace(brief=args.brief, json=args.json))
    return args._fn(args)  # type: ignore[attr-defined]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
