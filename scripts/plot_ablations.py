#!/usr/bin/env python3
# ruff: noqa: I001
"""
plot_ablations.py
-----------------

Purpose
=======
A fast, dependency‑light plotting utility for **ablation sweeps** in this repo.
It is designed to “just work” with the CSV artifacts produced by scripts like
``scripts/run_ablations_diffusion.py`` (and similarly formatted CSVs), while
remaining flexible enough to handle lightly different column names and multiple
series per figure.

**Why this exists**
- The course project emphasizes *clear, reproducible comparisons* with minimal
  setup overhead. This script requires only Matplotlib, runs on CPU, and handles
  common CSV variations (with/without header, different separators).
- It enforces sensible defaults (axis scaling, labels, error bars) that match
  the semantics of STL robustness monitoring and physics‑ML metrics we use.

Key features
------------
- **Auto‑ingest CSVs**: header/no‑header tolerant; delimiter sniffing; `#` comments.
- **Column inference**: finds `x` (ablation parameter) and `y` (metric) using
  robust name heuristics (e.g., `lambda`, `stl_weight`, `robustness`, `mse`, …).
- **Aggregation**: groups by identical `x` and computes mean/std/n across repeats.
- **Error bars**: std, sem, or 95% CI; configurable.
- **Autoscale**: optional log‑x when values span orders of magnitude.
- **Pretty output**: tight layout, outside legend option, multi‑format save (png/pdf/svg).
- **Summary table**: optional CSV with the *best setting* per series.

Examples
--------
Basic use (plot robustness vs. λ for one or more ablation CSVs)::

    python scripts/plot_ablations.py results/ablations_diffusion.csv

Multiple series on one plot (labels auto‑derived from filenames)::

    python scripts/plot_ablations.py results/*.csv -o figs/ablations.png

Explicit columns, CI shading, and summary csv::

    python scripts/plot_ablations.py \
        results/ablations_*.csv \
        --xcol lambda --ycol robustness --err ci95 \
        --title "STL robustness vs. λ" \
        --summary figs/ablations_summary.csv

Input format
------------
- CSV **with or without** a header line. If no header, the first two numeric
  columns are taken as `x` and `y`.
- Delimiters: comma, tab, or semicolon are auto‑sniffed. Lines beginning with
  `#` are ignored as comments.
- Typical files produced by this repo have rows like: ``lambda, robustness`` or
  (no header) ``<lambda>, <mean_robustness>``. Multiple repeats may appear as
  repeated rows for the same λ.

Outputs
-------
- A figure (PNG by default) showing **mean±error** per series.
- (Optional) a summary CSV with, per input file/series: **best x**, best y,
  number of points, whether higher is better, etc.

Design decisions
----------------
- **Speed**: pure‑Python parsing, no pandas. Matplotlib backend forced to **Agg**
  for headless reliability on servers/CI.
- **Correctness/robustness**: careful inference of columns and types. Reasonable
  defaults for STL semantics: *robustness higher is better*; *loss lower is better*.
- **Portability**: no reliance on local fonts/styles; works with base Matplotlib.

This script is intentionally **self‑contained** to avoid surprising imports from
elsewhere in the repository and to keep it easy to copy for quick experiments.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import math
import re
import statistics
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")  # headless, reproducible figures
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers: data ingestion
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"^\s*#")
_WHITESPACE_RE = re.compile(r"\s+")

# Common names for ablation parameter (x) and metric (y)
_X_CANDIDATES = (
    "lambda", "lam", "stl_weight", "weight", "alpha", "beta",
    "coef", "penalty", "reg", "regularizer", "kappa",
    "param", "x",
)
_Y_CANDIDATES = (
    # Positive-is-better
    "robustness", "stl_robustness", "satisfaction", "accuracy", "acc",
    "iou", "f1", "psnr", "r2",
    # Negative-is-better
    "loss", "val_loss", "train_loss", "mse", "rmse", "mae",
    "violation", "violation_rate", "error",
    # Fallback
    "y",
)

_ERR_CHOICES = ("none", "std", "sem", "ci95")


@dataclass(frozen=True)
class SeriesStats:
    x: list[float]
    y_mean: list[float]
    y_std: list[float]
    n: list[int]
    label: str
    y_name: str
    x_name: str


def _read_rows(path: Path) -> tuple[list[str] | None, list[list[str]]]:
    """
    Return (header, rows-as-strings).

    Tolerates '#' comments and auto-detects delimiter with csv.Sniffer, falling
    back to comma. Header is None if we don't detect any.
    """
    text_lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if _COMMENT_RE.match(raw):
                continue
            # Normalize tabs and stray whitespace-only lines
            if raw.strip() == "":
                continue
            text_lines.append(raw)

    if not text_lines:
        return None, []

    sample = "".join(text_lines[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
    except Exception:
        dialect = csv.excel
        dialect.delimiter = ","  # type: ignore[attr-defined]

    # Header detection
    has_header = False
    try:
        has_header = csv.Sniffer().has_header(sample)
    except Exception:
        # Heuristic: if the first non-empty line contains any letters, assume header
        has_header = any(c.isalpha() for c in text_lines[0])

    reader = csv.reader(text_lines, dialect=dialect)  # type: ignore[arg-type]
    rows = list(reader)
    header: list[str] | None = None
    if has_header:
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]

    return header, rows


def _to_float(s: str) -> float | None:
    try:
        # Strip common junk
        s = s.strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        # Replace unicode minus
        s = s.replace("−", "-")
        return float(s)
    except Exception:
        return None


def _infer_col_indices(header: list[str] | None, rows: list[list[str]]) -> tuple[int, int, str, str]:
    """
    Infer (x_idx, y_idx, x_name, y_name).

    Strategy:
    - With header: pick best matches from candidate lists.
    - Without header:
        * If there are exactly 2 columns and both parsable -> (0, 1).
        * Else choose the left-most column that looks numeric for x,
          and the next numeric column for y.
    """
    if header:
        lower = [h.strip().lower() for h in header]
        # Direct matches
        x_idx, y_idx = None, None
        for cand in _X_CANDIDATES:
            if cand in lower:
                x_idx = lower.index(cand)
                break
        for cand in _Y_CANDIDATES:
            if cand in lower:
                y_idx = lower.index(cand)
                break
        # If still missing, default to first two numeric columns
        if x_idx is None or y_idx is None or x_idx == y_idx:
            # Find numeric-looking columns using the first few rows
            numeric_cols: list[int] = []
            for j in range(len(lower)):
                # look across up to 10 rows
                col_vals = [_to_float(r[j]) for r in rows[: min(10, len(rows))] if j < len(r)]
                if col_vals and sum(v is not None for v in col_vals) >= max(1, len(col_vals) // 2):
                    numeric_cols.append(j)
            if x_idx is None and numeric_cols:
                x_idx = numeric_cols[0]
            if y_idx is None and len(numeric_cols) >= 2:
                y_idx = numeric_cols[1]
        if x_idx is None or y_idx is None or x_idx == y_idx:
            raise ValueError(f"Could not infer x/y columns from header={header!r}")
        return x_idx, y_idx, header[x_idx], header[y_idx]

    # No header case
    if not rows:
        raise ValueError("Empty file; cannot infer columns.")
    max_cols = max(len(r) for r in rows)
    # Common case: two columns
    if max_cols == 2:
        # ensure both numeric
        x0 = _to_float(rows[0][0]) if len(rows[0]) > 0 else None
        y0 = _to_float(rows[0][1]) if len(rows[0]) > 1 else None
        if x0 is not None and y0 is not None:
            return 0, 1, "x", "y"
    # Otherwise: choose first two numeric columns
    for j in range(max_cols - 1):
        x_try = _to_float(rows[0][j]) if j < len(rows[0]) else None
        if x_try is None:
            continue
        for k in range(j + 1, max_cols):
            y_try = _to_float(rows[0][k]) if k < len(rows[0]) else None
            if y_try is not None:
                return j, k, "x", "y"
    raise ValueError("Could not infer numeric x/y columns from CSV rows.")


def _sanitize_label_from_filename(p: Path) -> str:
    name = p.stem
    # Remove common prefixes/suffixes
    name = re.sub(r"^(results?_)?", "", name, flags=re.I)
    name = re.sub(r"(_?ablations?|_?diffusion|_?heat2d|_?burgers?)", "", name, flags=re.I)
    name = re.sub(r"[-_]+", " ", name).strip()
    return name or p.stem


def _aggregate_xy(rows: list[list[str]], x_idx: int, y_idx: int) -> tuple[list[float], list[float], list[float], list[int]]:
    """
    Group by identical x (float) and compute mean/std/n of y per group.
    Returns (xs_sorted, y_mean_sorted, y_std_sorted, n_sorted).
    """
    buckets: dict[float, list[float]] = {}
    for r in rows:
        if len(r) <= max(x_idx, y_idx):
            continue
        x = _to_float(r[x_idx])
        y = _to_float(r[y_idx])
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            continue
        buckets.setdefault(float(x), []).append(float(y))

    xs = sorted(buckets.keys())
    y_mean: list[float] = []
    y_std: list[float] = []
    n: list[int] = []

    for x in xs:
        vals = buckets[x]
        n.append(len(vals))
        if len(vals) == 1:
            y_mean.append(vals[0])
            y_std.append(0.0)
        else:
            y_mean.append(float(statistics.fmean(vals)))
            # Use population std if n>1 to avoid overestimating error with small n
            try:
                y_std.append(float(statistics.pstdev(vals)))
            except statistics.StatisticsError:
                y_std.append(0.0)
    return xs, y_mean, y_std, n


def _load_series(path: Path, xcol: str | None, ycol: str | None) -> SeriesStats:
    header, rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path}: file is empty after filtering comments/blank lines.")
    if header:
        # Allow user override
        if xcol is not None and ycol is not None:
            lower = [h.strip().lower() for h in header]
            if xcol.lower() not in lower or ycol.lower() not in lower:
                raise ValueError(f"{path}: requested columns {xcol}/{ycol} not in header {header}")
            x_idx = lower.index(xcol.lower())
            y_idx = lower.index(ycol.lower())
            x_name, y_name = header[x_idx], header[y_idx]
        else:
            x_idx, y_idx, x_name, y_name = _infer_col_indices(header, rows)
    else:
        x_idx, y_idx, x_name, y_name = _infer_col_indices(header, rows)

    xs, y_mean, y_std, n = _aggregate_xy(rows, x_idx, y_idx)
    label = _sanitize_label_from_filename(path)
    return SeriesStats(x=xs, y_mean=y_mean, y_std=y_std, n=n, label=label, y_name=y_name, x_name=x_name)


def _ci_multiplier(kind: str, n: int) -> float:
    """
    Return multiplier for the requested error kind.
    - 'std': 1.0 * std
    - 'sem': std / sqrt(n)
    - 'ci95': 1.96 * std / sqrt(n)  (Gaussian approx; good for n>=3)
    """
    if kind == "none":
        return 0.0
    if kind == "std":
        return 1.0
    if kind == "sem":
        return 1.0 / math.sqrt(max(1, n))
    if kind == "ci95":
        return 1.96 / math.sqrt(max(1, n))
    raise ValueError(f"Unknown error kind: {kind}")


def _infer_better_is_higher(y_name: str) -> bool:
    name = y_name.lower()
    if any(k in name for k in ("loss", "mse", "rmse", "mae", "error", "violation")):
        return False
    return True


def _maybe_log_x(xs: Sequence[float], requested: str | None) -> str:
    if requested in ("linear", "log"):
        return requested
    positives = [x for x in xs if x > 0]
    if not positives or len(positives) < 2:
        return "linear"
    span = max(positives) / min(positives)
    return "log" if span >= 100 else "linear"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_series(
    series_list: list[SeriesStats],
    out_base: Path,
    title: str | None = None,
    xscale: str | None = None,
    err: str = "ci95",
    figsize: tuple[float, float] = (6.0, 4.0),
    dpi: int = 160,
    legend_outside: bool = True,
    grid: bool = True,
    ylabel: str | None = None,
    xlabel: str | None = None,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """
    Render a single plot with all provided series and save to ``out_base`` with
    the requested formats. Returns the list of paths written.
    """
    if not series_list:
        raise ValueError("No data to plot. Provide at least one input file.")

    # Axis labels
    inferred_ylabel = series_list[0].y_name
    inferred_xlabel = series_list[0].x_name
    ylab = ylabel or inferred_ylabel
    xlab = xlabel or inferred_xlabel

    # Choose x-scale
    all_x = [x for s in series_list for x in s.x]
    scale = _maybe_log_x(all_x, xscale)

    # Create figure
    fig, ax = plt.subplots(figsize=tuple(figsize), dpi=dpi)
    for s in series_list:
        xs = s.x
        ys = s.y_mean
        yerrs = []
        if err != "none":
            for j in range(len(xs)):
                m = _ci_multiplier(err, s.n[j])
                yerrs.append(m * s.y_std[j])
        # Plot
        ax.errorbar(xs, ys, yerr=yerrs if err != "none" else None,
                    marker="o", capsize=3, linewidth=1.8, markersize=4.5,
                    label=s.label)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)

    # Scale/grid/legend
    ax.set_xscale(scale)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.35, which="both")
    if legend_outside and len(series_list) > 1:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout(rect=[0.0, 0.0, 0.85, 1.0])
    else:
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()

    # Save
    out_paths: list[Path] = []
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = out_base.with_suffix("." + fmt.lstrip("."))
        fig.savefig(target, bbox_inches="tight")
        out_paths.append(target)

    plt.close(fig)
    return out_paths


# ---------------------------------------------------------------------------
# Summary CSV (optional)
# ---------------------------------------------------------------------------

def write_summary_csv(series_list: list[SeriesStats], out_csv: Path, prefer_higher: bool | None = None) -> Path:
    """
    Write a CSV with, per series, the best x and best y, where “best” is chosen
    using the metric name heuristic (robustness↑, loss↓) unless overridden by
    ``prefer_higher``.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[str | float | int]] = []
    header = ["series", "metric", "higher_is_better", "best_x", "best_y", "num_points", "avg_n_repeats"]
    rows.append(header)
    for s in series_list:
        higher = _infer_better_is_higher(s.y_name) if prefer_higher is None else prefer_higher
        idx = max(range(len(s.y_mean)), key=lambda j: s.y_mean[j]) if higher else min(
            range(len(s.y_mean)), key=lambda j: s.y_mean[j]
        )
        best_x = s.x[idx]
        best_y = s.y_mean[idx]
        avg_n = float(statistics.fmean(s.n)) if s.n else 1.0
        rows.append([s.label, s.y_name, bool(higher), float(best_x), float(best_y), int(len(s.x)), float(avg_n)])
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return out_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _path_list(arg_vals: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for a in arg_vals:
        p = Path(a)
        if p.is_dir():
            out.extend(sorted(p.glob("*.csv")))
        else:
            # Allow globs
            if any(c in a for c in "*?[]"):
                out.extend(sorted(Path().glob(a)))
            else:
                out.append(p)
    # Deduplicate while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot ablation curves (mean±error) from CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="Input CSVs or directories (globs allowed).")
    p.add_argument("-o", "--out", default="figs/ablations", help="Output path base (extension is added per format).")
    p.add_argument("--formats", nargs="+", default=["png"], help="Image formats to save (e.g., png pdf svg).")

    # Column handling
    p.add_argument("--xcol", default=None, help="Override inferred x column name (header CSVs only).")
    p.add_argument("--ycol", default=None, help="Override inferred y column name (header CSVs only).")

    # Plot appearance / behavior
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--xlabel", default=None, help="X-axis label (overrides inferred).")
    p.add_argument("--ylabel", default=None, help="Y-axis label (overrides inferred).")
    p.add_argument("--xscale", choices=("linear", "log"), default=None, help="Force x-scale (else auto).")
    p.add_argument("--err", choices=_ERR_CHOICES, default="ci95", help="Error bar / band type.")
    p.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    p.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"), default=(6.0, 4.0), help="Figure size inches.")
    p.add_argument("--legend-inside", action="store_true", help="Place legend inside axes (default outside if >1 series).")
    p.add_argument("--no-grid", action="store_true", help="Disable background grid.")

    # Summary CSV
    p.add_argument("--summary", default=None, help="Optional path to write a CSV summary of best settings.")
    p.add_argument("--prefer-higher", type=str, choices=("auto", "higher", "lower"), default="auto",
                   help="Override metric direction when computing best rows in summary.")

    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Resolve inputs
    paths = _path_list(args.inputs)
    if not paths:
        raise SystemExit("No input files matched.")

    # Load series
    series_list: list[SeriesStats] = []
    for p in paths:
        try:
            s = _load_series(p, args.xcol, args.ycol)
            series_list.append(s)
        except Exception as e:
            raise SystemExit(f"{p}: {e}")

    # Plot
    out_paths = plot_series(
        series_list=series_list,
        out_base=Path(args.out),
        title=args.title,
        xscale=args.xscale,
        err=args.err,
        figsize=tuple(args.figsize),
        dpi=int(args.dpi),
        legend_outside=not bool(args.legend_inside),
        grid=not bool(args.no_grid),
        ylabel=args.ylabel,
        xlabel=args.xlabel,
        formats=tuple(args.formats),
    )

    # Summary?
    if args.summary:
        if args.prefer_higher == "auto":
            pref: bool | None = None
        elif args.prefer_higher == "higher":
            pref = True
        else:
            pref = False
        csv_path = write_summary_csv(series_list, Path(args.summary), prefer_higher=pref)
        print(f"Wrote summary: {csv_path}")

    # Friendly message
    if len(out_paths) == 1:
        print(f"Wrote {out_paths[0]}")
    else:
        print("Wrote:")
        for p in out_paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
