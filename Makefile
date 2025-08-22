# Physical-AI + STL research sandbox
#
# This Makefile is intentionally "demo-first":
#  - one-command reproducibility for the professor demo (plots + monitors)
#  - CPU-friendly defaults (override DEVICE=cuda when available)
#  - stable, discoverable artifacts (results/, figs/, assets/)
#
# Usage:
#   make help
#   make quickstart
#   make demo
#   make diffusion1d rtamt-eval
#   make heat2d moonlight-eval
#
# Notes:
#   * Targets never assume optional frameworks are installed unless you explicitly
#     request them (Neuromancer / PhysicsNeMo / TorchPhysics).
#   * Figures are regenerated from *actually-run* artifacts (CSV logs / .pt fields).
#   * Monitoring outputs are written to results/ and (when possible) copied into
#     the latest run directory created by scripts/run_experiment.py.
#
# ---------------------------------------------------------------------------

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

.DEFAULT_GOAL := help

# ---- knobs (override like: `make DEVICE=cuda diffusion1d`) -----------------

PYTHON ?= python3
VENV_DIR ?= .venv

# runtime outputs
RESULTS_DIR ?= results
FIGS_DIR    ?= figs
ASSETS_DIR  ?= assets
LOGS_DIR    ?= logs
TMP_DIR     ?= .tmp

# device defaults
DEVICE ?= cpu

# fast demo epoch counts (keep small for live demos)
DEMO_EPOCHS_DIFF ?= 40
DEMO_EPOCHS_HEAT ?= 40

# full-ish reproduction epoch counts (still laptop-friendly; override as needed)
FULL_EPOCHS_DIFF ?= 400
FULL_EPOCHS_HEAT ?= 200

# quick benchmark epoch count (for runtime comparisons in the report)
BENCH_EPOCHS ?= 100

# Torch wheel channel for CPU/GPU installs
TORCH_CHANNEL ?= cpu
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/$(TORCH_CHANNEL)

# source layout
PY_SRC ?= src

# configs
CFG_DIFF_BASELINE ?= configs/diffusion1d_baseline.yaml
CFG_DIFF_STL      ?= configs/diffusion1d_stl.yaml
CFG_HEAT2D        ?= configs/heat2d_baseline.yaml
CFG_NEURO_SINE    ?= configs/neuromancer_sine_bound.yaml

# heat2d STREL / MoonLight spec
HEAT2D_MLS ?= scripts/specs/contain_hotspot.mls
HEAT2D_MOONLIGHT_FORMULA ?= contain_hotspot

# ---- derived ---------------------------------------------------------------

VENV_PY := $(VENV_DIR)/bin/python
PIP     := $(VENV_PY) -m pip

# Make local imports work without requiring an editable install.
export PYTHONPATH := $(PY_SRC)$(if $(PYTHONPATH),:$(PYTHONPATH))

# ---- helpers ---------------------------------------------------------------

define ACTIVATE
if [ -f "$(VENV_DIR)/bin/activate" ]; then \
  source "$(VENV_DIR)/bin/activate"; \
else \
  echo "ERROR: venv not found at $(VENV_DIR). Run: make venv"; \
  exit 2; \
fi
endef

define REQUIRE_CMD
command -v "$(1)" >/dev/null 2>&1 || { echo "ERROR: required command not found: $(1)"; exit 2; }
endef

# Find newest file/dir by mtime (portable-ish; uses python).
define PY_LATEST
$(call ACTIVATE); $(VENV_PY) - <<'PY'
	import glob, os
	pat = r'$(1)'
	paths = glob.glob(pat, recursive=True)
	paths = [p for p in paths if os.path.exists(p)]
	if not paths:
	    raise SystemExit(1)
	paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	print(paths[0])
	PY
endef

# ---------------------------------------------------------------------------
# Help / meta
# ---------------------------------------------------------------------------

help: ## Show this help.
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | grep -v '^_' | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}' | sort

dirs: ## Create common output directories.
	@mkdir -p "$(RESULTS_DIR)" "$(FIGS_DIR)" "$(ASSETS_DIR)" "$(LOGS_DIR)" "$(TMP_DIR)"

clean: ## Remove caches and build artifacts (does NOT delete results/ or assets/).
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage dist build *.egg-info
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

venv: ## Create local virtual environment at $(VENV_DIR).
	@$(PYTHON) -m venv "$(VENV_DIR)"
	@$(call ACTIVATE); $(PIP) install --upgrade pip wheel

install: venv ## Install runtime requirements (CPU-friendly).
	@$(call ACTIVATE); $(PIP) install -r requirements.txt

install-dev: venv ## Install dev requirements (linters/tests + hw probing).
	@$(call ACTIVATE); $(PIP) install -r requirements-dev.txt

install-extra: venv ## Install optional framework extras (Neuromancer / PhysicsNeMo / TorchPhysics).
	@$(call ACTIVATE); $(PIP) install -r requirements-extra.txt

install-all: install install-dev install-extra ## Install everything (runtime + dev + extras).

install-torch-cpu: venv ## Install PyTorch CPU wheels (overwrites any existing torch install).
	@$(call ACTIVATE); $(PIP) install --upgrade --index-url "$(TORCH_INDEX_URL)" torch torchvision torchaudio

install-torch-gpu: venv ## Install PyTorch GPU wheels (set TORCH_CHANNEL=cu121, etc).
	@$(call ACTIVATE); $(PIP) install --upgrade --index-url "$(TORCH_INDEX_URL)" torch torchvision torchaudio

# Convenience: keep these names because they are referenced in docs/ and email.
install-rtamt: install ## Ensure RTAMT is installed.
	@$(call ACTIVATE); python -c "import rtamt; print('rtamt', getattr(rtamt, '__version__', '?'))"

install-moonlight: install ## Ensure MoonLight python wrapper is installed.
	@$(call ACTIVATE); python -c "import moonlight; print('moonlight import OK')"
	@if command -v java >/dev/null 2>&1; then java -version || true; else echo "[moonlight] WARNING: java not found; MoonLight requires Java 21+ (see MoonLight README)."; fi

install-neuromancer: venv ## Install Neuromancer (optional; uses pip).
	@$(call ACTIVATE); $(PIP) install neuromancer

quickstart: install install-dev ## Minimal working demo setup (runtime + dev). Use DEVICE=cuda optionally.
	@echo "✅ quickstart complete"
	@echo "Next: make doctor  (environment)  |  make demo  (end-to-end)"

# ---------------------------------------------------------------------------
# QA (format/lint/test)
# ---------------------------------------------------------------------------

fmt: ## Auto-format (ruff).
	@$(call ACTIVATE); ruff format .

lint: ## Lint (ruff).
	@$(call ACTIVATE); ruff check .

typecheck: ## Type check (mypy).
	@$(call ACTIVATE); mypy src

test: ## Run full test suite (pytest).
	@$(call ACTIVATE); pytest -q

test-fast: ## Run quick sanity tests only.
	@$(call ACTIVATE); pytest -q tests/test_stl_soft.py tests/test_specs.py

ci: ## CI-style checks: format, lint, typecheck, tests.
	@$(MAKE) fmt
	@$(MAKE) lint
	@$(MAKE) typecheck
	@$(MAKE) test

# ---------------------------------------------------------------------------
# Introspection (useful for report: hardware + dependency matrix)
# ---------------------------------------------------------------------------

doctor: dirs venv ## Print environment + hardware summary (and write to results/doctor.json).
	@$(call ACTIVATE); \
	$(VENV_PY) -m physical_ai_stl doctor --json | tee "$(RESULTS_DIR)/doctor.json" >/dev/null || true
	@$(call ACTIVATE); \
	$(VENV_PY) - <<'PY'
	import json, os, platform, sys
	from pathlib import Path
	
	out = {
	  "python": sys.version.replace("\n", " "),
	  "platform": platform.platform(),
	  "processor": platform.processor(),
	  "machine": platform.machine(),
	  "cpu_count_logical": os.cpu_count(),
	}
	
	# Optional (installed via requirements-dev.txt): richer HW probing
	try:
	    import psutil  # type: ignore
	    out["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 3)
	except Exception as e:
	    out["ram_total_gb"] = None
	    out["psutil_error"] = str(e)
	
	try:
	    import cpuinfo  # type: ignore
	    out["cpuinfo_brand_raw"] = cpuinfo.get_cpu_info().get("brand_raw")
	except Exception as e:
	    out["cpuinfo_brand_raw"] = None
	    out["cpuinfo_error"] = str(e)
	
	Path("results").mkdir(parents=True, exist_ok=True)
	Path("results/hardware.json").write_text(json.dumps(out, indent=2))
	print("✅ wrote results/hardware.json")
	PY

check: dirs venv ## Dependency probe matrix (prints + writes markdown to results/deps.md).
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/check_env.py --md --extended > "$(RESULTS_DIR)/deps.md" || true
	@echo "✅ wrote $(RESULTS_DIR)/deps.md"

# ---------------------------------------------------------------------------
# Experiments (training / rollouts)
# ---------------------------------------------------------------------------

# --- diffusion 1D -----------------------------------------------------------

diffusion1d-baseline: dirs install ## Train 1D diffusion PINN baseline (full epochs).
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/run_experiment.py -c "$(CFG_DIFF_BASELINE)" --device "$(DEVICE)" \
	  --set optim.epochs="$(FULL_EPOCHS_DIFF)"
	@$(MAKE) _sync_latest_run_artifacts EXP=diffusion1d TAG=baseline
	@$(MAKE) diffusion1d-figs

diffusion1d-stl: dirs install ## Train 1D diffusion PINN with STL penalty (full epochs).
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/run_experiment.py -c "$(CFG_DIFF_STL)" --device "$(DEVICE)" \
	  --set optim.epochs="$(FULL_EPOCHS_DIFF)"
	@$(MAKE) _sync_latest_run_artifacts EXP=diffusion1d TAG=stl
	@$(MAKE) diffusion1d-figs

diffusion1d-demo: dirs install ## Fast diffusion demo (short epochs) for live walkthroughs.
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/run_experiment.py -c "$(CFG_DIFF_BASELINE)" --device "$(DEVICE)" \
	  --set optim.epochs="$(DEMO_EPOCHS_DIFF)"
	@$(MAKE) _sync_latest_run_artifacts EXP=diffusion1d TAG=baseline
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/run_experiment.py -c "$(CFG_DIFF_STL)" --device "$(DEVICE)" \
	  --set optim.epochs="$(DEMO_EPOCHS_DIFF)"
	@$(MAKE) _sync_latest_run_artifacts EXP=diffusion1d TAG=stl
	@$(MAKE) diffusion1d-figs
	@$(MAKE) rtamt-eval

diffusion1d: diffusion1d-baseline diffusion1d-stl ## Run both diffusion1d baselines + STL (full epochs).

diffusion1d-figs: dirs install ## Regenerate diffusion1d plots into assets/ (fields + loss + robustness).
	@$(call ACTIVATE); \
	$(VENV_PY) - <<'PY'
	from __future__ import annotations
	import csv
	from pathlib import Path
	import glob
	import torch
	import matplotlib.pyplot as plt
	
	# Allow importing scripts/utils_plot.py as a plain module.
	import sys
	sys.path.insert(0, str(Path.cwd() / "scripts"))
	import utils_plot  # type: ignore
	
	results = Path("results")
	assets  = Path("assets")
	assets.mkdir(parents=True, exist_ok=True)
	
	def latest(pattern: str) -> Path | None:
	    paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
	    paths = [p for p in paths if p.exists()]
	    if not paths:
	        return None
	    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	    return paths[0]
	
	# Prefer the "stable" filenames if present, else fall back to newest run dir.
	baseline_field = (results / "diffusion1d_baseline_field.pt")
	stl_field      = (results / "diffusion1d_stl_field.pt")
	
	if not baseline_field.exists():
	    p = latest(str(results / "**/diffusion1d_baseline_field.pt"))
	    if p: baseline_field = p
	if not stl_field.exists():
	    p = latest(str(results / "**/diffusion1d_stl_field.pt"))
	    if p: stl_field = p
	
	def plot_field(pt: Path, out_png: Path, title: str) -> None:
	    obj = torch.load(pt, map_location="cpu")
	    u = obj["u"].detach().cpu().numpy()
	    x = obj["x"].detach().cpu().numpy()
	    t = obj["t"].detach().cpu().numpy()
	    utils_plot.plot_u_xt(u, x, t, out=out_png, title=title)
	
	if baseline_field.exists():
	    plot_field(baseline_field, assets / "diffusion1d_baseline_field.png", "1D Diffusion PINN (baseline)")
	else:
	    print("[warn] could not find diffusion1d_baseline_field.pt; skipping field plot")
	
	if stl_field.exists():
	    plot_field(stl_field, assets / "diffusion1d_stl_field.png", "1D Diffusion PINN (+STL penalty)")
	else:
	    print("[warn] could not find diffusion1d_stl_field.pt; skipping field plot")
	
	# ---- training curves -------------------------------------------------------
	def read_csv(path: Path) -> list[dict[str, str]]:
	    with path.open(newline="") as f:
	        return list(csv.DictReader(f))
	
	baseline_csv = results / "diffusion1d_baseline.csv"
	stl_csv      = results / "diffusion1d_stl.csv"
	
	if baseline_csv.exists() and stl_csv.exists():
	    rows_b = read_csv(baseline_csv)
	    rows_s = read_csv(stl_csv)
	
	    def col(rows, k):
	        out=[]
	        for r in rows:
	            try:
	                out.append(float(r.get(k, "nan")))
	            except Exception:
	                out.append(float("nan"))
	        return out
	
	    eb = col(rows_b, "epoch")
	    es = col(rows_s, "epoch")
	
	    # total loss (baseline vs stl)
	    plt.figure()
	    plt.plot(eb, col(rows_b, "loss_total"), label="baseline")
	    plt.plot(es, col(rows_s, "loss_total"), label="STL-reg")
	    plt.yscale("log")
	    plt.xlabel("epoch")
	    plt.ylabel("loss_total")
	    plt.legend()
	    plt.tight_layout()
	    plt.savefig(assets / "diffusion1d_training_loss.png", dpi=160)
	    plt.close()
	
	    # STL run loss components
	    plt.figure()
	    plt.plot(es, col(rows_s, "loss_pde"), label="loss_pde")
	    plt.plot(es, col(rows_s, "loss_bcic"), label="loss_bcic")
	    plt.plot(es, col(rows_s, "loss_stl"), label="loss_stl")
	    plt.plot(es, col(rows_s, "loss_total"), label="loss_total")
	    plt.yscale("log")
	    plt.xlabel("epoch")
	    plt.ylabel("loss")
	    plt.legend()
	    plt.tight_layout()
	    plt.savefig(assets / "diffusion1d_training_loss_components_stl.png", dpi=160)
	    plt.close()
	
	    # robustness over epochs (STL only)
	    rob = col(rows_s, "robustness")
	    plt.figure()
	    plt.plot(es, rob)
	    plt.xlabel("epoch")
	    plt.ylabel("robustness (rho)")
	    plt.tight_layout()
	    plt.savefig(assets / "diffusion1d_training_robustness.png", dpi=160)
	    plt.close()
	else:
	    print("[warn] missing diffusion CSV logs; skipping training curve plots")
	
	print("✅ wrote diffusion figures under assets/")
	PY

# --- heat 2D ----------------------------------------------------------------

heat2d: dirs install ## Train 2D heat PINN rollout (baseline config; can enable STL via --set stl.use=true).
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/run_experiment.py -c "$(CFG_HEAT2D)" --device "$(DEVICE)" \
	  --set optim.epochs="$(FULL_EPOCHS_HEAT)"
	@$(MAKE) _sync_latest_run_artifacts EXP=heat2d TAG=baseline
	@$(MAKE) heat2d-figs

heat2d-demo: dirs install ## Fast 2D heat demo (short epochs).
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/run_experiment.py -c "$(CFG_HEAT2D)" --device "$(DEVICE)" \
	  --set optim.epochs="$(DEMO_EPOCHS_HEAT)"
	@$(MAKE) _sync_latest_run_artifacts EXP=heat2d TAG=baseline
	@$(MAKE) heat2d-figs
	@$(MAKE) moonlight-eval

heat2d-figs: dirs install ## Generate a simple 2D heat snapshot figure and a packed .npy field for STREL.
	@$(call ACTIVATE); \
	$(VENV_PY) - <<'PY'
	from __future__ import annotations
	import glob
	import json
	from pathlib import Path
	import numpy as np
	import torch
	import matplotlib.pyplot as plt
	
	results = Path("results")
	figs = Path("figs")
	assets = Path("assets")
	figs.mkdir(parents=True, exist_ok=True)
	assets.mkdir(parents=True, exist_ok=True)
	
	def latest(pattern: str) -> Path | None:
	    paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
	    paths = [p for p in paths if p.exists()]
	    if not paths:
	        return None
	    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	    return paths[0]
	
	field_pt = results / "heat2d_baseline_field.pt"
	if not field_pt.exists():
	    p = latest(str(results / "**/heat2d_*_field.pt"))
	    if p:
	        field_pt = p
	
	u = None
	dt = None
	src = None
	
	if field_pt.exists():
	    obj = torch.load(field_pt, map_location="cpu")
	    u = obj["u"].detach().cpu().numpy()   # (nx, ny, nt)
	    src = str(field_pt)
	    try:
	        t = obj.get("t")
	        if t is not None:
	            t = t.detach().cpu().numpy()
	            if len(t) >= 2:
	                dt = float(t[1] - t[0])
	    except Exception:
	        dt = None
	else:
	    asset_npy = assets / "heat2d_scalar" / "field_xy_t.npy"
	    if asset_npy.exists():
	        u = np.load(asset_npy)
	        src = str(asset_npy)
	        meta = assets / "heat2d_scalar" / "meta.json"
	        if meta.exists():
	            try:
	                dt = float(json.loads(meta.read_text()).get("dt"))
	            except Exception:
	                dt = None
	
	if u is None:
	    print("[warn] heat2d field not found (neither trained .pt nor assets/heat2d_scalar); skipping heat2d-figs")
	    raise SystemExit(0)
	
	# Save packed array for MoonLight (layout xy_t).
	packed = results / "heat2d_field_xy_t.npy"
	np.save(packed, u.astype(np.float32))
	print(f"✅ wrote {packed}  shape={u.shape}  src={src}")
	
	if dt is not None:
	    (results / "heat2d_dt.txt").write_text(f"{dt}\n")
	    print(f"✅ wrote results/heat2d_dt.txt = {dt}")
	
	nx, ny, nt = u.shape
	idxs = [0, nt // 2, nt - 1]
	plt.figure(figsize=(10, 3))
	for i, k in enumerate(idxs, start=1):
	    ax = plt.subplot(1, 3, i)
	    im = ax.imshow(u[:, :, k].T, origin="lower", aspect="auto")
	    ax.set_title(f"u(x,y) @ t[{k}]")
	    ax.set_xlabel("x-index")
	    ax.set_ylabel("y-index")
	    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	plt.tight_layout()
	out = figs / "heat2d_u_frames.png"
	plt.savefig(out, dpi=160)
	plt.close()
	print(f"✅ wrote {out}")
	PY

# ---------------------------------------------------------------------------
# Monitoring (STL / STREL)
# ---------------------------------------------------------------------------

rtamt-eval: dirs install ## Monitor diffusion1d fields with RTAMT (always-bound + eventually examples).
	@$(call ACTIVATE); \
	# Evaluate both baseline and STL runs if present.
	for tag in baseline stl; do \
	  ckpt="$(RESULTS_DIR)/diffusion1d_$${tag}_field.pt"; \
	  if [ ! -f "$$ckpt" ]; then \
	    # fall back to newest in run directories
	    ckpt="$$(ls -t $(RESULTS_DIR)/diffusion1d--$${tag}--*/diffusion1d_$${tag}_field.pt 2>/dev/null | head -n 1 || true)"; \
	  fi; \
	  if [ -n "$$ckpt" ] && [ -f "$$ckpt" ]; then \
	    echo "[rtamt] $$tag  ckpt=$$ckpt"; \
	    $(VENV_PY) scripts/eval_diffusion_rtamt.py --ckpt "$$ckpt" \
	      --signal max_u \
	      --stl "always[0,1](s <= 0.6)" \
	      --out "$(RESULTS_DIR)/diffusion1d_$${tag}_rtamt.json" || true; \
	    $(VENV_PY) scripts/eval_diffusion_rtamt.py --ckpt "$$ckpt" \
	      --signal max_u \
	      --stl "eventually[0.2,1](s <= 0.4)" \
	      --out "$(RESULTS_DIR)/diffusion1d_$${tag}_rtamt_eventually.json" || true; \
	  else \
	    echo "[rtamt] WARN: no ckpt found for $$tag; run 'make diffusion1d' first."; \
	  fi; \
	done

moonlight-eval: dirs install ## Run MoonLight/STREL monitoring on the latest heat2d field (or assets/heat2d_scalar fallback).
	@$(call ACTIVATE); \
	$(VENV_PY) - <<'PY'
	from __future__ import annotations
	import json
	import subprocess
	import sys
	from pathlib import Path
	
	results = Path("$(RESULTS_DIR)")
	assets  = Path("$(ASSETS_DIR)")
	
	# Prefer a freshly-generated packed field; fall back to the committed scalar rollout.
	field = results / "heat2d_field_xy_t.npy"
	src = "results"
	if not field.exists():
	    field = assets / "heat2d_scalar" / "field_xy_t.npy"
	    src = "assets"
	if not field.exists():
	    print("[moonlight] ERROR: no heat2d field found. Run 'make heat2d-figs' or ensure assets/heat2d_scalar exists.")
	    raise SystemExit(2)
	
	# Best-effort dt: use results/heat2d_dt.txt if present; else assets/heat2d_scalar/meta.json when using fallback.
	dt = 1.0
	dt_file = results / "heat2d_dt.txt"
	if dt_file.exists():
	    try:
	        dt = float(dt_file.read_text().strip())
	    except Exception:
	        dt = 1.0
	elif src == "assets":
	    meta = assets / "heat2d_scalar" / "meta.json"
	    if meta.exists():
	        try:
	            dt = float(json.loads(meta.read_text()).get("dt", dt))
	        except Exception:
	            dt = dt
	
	out_json = results / "heat2d_moonlight.json"
	cmd = [
	    sys.executable, "scripts/eval_heat2d_moonlight.py",
	    "--field", str(field),
	    "--layout", "xy_t",
	    "--mls", "$(HEAT2D_MLS)",
	    "--formula", "$(HEAT2D_MOONLIGHT_FORMULA)",
	    "--binarize", "--quantile", "0.90",
	    "--dt", str(dt),
	    "--out-json", str(out_json),
	]
	print("[moonlight] src=", src)
	print("[moonlight] dt =", dt)
	print("[moonlight] cmd:", " ".join(cmd))
	subprocess.run(cmd, check=False)
	print(f"✅ wrote {out_json}")
	# Best-effort: copy into the latest heat2d baseline run directory (if any).
	try:
	    run_dirs = sorted(results.glob("heat2d--baseline--*"), key=lambda p: p.stat().st_mtime, reverse=True)
	    if run_dirs:
	        dst = run_dirs[0] / out_json.name
	        dst.write_bytes(out_json.read_bytes())
	        print("[moonlight] copied to", dst)
	except Exception:
	    pass
	PY
# ---------------------------------------------------------------------------
# Benchmarks (wall-time / cost)
# ---------------------------------------------------------------------------

benchmark: dirs install ## Quick wall-time benchmark (baseline vs STL training) -> results/benchmark_training.csv.
	@$(call ACTIVATE); \
	$(VENV_PY) - <<'PY'
	from __future__ import annotations
	import csv
	import json
	import os
	import subprocess
	import sys
	time = __import__("time")
	from pathlib import Path
	
	results = Path("$(RESULTS_DIR)")
	results.mkdir(parents=True, exist_ok=True)
	
	device = "$(DEVICE)"
	epochs = int("$(BENCH_EPOCHS)")
	
	def run(cfg: str, tag: str) -> dict:
	    cmd = [
	        sys.executable, "scripts/run_experiment.py",
	        "-c", cfg,
	        "--device", device,
	        "--set", f"io.tag={tag}",
	        "--set", f"optim.epochs={epochs}",
	    ]
	    print("\n[benchmark] running:", " ".join(cmd))
	    t0 = time.perf_counter()
	    proc = subprocess.run(cmd, check=False)
	    t1 = time.perf_counter()
	    return {
	        "returncode": proc.returncode,
	        "config": cfg,
	        "tag": tag,
	        "device": device,
	        "epochs": epochs,
	        "wall_time_s": round(t1 - t0, 6),
	    }
	
	rows = []
	rows.append(run("$(CFG_DIFF_BASELINE)", "bench_baseline"))
	rows.append(run("$(CFG_DIFF_STL)", "bench_stl"))
	
	out_csv = results / "benchmark_training.csv"
	with out_csv.open("w", newline="") as f:
	    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
	    w.writeheader()
	    w.writerows(rows)
	print("✅ wrote", out_csv)
	
	# Also emit a richer JSON blob with best-effort hardware context (if available).
	blob = {"runs": rows}
	for extra in ["doctor.json", "hardware.json"]:
	    p = results / extra
	    if p.exists():
	        try:
	            blob[extra] = json.loads(p.read_text())
	        except Exception:
	            blob[extra] = p.read_text()
	out_json = results / "benchmark_training.json"
	out_json.write_text(json.dumps(blob, indent=2))
	print("✅ wrote", out_json)
	PY

# ---------------------------------------------------------------------------
# Additional demos / tooling
# ---------------------------------------------------------------------------

neuromancer-sine: dirs install ## Toy "framework demo": sine dynamics w/ STL loss (runs in pure torch mode by default).
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/train_neuromancer_stl.py --config "$(CFG_NEURO_SINE)" --mode torch --device "$(DEVICE)" || true

ablations: dirs install ## Sweep STL weight (lambda) for diffusion1d and generate figs/diffusion1d_ablations.png.
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/run_ablations_diffusion.py \
	  --config "$(CFG_DIFF_STL)" \
	  --weights 0 1 2 4 6 8 10 \
	  --device "$(DEVICE)" \
	  --out "$(RESULTS_DIR)/diffusion1d_ablations.csv"
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/plot_ablations.py "$(RESULTS_DIR)/diffusion1d_ablations.csv" \
	  -o "$(FIGS_DIR)/diffusion1d_ablations" \
	  --summary "$(FIGS_DIR)/diffusion1d_ablations_summary.csv"
	@# Keep a copy in assets for the report/slides if desired.
	@cp -f "$(FIGS_DIR)/diffusion1d_ablations.png" "$(ASSETS_DIR)/diffusion1d_robust_vs_lambda.png" 2>/dev/null || true
	@echo "✅ wrote $(FIGS_DIR)/diffusion1d_ablations.png and $(FIGS_DIR)/diffusion1d_ablations_summary.csv"

survey: dirs venv ## Generate/update docs/framework_survey.md (best-effort).
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/framework_survey.py > docs/framework_survey.md || true
	@echo "✅ wrote docs/framework_survey.md (best-effort)"

diagrams: dirs ## Generate docs/diagrams.md (Mermaid) with framework + diffusion1d dataflow diagrams.
	@mkdir -p docs
	@cat > docs/diagrams.md <<'EOF'
	# Diagrams
	
	This file is generated by `make diagrams` and is meant to be copy-pasteable into the report/slides.
	
	## High-level framework connections
	
	```mermaid
	flowchart LR
	  A[PDE / ODE / Hybrid model\n(IC/BC, parameters, domain)] --> B[Physics-ML framework\n(Neuromancer / TorchPhysics / PhysicsNeMo)]
	  B --> C[Neural surrogate\n(PINN / Neural ODE / Neural PDE)]
	  C --> D[Rollout / field samples\n(u(x,t), u(x,y,t), ...)]
	  D --> E[Spec monitor\nSTL (RTAMT) / STREL (MoonLight)]
	  E --> F[Robustness signal ρ(t)\n(positive = satisfied)]
	  F --> G[Training objective\nL = L_data + L_PDE + λ·L_STL]
	  G --> C
	```
	
	## Diffusion1D dataflow (example)
	
	```mermaid
	flowchart TD
	  Cfg[config.yaml\nPDE + net + STL spec + λ] --> Samp[Sample collocation points\n(x,t)]
	  Samp --> Net[PINN forward pass\nû(x,t)]
	  Net --> PDE[Compute PDE residual\n∂t û - k·∂xx û]
	  Net --> BCIC[Compute IC/BC residuals]
	  Net --> TS[Reduce field → time-series s(t)\n(e.g., maxₓ û(x,t))]
	  TS --> Mon[STL monitor / soft robustness\nρ = ρ(φ, s)]
	  PDE --> Lpde[L_pde]
	  BCIC --> Lbcic[L_bcic]
	  Mon --> Lstl[L_stl = softplus(-ρ)]
	  Lpde --> Sum[L_total = L_pde + L_bcic + λ·L_stl]
	  Lbcic --> Sum
	  Lstl --> Sum
	  Sum --> Opt[Optimizer step]
	  Opt --> Net
	```
	EOF
	@echo "✅ wrote docs/diagrams.md"
	@echo "Tip: GitHub renders Mermaid blocks; for LaTeX, screenshot or render via mermaid-cli."
	
dry-run: venv ## Validate configs without running training (uses scripts/run_experiment.py --dry-run).
	@$(call ACTIVATE); \
	$(VENV_PY) scripts/run_experiment.py -c "$(CFG_DIFF_STL)" --dry-run

demo: ## Professor demo: env + tests + quick diffusion + monitors + key figs.
	@$(MAKE) quickstart
	@$(MAKE) doctor
	@$(MAKE) check
	@$(MAKE) test-fast
	@$(MAKE) diffusion1d-demo
	@$(MAKE) heat2d-figs
	@$(MAKE) moonlight-eval
	@echo "✅ demo complete. Key artifacts:"
	@echo "  - assets/ (figures for report/slides)"
	@echo "  - results/ (fields + monitoring JSON)"
	@echo "  - figs/    (ablations + heat2d frames)"

# ---------------------------------------------------------------------------
# Internal helper targets
# ---------------------------------------------------------------------------

_sync_latest_run_artifacts: ## (internal) Copy stable result files into newest run dir for EXP/TAG.
	@$(call ACTIVATE); \
	exp="$(EXP)"; tag="$(TAG)"; \
	run_dir="$$(ls -td $(RESULTS_DIR)/$${exp}--$${tag}--* 2>/dev/null | head -n 1 || true)"; \
	if [ -z "$$run_dir" ]; then \
	  echo "[sync] WARN: could not find latest run dir for $${exp}--$${tag}"; \
	  exit 0; \
	fi; \
	echo "[sync] $$run_dir"; \
	# Candidate artifact names (keep in sync with experiments/*.py)
	for f in \
	  "$(RESULTS_DIR)/$${exp}_$${tag}.csv" \
	  "$(RESULTS_DIR)/$${exp}_$${tag}.pt" \
	  "$(RESULTS_DIR)/$${exp}_$${tag}_field.pt" \
	  "$(RESULTS_DIR)/$${exp}_$${tag}_rtamt.json" \
	  "$(RESULTS_DIR)/$${exp}_$${tag}_rtamt_eventually.json" \
	; do \
	  if [ -f "$$f" ]; then cp -f "$$f" "$$run_dir/"; fi; \
	done; \
	# Also write "stable" symlink-like copies for scripts/docs that expect fixed names.
	if [ -f "$(RESULTS_DIR)/diffusion1d_stl_field.pt" ]; then \
	  cp -f "$(RESULTS_DIR)/diffusion1d_stl_field.pt" "$(RESULTS_DIR)/diffusion1d_field.pt" 2>/dev/null || true; \
	fi

.PHONY: help dirs clean venv install install-dev install-extra install-all \
        install-torch-cpu install-torch-gpu install-rtamt install-moonlight install-neuromancer quickstart \
        fmt lint typecheck test test-fast ci doctor check \
        diffusion1d diffusion1d-baseline diffusion1d-stl diffusion1d-demo diffusion1d-figs \
        heat2d heat2d-demo heat2d-figs \
        rtamt-eval moonlight-eval benchmark neuromancer-sine ablations survey diagrams dry-run demo _sync_latest_run_artifacts
