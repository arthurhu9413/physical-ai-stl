# Reproducibility Playbook

This file is the **single source of truth** for reproducing *everything that is “claim-worthy”* in this repository:

- runnable examples (1D diffusion PINN, 2D heat PINN, Neuromancer toy),
- monitored specs (STL + STREL),
- plots/figures used in slides/report,
- and basic **computational cost** + **hardware** reporting.

It is written to be **professor / artifact‑reviewer friendly**: copy‑paste commands, clear pass/fail checks, and stable output locations.

---

## 0) TL;DR (if you only do one thing)

### One-command end‑to‑end demo (CPU by default)

```bash
make demo
```

What this does:
- creates a local venv (`.venv/`)
- installs dependencies (lightweight by default)
- prints a **hardware + dependency matrix**
- runs fast tests
- runs a short Diffusion1D baseline + STL‑regularized training
- generates diffusion figures in `assets/`
- generates a Heat2D snapshot + STREL audit payload and runs MoonLight (if available)

✅ If `make demo` finishes, you have a complete “showcaseable” artifact trail under:
- `assets/` (figures)
- `results/` (run directories + monitoring JSON)
- `figs/` (ablations figure/table, if you ran ablations)

### Full “report/paper” reproduction (longer)

```bash
make doctor
make check
make diffusion1d
make ablations
make heat2d
make neuromancer-sine
make benchmark
make diagrams
```

> Tip: set `DEVICE=cuda` to use a GPU:
> `DEVICE=cuda make diffusion1d`

---

## 1) What “reproducible” means here

### 1.1 Scope

We aim for:
- **repeatability** on a fixed machine (same OS + Python + library versions + hardware), and
- **high‑level reproducibility** across machines (same qualitative trends; small numeric drift is acceptable).

This repo is intentionally layered:

- **Always works (core):** monitoring utilities + plotting + small tests.
- **Optional heavy stacks:** PyTorch training, Neuromancer / TorchPhysics / PhysicsNeMo.
- **Optional spatial backend:** MoonLight (STREL) requires **Java** at runtime.

### 1.2 Artifact map (where outputs go)

| Folder | What it contains | How to regenerate |
|---|---|---|
| `results/` | run directories (configs + env + tensors) + stable copies of key logs/weights | `make diffusion1d`, `make heat2d`, `make neuromancer-sine`, `make benchmark` |
| `assets/` | **report/slide-ready figures** + small packaged fields | `make diffusion1d-figs`, `make heat2d-figs` |
| `figs/` | ablation plots + summary CSV | `make ablations` |
| `docs/` | surveys + this playbook | `make survey`, edit docs |

---

## 2) Conceptual overview (what connects to what)

Prof. Johnson’s key request was **clarity about connections** and **dataflow**. The diagrams below are copy‑pasteable into a report/slides.

### 2.1 High-level framework connections

```mermaid
flowchart LR
  A[PDE / ODE / Hybrid model<br/>(IC/BC, parameters, domain)] --> B[Physics-ML framework<br/>(Neuromancer / TorchPhysics / PhysicsNeMo)]
  B --> C[Neural surrogate<br/>(PINN / Neural ODE / Neural PDE)]
  C --> D[Rollout / field samples<br/>(u(x,t), u(x,y,t), ...)]
  D --> E[Spec monitor<br/>STL (RTAMT) / STREL (MoonLight)]
  E --> F[Robustness signal ρ<br/>(positive = satisfied)]
  F --> G[Training objective<br/>L = L_data + L_PDE + λ·L_spec]
  G --> C
```

### 2.2 Example dataflow (Diffusion1D PINN + STL)

```mermaid
flowchart TD
  Cfg[config.yaml<br/>PDE + net + spec + λ] --> Samp[Sample collocation points<br/>(x,t)]
  Samp --> Net[PINN forward pass<br/>û(x,t)]
  Net --> PDE[Compute PDE residual<br/>∂t û - k·∂xx û]
  Net --> BCIC[Compute IC/BC residuals]
  Net --> TS[Reduce field → time-series s(t)<br/>(e.g., maxₓ û(x,t))]
  TS --> Mon[STL monitor / soft robustness<br/>ρ = ρ(φ, s)]
  PDE --> Loss[Total loss]
  BCIC --> Loss
  Mon --> Loss
  Loss --> Opt[Optimizer step]
  Opt --> Net
```

**Inputs:** a PDE + IC/BC, a neural architecture (PINN), and a spec (STL/STREL).  
**Outputs:** trained parameters + saved rollout/field tensors + monitoring/robustness JSON + figures.

---

## 3) Specifications used in this repo (STL + STREL)

A recurring instructor requirement: *“write the specs out explicitly.”* This section lists the exact specs used by the runnable examples.

### 3.1 Notation and the “λ (lambda)” parameter

Throughout the repo and report:

- **λ (“lambda”) = `stl.weight` / `stl_weight`**  
  It scales the spec penalty term inside the training objective:

```text
L_total = L_physics_or_data + λ * L_spec
```

- **Robustness sign convention (important):**  
  The monitoring utilities follow the standard convention:

```text
ρ >= 0  ⇒  spec satisfied
ρ <  0  ⇒  spec falsified
```

### 3.2 Diffusion1D safety spec (used in training)

Let the learned field be **u(x,t)**, typically interpretable as temperature/concentration at location x and time t.

We use a *global upper bound* safety spec:

- Define a scalar time-series via a spatial reduction:
  - `s(t) := max_x u(x,t)`
  - In code we approximate `max` on a finite grid of `stl.n_x` points.

- Safety spec:
  - `φ_safe := G_[0,T] ( s(t) <= U_max )`

Config references:
- Baseline (monitor-only): `configs/diffusion1d_baseline.yaml` (`stl.weight = 0`)
- STL-regularized: `configs/diffusion1d_stl.yaml` (`stl.weight > 0`)

### 3.3 Diffusion1D “eventually cooling” examples (evaluation via RTAMT)

To demonstrate *liveness* / “eventually” properties (as requested):

- `φ_cool := F_[0.8,1] ( s(t) <= 0.40 )`

RTAMT strings (already embedded in `configs/diffusion1d_stl.yaml`):

- `"eventually[0.8,1](s <= 0.40)"`
- `"eventually[0.8,1](s <= 0.30)"`

### 3.4 Heat2D spatial spec (STREL via MoonLight)

The Heat2D example produces a 3D field **u(x,y,t)** and (optionally) a Boolean predicate:

- `hot(x,y,t) := (u(x,y,t) >= threshold)`

We then evaluate a STREL-style containment property using MoonLight.

Spec file:
- `scripts/specs/contain_hotspot.mls`

Key formula family (see the `.mls` file for full definitions):

- `contain_within(deadline, tau) = eventually[0, deadline](globally[0, tau](nowhere_hot))`

Meaning (informal): “within `deadline`, the system enters a mode where for the next `tau` time units there is *nowhere* a hotspot.”

The default audit used by `configs/heat2d_baseline.yaml`:

- `formula: contain_within(0.75, 0.25)`
- `threshold: 0.25`
- `binarize: true`

---

## 4) Installation

### 4.1 Prerequisites

- Python **3.10+** recommended (3.11 is a safe choice)
- `make` (Linux/macOS; on Windows use the “manual commands” sections below)

Optional:
- CUDA GPU drivers if using `DEVICE=cuda`
- Java runtime (**Java 21+**) if you want STREL monitoring via MoonLight

### 4.2 Recommended install (Makefile)

```bash
make quickstart
```

This creates `.venv/` and installs:
- `requirements.txt` (core)
- `requirements-dev.txt` (dev/test tooling)

### 4.3 Install “everything” (framework comparisons + profiling)

```bash
make install-all
```

Equivalent manual install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-extra.txt
```

### 4.4 PyTorch install notes (CPU vs CUDA)

- CPU-only (recommended for laptops):
  ```bash
  make install-torch-cpu
  ```

- CUDA (only if your GPU/CUDA stack matches):
  ```bash
  make install-torch-gpu
  ```
  Or follow the official selector: https://pytorch.org/get-started/locally/

---

## 5) Environment + dependency reporting (for the report)

Prof. Johnson requested explicit **compute/hardware setup**. These commands generate machine-readable artifacts you can cite in the report.

```bash
make doctor
make check
```

Outputs:
- `results/hardware.json` (CPU brand, RAM, OS, etc.)
- `results/doctor.json` (Python + package versions, CUDA availability)
- `results/deps.md` (dependency availability matrix)

---

## 6) Sanity tests (fast)

```bash
make test-fast
```

If you want the full suite:

```bash
make test
```

---

## 7) Reproduce experiments + monitoring

This section is organized so you can run **toy → 1D → 2D**, each producing:
1) a rollout/field artifact,
2) a spec robustness evaluation,
3) plots.

### 7.1 List available experiments

```bash
python scripts/run_experiment.py --list
```

### 7.2 Diffusion1D (baseline vs STL) — the “core” example

#### Quick run (demo-length)

```bash
make diffusion1d-demo
```

#### Full run (report-length)

```bash
make diffusion1d
```

Key outputs:
- diffusion field tensors: `results/**/diffusion1d_*_field.pt`
- per-epoch logs: `results/diffusion1d_baseline.csv`, `results/diffusion1d_stl.csv`
- RTAMT monitoring JSON (generated by `make rtamt-eval`): `results/diffusion1d_*_rtamt.json`
- plots: `assets/diffusion1d_*.png`

Typical / expected qualitative outcome (what to look for):
- baseline: may slightly violate the upper bound (ρ can be negative)
- STL-regularized: robustness tends to move positive as training progresses
- ablation: increasing λ generally increases robustness, with diminishing returns


#### External monitoring with RTAMT (includes “eventually”)

**Recommended (robust path):**

```bash
make rtamt-eval
```

This monitors *both* baseline and STL runs (if present), using:
- `always[0,1](s <= U_max)` (safety), and
- example `eventually[...]` specs from `configs/diffusion1d_stl.yaml`.

**Manual (if you want a custom formula):**

```bash
# pick the newest STL field checkpoint
FIELD="$(ls -t results/diffusion1d--stl--*/diffusion1d_stl_field.pt | head -n 1)"

python scripts/eval_diffusion_rtamt.py \
  --field "$FIELD" \
  --agg amax \
  --stl "eventually[0.8,1](s <= 0.40)"
```

> Note: `--agg amax` corresponds to the spatial reduction `s(t)=max_x u(x,t)`.
> This is the bridge between a spatial field and a 1D STL monitor.

### 7.3 Ablation study: sweep λ (STL weight)

This produces both a CSV and a figure (good for the “results” section of the report).

```bash
make ablations
```

Outputs:
- `results/diffusion1d_ablations.csv`
- `figs/diffusion1d_ablations.png`
- `figs/diffusion1d_ablations_summary.csv`

### 7.4 Heat2D + STREL audit (MoonLight)

#### Quick run (demo-length)

```bash
make heat2d-demo
```

#### Full run (report-length)

```bash
make heat2d
```

Then generate the STREL payload + a snapshot figure:

```bash
make heat2d-figs
```

Finally evaluate the STREL property with MoonLight:

```bash
make moonlight-eval
```

Key outputs:
- Heat2D field tensor: `results/**/heat2d_*_field.pt`
- Packed NumPy field + metadata (for MoonLight): `assets/heat2d_scalar/`
- MoonLight JSON: `results/heat2d_moonlight.json`

### 7.5 Neuromancer “toy” demo: sine regression + STL safety bound

This is the simplest end-to-end demonstration of:
data → model → STL penalty → offline monitoring.

```bash
make neuromancer-sine
```

Under the hood, this runs:
- `scripts/train_neuromancer_stl.py`
- spec: `G(y(t) <= bound)` (ReLU-based soft penalty during training)
- optional independent RTAMT robustness

Outputs are written under `runs/neuromancer_sine_bound/` (JSON + optional saved model).

---

## 8) Generate figures (what goes into slides/report)

### 8.1 Diffusion figures

```bash
make diffusion1d-figs
```

Writes into `assets/`:
- `diffusion1d_baseline_field.png`
- `diffusion1d_stl_field.png`
- `diffusion1d_training_loss.png`
- `diffusion1d_training_loss_components_stl.png`
- `diffusion1d_training_robustness.png`
- `diffusion1d_robust_vs_lambda.png` (if ablation CSV exists)

### 8.2 Heat2D snapshot + STREL payload

```bash
make heat2d-figs
```

Writes into `assets/heat2d_scalar/` (and a snapshot figure if configured).

### 8.3 Diagrams (copy‑paste into the report)

```bash
make diagrams
```

Writes `docs/diagrams.md` (Mermaid) containing the same diagrams as Section 2.

---

## 9) Computational cost (runtimes) — baseline vs STL

Prof. Johnson requested explicit compute cost reporting.

```bash
make benchmark
```

This writes:
- `results/benchmark_training.csv` (wall-time per training run)
- and includes any available hardware blobs (`results/doctor.json`, `results/hardware.json`)

For monitoring/inference-side cost (not training), you can time just the monitors:

```bash
time make rtamt-eval
time make moonlight-eval
```

For more fine-grained timing, you can also wrap any command with:

```bash
/usr/bin/time -v python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml --set optim.epochs=200
```

---

## 10) Determinism / seeds

### 10.1 What we control

- A single “master seed” flows into NumPy and PyTorch.
- Configs avoid nondeterministic training tricks by default (AMP off, compile off).

### 10.2 What we *do not* guarantee

Even with seeds fixed, **exact bit‑for‑bit equality across machines** is not guaranteed due to:
- different BLAS / CPU instruction sets,
- GPU nondeterminism unless strict flags are enabled,
- library version drift.

If you need strict determinism on CUDA, consult PyTorch’s reproducibility notes:
https://docs.pytorch.org/docs/stable/notes/randomness.html

---

## 11) Troubleshooting

### 11.1 MoonLight / Java

If `make moonlight-eval` fails:
- confirm `java -version` works
- install a Java 21+ runtime and ensure it is on your PATH

### 11.2 “I don’t have make” (Windows)

All Makefile commands correspond to plain Python commands. For example:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-extra.txt
python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml --set optim.epochs=100
```

### 11.3 “RTAMT import error”

Install RTAMT:

```bash
python -m pip install rtamt
```

---

## 12) Reference artifact hashes (optional but useful)

The repo includes **reference artifacts** (figures + logs) so a reviewer can verify
they have the same files *without rerunning training*.

> These hashes are for the committed artifacts in this repository snapshot.

```text
results/diffusion1d_baseline.csv  3ca71dd2135163e312af1b6b60974d618d1b8f7a35e17693b2a30f9957dd3b6c
results/diffusion1d_baseline.pt  317e161a117a1ad088062889de4f9d8cd843c4933e68aaf82692d4c93cd7d135
results/diffusion1d_baseline_rtamt.json  afac2d378a1f9a50a7a55bf29d5e2981512c48e790e768ce1d7a185e28523d77
results/diffusion1d_stl.csv  3c391d39a0612a3c6590285f1452943e5a206d8643053622321f35c0e450e793
results/diffusion1d_stl.pt  85aff4c97ca05cc2edb355b7dc295b9741291c9575a66f9e14763a1f53e10be1
results/diffusion1d_stl_rtamt.json  6952861420ccb2d75b92a84ea8b7c39049d601cf8bb5d28a7412e819240a37aa
results/diffusion1d_ablations.csv  859ba42fc823a5fe602b61cd3a40fbb6b6660f7397dd2e1b4649b523de15e428
assets/diffusion1d_baseline_field.png  8ee61aadcaf00ea7e6869a9a6ab1b028371c7b5ae25023ba20179497b3f699af
assets/diffusion1d_stl_field.png  61a9cc26b167d3ff8d5729a7704bfae637494c4f66843817afeb0cdbcea82df8
assets/diffusion1d_training_loss.png  92a47a97a9677f7abf7b352eaefc4c56d392aa2f65b11c6a6b977e94d4e2ba9a
assets/diffusion1d_training_loss_components_stl.png  0217465ea7b94a8c7016b12e915fbd2c860f09dc33b6ebeecf15b13e2df399d9
assets/diffusion1d_training_robustness.png  04d800256ae87cab9ba6dd36adbf9cd93755792f7c4d1a16090b5b122f8cb996
assets/diffusion1d_robust_vs_lambda.png  da3cb7d1cf0b9a8380613e1c350cf66c3bf8493846905ed4a704b3a24b0c1763
figs/diffusion1d_ablations.png  fbff583422f79f8b3222abff10522923b2aa864fcdd636c4f06d14199119ae1d
figs/diffusion1d_ablations_summary.csv  64ae025f1aeff7ed79d54585b9ad1f78220fe4baba37209142ef5a2ce89bbab0
assets/heat2d_scalar/meta.json  b691c78d665eedfbd5c3075b34905bdc2625b34c184e0a8e3cc4eb14e7edff1d
assets/heat2d_scalar/field_xy_t.npy  1165ce2cb1785a38d47c12db3551686db938220baa28308366df8fd8010c12a7
results/diffusion1d--baseline--20251117-164754-066122/diffusion1d_baseline_field.pt  7e04865bf7157195ec7c671ef4557e32c7e628b6132e58bb226cdfdcd1034299
results/diffusion1d--stl--20251117-170002-822925/diffusion1d_stl_field.pt  4cf82314dcef38c337b54b7dd7d1239b63cb1be3ac844db3fe5c4f0fc7786070
```
