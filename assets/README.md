# `assets/` — figures, diagrams, and tiny reproducible data

This folder is the **artifact shelf** for the project: it contains the **exact plots used in the report/slides** (PNG) plus **small numeric rollouts** used for monitoring demos. Everything here is intentionally small and deterministic so a reader (or Prof. Johnson) can immediately see:

- what examples were actually run,
- what specifications were monitored (STL / STREL),
- whether they were satisfied or falsified,
- and where the corresponding figures live.

> Want the project overview / framework survey? Start at [`../README.md`](../README.md).  
> Want exact commands to reproduce runs? See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

---

## Quick index (what to open first)

### Diffusion‑1D (PINN) — baseline vs +STL regularization

- Field plots (u(x,t)):
  - `diffusion1d_baseline_field.png`
  - `diffusion1d_stl_field.png`
- Training curves:
  - `diffusion1d_training_loss.png`
  - `diffusion1d_training_loss_components_stl.png`
  - `diffusion1d_training_robustness.png`
- λ‑sweep / trade‑off:
  - `diffusion1d_robust_vs_lambda.png` (copy of `../figs/diffusion1d_ablations.png`)

### Heat‑2D (scalar rollout) — STREL monitoring input

- Packed rollout for MoonLight:
  - `heat2d_scalar/field_xy_t.npy` (shape `(nx, ny, nt)`)
  - `heat2d_scalar/meta.json` (provenance)

---

## Diagrams requested by Prof. Johnson

### 1) High‑level “what connects to what”

```mermaid
flowchart LR
  PDE[PDE / ODE / Hybrid model\n(IC/BC, parameters, domain)] --> F[Physics-ML framework\n(Neuromancer / TorchPhysics / PhysicsNeMo)]
  F --> NN[Neural surrogate\n(PINN / Neural ODE / Neural PDE)]
  NN --> Field[Rollout / sampled field\nu(x,t), u(x,y,t), ...]
  Spec[Spec(s)\nSTL / STREL] --> Mon[Monitor\nRTAMT (STL) / MoonLight (STREL)]
  Field --> Mon
  Mon --> Rho[Robustness / satisfaction\nρ > 0 ⇒ satisfied]
  Rho --> Loss[Optional: differentiable penalty\nL_STL = ℓ(ρ) and total loss L = L_phys + λ·L_STL]
  Loss --> NN
```

### 2) Diffusion‑1D dataflow (what are the inputs/outputs?)

```mermaid
flowchart TD
  Cfg[Input: config\n(PDE + IC/BC + NN + spec params + λ)] --> Train[scripts/run_experiment.py\n(train PINN)]
  Train --> Model[Output: trained PINN\nresults/diffusion1d_*.pt]
  Train --> Field[Output: saved field u(x,t)\nresults/**/diffusion1d_*_field.pt]
  Train --> Logs[Output: per-epoch logs\nresults/diffusion1d_*.csv\n(losses + robustness)]
  Field --> RTAMT[Optional audit: scripts/eval_diffusion_rtamt.py\nreduce_x → s(t) then monitor STL]
  RTAMT --> JSON[Output: robustness summary\nresults/diffusion1d_*_rtamt.json]
  Field --> Fig[Plotting scripts / notebooks\n(see commands below)]
  Logs --> Fig
  Fig --> PNGs[Output figures\n(this folder: assets/*.png)]
```

---

## Example A — Diffusion‑1D PINN + STL (what the figures mean)

### Physical meaning

The saved field is a scalar PDE solution:

- `u(x,t)` = **temperature** (or any diffusing scalar) at location `x ∈ [0,1]` and time `t ∈ [0,1]`.
- PDE: `u_t = α u_xx` (1‑D heat/diffusion).
- The PINN `u_θ(x,t)` is trained from the PDE residual + IC/BC residuals.

Configs:
- Baseline (monitor‑only, λ = 0): [`../configs/diffusion1d_baseline.yaml`](../configs/diffusion1d_baseline.yaml)
- +STL regularized (λ > 0): [`../configs/diffusion1d_stl.yaml`](../configs/diffusion1d_stl.yaml)

### Spec monitored (written out)

We use a safety‑style **global upper bound**:

$$
\varphi_{\text{bound}} := \mathbf{G}_{[0,1]}\big(s(t) \le U_{\max}\big),
\quad s(t) := \max_{x \in [0,1]} u(x,t).
$$

- In practice, the spatial max is approximated on a grid:
  - **training‑time**: `stl.spatial = softmax` (smooth max, differentiable) or `amax` (hard max, nondifferentiable)
  - **monitor‑only / RTAMT audit**: `--agg amax` gives `s(t) = max_x u(x,t)` on the exported grid
- The temporal `G` operator is approximated using a soft‑min across sampled times (temperature `stl.temp`).

### What is “λ” / “stl_weight”?

In the code and configs, λ is the STL penalty weight:

- YAML: `stl.weight`
- Code: `stl_weight` (in `src/physical_ai_stl/experiments/diffusion1d.py`)
- Training objective:

$$
L_{\text{total}} = L_{\text{PDE}} + L_{\text{IC/BC}} + \lambda\,L_{\text{STL}},
$$

where `L_STL` is a smooth penalty that drives robustness positive (softplus/hinge on `ρ`).

### Monitoring results (already computed)

These JSON files record the monitored spec and robustness values:

- `../results/diffusion1d_baseline_rtamt.json`
- `../results/diffusion1d_stl_rtamt.json`

For the audited predicate `s(t) ≤ 1.0` (outer `always`):

| run | robustness ρ | satisfied? | note |
|---|---:|:---:|---|
| baseline | **−0.00397** | ✗ | slight overshoot above 1.0 |
| +STL | **+0.34378** | ✓ | clear safety margin |

(Positive robustness ⇒ satisfied; negative ⇒ violated.)

### Figures (open these)

#### Field plots `u(x,t)`

![Diffusion 1D baseline field](diffusion1d_baseline_field.png)

![Diffusion 1D +STL field](diffusion1d_stl_field.png)

#### Training curves

![Diffusion 1D training loss](diffusion1d_training_loss.png)

![Diffusion 1D loss components (STL run)](diffusion1d_training_loss_components_stl.png)

![Diffusion 1D robustness over epochs (STL run)](diffusion1d_training_robustness.png)

#### λ sweep (trade‑offs)

![Diffusion 1D robustness vs lambda](diffusion1d_robust_vs_lambda.png)

### Additional spec patterns (recommended in the report)

A liveness‑style “cooling” requirement can be expressed as:

$$
\varphi_{\text{cool}} := \mathbf{F}_{[t_c,1]}\big(s(t) \le U_{\text{cool}}\big)
$$

Example RTAMT formulas (variable name is `s` after spatial reduction):

- `always[0,1](s <= 1.0)`  (safety bound; used above)
- `eventually[0.8,1](s <= 0.40)`  (cooling; likely satisfiable for α=0.1)
- `eventually[0.8,1](s <= 0.30)`  (cooling; likely falsified on horizon [0,1])

---

## Example B — Heat‑2D rollout + STREL (MoonLight input asset)

This folder includes a **tiny, deterministic 2‑D heat rollout** intended as the simplest “spatial STL” demo.

### What is stored

- `heat2d_scalar/field_xy_t.npy` is a NumPy array with shape `(nx, ny, nt)` (layout `xy_t`).
- `heat2d_scalar/meta.json` includes parameters (grid size, dt, α, BC, init, seed, etc.).

### STREL spec (written out)

The repository’s MoonLight script is:

- [`../scripts/specs/contain_hotspot.mls`](../scripts/specs/contain_hotspot.mls)

Key formulas (verbatim, in MoonLight syntax):

```text
nowhere_hot     = !(somewhere[0, 100](hot));
quench          = globally(nowhere_hot);
contain         = eventually(quench);
contain_hotspot = contain;
```

Interpretation:

- `hot` is a boolean signal per grid cell (typically `hot := (u >= θ)` after thresholding).
- `somewhere[0,100](hot)` means “there exists a location within distance 0..100 that is hot”.
- `nowhere_hot` means “no hot cells exist anywhere (within the chosen reach)”.
- `contain_hotspot` requires that **eventually** the system reaches a state where **globally** no cell is hot (a quench).

> Why this is a good toy example  
> It makes the spatial aspect obvious: the spec talks about *where* hot spots exist, not just their time trace.

### How to (re)run STREL monitoring

From the repo root (after installing extras + a JDK):

```bash
python scripts/eval_heat2d_moonlight.py \
  --field assets/heat2d_scalar/field_xy_t.npy \
  --layout xy_t \
  --mls scripts/specs/contain_hotspot.mls \
  --formula contain_hotspot \
  --quantile 0.95 \
  --out-json results/heat2d_contain_hotspot.json
```

This produces a JSON summary (pass/fail or robustness depending on the spec domain).

### How to generate a 2‑D figure from the `.npy`

To make a slide‑ready snapshot (pick a time index `k`):

```bash
python - <<'PY'
import numpy as np
from scripts.utils_plot import plot_u_xy_frame

u = np.load('assets/heat2d_scalar/field_xy_t.npy')  # (nx, ny, nt)
for k in [0, u.shape[2]//2, u.shape[2]-1]:
    plot_u_xy_frame(u[:, :, k].T, out=f'figs/heat2d_t{k:03d}.png', title=f'Heat2D @ t[{k}]')
print('✅ wrote figs/heat2d_t*.png')
PY
```

---

## How these assets are generated (repro steps)

### 1) Run the experiments

```bash
# Baseline + STL-regularized diffusion PINN
python scripts/run_experiment.py -c configs/diffusion1d_baseline.yaml
python scripts/run_experiment.py -c configs/diffusion1d_stl.yaml

# λ sweep (creates results/diffusion1d_ablations.csv + figs/*)
python scripts/run_ablations_diffusion.py \
  --config configs/diffusion1d_stl.yaml \
  --weights 0 1 2 4 6 8 10 \
  --out results/diffusion1d_ablations.csv
python scripts/plot_ablations.py results/diffusion1d_ablations.csv -o figs/diffusion1d_ablations
```

### 2) Run monitors (audit)

```bash
# RTAMT / fallback robustness for diffusion (monitor s(t)=max_x u(x,t))
python scripts/eval_diffusion_rtamt.py \
  --ckpt results/diffusion1d--baseline--*/diffusion1d_baseline_field.pt \
  --semantics discrete --agg amax --spec upper --u-max 1.0 \
  --json-out results/diffusion1d_baseline_rtamt.json

python scripts/eval_diffusion_rtamt.py \
  --ckpt results/diffusion1d--stl--*/diffusion1d_stl_field.pt \
  --semantics discrete --agg amax --spec upper --u-max 1.0 \
  --json-out results/diffusion1d_stl_rtamt.json
```

### 3) Plot into `assets/`

The **committed PNGs in this folder** were generated from the saved `*_field.pt` and `*.csv` logs.
If you regenerate runs locally, you can recreate these images using the plotting helpers in:

- [`../scripts/utils_plot.py`](../scripts/utils_plot.py)

(Keeping plots in `assets/` makes the report/slides reproducible and keeps Git history clean.)

---

## House rules (so the repo stays lightweight)

✅ Put here:
- Figures (PNG/PDF) that appear in the report/slides
- Tiny `.npy` rollouts for monitoring demos (≤ a few MB) + a `meta.json`

❌ Do not put here:
- Large datasets, raw logs, or checkpoints (use `results/` or download scripts)
- Anything that makes `git clone` painful

---

## Attributions & upstream projects

This project connects to (and is inspired by) the following open-source tools:

- **Neuromancer** (physics-based ML / constrained optimization): https://github.com/pnnl/neuromancer
- **RTAMT** (STL monitoring): https://github.com/nickovic/rtamt
- **MoonLight** (STREL / spatial monitoring): https://github.com/MoonLightSuite/moonlight
- **SpaTiaL** (spatio-temporal logic tooling): https://github.com/KTH-RPL-Planiacs/SpaTiaL
- **NVIDIA PhysicsNeMo** (physics AI framework): https://github.com/NVIDIA/physicsnemo
- **Bosch TorchPhysics** (physics-informed deep learning): https://github.com/boschresearch/torchphysics

Please cite/acknowledge upstream projects as appropriate in the final report/paper.
