# Installing Optional Extras (Frameworks + Spatial Backends)

This repository is intentionally **layered**:

- **Core (always works):** lightweight STL monitoring + plotting + CLI utilities.
- **Training (optional):** PyTorch (CPU or CUDA) to train PINNs / neural models.
- **Framework comparisons (optional):** NeuroMANCER, TorchPhysics, PhysicsNeMo.
- **Spatial/STREL backends (optional):**
  - **MoonLight (STREL):** `moonlight` Python wrapper **+ Java 21+**
  - **SpaTiaL (spatial-lib):** installed from the upstream repo (non‑Windows)
  - **MONA:** only needed for SpaTiaL’s *automaton-based* planning utilities

If you are preparing a **demo** or a **report**, this page also points to the
exact commands that generate the figures, monitoring outputs, and benchmark
runtime tables that the repo’s configs expect.

---

## 0) One-glance: what do you need?

| Goal | Installs you need | Recommended command(s) |
|---|---|---|
| **Read docs / run monitoring-only code** | Core only | `make install` |
| **Run CPU training demos (Diffusion1D / Heat2D)** | Core + **PyTorch CPU** | `make install` then `make install-torch-cpu` |
| **Run full “extras” stack (Neuromancer / TorchPhysics / PhysicsNeMo)** | Core + PyTorch + frameworks | `make install-extra` *(Linux: add NVIDIA index url; see below)* |
| **Run STREL/MoonLight demo** | Core + **Java 21+** | install Java → `make install-moonlight` |
| **Run SpaTiaL geometry demo (no automata)** | Core + `spatial-lib` (git) | `make install-extra` *(non‑Windows)* |
| **Run SpaTiaL automaton planning** | Above + **MONA** | install MONA → run planning features |

> **Platform note**  
> Windows is supported for *core* and most CPU demos, but **SpaTiaL spatial-lib**
> and some toolchains are easiest via **WSL2**.

---

## 1) Repo-level “what connects to what” (block diagram)

This diagram is meant to make the codebase navigation and demo story clear.

```mermaid
flowchart LR
  %% Inputs
  PDE[PDE / dynamics + IC/BC] --> Sampler[Collocation / rollout sampler]
  Spec[STL / STREL spec φ] --> Monitor

  %% Core training loop
  Sampler --> Model[Neural model (PINN / NN)]
  Model --> PhysLoss[Physics loss<br/>(residual + IC/BC)]
  Model --> Signal[Signals / fields<br/>(time-series or u(x,t), u(x,y,t))]
  Signal --> Monitor[Monitoring / robustness ρφ]

  %% Monitoring backends
  subgraph STL_Backends[Temporal monitoring backends]
    RTAMT[RTAMT (STL)]
    SoftSTL[Differentiable STL penalty<br/>(soft robustness)]
  end
  subgraph Spatial_Backends[Spatial / STREL backends]
    MoonLight[MoonLight (STREL)<br/>Python + Java 21+]
    SpaTiaL[ SpaTiaL spatial-lib<br/>(git install) ]
  end

  Monitor -->|uses| RTAMT
  Monitor -->|or uses| SoftSTL
  Monitor -->|spatial| MoonLight
  Monitor -->|geometry| SpaTiaL

  %% Optimization
  PhysLoss --> TotalLoss
  Monitor --> STLLoss[STL/STREL penalty<br/>(from robustness)]
  STLLoss --> TotalLoss[Total loss<br/>L = L_phys + λ·L_spec]
  TotalLoss --> Optim[Optimizer]
  Optim --> Model

  %% Outputs
  Model --> Artifacts[Artifacts<br/>CSV logs, .pt fields, JSON monitor output, plots]
```

**Key takeaway:** the *inputs* are “physics + model + spec”, and the *output* is a
trained model plus **monitoring results** (robustness, falsification, plots).

---

## 2) Recommended install paths

All commands below assume you are at the **repo root**.

### 2.1 Create a virtual environment

```bash
make venv
# or: python3 -m venv .venv
```

### 2.2 Install the core (CPU-friendly)

```bash
make install
```

Core gives you: RTAMT + MoonLight Python wrapper (but **not Java**) + plotting +
the `physical_ai_stl` CLI utilities.

---

## 3) PyTorch (CPU or GPU)

PyTorch is treated as **optional** because it is the heaviest dependency.

### 3.1 CPU wheels (recommended for laptops + quick demos)

```bash
make install-torch-cpu
```

Equivalent manual install:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### 3.2 CUDA wheels (only if you have a compatible NVIDIA setup)

Use the official selector (recommended): https://pytorch.org/get-started/locally/

Or via Makefile:

```bash
# Example: CUDA 12.1 wheels
make TORCH_CHANNEL=cu121 install-torch-gpu
```

### 3.3 Verify torch

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available())"
```

---

## 4) MoonLight STREL backend (Java 21+)

MoonLight monitoring is triggered by scripts that call the Java backend through
the `moonlight` Python wrapper.

### 4.1 Install Java 21+

- **macOS (Homebrew):** `brew install openjdk@21`
- **Ubuntu/WSL:** `sudo apt-get update && sudo apt-get install -y openjdk-21-jre`
- **Other OS:** install any **Java 21+** runtime and ensure `java -version` works.

Verify:

```bash
java -version
```

### 4.2 Verify MoonLight import + Java presence

```bash
make install-moonlight
```

### 4.3 Run the “hello STREL” check

```bash
python -c "from physical_ai_stl.monitors.moonlight_strel_hello import strel_hello; r = strel_hello(); print(r.shape, float(r.min()), float(r.max()))"
```

---

## 5) SpaTiaL (two different installs)

Upstream SpaTiaL has **(at least) two Python surfaces** you may encounter:

1. **`spatial-spec` (PyPI)** → import `spatial_spec`  
   Lightweight “spec” distribution. Included in the **core** requirements.

2. **`spatial-lib` (from the SpaTiaL repo)** → import `spatial`  
   This is the geometry + monitoring library used by `spatial_demo.py`.  
   Installed by `requirements-extra.txt` on **non‑Windows** platforms.

### 5.1 Install spatial-lib (recommended)

If you installed extras (`make install-extra`), you already have it on non‑Windows.
Manual install:

```bash
python -m pip install "git+https://github.com/KTH-RPL-Planiacs/SpaTiaL.git#subdirectory=spatial-lib"
```

Quick check:

```bash
python -c "import spatial; print('spatial import OK')"
```

### 5.2 Optional: install MONA (only for automaton planning)

If you use `ltlf2dfa` → DFA conversion via MONA:

- **Ubuntu/WSL:** `sudo apt install mona`
- Otherwise: see https://www.brics.dk/mona/

---

## 6) Framework extras (Neuromancer / TorchPhysics / PhysicsNeMo)

### 6.1 One-command “extras” install

This installs torch + Neuromancer + TorchPhysics + (Linux) PhysicsNeMo, plus
spatial-lib and a few other research libraries.

```bash
# Easiest (all platforms): uses the Makefile (may skip PhysicsNeMo on non-Linux)
make install-extra
```

**Linux + PhysicsNeMo:** some `nvidia-*` distributions are hosted on NVIDIA’s index.
If you see resolver errors for `nvidia-physicsnemo`, install with one of:

```bash
# Option A (one-off command)
python -m pip install --extra-index-url https://pypi.nvidia.com -r requirements-extra.txt

# Option B (Makefile, using pip env var)
PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com make install-extra
```

> **CUDA note**  
> `requirements-extra.txt` only requires `torch>=...` and the Makefile does **not** pass `--upgrade`,
> so if you install a CUDA-enabled PyTorch first, `make install-extra` should not downgrade it.
> Recommended GPU flow:
>
> 1. `make TORCH_CHANNEL=cu121 install-torch-gpu` (or use the official selector)
> 2. `make install-extra`

> **PhysicsNeMo note (Linux-first)**  
> PhysicsNeMo is easiest on Linux and often easiest via NVIDIA’s containers.
> The repo keeps PhysicsNeMo behind a Linux marker to avoid breaking macOS/Windows.

### 6.2 Minimal per-framework installs (if you want to stay lean)

```bash
python -m pip install neuromancer
python -m pip install torchphysics
# Linux only (may require NVIDIA index):
python -m pip install --extra-index-url https://pypi.nvidia.com nvidia-physicsnemo
```

### 6.3 Quick “hello” checks (zero-training)

These are intentionally fast and CPU-only.

```bash
python -c "from physical_ai_stl.frameworks.neuromancer_hello import neuromancer_version; print('neuromancer', neuromancer_version())"
python -c "from physical_ai_stl.frameworks.torchphysics_hello import torchphysics_version; print('torchphysics', torchphysics_version())"
python -c "from physical_ai_stl.frameworks.physicsnemo_hello import physicsnemo_version; print('physicsnemo', physicsnemo_version())"
```

---

## 7) Demo + report support commands (plots, specs, runtime, hardware)

These commands are the *fastest path* to producing the artifacts that tend to
matter in a repo walkthrough or paper-style report.

### 7.1 Capture hardware + dependency versions (for the report)

Professor-facing reports typically require CPU/GPU/RAM + library versions.

```bash
make doctor
# writes: results/doctor.json
```

### 7.2 Generate example plots and monitoring outputs

**Diffusion1D (time-series + field plots + robustness):**

```bash
make diffusion1d-demo
make diffusion1d-figs
make rtamt-eval
```

**Heat2D (field snapshots + STREL audit):**

```bash
make heat2d-demo
make heat2d-figs
make moonlight-eval
```

### 7.3 Benchmark runtime cost (baseline vs STL)

This produces a small CSV table you can paste into a report:

```bash
make benchmark
# writes: results/benchmark_training.csv
```

---

## 8) Example-level “dataflow” diagram (Diffusion1D PINN + STL)

This is the “tied to an example” view the professor asked for.

```mermaid
flowchart TB
  PDE[u_t = α u_xx<br/>IC/BC] --> Samples[Sample (x,t) collocation points]
  Samples --> PINN[u_θ(x,t)]
  PINN --> Residual[Compute PDE residual + IC/BC loss]
  PINN --> Signal[u_θ on grid (x,t)]
  Signal --> Robust[Compute robustness ρ_φ via STL]
  Robust --> Penalty[STL penalty L_STL = softplus(-ρ_φ)]
  Residual --> Total[L_total = L_phys + λ·L_STL]
  Penalty --> Total
  Total --> Opt[Optimizer step]
  Opt --> PINN
  PINN --> Out[Outputs:<br/>trained weights + CSV + plots + JSON monitor]
```

**What is λ?**  
In this repo, λ is the STL regularization weight (see `stl.weight` in the YAML
configs). It scales the STL penalty term inside the total loss:

$$
L_{total} = L_{phys} + \lambda \cdot L_{STL}
$$

---

## 9) Troubleshooting (common issues)

### 9.1 “moonlight import works but STREL evaluation fails”

- Check `java -version` → must be **21+**
- Re-run: `make install-moonlight`
- If you are on Windows, prefer **WSL2** for the Java tooling.

### 9.2 “PhysicsNeMo won’t install”

- Linux is strongly recommended.
- Use NVIDIA’s index: `--extra-index-url https://pypi.nvidia.com`
- Consider using NVIDIA’s container workflow from the official docs:
  https://docs.nvidia.com/physicsnemo/

### 9.3 “SpaTiaL/ltlf2dfa doesn’t work on Windows”

That is an upstream limitation: SpaTiaL’s MONA/ltlf2dfa toolchain is
not Windows-friendly. Use WSL2 or skip automaton planning features.

### 9.4 Quick sanity: let the repo tell you what’s missing

```bash
python -m physical_ai_stl doctor
# or: python -m physical_ai_stl doctor --require physics stl
```

The doctor output prints **which optional deps are present** and provides a
**pip hint** for anything missing.



---

## Appendix A) Spec patterns you can reuse (STL + STREL)

This section is **not required** to install anything, but it is useful when you
need to *explain* what you monitored/trained against (slides/report), and it
directly addresses the “write the specs explicitly” feedback.

### A.1 Temporal STL (RTAMT / soft STL)

- **Safety bound (invariance):**  
  “temperature never exceeds a max”
  - Continuous intuition:  \(\forall t \in [0,T],\ \max_x u(x,t) \le U_{max}\)
  - STL sketch: `G[0,T] (max_x u(x,t) <= U_max)`

- **Cooling (eventual):**  
  “after some time, a probe point cools below a threshold”
  - Continuous intuition: \(\exists t \ge t_0 : u(x^*, t) \le T_{cool}\)
  - STL sketch: `F[t0,T] (u(x*, t) <= T_cool)`

**Where this plugs in (this repo):**
- Diffusion1D uses a built-in spec described in `src/physical_ai_stl/experiments/diffusion1d.py`
  and parameterized in `configs/diffusion1d_stl.yaml` (see `stl.weight`, `stl.u_max`).
- The STL penalty uses a robustness value ρ and converts it to a soft loss via:
  `L_STL = softplus(-ρ)` (so violations increase loss smoothly).

### A.2 Spatial / STREL (MoonLight)

The repo’s Heat2D audit uses MoonLight on the script:

- `scripts/specs/contain_hotspot.mls` (formula: `contain_hotspot`)

Intuition: a “hotspot” appears somewhere, but **eventually** the field becomes
“nowhere hot” for a sustained duration.

**Run it (after Java 21+):**
```bash
make heat2d-figs
make moonlight-eval
```

---

## Appendix B) Pointers to upstream docs (for installation + citation)

- PyTorch install selector: https://pytorch.org/get-started/locally/
- NeuroMANCER: https://github.com/pnnl/neuromancer
- TorchPhysics: https://github.com/boschresearch/torchphysics
- PhysicsNeMo docs: https://docs.nvidia.com/physicsnemo/
- RTAMT: https://github.com/nickovic/rtamt
- MoonLight: https://github.com/MoonLightSuite/moonlight
- SpaTiaL: https://github.com/KTH-RPL-Planiacs/SpaTiaL
- MONA: https://www.brics.dk/mona/
