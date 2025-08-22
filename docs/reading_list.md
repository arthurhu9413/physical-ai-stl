# Reading List — Physical AI + (Spatial) Signal Temporal Logic (STL)  _(living document)_

> Curated readings, docs, and repos to support **monitoring/enforcing STL & spatial STL (STREL)** in **physics‑based ML** (Neuromancer, NVIDIA PhysicsNeMo, TorchPhysics) and to pick **problem spaces/datasets**. Ordered so you can ramp fast, then go deep.

---

## 0) TL;DR — first week “must‑reads”

1. **STL foundations & monitoring**
   - Maler & Ničković, *Monitoring Temporal Properties of Continuous Signals*, **FORMATS/FTRTFT 2004** — introduces STL & finite‑trace monitoring. [PDF](https://www-verimag.imag.fr/~maler/Papers/monitor.pdf).
   - Fainekos & Pappas, *Robustness of Temporal Logic Specifications for Continuous‑Time Signals*, **TCS 2009** — quantitative (robust) semantics used by most tools. [Paper](https://www.sciencedirect.com/science/article/pii/S0304397509004149).
   - Donzé, *Efficient Robust Monitoring for STL*, **RV 2013** — fast sliding‑window monitoring. [PDF](https://www-verimag.imag.fr/~maler/Papers/STLRobustAlgo.pdf).

2. **Spatial STL**
   - Nenzi et al., *STREL: Spatio‑Temporal Reach and Escape Logic*, **LMCS 2022** — logic & offline monitor; basis of MoonLight. [PDF](https://lmcs.episciences.org/8936/pdf).

3. **Tools you’ll actually use**
   - **RTAMT** (Python) — offline/online STL monitors; dense & discrete time; optimized C++ backend for some modes. [GitHub](https://github.com/nickovic/rtamt) · [2025 arXiv](https://arxiv.org/abs/2501.18608).
   - **MoonLight** (Java tool with Python bindings) — monitors temporal, spatial (STREL) and spatio‑temporal specs. [GitHub](https://github.com/MoonLightSuite/moonlight) · [Docs](https://moonlightsuite.github.io/moonlight/).
   - **SpaTiaL** (Python) — spatio‑temporal spec & planning for object relations. [GitHub](https://github.com/KTH-RPL-Planiacs/SpaTiaL) · [API docs](https://kth-rpl-planiacs.github.io/SpaTiaL/) · [PyPI](https://pypi.org/project/spatial-spec).

4. **Physical‑AI frameworks to compare**
   - **Neuromancer** (PNNL) — PyTorch SciML for constrained optimization, physics‑informed ID, and differentiable MPC. [GitHub](https://github.com/pnnl/neuromancer) · [Docs](https://pnnl.github.io/neuromancer/) · [overview](https://drgona.github.io/projects/1_project/).
   - **NVIDIA PhysicsNeMo** — physics‑ML at scale; containerized; submodules: `physicsnemo‑sym`, `‑cfd`, `‑curator`. [Docs](https://docs.nvidia.com/physicsnemo/index.html) · [GitHub](https://github.com/NVIDIA/physicsnemo).
   - **TorchPhysics** — PINNs/DeepRitz/DeepONet/FNO; mesh‑free sampling. [GitHub](https://github.com/boschresearch/torchphysics) · [Docs](https://torchphysics.ai/).

5. **Operator‑/physics‑ML essentials**
   - **PINNs** — Raissi et al., *JCP 2019*. [Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125).
   - **Neural ODEs** — Chen et al., *NeurIPS 2018*. [Paper](https://arxiv.org/abs/1806.07366).
   - **Neural Operators** — (i) **FNO**: Li et al., 2020. [arXiv](https://arxiv.org/abs/2010.08895); (ii) **Survey**: Kovachki et al., *JMLR 2023*. [PDF](https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf); (iii) **DeepONet**: Lu et al., *Nat. Mach. Intell. 2021*. [Article](https://www.nature.com/articles/s42256-021-00302-5) · [arXiv 2019](https://arxiv.org/abs/1910.03193) · [code](https://github.com/lululxvi/deeponet).

---

## 1) STL / STREL / SaSTL — theory ➜ practice

### Core theory
- **STL (finite‑trace monitoring):**
  - Maler & Ničković, 2004 — canonical STL & monitor generation. [PDF](https://www-verimag.imag.fr/~maler/Papers/monitor.pdf)
  - Fainekos & Pappas, 2009 — **robustness** semantics. [Paper](https://www.sciencedirect.com/science/article/pii/S0304397509004149)
  - Deshmukh et al., 2015 — **robust online** monitoring. [PDF](https://people.eecs.berkeley.edu/~sseshia/pubdir/rv15.pdf)
  - Donzé, 2013 — efficient robust monitoring. [PDF](https://www-verimag.imag.fr/~maler/Papers/STLRobustAlgo.pdf)
- **Spatial extensions:**
  - **STREL** — Nenzi et al., 2022 (LMCS). [PDF](https://lmcs.episciences.org/8936/pdf)
  - **SaSTL** — Ma et al., ICCPS 2020 [PDF](https://www.cs.virginia.edu/~stankovic/psfiles/SaSTL_ICCPS_Camera_Ready.pdf) · extended **IEEE IoT‑J 2021** [PDF](https://www.cs.virginia.edu/~stankovic/psfiles/SaSTL_IoT_Journal.pdf)

### Tools & libraries
- **RTAMT** — [GitHub](https://github.com/nickovic/rtamt) · [2025 arXiv](https://arxiv.org/abs/2501.18608) · `pip install rtamt`.
- **MoonLight** — [GitHub](https://github.com/MoonLightSuite/moonlight) · [Docs](https://moonlightsuite.github.io/moonlight/).  
  _Env_: JDK 17–21; Python binding: `pip install moonlight`.
- **SpaTiaL** — [GitHub](https://github.com/KTH-RPL-Planiacs/SpaTiaL) · [API docs](https://kth-rpl-planiacs.github.io/SpaTiaL/) · [PyPI](https://pypi.org/project/spatial-spec).  
  _Note_: Automaton planning requires **MONA** via `ltlf2dfa` (Linux/macOS).
- **Also useful**:
  - **stlpy** — [docs](https://stlpy.readthedocs.io/en/latest/) — control/planning from STL (MIP & optimization encodings).
  - **STLCG / STLCG++** — [project page](https://uw-ctrl.github.io/stlcg/) — differentiable robustness; helpful for gradient‑based learning.
  - **PSY‑TaLiRo** — [paper](https://par.nsf.gov/servlets/purl/10299601); **Breach** — [GitHub](https://github.com/decyphir/breach) (background).

### How this maps to our project
- **Monitoring during training**: use robustness values as **loss terms** (soft constraints). For differentiability, prefer smooth/approx robustness (e.g., STLCG++); otherwise treat as penalties with sub‑gradients/STE.
- **Spatio‑temporal**: for PDE grids/meshes, build adjacency/graph and monitor **STREL** properties (MoonLight) over nodes/edges; for traffic/air‑quality, use sensor graphs.

---

## 2) Physical‑AI frameworks (to evaluate + choose)

### Neuromancer (PNNL, PyTorch)
- **What it is**: SciML + differentiable programming for **parametric constrained optimization**, physics‑informed system identification, and **differentiable MPC**.
- **Why we care**: Easy to inject **custom losses** (STL robustness), constraints, and differentiable models (Neural ODE/PDE blocks).
- **Links**: [GitHub](https://github.com/pnnl/neuromancer) · [Docs](https://pnnl.github.io/neuromancer/) · [Overview](https://drgona.github.io/projects/1_project/).

### NVIDIA PhysicsNeMo
- **What it is**: Open‑source **physics‑ML** framework with containerized training/inference; supports **symbolic PDE residuals** ([physicsnemo‑sym](https://github.com/NVIDIA/physicsnemo-sym)), **CFD** workflows ([physicsnemo‑cfd](https://github.com/NVIDIA/physicsnemo-cfd)), and data curation ([physicsnemo‑curator](https://github.com/NVIDIA/physicsnemo-curator)).
- **Why we care**: Scales well on NVIDIA stack; good exemplar for PDE surrogate/Neural‑Operator training where STL monitoring can wrap training/validation.
- **Links**: [Docs](https://docs.nvidia.com/physicsnemo/index.html) · [GitHub](https://github.com/NVIDIA/physicsnemo).  
- **Install tip**: Prefer official **container** for CUDA/driver alignment.

### TorchPhysics (Bosch Research + Univ. Bremen)
- **What it is**: PyTorch library for **PINNs, DeepRitz, DeepONet, FNOs**; mesh‑free sampling, inverse problems, parameter studies.
- **Why we care**: Clean API for PDE definitions & BC/IC; straightforward to add **STL‑as‑loss** hooks in training loops.
- **Links**: [GitHub](https://github.com/boschresearch/torchphysics) · [Docs](https://torchphysics.ai/).

### Worth knowing (comparators, not primary targets)
- **DeepXDE** (PINNs), **PINA** (JOSS 2023), **torchdiffeq** (Neural ODE solvers).

---

## 3) Operator learning & physics‑ML primers

- **PINNs** — Raissi, Perdikaris, Karniadakis, *J. Comput. Phys.* 2019. [Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)  
- **Neural ODEs** — Chen, Rubanova, Bettencourt, Duvenaud, *NeurIPS* 2018. [Paper](https://arxiv.org/abs/1806.07366)  
- **Neural Operators**  
  - **FNO** — Li et al., 2020. [arXiv](https://arxiv.org/abs/2010.08895)  
  - **Survey/tutorial** — Kovachki et al., *JMLR* 2023. [PDF](https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf)  
  - **DeepONet** — Lu et al., *Nat. Mach. Intell.* 2021. [Article](https://www.nature.com/articles/s42256-021-00302-5) · [arXiv 2019](https://arxiv.org/abs/1910.03193) · [Code](https://github.com/lululxvi/deeponet)

---

## 4) Exemplars tying logic ↔ learning

- **STLnet** — Ma et al., *NeurIPS 2020*: end‑to‑end enforcement of STL specs during sequence model training. [Paper](https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf) · [Code](https://github.com/meiyima/STLnet)  
  *Takeaway*: use teacher signals that are the “closest” spec‑satisfying traces to guide the student network.
- **SaSTL** — spatial aggregation logic for smart cities; ICCPS 2020 [PDF](https://www.cs.virginia.edu/~stankovic/psfiles/SaSTL_ICCPS_Camera_Ready.pdf) + IoT‑J 2021 [PDF](https://www.cs.virginia.edu/~stankovic/psfiles/SaSTL_IoT_Journal.pdf).  
- **RL + STL** — online robustness in RL: [rlstl repository](https://github.com/iitkcpslab/rlstl).  
- **Differentiable STL** — **STLCG / STLCG++** (2025): [project page](https://uw-ctrl.github.io/stlcg/) — smooth robustness for gradient‑based learning.

---

## 5) Datasets & problem spaces (STL/STREL angles)

> Aim for **one PDE** and **one spatio‑temporal sensor graph** dataset.

**PDE / CFD benchmarks**
- **PDEBench** (NeurIPS 2022 D&B) — code [GitHub](https://github.com/pdebench/PDEBench) · data [DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986).  
  *STL ideas*: bounds on fields (e.g., temperature/pressure), event timing (e.g., shock onset within \([t_1,t_2]\)), spatial containment (“hotspot does not escape region R”).  
- **FNO Navier–Stokes** datasets — convenient sources [package datasets](https://pypi.org/project/fourier-neural-operator/).  
- **Canonical diffusion/heat** (already in repo) — ideal for first STL‑as‑loss prototype (global envelopes, spatial gradient limits).

**Urban sensing / traffic**
- **METR‑LA / PEMS‑BAY** traffic datasets — standard graph time‑series for forecasting; see [DCRNN repo (data links)](https://github.com/liyaguang/DCRNN).  
  *STL ideas*: “no congestion > X minutes on corridor C,” “recovery within 10 min after incident,” **SaSTL** aggregations over sensor regions.

**Air quality (STLnet)**
- **Air‑quality multivariate time series** — datasets and scripts in [STLnet repo](https://github.com/meiyima/STLnet).  
  *STL ideas*: “PM2.5 stays below threshold after mitigation within \([0,1]\) h,” “if NO₂ spikes, O₃ stays within bounds 15–45 min later.”

**Other spatio‑temporal**
- **MoonLight** examples (sensor networks, bike‑sharing); **epidemic** STREL case study (dynamic graphs) — see [MoonLight docs](https://moonlightsuite.github.io/moonlight/) and [STREL paper](https://lmcs.episciences.org/8936/pdf).

---

## 6) Practical setup notes (so nothing blocks you)

- **Python** ≥ 3.10; `pip install -r requirements.txt` (this repo).  
- **RTAMT**: `pip install rtamt` — note API differences among versions.  
- **MoonLight**: install **Java (JDK 17–21)**; Python binding: `pip install moonlight`. If local Java is painful, use the project Dockerfile with `WITH_JAVA=1`.  
- **SpaTiaL**: `pip install spatial-spec`; automaton planning needs **MONA** (`apt install mona`) via `ltlf2dfa` (Linux/macOS).  
- **PhysicsNeMo**: use NVIDIA’s **container**; see [install docs](https://docs.nvidia.com/physicsnemo/index.html).  
- **TorchPhysics / Neuromancer**: standard PyTorch stacks; check their docs for exact versions ([TP docs](https://torchphysics.ai/), [Neuromancer docs](https://pnnl.github.io/neuromancer/)).

---

## 7) Stretch/background (skim as needed)

- Operator‑learning surveys & tutorials (2024–2025): e.g., [Kovachki JMLR 2023](https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf); newer perspectives appear regularly.  
- Verification context: **NNV 2.0** (CAV 2023) and **reachability of Neural ODEs** (FORMATS 2022) — useful for broader guarantees (adjacent to earlier topic).

---

## 8) Bib‑style quick references (by theme)

**STL & monitoring**
- Maler, O.; Ničković, D. **Monitoring Temporal Properties of Continuous Signals.** FORMATS/FTRTFT, 2004. [PDF](https://www-verimag.imag.fr/~maler/Papers/monitor.pdf)  
- Fainekos, G.; Pappas, G. **Robustness of Temporal Logic Specifications for Continuous‑Time Signals.** *Theor. Comput. Sci.*, 2009. [Paper](https://www.sciencedirect.com/science/article/pii/S0304397509004149)  
- Donzé, A. **Efficient Robust Monitoring for STL.** RV, 2013. [PDF](https://www-verimag.imag.fr/~maler/Papers/STLRobustAlgo.pdf)  
- Deshmukh, J. et al. **Robust Online Monitoring of STL.** RV, 2015. [PDF](https://people.eecs.berkeley.edu/~sseshia/pubdir/rv15.pdf)  
- **RTAMT** — [GitHub](https://github.com/nickovic/rtamt) · [2025 arXiv](https://arxiv.org/abs/2501.18608)

**Spatial STL / SaSTL**
- Nenzi, L. et al. **STREL.** *LMCS*, 2022. [PDF](https://lmcs.episciences.org/8936/pdf)  
- Ma, M. et al. **SaSTL.** ICCPS, 2020; **IEEE IoT‑J**, 2021. [ICCPS PDF](https://www.cs.virginia.edu/~stankovic/psfiles/SaSTL_ICCPS_Camera_Ready.pdf) · [IoT‑J PDF](https://www.cs.virginia.edu/~stankovic/psfiles/SaSTL_IoT_Journal.pdf)  
- **MoonLight** — [GitHub](https://github.com/MoonLightSuite/moonlight) · [Docs](https://moonlightsuite.github.io/moonlight/)

**Physical‑AI frameworks**
- **Neuromancer** — [GitHub](https://github.com/pnnl/neuromancer) · [Docs](https://pnnl.github.io/neuromancer/)  
- **NVIDIA PhysicsNeMo** — [Docs](https://docs.nvidia.com/physicsnemo/index.html) · [GitHub](https://github.com/NVIDIA/physicsnemo)  
- **TorchPhysics** — [GitHub](https://github.com/boschresearch/torchphysics) · [Docs](https://torchphysics.ai/)

**Physics‑ML primers**
- Raissi, M.; Perdikaris, P.; Karniadakis, G. **PINNs.** *JCP*, 2019. [Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)  
- Chen, R.T.Q. et al. **Neural ODEs.** *NeurIPS*, 2018. [Paper](https://arxiv.org/abs/1806.07366)  
- Li, Z. et al. **FNO.** 2020. [arXiv](https://arxiv.org/abs/2010.08895) · Kovachki, N. et al., **JMLR 2023** [PDF](https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf)  
- Lu, L. et al. **DeepONet.** *Nat. Mach. Intell.*, 2021. [Article](https://www.nature.com/articles/s42256-021-00302-5)

**Exemplars**
- Ma, M. et al. **STLnet.** *NeurIPS*, 2020. [Paper](https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf) · [Code](https://github.com/meiyima/STLnet)  
- RL‑STL integration — [rlstl GitHub](https://github.com/iitkcpslab/rlstl)  
- Differentiable STL — [STLCG / STLCG++](https://uw-ctrl.github.io/stlcg/)

---

### How to use this list
- Treat Sections **0–2** as week‑1 ramp.  
- Pick **one PDE** (PDEBench or our heat/diffusion) and **one graph dataset** (METR‑LA/PEMS or air‑quality) from Section 5, then map **2–3 STL/STREL properties** onto each.  
- While prototyping, keep RTAMT/MoonLight API docs open alongside your training loop code.

> _Maintainer note_: Keep this file precise and lean. If you add items, include one‑line **“why it matters here”** and a **working URL/DOI**.
