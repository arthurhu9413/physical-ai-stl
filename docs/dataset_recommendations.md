# Dataset recommendations (STL/STREL-friendly)

This doc proposes **problem spaces / datasets** that make it easy to:
1) **demo + plot** spatio-temporal behavior clearly (time-series + PDE field plots),
2) **monitor STL / STREL** specifications (incl. falsification-style “what violated where/when?”),
3) **evaluate logic-guided training** (e.g., STL penalty / constraints) with meaningful baselines,
4) optionally support an **aerospace-facing narrative** (e.g., for an NFM-style paper target).

---

## TL;DR shortlist (what I would actually run)

| Tier | Recommended case study | Why it is STL/STREL-friendly | Typical “paper-grade” figures |
|---|---|---|---|
| **T0 (must)** | **2D heat / diffusion (synthetic PDE field)** | Clean spatio-temporal field; easiest on-ramp for **STREL** (containment/reach/surround) + STL (bounds + eventual cooling). | Field snapshots over time, robustness maps/curves, satisfaction rate, ablations over STL weight (λ), runtime overhead. |
| **T1 (must)** | **Traffic forecasting (METR-LA or PEMS-BAY; graph)** | “Space” is the sensor graph; standard spatio-temporal benchmark; natural specs like “congestion resolves within Δ”. | Time-series at selected nodes, graph heatmaps, robustness distribution, baseline vs STL-regularized comparisons. |
| **T2 (aerospace tie)** | **NASA CMAPSS turbofan (fleet time-series)** | Canonical **aeronautics** prognostics dataset; STL specs are natural (bounds, degrade-then-fail patterns). | RUL prediction vs spec satisfaction, robustness vs horizon, compute cost / hardware notes. |
| **Stretch (benchmark)** | **PDEBench (incl. 2D/3D compressible Navier–Stokes)** | Standard SciML benchmark; compressible flows can be framed as aero/CFD; has real “field” outputs. | Error + robustness across seeds, runtime scaling vs grid/time resolution, STL/STREL violation localization plots. |

**If time is tight:** do **T0 + T1** well.  
**If aiming for a submission-quality extension:** add **T2** and (if feasible) **PDEBench compressible NS**.

---

## Selection criteria (STL/STREL first)

A dataset/problem is “good” for this project if we can cleanly answer **yes** to most of:

1) **Observable semantics:** the signals/fields correspond to quantities with safety meaning (temperature, density, speed, pressure, etc.).
2) **Natural spec patterns:** we can write at least:
   - **Invariant / safety:** `G[a,b] (signal ≤ c)` or `G[a,b] (c1 ≤ signal ≤ c2)`
   - **Liveness / recovery:** `F[a,b] (signal ≤ c)` (eventual cooling / recovery)
   - **Rate-of-change / smoothness:** `G[a,b] (|Δsignal| ≤ c)` (optional)
   - **Spatial variants (STREL/SSTL):** “in region R”, “within distance d”, “surrounded by”, etc.
3) **Plotability:** we can show a compelling plot in one slide:
   - 1D/2D/3D fields (heatmaps, contours), or
   - clear multi-sensor time-series with a spatial layout (map/graph).
4) **Integration friction:** small enough to run on CPU for demos (or has a tiny subset), simple download/licensing, stable preprocessing.
5) **Baselines exist:** standard forecasters/operators exist so “STL helps” can be evaluated honestly.

---

## How we represent “space” (so STREL makes sense)

We need a **spatial structure** for STREL. We standardize on:

- **Grid PDE**: nodes are grid points; edges connect 4- or 8-neighborhood; distance is Euclidean or graph shortest-path.
- **Sensor networks / traffic**: nodes are sensors; edges are road-network adjacency; distance can be hop-distance or weighted by physical distance.
- **Irregular meshes (CFD)**: nodes are mesh vertices/cells; edges follow mesh connectivity; distance is geodesic/shortest-path on mesh graph.

This lets us apply STREL-style operators like “reach within distance d” or “surrounded by region satisfying φ”.

---

## Candidate datasets + spec templates

### 1) 2D Heat / Diffusion (synthetic; **core STREL demo**)

**Data type:** spatio-temporal field `u(x, y, t)` (temperature/concentration).  
**Why it’s great:** simplest “hello world” where STL + STREL are meaningful and plots are easy.

**STL specs (examples):**
- **Safety bound everywhere (approx. via max over sampled grid):**  
  `G[0,T] ( u(x,y,t) ≤ U_max )`
- **Eventual cooling at a point or region:**  
  `F[t_cool, T] ( u(x*,y*,t) ≤ U_cool )`  
  Region version (grid approximation): `F[t_cool,T] ( max_{(x,y)∈R} u(x,y,t) ≤ U_cool )`
- **Bounded overshoot after a step input:**  
  `G[0,T] ( u(x*,y*,t) ≤ U_set + δ )`

**STREL specs (examples; informal, to be instantiated in MoonLight/STREL syntax):**
- **Containment:** hotspots should not “reach” a protected region within distance d:  
  `G[0,T] ( Hot → ¬Reach_{≤d}(Protected ∧ Hot) )`
- **Surround:** if a hotspot exists, it must be surrounded by “safe” nodes (buffer ring):  
  `G[0,T] ( Hot → Safe Surround Hot )`
- **Spatial liveness:** any hotspot must disappear within Δ in its neighborhood:  
  `G[0,T] ( Hot → F[0,Δ] (¬Hot) )` evaluated over a local region.

**Framework fit:** TorchPhysics, PhysicsNeMo Sym, Neuromancer can all express/learn PDE surrogates / PINNs.

**What to plot (minimum):**
- `u(x,y,t)` snapshots at 4–6 times (heatmaps)
- STL robustness over time (and over epochs if using STL penalty in training)
- “Violation localization”: show where on the grid robustness is worst

---

### 2) 2D Reaction–Diffusion / Gray–Scott (synthetic; richer spatial patterns)

**Data type:** two fields `u(x,y,t), v(x,y,t)` with pattern formation.  
**Why it’s useful:** gives more interesting spatial logic than simple diffusion.

**Specs (examples):**
- **Invariant:** `G[0,T] ( 0 ≤ u ≤ 1 ∧ 0 ≤ v ≤ 1 )`
- **Eventual pattern constraint:** “by time t0, pattern intensity in region R exceeds θ”:  
  `F[t0,T] ( mean_{(x,y)∈R} v(x,y,t) ≥ θ )`
- **STREL shape constraint:** “high-v region stays inside ROI and does not leak”.

**Framework fit:** PhysicsNeMo has examples for Gray–Scott; also doable in TorchPhysics.

---

### 3) Traffic forecasting: METR-LA / PEMS-BAY (real; **graph space**)

**Data type:** time-series `v_i(t)` on a fixed sensor graph (nodes=sensors).  
**Good STL angle:** “congestion constraints / recovery within a time window”.

**Where to get it:** widely mirrored; one convenient open packaging is the Zenodo CSV release.

**STL specs (examples):**
- **No impossible speeds:** `G[0,T] (0 ≤ v_i(t) ≤ v_max)`
- **Recovery after congestion (per sensor):**  
  `G[0,T-Δ] ( (v_i(t) ≤ v_low) → F[0,Δ] (v_i(t) ≥ v_rec) )`
- **Rush-hour envelope (soft):**  
  `G[t1,t2] ( v_i(t) ≥ v_rush_min )` for selected corridors.

**Spatial / STREL angle (examples):**
- **Neighborhood congestion propagation:** if a node is congested, within K hops there exists a “witness” of congestion within τ:  
  (graph reachability with hop-distance)
- **Containment:** congestion should not spread beyond a subgraph (e.g., downtown region).

**Baselines to compare against:** DCRNN-style forecasters; STLnet is directly relevant as a logic-guided training baseline.

**What to plot:**
- time-series at a few representative sensors + overlay “spec satisfied/violated”
- heatmap over sensors (speed) at a timestamp + robustness overlay
- robustness distribution across nodes / time windows

---

### 4) Beijing multi-site air quality (real; small N; good for fast iteration)

**Data type:** pollutant time-series at 12 monitoring sites + meteorology.  
**Why it’s useful:** smaller than traffic; good for rapid “STL helps prediction” ablations.

**STL specs (examples):**
- **Safety:** `G[0,T] ( PM2.5(t) ≤ θ )` for WHO-style thresholds (choose θ carefully)
- **Eventual improvement:** `F[t0,T] ( PM2.5(t) ≤ θ_good )`
- **Multi-signal coupling:** e.g., `G[0,T] ( (wind_speed ≥ w0) → F[0,Δ] (PM2.5 decreases) )` (exploratory)

**Spatial angle:** small graph with geographic distances (STREL is possible but less compelling than traffic).

---

### 5) **NASA CMAPSS turbofan** (real; aerospace-facing; STL-centric)

**Data type:** multiple run-to-failure trajectories per engine; sensors + operational settings; RUL labels for test.  
**Why it’s great for “NFM-style” narrative:** it is explicitly aeronautics/prognostics; has established ML baselines; specs are meaningful.

**Candidate tasks:**
- RUL regression / early-warning classification
- multi-horizon forecasting with spec monitoring

**STL specs (examples):**
- **Always-safe sensor envelope:** `G[0,T] ( s_j(t) ≤ c_j )` for selected sensors  
  (choose sensors with clear semantics; document scaling/normalization!)
- **Degradation pattern:** “once a sensor crosses a warning band, it stays bad until failure”:  
  `G[0,T] ( (s_j ≥ warn) → (s_j ≥ warn) U[0,T] fail )`
- **Early-warning liveness:** `F[T-Δ, T] fail` (on simulated trajectories) to sanity-check monitors.

**What to plot:**
- a few sensor traces with detected violation intervals
- robustness vs time-to-failure and robustness vs prediction horizon
- runtime/overhead of monitoring per trajectory

---

### 6) PDEBench (benchmark; includes **compressible Navier–Stokes**)

**Data type:** HDF5 trajectories with standardized layout `[b, t, x1, ..., xd, v]`.  
**Why it’s useful:** credible benchmark suite; contains multiple PDE families; includes compressible NS that can be framed as aero/CFD.

**STL specs (examples):**
- **Physical plausibility:** `G[0,T] ( density > 0 ∧ pressure > 0 )`
- **Boundedness / stability:** `G[0,T] ( ||u|| ≤ c )` (choose field component + norm)
- **Eventual settling:** `F[t0,T] ( ||u(t)|| ≤ ε )` (problem-dependent)

**STREL angle:**
- “high-density region stays inside ROI” (containment),
- “shock-like feature does not reach boundary within d” (if applicable),
- “region is surrounded by safe buffer”.

**Recommendation:** start with a **small 2D case** (Darcy/shallow water) for integration, then move to **compressible NS** for a stronger benchmark.

---

### 7) Optional aerospace “fields” (if we want external aerodynamics beyond PDEBench)

If we want a stronger external-aero story (mesh/field surrogates), consider:
- PhysicsNeMo CFD examples (vortex shedding, external aerodynamic evaluation) as an entry point.
- Open airfoil CFD datasets (often large; best used via a small curated subset).

**Caveat:** many airfoil datasets are **steady-state** (no time), so they are better for *spatial* logic (STREL/SSTL) than full spatio-temporal logic unless we treat parameter sweeps as pseudo-time.

---

## Practical recommendation (for this semester)

**Minimum “A+ package”:**
1) **2D heat** with at least:
   - one safety spec (`G` bound),
   - one **eventually** spec (`F` cooling),
   - one **spatial** spec (STREL containment or surround),
   - plots + runtime notes.
2) One **real** spatio-temporal dataset:
   - **traffic** if you want graph space + stronger spatial story, or
   - **air quality** if you want faster iteration and smaller data.
3) If targeting a paper: add **NASA CMAPSS** as the explicit aerospace dataset, and/or add **PDEBench compressible NS** as a strong SciML benchmark.

---

## Pointers (official pages / docs)

**Physical AI frameworks**
- Neuromancer: https://github.com/pnnl/neuromancer + https://pnnl.github.io/neuromancer/
- NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo + https://docs.nvidia.com/physicsnemo/
- Bosch TorchPhysics: https://github.com/boschresearch/torchphysics + https://boschresearch.github.io/torchphysics/

**STL / STREL tooling**
- RTAMT (STL monitoring): https://github.com/nickovic/rtamt
- MoonLight (STREL monitoring): https://github.com/MoonLightSuite/moonlight (see Python interface docs)
- SpaTiaL (spatial-temporal relations; alternative approach): https://github.com/KTH-RPL-Planiacs/SpaTiaL

**Benchmarks / datasets**
- PDEBench (datasets + code): https://github.com/pdebench/PDEBench and https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
- Traffic (METR-LA / PEMS-BAY CSV packaging): https://zenodo.org/records/5146275
- Beijing air quality (UCI): https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata
- NASA CMAPSS (Open Data Portal): https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

---

## Appendix: quick “spec templates” to reuse

**STL (scalar)**
- Safety: `G[a,b] (x ≤ c)` / `G[a,b] (c1 ≤ x ≤ c2)`
- Recovery: `G[a,b-Δ] (bad → F[0,Δ] good)`
- Overshoot: `G[a,b] (x ≤ setpoint + δ)`

**Spatial quantification (grid)**
- Approx `∀(x,y)∈R`: use `max_{(x,y)∈R} x(x,y,t) ≤ c`
- Approx `∃(x,y)∈R`: use `min_{(x,y)∈R} x(x,y,t) ≤ c`

**Where spatial logic helps**
- “hotspot contained in ROI”
- “unsafe region does not reach boundary”
- “unsafe region is surrounded by a safe buffer ring”
