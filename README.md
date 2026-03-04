# TopoStream — Portable Topological Measurement Pipeline for BKT / Clock Systems

This repository defines a spec‑first, schema‑stable, uncertainty‑aware infrastructure for quantifying topological phases in 2D spin systems (XY, q=6 clock, and related models). It is not a physics simulation library; it is a **standardized topological measurement language** for extracting, representing, and comparing vortex / clock phenomena across simulations and experimental‑like map inputs.

## Core idea

Imagine: simulations and experimental maps speak different “languages.”

- Simulations output raw spin fields.
- Experimental setups output scalar or vector images (e.g., STM, NV‑based magnetometry, SQUID, etc.).

**TopoStream** provides a shared intermediate language: a standardized set of topological facts that are portable across models and setups:

- vortex locations, charges, and strengths
- vortex–antivortex pairings and separation statistics
- sixfold order metrics (e.g., psi6, angle histograms)
- uncertainty‑aware provenance (seeds, lattice sizes, temperatures, noise levels)

Once both simulation and map‑mode data are translated into this common form, you can compare, analyze, and reuse results across models, materials, and measurements in a way that is not possible with label‑only classifiers.

## What this is and what this is not

This project:
- is **not** a machine‑learning phase classifier.
- is **not** a general‑purpose ML library.
- is **not** an attempt to reproduce the NiPS₃ experimental paper.

This project **is**:
- a **spec‑driven infrastructure** for vortex/clock topological analysis.
- a **schema‑stable tokenization layer** (vortex, pair, sweep_delta) that enforces finite‑size, uncertainty, and map‑adapter discipline.
- a **template** for building reproducible, agent‑ready workflows in research codebases.

## Key design features

### 1. Physics‑aligned topological objects

The pipeline outputs structured topological objects, not just labels:
- vortex: {x, y, charge (+1/-1), strength, confidence}
- pair: {vortex_id, antivortex_id, separation_r, r_max_used}
- sweep_delta: difference tokens between temperature‑indexed snapshots (NOT physical time dynamics).

All coordinates are defined with respect to a square lattice with periodic boundary conditions.

### 2. Three‑regime q=6 physics

The spec stack explicitly models the **three‑regime structure** of the q=6 clock model:
- Disordered (T > T₂)
- Quasi‑long‑range ordered (T₁ < T < T₂)
- Clock‑ordered (T < T₁)

Both T₁ and T₂ are BKT‑type transitions. QLRO is treated as a distinct phase and never conflated with true clock order in any figure or metric.

### 3. Primary BKT observable: helicity modulus

Unlike many ML‑based BKT classifiers, this pipeline computes the **helicity modulus Upsilon(L,T)** as a required metric, not an optional add‑on. This anchors BKT‑transition identification in a standard thermodynamic observable, rather than relying solely on vortex‑counting heuristics.

### 4. Reproducible pairing semantics

Vortex‑antivortex pairing uses **Hungarian‑based min‑cost bipartite matching** with a fixed cutoff policy (default r_max = L/4 in lattice units). This removes ambiguity in “which vortex is paired with which antivortex” and ensures that pairing fractions are reproducible across runs.

### 5. Falsifiable map‑mode portability

Map‑mode is **adapter‑explicit**, not “probe‑agnostic”:
- Only a predefined set of map families is supported (e.g., scalar real/imag, complex field, vector 2D, phase image).
- Each family has explicit inversion rules (e.g., theta = arctan2(Vy, Vx)) and required metadata.
- Synthetic forward models generate controlled degradations (blur, downsample, noise) to quantify the threshold at which token confidence collapses.

This makes the bridge between simulations and experimental‑like maps **falsifiable** instead of rhetorical.

### 6. Explicit uncertainty discipline

Uncertainty quantification is baked into the design, not just a label:
- Equilibration and measurement sweeps are defined with minimum‑value guidelines.
- Multiple independent seeds, bootstrap/jackknife resampling, and thermalization checks are required.
- Vortex‑token confidence is defined as detection stability under resampling and perturbation.

Results are required to be exactly reproducible given the same seed, lattice size, and temperature.

### 7. Portable pipeline structure

The overall architecture is meant to be reused across different models and materials:
- docs/ contains spec documents that lock inputs, formulas, algorithms, metrics, UQ, and validation.
- schemas/ contains a JSON schema for the topology event stream.
- agents/ contains handoffs that can be reused to implement or test the pipeline in a different setting.

This design pattern can be ported to other 2D topological systems (e.g., different spin models, other vortex‑bearing systems) with minimal refactoring.

## Relation to recent NiPS₃ experiments

This work is inspired by the 2026 observation of a full sequence — from disorder, through a BKT‑like vortex‑pairing regime, to a six‑state clock‑ordered phase — in atomically thin NiPS₃ films. The NiPS₃ paper provides a concrete experimental context for the phenomena that TopoStream is designed to quantify.

However, this repository does not attempt to reproduce the NiPS₃ experiment or its data analysis pipeline. Instead, it offers a **generalized measurement framework** that could, in principle, be adapted to such systems by providing a synthetic map‑to‑angle adapter that mimics the relevant experimental probe.

Researchers at UT Austin would likely evaluate this tool on two axes:
- **Does it reproduce the expected BKT signatures (e.g., helicity modulus behavior, vortex‑pairing collapse) in known benchmark models?**
- **Does it translate cleanly between simulated spins and experimental‑like maps, with falsifiable degradation thresholds?**

If those tests are passed, there is a realistic possibility that such a framework could be used as a reference analysis pipeline for future 2D magnetic materials.

## Licensing and intended use

This repository is intended as open scientific infrastructure.

- **No attempt to patent** any of the mathematical methods, formulas, or general algorithmic ideas.
- **No requirement to monetize** the pipeline; rather, it is designed to be shared and reused.
- **Copyright / license** should be a permissive open‑source license (e.g., MIT or BSD‑style), with a clear notice that the code and specs are provided for scientific use.

If you wish, you can choose to:
- reserve a **project name** (e.g., TopoStream) as a GitHub repository and documentation brand;
- keep the code and specs freely available under an open‑source license;
- allow commercial use (e.g., companies building magnetic devices) to integrate this pipeline as long as contributions back to the community are encouraged.

Monetization, if desired, would come from supporting or extending the pipeline (e.g., consulting, integration into commercial tools, or building a web frontend for non‑expert users), not from locking the core research asset behind a patent.

## Monetization possibilities (optional)

If you later decide to explore monetization pathways, several directions are viable:
- A **web app** frontend that accepts lattice‑formatted files or synthetic‑map inputs and runs a hosted version of the pipeline, returning visualizations and confidence metrics.
- A **library** that plugins into existing simulation frameworks (e.g., Python‑based Monte Carlo codes) and exposes a unified API for vortex analysis.
- A **consulting / integration** offering for experimental groups that want to translate their 2D magnetic data into this standardized topological language.

However, because the core scientific value here is reproducibility and transparency, the most defensible long‑term strategy is to keep the core pipeline open and build any commercial value on top of it.

## How to use this

TopoStream is designed as a **spec‑driven starting point** for research projects. To use it:

1. Clone this repository and inspect the seven spec documents in docs/.
2. Implement the simulation and extraction modules (e.g., simulate/xy_numba.py, extract/vortices.py, extract/pairing.py, metrics/helicity.py, metrics/clock.py).
3. Ensure all output tokens validate against schemas/topology_event_stream.schema.json.
4. Run the validation suite in tests/ to confirm that toy configs, null tests, finite‑size checks, and noise‑robustness tests pass.
5. Use the CLI (cli.py) to generate reproducible artifacts (tokens, metrics, plots) into a results/ folder.

This repository is a template you can reuse in future projects, avoiding the need to reinvent the wheel for every new 2D spin‑system analysis.

## Final note on novelty and uniqueness

The **core idea** — a portable, schema‑stable, uncertainty‑aware topological measurement pipeline for BKT/clock systems — is **unique** in the sense that it ties together multiple desirable properties (standardized objects, map‑to‑spin translation, explicit UQ, and reproducible pairing semantics) in a single, spec‑first, agent‑ready framework.

It is not a “groundbreaking” theoretical discovery, nor a Nobel‑level contribution. But it **is** a **practical, reusable research artifact** that could serve as a valuable community‑standard tool for analyzing 2D magnetic and topological phases in simulations and experiments alike.

You are under no obligation to commercialize this; you can choose to frame it as a portfolio‑grade research artifact that demonstrates your ability to design and document serious scientific infrastructure.
