# topostream

**Topology Event Stream** toolkit for 2D XY and q=6 clock-model workflows.

This repository defines and implements a **portable event-stream layer** for topological-defect analysis:
- simulate spin fields
- extract vortices / antivortices
- pair defects
- compute summary observables
- emit schema-validated `vortex`, `pair`, and `sweep_delta` tokens with provenance

The current contribution is best understood as a **reproducible reference pipeline + artifact contract**.  
It is **not yet** a claim of new physics, experimental reproduction, or probe-agnostic inversion.

---

## What this is

This repo currently provides:

- **Reference implementations** for:
  - 2D XY Metropolis simulation (Numba)
  - q=6 clock Metropolis simulation (Numba)
  - plaquette-based vortex extraction
  - Hungarian min-cost vortex/antivortex pairing with `r_max` policy
  - core observables and summary outputs
- A **schema-defined token layer**:
  - `vortex`
  - `pair`
  - `sweep_delta`
- A **CLI workflow** for:
  - `reproduce`
  - `sweep`
  - `validate`
  - `plot`
- A **synthetic-first map-mode scaffold**:
  - forward models
  - explicit adapters
  - controlled degradation testing

---

## What this is not

This repo does **not** currently claim:

- a new physical law, phase-discovery method, or novel simulator
- reproduction of any specific experimental dataset end-to-end
- a general-purpose image inversion tool
- proof that the event-stream abstraction is already superior to raw-field workflows in all settings

Map-mode remains **adapter-explicit** and **synthetic-first**.

---

## Current status

### Implemented and test-covered
- XY simulator
- q=6 clock simulator
- vortex extraction
- Hungarian pairing
- helicity modulus
- clock-model metrics
- schema validation helpers
- CLI (`reproduce`, `sweep`, `validate`, `plot`)
- synthetic map forward/adapters
- determinism and schema-validation tests

### Implemented but still methodologically incomplete
- `vortex.confidence` field exists in the schema
- current extractor uses a **single-run default** confidence value
- multi-seed / re-detection confidence aggregation is **not yet wired**

### Important limitations
- the event-stream layer is implemented, but its **material advantage** over ordinary raw-field outputs is **not yet demonstrated**
- clock6 summaries should use a **clock-specific order observable** as the primary summary, rather than presenting `|ψ₆|` as the main signal
- packaging metadata should be aligned with runtime reality for CLI/plot dependencies

---

## Why the token stream exists

The point of the token stream is **not** to replace raw fields.

The point is to provide a **stable interchange layer** for downstream tasks that need:
- cross-pipeline interoperability
- compact defect-level analytics
- reproducible benchmarking
- map/simulation comparison without assuming identical raw inputs

The key question this repo is trying to answer is:

> Can topological defects be represented in a way that is portable across simulation outputs and constrained map-like inputs, without forcing every downstream tool to re-interpret raw arrays?

That question is only **partially answered** today:
- the representation exists
- the validation exists
- the proof of downstream advantage still needs to be shown

---

## Repository layout

| Path | Description |
|------|-------------|
| `docs/` | Locked specifications for inputs, algorithms, formulae, metrics, UQ, validation |
| `schemas/` | JSON schema for token outputs |
| `agents/` | Handoff contracts and gate ordering |
| `configs/` | Example reproduction config(s) |
| `src/topostream/simulate/` | XY and q=6 clock simulation |
| `src/topostream/extract/` | Vortex extraction and pairing |
| `src/topostream/metrics/` | Helicity, clock metrics, regime helpers |
| `src/topostream/map/` | Synthetic forward models and adapters |
| `src/topostream/io/` | Schema validation helpers |
| `tests/` | Gate tests and validation tests |
| `results/` | Run outputs (gitignored) |

---

## Installation

### Base install

```bash
python -m venv .venv
source .venv/bin/activate          # mac / linux
# .venv\Scripts\Activate.ps1      # windows powershell

python -m pip install -U pip
python -m pip install -e .
```

### CLI config support

The CLI reads YAML configs, so install:

```bash
python -m pip install pyyaml
```

### Plotting support

Plot generation requires:

```bash
python -m pip install matplotlib
```

If `matplotlib` is absent, the `plot` subcommand is skipped gracefully.

---

## Quick start

### Run tests

```bash
python -m pytest -q
```

### Run one sweep

```bash
python -m topostream.cli sweep --model XY --L 16 --T 0.9 --seed 42
```

### Run full reproduce config

```bash
python -m topostream.cli reproduce --config configs/default.yaml
```

### Validate all emitted tokens

```bash
python -m topostream.cli validate --results-dir results/
```

### Generate figures from summaries

```bash
python -m topostream.cli plot --results-dir results/ --output figures/
```

---

## Outputs

### Canonical token schema

Defined in:

```
schemas/topology_event_stream.schema.json
```

### Token types

#### `vortex`

```json
{ "id": "...", "x": 0, "y": 0, "charge": 1, "strength": 0.0, "confidence": 1.0 }
```

#### `pair`

```json
{ "pair_id": "...", "vortex_id": "...", "antivortex_id": "...", "separation_r": 0.0, "r_max_used": 0.0 }
```

#### `sweep_delta`

- Derived deltas between consecutive temperature-indexed snapshots
- **Not** physical-time dynamics

### Token requirements

All tokens must:
- validate against the schema
- include provenance
- be reproducible from the same seed + parameters

---

## Reproducibility stance

This repo is organized as a research artifact:

- specs are explicit
- outputs are schema-validated
- seeds are recorded
- tests enforce determinism and contract behavior

**What is still missing** is the next step:
- a frozen benchmark artifact
- real multi-seed confidence aggregation
- a downstream demonstration that consumes tokens directly and shows why the token layer matters

---

## Strongest honest positioning

> Today, **topostream** is best described as: a reproducible, schema-driven reference pipeline for XY / q=6 topological-defect workflows, with a portable token layer and synthetic-first map-mode scaffolding — but without a completed proof yet that the token abstraction materially outperforms ordinary raw-field workflows.

---

## Collaboration stance

This repository is designed to be readable by:
- technical reviewers
- future collaborators
- researchers who want a reproducible contract around defect extraction outputs

The best near-term collaboration target is **not** "new physics claims."  
It is:
- interoperability
- robustness benchmarking
- defect-level comparison across heterogeneous producers

---

## Immediate priorities

1. Remove package / README / output drift
2. Implement real multi-seed token confidence
3. Freeze one benchmark bundle
4. Add one downstream token-only analysis that proves the abstraction earns its keep

---

*Last updated: 2026-03-05*
