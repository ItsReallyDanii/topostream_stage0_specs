# topostream

**Topology Event Stream** toolkit for 2D XY / q=6 clock-style physics: extracts **vortex / pair / sweep_delta** tokens from simulated spin fields (and, later, constrained “map-mode” inputs), with **schema validation** and **reproducibility gates**.

This repo is structured as a **research artifact**: specs are locked, implementations are test-gated, and outputs are designed to be portable across pipelines.

## What this is (and is not)

**This is:**
- A **portable representation layer** (“topological event stream”) + reference implementation.
- CPU-first (Numba) simulation + extraction + pairing + metrics (**Υ helicity modulus**, **ψ₆**) with validation.

**This is not:**
- A claim of reproducing any specific experimental dataset (e.g., NiPS₃) end-to-end.
- A general-purpose “probe-agnostic” inversion tool for arbitrary images (map-mode is adapter-explicit and synthetic-first).

## Current status

**Locked specs (Stage 0):**
- `docs/` — normative inputs, formulae, algorithms, metrics, UQ, validation
- `schemas/` — `topology_event_stream.schema.json`
- `agents/` — handoff contracts and gate order

**Implemented + passing gates:**
- Agent 02 — vortex extraction ✅
- Agent 03 — Hungarian pairing + r_max policy ✅
- Agent 04 — metrics: helicity modulus Υ + ψ₆ + histogram + regime labeling ✅
- Agent 01 — Numba XY Metropolis simulator ✅
- Agent 05 — validation suite ✅

**Pending:**
- Agent 06 — CLI `reproduce` wiring ⏳

## Repo layout (high level)

- `docs/` — locked specifications (do not edit casually)
- `schemas/` — JSON schema for emitted tokens
- `agents/` — agent handoffs + gating rules
- `src/topostream/`
  - `simulate/` — XY (Numba) simulation
  - `extract/` — vortex extraction + pairing
  - `metrics/` — helicity modulus Υ, ψ₆, histograms, regime labeling
  - `map/` — (future) synthetic-forward models + adapters
  - `io/` — (future) schema validation helpers / persistence
- `tests/` — gate tests + validation suite
- `results/` — run outputs (ignored by git)

## Install (local)

```bash
python -m venv .venv
# mac/linux:
source .venv/bin/activate
# windows powershell:
# .venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install -e .
```

## Run tests (current gates)

```bash
python -m pytest -q
```

Or per gate:

```bash
python -m pytest -q tests/test_extract_vortices.py
python -m pytest -q tests/test_pairing.py
python -m pytest -q tests/test_metrics_helicity.py tests/test_metrics_clock.py
python -m pytest -q tests/test_sim_xy.py
python -m pytest -q tests/test_validation_suite.py
```

## Outputs (token stream)

The canonical output format is defined by:

- `schemas/topology_event_stream.schema.json`

Token types:
- `vortex` — `{id, x, y, charge, strength, confidence}`
- `pair` — `{pair_id, vortex_id, antivortex_id, separation_r, r_max_used}`
- `sweep_delta` — temperature-indexed snapshot deltas (not physical-time dynamics)

All emitted tokens should validate against the schema and include provenance metadata.

## Contact / collaboration

This repository is intended to be readable and reusable by other researchers. If you want to compare against experimental map-like outputs, the preferred workflow is:

1) provide a small sample of processed “map-mode” inputs + metadata  
2) define the supported map family and adapter assumptions explicitly  
3) evaluate token stability under controlled degradations (synthetic-first)

_Last updated: 2026-03-03_
