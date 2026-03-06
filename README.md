# TopoStream — Topological Defect Analysis Pipeline for 2D Spin Systems

Spec-driven, schema-stable pipeline for extracting, representing, and comparing
vortex / clock topological defects in 2D spin systems (XY, q=6 clock, and
synthetic map-mode inputs).

---

## What this repository contains

| Component | Status |
|-----------|--------|
| XY / clock6 Monte Carlo simulator (Numba) | ✅ implemented |
| Vortex extraction (plaquette winding number) | ✅ implemented |
| Vortex–antivortex pairing (Hungarian, PBC) | ✅ implemented |
| Schema-validated token event stream (JSONL) | ✅ implemented |
| Helicity modulus Υ(L,T) | ✅ implemented |
| Multi-seed confidence aggregation | ✅ implemented |
| clock6 model-aware summary (clock6_order primary) | ✅ implemented |
| Synthetic map-mode forward models + adapters | ✅ implemented (synthetic only) |
| Token-only downstream cross-pipeline benchmark | ✅ implemented |
| Frozen regression benchmark artifact | ✅ committed in `benchmarks/` |
| CLI (`reproduce`, `sweep`, `validate`, `plot`, `aggregate`) | ✅ implemented |
| Real experimental map integration | ❌ not implemented |

---

## Token schema

All pipeline outputs conform to `schemas/topology_event_stream.schema.json` (v1.1.0).

Three token types:

```
vortex      {id, x, y, charge (+1/-1), strength, confidence}
pair        {pair_id, vortex_id, antivortex_id, separation_r, r_max_used}
sweep_delta {delta_type, T_from, T_to, delta_value}
```

Every token carries a `provenance` block: model, L, T, seed, sweep_index, schema_version.
Tokens are written to JSONL files, one token per line.

---

## Quick start

```bash
pip install -e .                        # installs numpy, scipy, numba, jsonschema, pyyaml
pip install -e '.[plot]'                # also installs matplotlib for the plot command

# Single sweep
python -m topostream.cli sweep --model XY --L 16 --T 0.9 --seed 42

# Full reproduce run (uses configs/default.yaml)
python -m topostream.cli reproduce --config configs/default.yaml

# Validate all tokens in results/
python -m topostream.cli validate --results-dir results/

# Plot summaries
python -m topostream.cli plot --results-dir results/ --output figures/

# Multi-seed aggregation
python -m topostream.cli aggregate --results-dir results/

# Frozen benchmark check
make benchmark-check
```

---

## Dependencies

| Package | Role | Required |
|---------|------|----------|
| numpy | arrays | ✅ core |
| scipy | Hungarian assignment, Gaussian blur | ✅ core |
| numba | JIT-compiled MC | ✅ core |
| jsonschema | token schema validation | ✅ core |
| pyyaml | CLI config loading | ✅ core |
| matplotlib | `plot` sub-command | optional (`pip install 'topostream[plot]'`) |

---

## Key design choices

### Primary BKT observable: helicity modulus

Υ(L,T) is computed as a required metric for every sweep.
`psi6_mag` is retained as a diagnostic field in all summaries but is not
treated as the primary order parameter.

### Model-aware summary contract

Every `summary_*.json` file includes:
- `primary_order_name`: `"clock6_order"` for clock6, `"psi6_mag"` for XY
- `primary_order_value`: the value of that observable
- `psi6_mag`: always present as a diagnostic field

For clock6: `clock6_order` (population-concentration metric) is the meaningful
discriminant. `|ψ₆|` is algebraically near-1 for all discrete clock configs and
does not discriminate temperature — this is documented and not used as the primary.

### Pairing semantics

Hungarian min-cost bipartite matching with cutoff `r_max = L/4` (default).
Periodic boundary conditions via minimum-image distance.
`f_paired = (2 × n_pairs) / n_defects`.

### Multi-seed confidence aggregation

For each `(model, L, T)` condition, detection stability is computed across seeds
via cross-seed vortex clustering. Per-vortex confidence reflects how consistently
the vortex appears across independent seeds. Single-seed runs emit `confidence=1.0`
(documented as a placeholder, not a calibrated value).

### Map-mode

Map-mode is **synthetic-first only**. The pipeline includes:
- `forward_models.py`: theta → vector map, then blur / noise / masking
- `adapters.py`: vector map → theta (arctan2 inversion)

No real experimental map integration exists. The map-mode path is used only
to test that the token schema and downstream consumer are producer-agnostic.

### Token-only downstream benchmark

`src/topostream/analysis/token_benchmark.py` demonstrates producer-agnostic
downstream analysis: the same consumer compares simulation and degraded
map-mode outputs using only JSONL token files — never raw spin fields.

```bash
python scripts/run_token_benchmark.py
# results in results/token_benchmark/benchmark_results.json
```

What this proves: a single token-based downstream analysis can compare outputs
from both simulation and synthetic map-like producers without custom raw-data
logic for each producer.

What this does not prove: compatibility with real experimental data formats,
calibrated confidence across producers, or optimality of the matching heuristic.

### Frozen benchmark

`benchmarks/stage1_xy_single_sweep/frozen/` contains committed reference outputs
for a fixed XY run (L=16, T=0.9, seed=42). These are regression anchors, not
publication benchmarks.

```
benchmarks/stage1_xy_single_sweep/frozen/
  sim_XY_16x16_T0.9000_seed0042.npy
  tokens_XY_16x16_T0.9000_seed0042.jsonl
  summary_XY_16x16_T0.9000_seed0042.json
```

---

## Repository layout

```
src/topostream/
  simulate/       xy_numba.py, clock6_numba.py
  extract/        vortices.py, pairing.py
  metrics/        clock.py (psi6, clock6_order, helicity)
  aggregate/      confidence.py (multi-seed cross-seed clustering)
  map/            forward_models.py, adapters.py
  analysis/       token_benchmark.py (token-only downstream consumer)
  io/             schema_validate.py
  cli.py

benchmarks/stage1_xy_single_sweep/   frozen outputs + run_benchmark.py
schemas/                              topology_event_stream.schema.json
docs/                                 spec documents
configs/                              default.yaml
scripts/                              run_token_benchmark.py, physics_sanity_audit.py
tests/                                pytest suite (336 tests)
```

---

## Running tests

```bash
pytest tests/ -q
```

Expected: all tests pass except `test_load_default_config` if PyYAML is not
installed in the environment (the test calls `_load_config` on
`configs/default.yaml`; install `pyyaml` to fix it).

---

## What this is not

- Not a machine-learning phase classifier.
- Not a general-purpose Monte Carlo library.
- Not an attempt to reproduce any specific experimental paper.
- Map-mode does not connect to real experimental data; the adapter layer
  is synthetic-only and its domain of validity is documented but narrow.
