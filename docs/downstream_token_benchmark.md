# Downstream Token Benchmark: Cross-Pipeline Defect Stability

## What this demonstrates

A single downstream consumer (`src/topostream/analysis/token_benchmark.py`)
can compare outputs from both simulation and synthetic map-mode producers
**without any producer-specific raw-data logic**.

The consumer reads only schema-valid JSONL token files. It does not read:
- raw theta arrays
- raw vector maps
- raw scalar maps
- simulator internals

The analysis code is identical for all producers. The producer identity
is visible only in `provenance.model` (which the consumer ignores for
the purpose of matching and metrics).

---

## Producer paths

### 1. Clean simulation (reference)
```
run_xy(seed=42) → theta → extract_vortices → pair_vortices → tokens_ref.jsonl
```

### 2. Degraded synthetic map-mode (candidate)
```
run_xy(seed=42) → theta → to_vector_map → DEGRADE → vector_map_to_theta → extract_vortices → pair_vortices → tokens_cand_{label}.jsonl
```

The simulation step is shared. The key distinction is that the map-mode path
goes through forward_model degradation + adapter inversion before extraction.
The downstream consumer sees only the resulting tokens — not how they were made.

---

## Degradation ladder

| Condition   | Description |
|-------------|-------------|
| `clean`     | No degradation: round-trip through to_vector_map + vector_map_to_theta |
| `blur_s1`   | Gaussian blur σ=1.0 on vector map components |
| `blur_s2`   | Gaussian blur σ=2.0 on vector map components |
| `noise_s03` | Additive Gaussian noise σ=0.3 on vector map components |
| `noise_s05` | Additive Gaussian noise σ=0.5 on vector map components |
| `mask_10`   | 10% random NaN masking on vector map (same sites for Mx, My) |

---

## Metrics computed (all from tokens only)

| Metric | Definition |
|--------|------------|
| **recall** | (matched vortices) / (reference vortex count) |
| **precision** | (matched vortices) / (candidate vortex count) |
| **false_positive_rate** | 1 − precision |
| **paired_fraction_ref** | (2 × pair count) / (vortex count) for reference |
| **paired_fraction_cand** | same for candidate |
| **paired_fraction_error** | abs(cand − ref) |
| **mean_separation_ref** | mean pair.separation_r for reference |
| **mean_separation_cand** | same for candidate |
| **mean_separation_error** | abs(cand − ref) |
| **mean_confidence_ref** | mean vortex.confidence for reference (if present) |
| **mean_confidence_cand** | same for candidate |

---

## Matching rules

- **Same-charge only**: vortex (+1) matches vortex (+1), antivortex (−1) matches antivortex (−1).
- **Minimum-image PBC distance** on the L×L lattice.
- **Tolerance**: 1.5 lattice units (just over one plaquette diagonal).
- **Greedy nearest-first** matching (simple, conservative, deterministic).
- Documented explicitly — no hidden heuristics.

---

## Exact command

```bash
python scripts/run_token_benchmark.py
```

Outputs:
- `results/token_benchmark/tokens_ref_simulation.jsonl`
- `results/token_benchmark/tokens_cand_{clean,blur_s1,...}.jsonl`
- `results/token_benchmark/benchmark_results.json`

---

## Confidence calibration note

All tokens currently have `confidence = 1.0` (single-run default).
Meaningful confidence calibration requires multi-seed aggregation,
which is implemented in `src/topostream/aggregate/confidence.py` but
is not wired into the map-mode path yet.

This is stated plainly: confidence calibration across producers is
deferred until map-mode aggregation is implemented.

---

## What this proves

A single token-based downstream analysis can compare outputs from
both simulation and synthetic map-like producers without custom
raw-data logic for each producer.

The same `compare_token_streams()` function produces valid recall,
precision, paired-fraction, and separation metrics regardless of
whether the tokens came from a direct simulation or a degraded
forward-model + adapter pipeline.

## What this does not prove

- That the tokenisation abstraction will work for all future real
  experimental data formats without modifications.
- That the matching heuristic (greedy nearest-first) is optimal.
  A transport-based matching could give tighter bounds.
- That confidence values are calibrated across producers.
  Single-run confidence = 1.0 is a placeholder.
- That the degradation ladder covers all realistic measurement artifacts.
  Real instruments have spatially correlated noise, drift, and non-linear
  response — none of which are modelled here.
- That recall/precision are publication-ready metrics for vortex
  detection benchmarking. They are useful regression anchors, not
  peer-reviewed performance claims.
