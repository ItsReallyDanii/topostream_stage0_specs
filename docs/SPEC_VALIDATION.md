# SPEC_VALIDATION.md — Validation Suite Specification
schema_version: 1.0.0

All tests must pass before any Stage-N deliverable is marked complete.
Run with: `python -m pytest tests/ -v`

---

## 1. Toy configuration tests (gate for Stage 1)

### 1.1 Single vortex
- Config: θ(x,y) = arctan2(y − L/2, x − L/2), L=32
- Expected: exactly 1 vortex, charge=+1, position ≈ (L/2, L/2) ± 1 plaquette
- Assert: len(vortices)==1 and vortices[0].charge==+1

### 1.2 Single antivortex
- Config: θ(x,y) = −arctan2(y − L/2, x − L/2)
- Expected: exactly 1 antivortex, charge=−1

### 1.3 Bound pair (separation d=4)
- Config: θ = arctan2(y−yv, x−xv) − arctan2(y−ya, x−xa)
  with yv=ya=L/2, xv=L/2−2, xa=L/2+2
- Expected: 1 vortex, 1 antivortex, 1 pair, separation_r ∈ [3.5, 4.5]

### 1.4 Uniform field (vortex-free)
- Config: θ(x,y) = 0 everywhere
- Expected: 0 vortex tokens

---

## 2. Null tests

### 2.1 Random field
- Config: θ iid Uniform[−π, π), L=32
- Expected: vortex density ~ 0.5 (hot disordered)
- Expected: f_paired < 0.25 (free vortex plasma)

### 2.2 Pairing algorithm symmetry
- Inject N=10 pairs at known positions, separations r_k < r_max
- Assert: all 10 pairs recovered; separation_r within ±0.6 of truth
- Assert: 0 spurious pairs

---

## 3. Finite-size scaling tests (Stage 1 completion gate)

For L ∈ {16, 32, 64, 128}, T ∈ [0.5, 1.5], ≥20 temperature points:
- Assert: Υ(L, T=0.6) > Υ(L, T=1.2) for all L
- Assert: Υ(L, T=1.4) → 0 as L increases
- Known benchmark: T_BKT ≈ 0.893 J/k_B for XY; Υ(L,T) must show a feature near this value
- Plot Υ vs T for all L on same axes; include in Stage 1 deliverable

---

## 4. Noise and robustness tests

### 4.1 Spin perturbation
- Take converged low-T config; add Gaussian noise σ ∈ {0.05, 0.10, 0.20} rad/site
- Re-extract vortices; compute token confidence per SPEC_UQ §3
- Assert: vortex count change < 10% for σ=0.05; < 30% for σ=0.10

### 4.2 Map mode degradation (Stage 3 gate)
- Forward-generate map from known spin config
- Apply: blur σ ∈ {0.5, 1.0, 2.0}; downsample ∈ {2x, 4x, 8x}; additive noise σ ∈ {0.05, 0.20}
- Re-extract tokens; report confidence collapse threshold (confidence < 0.5)

---

## 5. Schema consistency test (Stage 2 gate)

- Run XY (Stage 1) and clock6 (Stage 2) pipelines to completion
- Load all output `.jsonl` files
- Validate every token against `schemas/topology_event_stream.schema.json`
- Assert: zero schema validation errors
- Assert: provenance fields (seed, L, T, schema_version) present and correct in every token

---

## 6. r_max sensitivity test

- Re-run pairing with r_max ∈ {L/8, L/4, L/2} for one L at 5 representative T values
- Report: how much f_paired changes; document sensitivity in results/