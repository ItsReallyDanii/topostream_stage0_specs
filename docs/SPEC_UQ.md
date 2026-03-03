# SPEC_UQ.md — Uncertainty Quantification Protocol
schema_version: 1.0.0

---

## 1. MC run parameters

| Parameter | Symbol | Fast (dev) | Preferred (pub) | Notes |
|---|---|---|---|---|
| Equilibration sweeps | N_equil | 10,000 | 50,000 | Discarded; not measured |
| Measurement sweeps | N_meas | 50,000 | 200,000 | After equilibration |
| Thinning cadence | N_thin | 50 | 100 | Keep every N_thin-th config |
| Independent seeds | N_seeds | 4 | 8 | Separate RNG initializations |
| Lattice sizes | L | {16,32,64} | {16,32,64,128} | All at same T grid |

Sweep = one full lattice update (L² Metropolis proposals).

---

## 2. Error bar computation (normative)

### 2.1 Primary: seed-level standard error
    O_mean = mean([O_seed_0, ..., O_seed_{N-1}])
    O_err  = std([O_seed_0, ..., O_seed_{N-1}]) / sqrt(N_seeds)

Required for all metric CSV error columns.

### 2.2 Bootstrap (if N_seeds < 8)
- n_bootstrap = 1000 resamples with replacement
- Report 16th–84th percentile interval (1σ equivalent)

### 2.3 Jackknife (for autocorrelated observables, e.g., Υ)
- Block size: N_meas / (10 × N_thin) measurements
- Standard jackknife variance formula

---

## 3. Token confidence definition (normative)

    confidence = 1 − σ_detection / μ_detection

- μ_detection = mean vortex count across N_seeds at temperature T
- σ_detection = std of vortex count across N_seeds at temperature T
- confidence ∈ [0, 1]; values < 0.5 flagged in token

For map-mode tokens:
    confidence_map = fraction of N_noise_trials=20 where this vortex is re-detected
    (noise level per SPEC_VALIDATION §4.1)

Stored in: `vortex.confidence` field of every vortex token.

---

## 4. Thermalization check (required, not optional)

For each run, compute:
    E_early = mean energy/spin over first 25% of measurement sweeps
    E_late  = mean energy/spin over last 25% of measurement sweeps
    Assert: |E_late − E_early| / |E_late| < 0.01

If assertion fails: double N_equil, re-run, log warning in token provenance.

---

## 5. Random seed policy

Default seed sequence: [42, 43, 44, 45, 46, 47, 48, 49] — use first N_seeds.
Seeds must be set explicitly and stored in every output token's provenance block.
Results must be exactly reproducible given the same seed + parameters.
Never call random.seed() without an explicit argument.