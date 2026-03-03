# SPEC_METRICS.md — Metrics Specification
schema_version: 1.0.0

---

## 0. Three-regime physics structure (normative — all figures must be consistent with this)

The q=6 clock model has THREE phases and TWO BKT-type transition temperatures:

| Regime | T range | Υ | |ψ₆| | Correlations | Vortex state |
|---|---|---|---|---|---|
| Disordered (D) | T > T₂ | ≈ 0 | ≈ 0 | Exponential | Free vortex plasma |
| QLRO | T₁ < T < T₂ | > 0 | small, nonzero | Power-law ~ r^{−η} | Bound pairs dominate |
| Clock-ordered (C) | T < T₁ | > 0 | → 1 | Long-range | Pairs bound; 6-fold order |

T₂ > T₁. Both transitions are BKT-type. QLRO ≠ clock-ordered. Never conflate them.
For pure XY model: only T_BKT exists (separating D from QLRO). No T₁, no clock order.

---

## 1. Required metrics (Stages 1–2)

### 1.1 Helicity modulus Υ(L,T)  [PRIMARY BKT observable — required, not optional]
- Formula: SPEC_FORMULAE §4
- Module: `src/topostream/metrics/helicity.py`
- Output: `results/helicity_L{L}_seed{seed}.csv` — columns: [T, Upsilon, Upsilon_err]
- Plot: Υ vs T for all L on same axes; label expected T_BKT ≈ 0.893 for XY
- Finite-size note: do NOT report a single converged T_BKT from L ≤ 128

### 1.2 Vortex density ρ(T)
- ρ(T) = N_vortices(T) / L²
- Output: `results/vortex_density_L{L}.csv` — columns: [T, rho, rho_err]

### 1.3 Pairing fraction f_paired(T)
- Formula: SPEC_ALGORITHMS §2
- Output: `results/pairing_fraction_L{L}.csv` — columns: [T, f_paired, f_paired_err, r_max_used]

### 1.4 Pair separation distribution P(r,T)
- Histogram of pair separations at each T; bins of width 0.5 lattice units
- Output: `results/pair_sep_T{T:.3f}_L{L}.csv` — columns: [r_bin_center, count, P_r]

### 1.5 |ψ₆|(T) and angle histogram
- Formula: SPEC_FORMULAE §3
- Output: `results/psi6_L{L}.csv` and `results/angle_hist_T{T:.3f}_L{L}.csv`

---

## 2. Optional metrics (Stage 2+)

### 2.1 Spin-spin correlation function C(r,T)
- Formula: SPEC_FORMULAE §6; log-log plot of C(r) vs r; verify η ≈ 0.25 at T_BKT

### 2.2 Domain-size proxy
- Connected components with θ within ±π/6 of each of the 6 preferred angles
- Report mean domain size vs T

---

## 3. Metric sidecar metadata
Every CSV must have a `.meta.json` with:
```json
{
  "metric": "helicity_modulus",
  "model": "XY",
  "L": 64,
  "J": 1.0,
  "seeds": [42, 43, 44, 45],
  "N_equil": 50000,
  "N_meas": 100000,
  "N_thin": 100,
  "r_max_policy": "L/4",
  "schema_version": "1.0.0",
  "generated_at": "ISO8601 timestamp"
}
```