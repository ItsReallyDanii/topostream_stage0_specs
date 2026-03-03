# SPEC_FORMULAE.md — Mathematical Definitions
schema_version: 1.0.0

All formulas below are normative. Any implementation deviation requires a schema version bump.

---

## 1. Angle wrapping (normative)

    wrap(Δθ) = arctan2(sin(Δθ), cos(Δθ))

This maps any real Δθ into (−π, π].
Implementation: `numpy.arctan2(numpy.sin(dtheta), numpy.cos(dtheta))`
This formulation is numerically stable and avoids modulo edge cases at ±π.
ALL angle differences in this codebase use this operator. No exceptions.

---

## 2. Vortex charge (winding number)

For plaquette (i,j) with corners traversed counter-clockwise:
  A = θ[i,   j  ]
  B = θ[i,   j+1 % L]
  C = θ[(i+1)%L, (j+1)%L]
  D = θ[(i+1)%L, j  ]

    W(i,j) = (1/2π) × [wrap(B−A) + wrap(C−B) + wrap(D−C) + wrap(A−D)]

W ∈ {−1, 0, +1} for well-defined spin fields.
- W = +1 → vortex (charge +1)
- W = −1 → antivortex (charge −1)
- W = 0  → no defect

Plaquette center position: x = j + 0.5, y = i + 0.5 (lattice units).

---

## 3. ψ₆ (sixfold order parameter)

Preferred approach — spin-angle based:

    ψ₆_spin = (1/N_sites) Σ_{x,y} exp(6i·θ(x,y))

|ψ₆_spin| → 1 in perfectly six-state clock-ordered phase.
|ψ₆_spin| → 0 in disordered phase.

Angle histogram: bin θ(x,y) into 36 bins over [−π, π).
Six peaks expected at multiples of π/3 in the clock-ordered phase.

---

## 4. Helicity modulus Υ(L,T)

For the XY model, H = −J Σ_{bonds} cos(θᵢ − θⱼ):

    Υ = (1/L²) × [ ⟨Σ_x cos(θᵢ−θⱼ)⟩ − (1/T) × ⟨(Σ_x sin(θᵢ−θⱼ))²⟩ ]

where Σ_x runs over all horizontal bonds only (twist in x-direction).

Universal BKT jump (infinite-L limit only):
    Υ(T_BKT⁻) = 2T_BKT/π,   Υ(T_BKT⁺) = 0

At finite L, Υ does not jump; use Nelson-Kosterlitz finite-size form for T_BKT estimates.
DO NOT claim a converged T_BKT from L ≤ 128 without explicit finite-size analysis.

For q=6 clock model: same estimator; two drops in Υ(L,T) locate T₂ and T₁.

---

## 5. Minimum-image pair separation

For matched pair (vortex at p⃗, antivortex at q⃗) on L×L lattice with PBC:

    r_x = min(|px − qx|, L − |px − qx|)
    r_y = min(|py − qy|, L − |py − qy|)
    r   = sqrt(r_x² + r_y²)

---

## 6. Spin-spin correlation function C(r,T)  [optional, Stage 2+]

    C(r,T) = ⟨cos(θ(0) − θ(r⃗))⟩

averaged over all site pairs at distance r ± 0.5.
BKT power law: C(r) ~ r^{−η(T)}, with η(T_BKT) = 1/4.