# SPEC_ALGORITHMS.md — Algorithm Specifications
schema_version: 1.0.0

---

## 1. Vortex extraction (normative pseudocode)

```
INPUT:  theta[L, L]  (angle field, float64, radians, PBC)
OUTPUT: list of {x, y, charge, strength}

FOR each plaquette (i, j) in L×L:
    A = theta[i,        j       ]
    B = theta[i,       (j+1)%L  ]
    C = theta[(i+1)%L, (j+1)%L  ]
    D = theta[(i+1)%L,  j       ]

    dAB = wrap(B - A);  dBC = wrap(C - B)
    dCD = wrap(D - C);  dDA = wrap(A - D)

    W_raw = (dAB + dBC + dCD + dDA) / (2π)
    charge = round(W_raw)   # must be in {-1, 0, +1}

    IF abs(W_raw - charge) > 0.01:
        LOG warning at (i,j); skip this plaquette
    ELIF charge != 0:
        EMIT token: x=j+0.5, y=i+0.5, charge=charge, strength=abs(W_raw)
```

Rules:
- L ≥ 8 required; raise ValueError otherwise.
- PBC wrapping is mandatory; open boundaries are rejected.
- NaN-corner plaquettes (map mode) are skipped; log count.

---

## 2. Vortex pairing — NORMATIVE ALGORITHM

This is the ONLY permitted pairing algorithm. Any change requires schema version bump.

```
INPUT:  vortices     = [{id, x, y, charge=+1}, ...]
        antivortices = [{id, x, y, charge=-1}, ...]
        r_max        (float, lattice units; default = L/4)
OUTPUT: pairs = [{pair_id, vortex_id, antivortex_id, separation_r, r_max_used}]
        unmatched_vortex_ids, unmatched_antivortex_ids

1. Build cost matrix C[i,j] = r_minimum_image(vortices[i], antivortices[j])
   using formula from SPEC_FORMULAE §5.

2. Mask: set C[i,j] = 1e9 if C[i,j] > r_max

3. Solve: row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)

4. For each (i, j) in solution:
   IF C[i,j] >= 1e9: mark both as unmatched
   ELSE: emit pair token with separation_r = C[i,j]

5. Remaining un-assigned indices → unmatched lists (logged, not errors)
```

r_max policy:
- Default: r_max = L / 4 (in lattice units)
- Rationale: pairs with r > L/4 are candidates for free (unbound) vortices at high T
- r_max is stored in every pair token and in provenance
- Sensitivity runs: r_max ∈ {L/8, L/4, L/2} required in validation suite

Pairing fraction:
    f_paired(T) = 2 × N_pairs(T) / (N_vortices(T) + N_antivortices(T))

---

## 3. Metropolis MC — XY model

```
INPUT:  L, T, J=1.0, N_equil, N_meas, N_thin, seed
OUTPUT: list of spin configurations (one per N_thin-th measurement sweep)

Initialize: theta[L,L] ~ Uniform[−π, π) using numpy RNG with given seed
step_size = π/3

EQUILIBRATION (N_equil sweeps):
  FOR sweep in range(N_equil):
    FOR each site in random order:
      dtheta = step_size × (2×rand() − 1)
      theta_new = wrap(theta[i,j] + dtheta)
      dE = −J × Σ_{nn k} [cos(theta_new − theta_k) − cos(theta[i,j] − theta_k)]
      IF dE < 0 OR rand() < exp(−dE/T):  accept

MEASUREMENT (N_meas sweeps):
  FOR sweep in range(N_meas):
    <one full sweep as above>
    IF sweep % N_thin == 0: configs.append(theta.copy())
RETURN configs
```

MANDATORY: the per-site inner loop MUST be Numba-jitted (@numba.njit) in production.
See agents/00_repo_rules.md §CPU for enforcement.

---

## 4. q=6 Clock model extension

Identical to XY MC above, with one modification:
After proposing theta_new, project to nearest q=6 state:
    allowed = [k × π/3 for k in range(6)]
    theta_new = allowed[argmin_k |wrap(theta_new − allowed[k])|]

All downstream algorithms (vortex extraction, psi6, helicity modulus) are identical.