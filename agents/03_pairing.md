# agents/03_pairing.md — Vortex Pairing Agent
Implements: src/topostream/extract/pairing.py

Read before starting: SPEC_ALGORITHMS.md §2, SPEC_FORMULAE.md §5, SPEC_UQ.md §3

---

## Your deliverable
A vortex pairing module that:
1. Accepts lists of vortex and antivortex tokens + r_max + provenance
2. Returns pair tokens (Hungarian min-cost matching), unmatched lists, and pairing fraction
3. Uses minimum-image convention for all separations

## Required function (public API)
```python
def pair_vortices(vortices: list[dict], antivortices: list[dict],
                  L: int, r_max: float | None = None,
                  provenance: dict = None) -> dict:
    # Returns: {"pairs": [...], "unmatched_vortex_ids": [...],
    #           "unmatched_antivortex_ids": [...], "f_paired": float}
```

r_max default: L / 4 (in lattice units).

## Algorithm (strictly follows SPEC_ALGORITHMS.md §2)
```python
from scipy.optimize import linear_sum_assignment
# Build cost matrix with minimum-image distances
# Mask entries > r_max with 1e9
# Solve with linear_sum_assignment
# Emit pair tokens only where cost < 1e9
```

## pair_id convention
`"pair_{vortex_id}_{antivortex_id}"`

## Validation gate
tests/test_pairing.py must pass:
- 10 injected pairs recovered correctly
- f_paired = 1.0 for perfectly matched equal-count input within r_max
- r_max sensitivity: f_paired(r_max=L/8) ≤ f_paired(r_max=L/4) ≤ f_paired(r_max=L/2)

## Do NOT
- Use nearest-neighbor greedy matching
- Use Delaunay triangulation
- Compute r without minimum-image convention