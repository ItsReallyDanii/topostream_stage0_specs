# agents/01_sim_xy.md — XY Model Simulation Agent
Implements: src/topostream/simulate/xy_numba.py

Read before starting: SPEC_INPUTS.md, SPEC_FORMULAE.md, SPEC_ALGORITHMS.md §3, SPEC_UQ.md

---

## Your deliverable
A Numba-jitted XY model Metropolis MC implementation that:
1. Accepts (L, T, J, N_equil, N_meas, N_thin, seed) as arguments
2. Returns a list of spin configuration arrays (float64, shape L×L, values in [−π,π))
3. Computes and returns helicity modulus Υ(L,T) from the measurement sweeps
4. Passes thermalization check defined in SPEC_UQ.md §4

## Required functions (public API)
```python
def run_xy(L: int, T: float, J: float = 1.0,
           N_equil: int = 10000, N_meas: int = 50000,
           N_thin: int = 50, seed: int = 42
           ) -> dict:
    # Returns: {"configs": [...], "helicity": float, "helicity_err": float,
    #           "energy_per_spin": [...], "provenance": {...}}
```

## Numba requirement
The per-site update loop MUST use @numba.njit. Wrap the full sweep function.
Reference pattern:
```python
@numba.njit
def _metropolis_sweep(theta, L, T, J, rng_state):
    ...
```

## Validation gate
Before this agent is done, tests/test_sim_xy.py must pass:
- Uniform initialization at T=0.1 stays ordered (mean |cos θ| > 0.9 after N_equil sweeps)
- At T=2.0, mean |cos θ| < 0.3 (disordered)
- Υ(L=32, T=0.6) > Υ(L=32, T=1.2) (low-T helicity exceeds high-T)
- Energy trace shows convergence (thermalization check passes)

## Do NOT
- Use open boundary conditions
- Implement a cluster algorithm (Metropolis only for now)
- Store all N_meas configs (only every N_thin-th)