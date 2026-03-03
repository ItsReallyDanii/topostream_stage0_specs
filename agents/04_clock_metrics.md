# agents/04_clock_metrics.md — Clock / ψ₆ Metrics Agent
Implements: src/topostream/metrics/clock.py, src/topostream/metrics/helicity.py

Read before starting: SPEC_FORMULAE.md §3–4, SPEC_METRICS.md §0–1

---

## Your deliverable

### helicity.py
```python
def compute_helicity(theta: np.ndarray, T: float, J: float = 1.0) -> tuple[float, float]:
    # Returns (Upsilon, Upsilon_jackknife_err) for a single config
    # Use horizontal bonds only (twist in x-direction)
    # Formula: SPEC_FORMULAE.md §4
```

### clock.py
```python
def compute_psi6(theta: np.ndarray) -> complex:
    # Returns complex ψ₆ = mean(exp(6i·θ)) over all sites
    # |psi6| → 1 in clock-ordered phase

def compute_angle_histogram(theta: np.ndarray, n_bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    # Returns (bin_centers, counts) over [−π, π)
    # Expected: 6 peaks at multiples of π/3 in ordered phase
```

## Regime labeling
Use this classification to label sweep_delta tokens:
- T > T₂: "disordered"
- T₁ < T < T₂: "QLRO"
- T < T₁: "clock_ordered"
Where T₂ and T₁ are user-supplied parameters (or "unknown" if not supplied).

## Validation gate
tests/test_clock_metrics.py must pass:
- Perfectly ordered θ=0 field → |ψ₆|=1.0
- Random field → |ψ₆| < 0.1 (with high probability)
- Six-state config (θ = k·π/3 randomly assigned) → |ψ₆| > 0.9
- Υ(T=0.1) > Υ(T=1.5) on a converged XY config

## Do NOT
- Conflate QLRO with clock_ordered in any output label or comment
- Use vortex positions for ψ₆ (use spin angles directly)