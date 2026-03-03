"""
tests/test_r_max_sensitivity.py
=================================
SPEC_VALIDATION §6 — r_max sensitivity sweep.

Re-runs pairing with r_max ∈ {L/8, L/4, L/2} at 5 representative T values
for L=16, and asserts f_paired monotonicity:
    f_paired(r_max=L/8) ≤ f_paired(r_max=L/4) ≤ f_paired(r_max=L/2)

Uses the real pipeline:
  - topostream.simulate.xy_numba.run_xy
  - topostream.extract.vortices.extract_vortices
  - topostream.extract.pairing.pair_vortices

Deterministic: all seeds fixed.

Run with:
    python -m pytest tests/test_r_max_sensitivity.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from topostream.simulate.xy_numba import run_xy
from topostream.extract.vortices import extract_vortices
from topostream.extract.pairing import pair_vortices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prov(L: int, T: float, seed: int = 42, sweep_index: int = 0) -> dict:
    return {
        "model": "XY",
        "L": L,
        "T": T,
        "seed": seed,
        "sweep_index": sweep_index,
        "schema_version": "1.0.0",
    }


def _extract_and_pair(theta: np.ndarray, L: int, T: float, r_max: float) -> float:
    """Extract vortices, run pairing at given r_max, return f_paired."""
    prov = _prov(L=L, T=T)
    tokens = extract_vortices(theta, prov)
    vortices = [t for t in tokens if t["vortex"]["charge"] == +1]
    antivortices = [t for t in tokens if t["vortex"]["charge"] == -1]
    if not vortices or not antivortices:
        return 0.0
    result = pair_vortices(vortices, antivortices, L, r_max=r_max, provenance=prov)
    return result["f_paired"]


# ---------------------------------------------------------------------------
# §6 r_max sensitivity — monotonicity
# ---------------------------------------------------------------------------

class TestRmaxSensitivitySweep:
    """f_paired(r_max=L/8) ≤ f_paired(r_max=L/4) ≤ f_paired(r_max=L/2)
    tested across 5 representative temperatures on real MC configs.
    """

    L = 16
    T_VALUES = [0.5, 0.7, 0.9, 1.1, 1.3]

    @pytest.fixture(scope="class")
    def configs(self):
        """Generate one MC config per temperature."""
        cfgs = {}
        for T in self.T_VALUES:
            result = run_xy(
                L=self.L, T=T,
                N_equil=200, N_meas=100, N_thin=50,
                seed=42,
            )
            cfgs[T] = result["configs"][-1]
        return cfgs

    @pytest.mark.parametrize("T", T_VALUES)
    def test_monotonicity(self, configs, T):
        """f_paired must be non-decreasing as r_max grows."""
        theta = configs[T]
        rmax_vals = [self.L / 8, self.L / 4, self.L / 2]
        f_vals = [_extract_and_pair(theta, self.L, T, rm) for rm in rmax_vals]

        assert f_vals[0] <= f_vals[1] + 1e-12, (
            f"T={T}: f_paired(L/8)={f_vals[0]:.3f} > f_paired(L/4)={f_vals[1]:.3f}"
        )
        assert f_vals[1] <= f_vals[2] + 1e-12, (
            f"T={T}: f_paired(L/4)={f_vals[1]:.3f} > f_paired(L/2)={f_vals[2]:.3f}"
        )

    def test_deterministic(self, configs):
        """Same config → same f_paired for repeated calls."""
        T = self.T_VALUES[2]  # pick T=0.9
        theta = configs[T]
        f1 = _extract_and_pair(theta, self.L, T, self.L / 4)
        f2 = _extract_and_pair(theta, self.L, T, self.L / 4)
        assert f1 == f2


# ---------------------------------------------------------------------------
# Random-field sensitivity (no MC needed)
# ---------------------------------------------------------------------------

class TestRmaxSensitivityRandom:
    """r_max monotonicity on a fully random field (hot plasma)."""

    def test_random_field_monotonicity(self):
        L = 16
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, size=(L, L))
        rmax_vals = [L / 8, L / 4, L / 2]
        f_vals = [_extract_and_pair(theta, L, T=5.0, r_max=rm) for rm in rmax_vals]
        assert f_vals[0] <= f_vals[1] + 1e-12
        assert f_vals[1] <= f_vals[2] + 1e-12
