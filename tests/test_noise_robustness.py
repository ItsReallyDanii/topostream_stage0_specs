"""
tests/test_noise_robustness.py
================================
SPEC_VALIDATION §4.1 — Spin perturbation noise robustness tests.

Takes a converged low-T config, adds Gaussian noise at σ ∈ {0.05, 0.10, 0.20},
re-extracts vortices, and asserts count stability thresholds from the spec:
  - σ = 0.05 → vortex count change < 10%
  - σ = 0.10 → vortex count change < 30%

Uses the real pipeline:
  - topostream.simulate.xy_numba.run_xy
  - topostream.extract.vortices.extract_vortices

Deterministic: all seeds fixed.

Run with:
    python -m pytest tests/test_noise_robustness.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from topostream.simulate.xy_numba import run_xy
from topostream.extract.vortices import extract_vortices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prov(L: int = 16, T: float = 0.5, seed: int = 42, sweep_index: int = 0) -> dict:
    return {
        "model": "XY",
        "L": L,
        "T": T,
        "seed": seed,
        "sweep_index": sweep_index,
        "schema_version": "1.0.0",
    }


def _get_cold_config(L: int = 16, seed: int = 42) -> np.ndarray:
    """Run a short MC at low T and return a converged config."""
    result = run_xy(L=L, T=0.3, N_equil=300, N_meas=100, N_thin=50, seed=seed)
    return result["configs"][-1]


def _count_vortices(theta: np.ndarray, L: int, T: float = 0.3) -> int:
    tokens = extract_vortices(theta, _prov(L=L, T=T))
    return len(tokens)


def _add_noise(theta: np.ndarray, sigma: float, seed: int = 99) -> np.ndarray:
    """Add Gaussian noise σ to spin angles (wrap back to [-π,π))."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=theta.shape)
    noisy = theta + noise
    # Wrap to [-π, π) using normative wrap
    return np.arctan2(np.sin(noisy), np.cos(noisy))


# ---------------------------------------------------------------------------
# §4.1 Spin perturbation
# ---------------------------------------------------------------------------

class TestSpinPerturbation:
    """SPEC_VALIDATION §4.1: add noise to converged config, re-extract."""

    L = 16

    @pytest.fixture(scope="class")
    def cold_config(self):
        return _get_cold_config(self.L, seed=42)

    @pytest.fixture(scope="class")
    def baseline_count(self, cold_config):
        return _count_vortices(cold_config, self.L)

    def test_sigma_005_count_change_lt_10pct(self, cold_config, baseline_count):
        """σ=0.05: vortex count change < 10%."""
        noisy = _add_noise(cold_config, sigma=0.05, seed=100)
        noisy_count = _count_vortices(noisy, self.L)
        if baseline_count == 0:
            # Cold config has 0 vortices; noisy should still have very few
            assert noisy_count <= 3, (
                f"σ=0.05: from 0 baseline, got {noisy_count} vortices"
            )
        else:
            pct_change = abs(noisy_count - baseline_count) / baseline_count
            assert pct_change < 0.10, (
                f"σ=0.05: count changed {pct_change:.1%} "
                f"(baseline={baseline_count}, noisy={noisy_count})"
            )

    def test_sigma_010_count_change_lt_30pct(self, cold_config, baseline_count):
        """σ=0.10: vortex count change < 30%."""
        noisy = _add_noise(cold_config, sigma=0.10, seed=101)
        noisy_count = _count_vortices(noisy, self.L)
        if baseline_count == 0:
            assert noisy_count <= 10, (
                f"σ=0.10: from 0 baseline, got {noisy_count} vortices"
            )
        else:
            pct_change = abs(noisy_count - baseline_count) / baseline_count
            assert pct_change < 0.30, (
                f"σ=0.10: count changed {pct_change:.1%} "
                f"(baseline={baseline_count}, noisy={noisy_count})"
            )

    def test_sigma_020_still_finite(self, cold_config):
        """σ=0.20: vortex count may grow but must remain finite and plausible."""
        noisy = _add_noise(cold_config, sigma=0.20, seed=102)
        noisy_count = _count_vortices(noisy, self.L)
        max_plaquettes = (self.L) * (self.L)
        assert 0 <= noisy_count <= max_plaquettes, (
            f"σ=0.20: implausible count {noisy_count}"
        )

    def test_noise_is_deterministic(self, cold_config):
        """Same noise seed → same result."""
        n1 = _add_noise(cold_config, 0.05, seed=200)
        n2 = _add_noise(cold_config, 0.05, seed=200)
        np.testing.assert_array_equal(n1, n2)
