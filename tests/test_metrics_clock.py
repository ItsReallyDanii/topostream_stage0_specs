"""
tests/test_metrics_clock.py
==============================
Gate tests for src/topostream/metrics/clock.py.

Handoff gate (agents/04_clock_metrics.md):
  - Perfectly ordered θ=0 field → |ψ₆| = 1.0
  - Random field → |ψ₆| < 0.1  (with high probability)
  - Six-state config (θ = k·π/3 randomly assigned) → |ψ₆| > 0.9
  - Angle histogram: 6 peaks for ideal six-state configuration.

Additional:
  - Determinism.
  - Regime labeling.
  - Input validation.

Run with:
    python -m pytest tests/test_metrics_clock.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from topostream.metrics.clock import (
    compute_psi6,
    compute_angle_histogram,
    label_regime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_field(L: int = 32, angle: float = 0.0) -> np.ndarray:
    return np.full((L, L), angle, dtype=np.float64)


def _random_field(L: int = 32, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-math.pi, math.pi, size=(L, L))


def _six_state_field(L: int = 32, seed: int = 42) -> np.ndarray:
    """Each site randomly assigned one of the 6 clock states k·π/3,
    wrapped into [−π, π) so histogram bins align correctly."""
    rng = np.random.default_rng(seed)
    allowed = np.array([k * math.pi / 3 for k in range(6)])
    # Wrap into [−π, π) using normative wrap (SPEC_FORMULAE §1)
    allowed = np.arctan2(np.sin(allowed), np.cos(allowed))
    choices = rng.integers(0, 6, size=(L, L))
    return allowed[choices]


# ===========================================================================
# ψ₆ tests
# ===========================================================================

class TestPsi6:
    """Gate: ψ₆ magnitude checks."""

    def test_uniform_zero_psi6_is_one(self):
        """θ = 0 everywhere → exp(6i·0)=1 → ψ₆ = 1.0."""
        psi = compute_psi6(_uniform_field(32, 0.0))
        assert abs(psi) == pytest.approx(1.0, abs=1e-12)

    def test_uniform_pi_over_3_psi6_is_one(self):
        """θ = π/3 everywhere → exp(6i·π/3) = exp(2πi) = 1 → |ψ₆| = 1."""
        psi = compute_psi6(_uniform_field(32, math.pi / 3))
        assert abs(psi) == pytest.approx(1.0, abs=1e-10)

    def test_random_field_low_psi6(self):
        """Random iid θ → |ψ₆| ≈ 0  (< 0.1 with high probability for L=32)."""
        psi = compute_psi6(_random_field(32, seed=42))
        assert abs(psi) < 0.1, f"|ψ₆| = {abs(psi):.4f}, expected < 0.1"

    def test_six_state_high_psi6(self):
        """All sites from {0, π/3, 2π/3, π, 4π/3, 5π/3} → |ψ₆| > 0.9.

        exp(6i·k·π/3) = exp(2πi·k) = 1  for all k, so ψ₆ = 1 exactly.
        """
        psi = compute_psi6(_six_state_field(32, seed=42))
        assert abs(psi) > 0.9, f"|ψ₆| = {abs(psi):.4f}, expected > 0.9"

    def test_six_state_exact_one(self):
        """More precisely, 6-state field should give |ψ₆| ≈ 1.0 exactly."""
        psi = compute_psi6(_six_state_field(32, seed=42))
        assert abs(psi) == pytest.approx(1.0, abs=1e-10)

    def test_returns_complex(self):
        psi = compute_psi6(_uniform_field(16))
        assert isinstance(psi, complex)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            compute_psi6(np.zeros((16, 32)))


# ===========================================================================
# Angle histogram tests
# ===========================================================================

class TestAngleHistogram:
    """Gate: histogram shape and 6-peak structure."""

    def test_returns_correct_shapes(self):
        centers, counts = compute_angle_histogram(_uniform_field(16), n_bins=36)
        assert centers.shape == (36,)
        assert counts.shape == (36,)

    def test_total_counts_equals_sites(self):
        L = 16
        _, counts = compute_angle_histogram(_uniform_field(L), n_bins=36)
        assert counts.sum() == L * L

    def test_uniform_one_peak(self):
        """Uniform θ=0 → all counts in a single bin."""
        _, counts = compute_angle_histogram(_uniform_field(32, 0.0), n_bins=36)
        assert counts.max() == 32 * 32
        assert np.count_nonzero(counts) == 1

    def test_six_state_six_peaks(self):
        """Six-state config should produce exactly 6 non-zero histogram bins
        (one per preferred angle at k·π/3)."""
        theta = _six_state_field(64, seed=42)
        centers, counts = compute_angle_histogram(theta, n_bins=36)
        # Each clock angle k·π/3 hits exactly one bin.
        # With 36 bins of width 2π/36 = π/18, the 6 states at 0, π/3, 2π/3,
        # π, 4π/3, 5π/3 each fall into a unique bin.
        nonzero_bins = np.count_nonzero(counts)
        assert nonzero_bins == 6, (
            f"Expected 6 non-zero bins, got {nonzero_bins}"
        )

    def test_random_field_broadly_distributed(self):
        """Random field should fill many bins (broad distribution)."""
        _, counts = compute_angle_histogram(_random_field(32), n_bins=36)
        # At least 30 of 36 bins should be non-empty for 1024 random points
        assert np.count_nonzero(counts) >= 30

    def test_custom_n_bins(self):
        centers, counts = compute_angle_histogram(_uniform_field(16), n_bins=72)
        assert centers.shape == (72,)
        assert counts.shape == (72,)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            compute_angle_histogram(np.zeros((16, 32)))


# ===========================================================================
# Regime labeling
# ===========================================================================

class TestRegimeLabeling:
    """Three q=6 regimes: disordered, QLRO, clock_ordered (never conflated)."""

    def test_disordered(self):
        assert label_regime(T=2.0, T1=0.7, T2=0.9) == "disordered"

    def test_qlro(self):
        assert label_regime(T=0.8, T1=0.7, T2=0.9) == "QLRO"

    def test_clock_ordered(self):
        assert label_regime(T=0.5, T1=0.7, T2=0.9) == "clock_ordered"

    def test_unknown_when_thresholds_missing(self):
        assert label_regime(T=0.5) == "unknown"
        assert label_regime(T=0.5, T1=0.7) == "unknown"
        assert label_regime(T=0.5, T2=0.9) == "unknown"

    def test_boundary_T2(self):
        """At T = T₂ exactly: T > T₂ is false, T > T₁ is true → QLRO."""
        assert label_regime(T=0.9, T1=0.7, T2=0.9) == "QLRO"

    def test_boundary_T1(self):
        """At T = T₁ exactly: T > T₁ is false → clock_ordered."""
        assert label_regime(T=0.7, T1=0.7, T2=0.9) == "clock_ordered"


# ===========================================================================
# Determinism
# ===========================================================================

class TestClockDeterminism:
    def test_psi6_deterministic(self):
        theta = _random_field(16, seed=77)
        p1 = compute_psi6(theta)
        p2 = compute_psi6(theta)
        assert p1 == p2

    def test_histogram_deterministic(self):
        theta = _random_field(16, seed=77)
        c1, h1 = compute_angle_histogram(theta)
        c2, h2 = compute_angle_histogram(theta)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(h1, h2)
