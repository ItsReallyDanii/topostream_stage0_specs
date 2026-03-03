"""
tests/test_metrics_helicity.py
================================
Gate tests for src/topostream/metrics/helicity.py.

Covers:
  - Υ returns finite values on uniform field.
  - Υ(low T) > Υ(high T) on physically motivated configs.
  - Determinism (same input → same output).
  - Ensemble + jackknife variant.

Run with:
    python -m pytest tests/test_metrics_helicity.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from topostream.metrics.helicity import compute_helicity, compute_helicity_ensemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_field(L: int = 32, angle: float = 0.0) -> np.ndarray:
    """All spins aligned at the same angle."""
    return np.full((L, L), angle, dtype=np.float64)


def _random_field(L: int = 32, seed: int = 42) -> np.ndarray:
    """iid Uniform[−π, π)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-math.pi, math.pi, size=(L, L))


def _slightly_disordered(L: int = 32, sigma: float = 0.1, seed: int = 42) -> np.ndarray:
    """Nearly ordered: θ ≈ 0 + small Gaussian noise."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(L, L))


# ---------------------------------------------------------------------------
# Finite values / no crash
# ---------------------------------------------------------------------------

class TestHelicityBasic:
    def test_returns_tuple(self):
        theta = _uniform_field(16)
        result = compute_helicity(theta, T=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_finite_on_uniform(self):
        theta = _uniform_field(16)
        ups, err = compute_helicity(theta, T=1.0)
        assert math.isfinite(ups)
        assert math.isfinite(err)

    def test_finite_on_random(self):
        theta = _random_field(16)
        ups, err = compute_helicity(theta, T=1.0)
        assert math.isfinite(ups)
        assert math.isfinite(err)

    def test_invalid_T_raises(self):
        theta = _uniform_field(16)
        with pytest.raises(ValueError, match="T must be"):
            compute_helicity(theta, T=0.0)
        with pytest.raises(ValueError, match="T must be"):
            compute_helicity(theta, T=-1.0)

    def test_non_square_raises(self):
        theta = np.zeros((16, 32))
        with pytest.raises(ValueError):
            compute_helicity(theta, T=1.0)


# ---------------------------------------------------------------------------
# Physical sanity: ordered field → high Υ
# ---------------------------------------------------------------------------

class TestHelicityPhysics:
    def test_uniform_field_high_upsilon(self):
        """A perfectly aligned field has cos(Δθ)=1 everywhere (max stiffness).
        Υ should be near J (=1.0) for any T, since the sin² term ≈ 0."""
        L = 32
        theta = _uniform_field(L, angle=0.0)
        ups, _ = compute_helicity(theta, T=1.0, J=1.0)
        # For uniform: Σcos = L², Σsin = 0 → Υ = (1/L²)×L² = 1.0
        assert ups == pytest.approx(1.0, abs=1e-10)

    def test_uniform_nonzero_angle_same(self):
        """Constant field at angle=π/4 → same Υ as angle=0 (all Δθ=0)."""
        L = 32
        ups0, _ = compute_helicity(_uniform_field(L, 0.0), T=1.0)
        ups_pi4, _ = compute_helicity(_uniform_field(L, math.pi / 4), T=1.0)
        assert ups0 == pytest.approx(ups_pi4, abs=1e-10)

    def test_low_T_higher_than_high_T(self):
        """Near-ordered config: Υ at low T should be larger because the
        −(1/T)sin² penalty is smaller when near-aligned AND the (1/T)
        prefactor makes it less severe at high T—wait, actually:

        For a near-ordered config, Σsin ≈ 0 and Σcos ≈ L², so Υ ≈ J
        regardless of T.  The T-dependence only matters for thermal
        fluctuations that produce nonzero Σsin.

        Instead, use a weakly disordered config where sin² > 0.
        At low T, the (1/T)sin² penalty is LARGER, but Σcos is also
        high.   Actually Υ = (1/L²)[Σcos − (1/T)(Σsin)²].
        For fixed config:  Υ(T_low) < Υ(T_high) if sin² > 0.

        The physical statement is: *ensemble-averaged* Υ decreases with T.
        For a single fixed config the T-dependence is only through (1/T).

        We test with a slightly noisy config to get nontrivial sin²:
        Υ(T=5.0) > Υ(T=0.1) for a fixed config (since −(1/T)sin² is
        less punishing at high T).

        But the HANDOFF says: Υ(T=0.1) > Υ(T=1.5) on a converged XY config.
        A 'converged XY config' at T=0.1 is nearly ordered → Σsin ≈ 0,
        and at T=1.5 it is disordered → Σcos ≈ 0.  So we need TWO
        different configs (one per temperature).
        """
        L = 32
        # Config 1: nearly ordered (T=0.1 equilibrated)
        theta_cold = _slightly_disordered(L, sigma=0.05, seed=42)
        # Config 2: fully disordered (T=1.5 equilibrated)
        theta_hot = _random_field(L, seed=43)

        ups_cold, _ = compute_helicity(theta_cold, T=0.1)
        ups_hot, _ = compute_helicity(theta_hot, T=1.5)

        assert ups_cold > ups_hot, (
            f"Υ(cold)={ups_cold:.4f} should exceed Υ(hot)={ups_hot:.4f}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestHelicityDeterminism:
    def test_same_input_same_output(self):
        theta = _random_field(16, seed=99)
        r1 = compute_helicity(theta, T=1.0)
        r2 = compute_helicity(theta, T=1.0)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]


# ---------------------------------------------------------------------------
# Ensemble / jackknife
# ---------------------------------------------------------------------------

class TestHelicityEnsemble:
    def test_single_config_zero_err(self):
        theta = _uniform_field(16)
        mean_val, err = compute_helicity_ensemble([theta], T=1.0)
        assert err == pytest.approx(0.0)

    def test_multiple_configs_finite_err(self):
        configs = [_slightly_disordered(16, sigma=0.1, seed=s) for s in range(4)]
        mean_val, err = compute_helicity_ensemble(configs, T=1.0)
        assert math.isfinite(mean_val)
        assert math.isfinite(err)
        assert err >= 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_helicity_ensemble([], T=1.0)
