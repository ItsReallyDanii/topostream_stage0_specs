"""
tests/test_metrics_clock6_order.py
=====================================
Gate tests for compute_clock6_order() in src/topostream/metrics/clock.py.

Tests:
  1. Perfect single-state config  → clock6_order == 1.0
  2. Uniform mixture of all 6     → clock6_order ≈ 1/6 (± tolerance)
  3. Random clock6 field (hot)    → clock6_order near 1/6
  4. Determinism (same input → same output)
  5. Shape-validation error
  6. Integration with run_clock6 — ordering decreases as T rises
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from topostream.metrics.clock import compute_clock6_order

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_K = [k * math.pi / 3.0 for k in range(6)]
# Canonical clock angles wrapped into (−π, π] via arctan2
_ALLOWED = np.array(
    [math.atan2(math.sin(a), math.cos(a)) for a in _ALLOWED_K],
    dtype=np.float64,
)

L = 16
N = L * L


def _single_state_config(state_idx: int, L: int = L) -> np.ndarray:
    """All spins in clock state `state_idx`."""
    return np.full((L, L), _ALLOWED[state_idx], dtype=np.float64)


def _uniform_mixture_config(L: int = L, seed: int = 0) -> np.ndarray:
    """Exactly one spin per (state × repetition), as uniformly as possible."""
    rng = np.random.default_rng(seed)
    base = np.tile(_ALLOWED, N // 6 + 1)[:N]
    rng.shuffle(base)
    return base.reshape(L, L)


def _random_clock6_config(L: int = L, seed: int = 42) -> np.ndarray:
    """Random clock6 config (each site i.i.d. uniform over the 6 states)."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 6, size=(L, L))
    return _ALLOWED[indices]


# ---------------------------------------------------------------------------
# 1. Perfect single-state configuration → clock6_order == 1.0
# ---------------------------------------------------------------------------

class TestPerfectOrder:
    @pytest.mark.parametrize("state_idx", range(6))
    def test_single_state_returns_one(self, state_idx: int):
        cfg = _single_state_config(state_idx)
        result = compute_clock6_order(cfg)
        assert result == pytest.approx(1.0, abs=1e-12), (
            f"state {state_idx}: expected 1.0, got {result}"
        )


# ---------------------------------------------------------------------------
# 2. Uniform mixture of all 6 states → clock6_order ≈ 1/6
# ---------------------------------------------------------------------------

class TestUniformMixture:
    def test_uniform_mixture_near_one_sixth(self):
        cfg = _uniform_mixture_config(seed=7)
        result = compute_clock6_order(cfg)
        expected = 1.0 / 6.0
        # Allow ±2/N tolerance for discrete binning rounding
        tol = 2.0 / N
        assert abs(result - expected) <= tol, (
            f"uniform mixture: clock6_order={result:.6f}, "
            f"expected ≈ {expected:.6f} ± {tol:.6f}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 5, 99])
    def test_multiple_seeds_near_one_sixth(self, seed: int):
        cfg = _uniform_mixture_config(seed=seed)
        result = compute_clock6_order(cfg)
        tol = 3.0 / N  # generous tolerance
        assert abs(result - 1.0 / 6.0) <= tol, (
            f"seed={seed}: clock6_order={result:.6f} too far from 1/6"
        )


# ---------------------------------------------------------------------------
# 3. Random i.i.d. clock6 field → near 1/6 with tolerance
# ---------------------------------------------------------------------------

class TestRandomClock6Field:
    def test_random_field_near_one_sixth(self):
        cfg = _random_clock6_config(L=32, seed=42)
        result = compute_clock6_order(cfg)
        # For L=32 (1024 spins) and true uniform, stddev ≈ sqrt(1/6·5/6 / 1024) ≈ 0.011
        # Allow 5σ ≈ 0.055 tolerance
        assert abs(result - 1.0 / 6.0) < 0.06, (
            f"random clock6 field: clock6_order={result:.6f} is too far from 1/6"
        )

    def test_result_in_valid_range(self):
        for seed in range(5):
            cfg = _random_clock6_config(seed=seed)
            result = compute_clock6_order(cfg)
            assert 0.0 <= result <= 1.0, (
                f"seed={seed}: clock6_order={result} outside [0, 1]"
            )


# ---------------------------------------------------------------------------
# 4. Determinism — same input produces same output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_config_same_result(self):
        cfg = _random_clock6_config(seed=42)
        r1 = compute_clock6_order(cfg)
        r2 = compute_clock6_order(cfg)
        assert r1 == r2, f"Determinism failure: {r1} vs {r2}"

    def test_copy_gives_same_result(self):
        cfg = _random_clock6_config(seed=1)
        r1 = compute_clock6_order(cfg)
        r2 = compute_clock6_order(cfg.copy())
        assert r1 == r2


# ---------------------------------------------------------------------------
# 5. Shape validation
# ---------------------------------------------------------------------------

class TestShapeValidation:
    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_clock6_order(np.zeros((16,)))

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            compute_clock6_order(np.zeros((8, 16)))


# ---------------------------------------------------------------------------
# 6. Integration: clock6_order falls as T rises (using run_clock6)
# ---------------------------------------------------------------------------

class TestTemperatureOrdering:
    """clock6_order should be higher at low T (frozen) than at high T (diffuse).

    Uses the energy-variance insight: at very low T the simulator freezes into
    one state (or one of a few), raising max_k p_k well above 1/6.
    At high T the system shuffles all 6 states roughly equally.
    """

    def test_low_T_more_concentrated_than_high_T(self):
        import logging
        logging.disable(logging.CRITICAL)

        from topostream.simulate.clock6_numba import run_clock6

        # Low T: system freezes → most spins in one or two states
        r_cold = run_clock6(L=16, T=0.3, N_equil=200, N_meas=100, N_thin=50,
                            seed=42)
        # High T: uniform mixing among 6 states
        r_hot = run_clock6(L=16, T=2.5, N_equil=200, N_meas=100, N_thin=50,
                           seed=42)

        # Average over measurement configs
        order_cold = float(np.mean([
            compute_clock6_order(cfg) for cfg in r_cold["configs"]
        ]))
        order_hot = float(np.mean([
            compute_clock6_order(cfg) for cfg in r_hot["configs"]
        ]))

        assert order_cold > order_hot, (
            f"clock6_order(T=0.3)={order_cold:.4f} should exceed "
            f"clock6_order(T=2.5)={order_hot:.4f}"
        )

    def test_high_T_near_uniform(self):
        """At T=2.5 the system mixes freely → clock6_order ≈ 1/6."""
        import logging
        logging.disable(logging.CRITICAL)

        from topostream.simulate.clock6_numba import run_clock6

        r_hot = run_clock6(L=16, T=2.5, N_equil=500, N_meas=500, N_thin=10,
                           seed=42)
        orders = [compute_clock6_order(cfg) for cfg in r_hot["configs"]]
        mean_order = float(np.mean(orders))

        # Should be close to uniform (1/6 ≈ 0.167), certainly below 0.35
        assert mean_order < 0.35, (
            f"High-T clock6_order={mean_order:.4f} should be near 1/6 (< 0.35)"
        )
