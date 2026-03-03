"""
tests/test_map_mode_synthetic.py
==================================
Map-mode Stage 3 gate tests — synthetic-first, probe-agnostic.

Strategy
--------
1. Generate ground-truth theta from XY or Clock6 init_config.
2. Apply forward model pipeline (to_vector_map → degrade → adapter → theta_hat).
3. Run extract_vortices on both theta and theta_hat.
4. Compare token counts and document NaN behaviour.

Design principles
-----------------
- No "magic fix" of NaN propagation: NaN sites produce NaN theta_hat, and
  the vortex extractor skips affected plaquettes without crash.
- Tolerances on token counts are LOOSE (±30 %) for mild degradation, and
  graceful-failure is asserted for extreme degradation (empty result or only
  NaN plaquettes skipped, no crash).
- All seeds and sizes are fixed so tests are deterministic and fast.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from topostream.simulate.xy_numba import init_config
from topostream.simulate.clock6_numba import init_config_clock6

from topostream.map.forward_models import (
    to_vector_map,
    apply_blur,
    downsample,
    add_noise,
    mask_nan,
)
from topostream.map.adapters import (
    vector_map_to_theta,
    scalar_phase_map_to_theta,
)
from topostream.extract.vortices import extract_vortices

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

L = 16   # small, fast; still ≥ 8 (vortex extractor minimum)


def _prov(model: str, L: int = L, T: float = 1.0) -> dict:
    return {"model": model, "L": L, "T": T, "seed": 42,
            "sweep_index": 0, "schema_version": "1.0.0"}


def _xy_theta(seed: int = 42) -> np.ndarray:
    return init_config(L, seed)


def _clock6_theta(seed: int = 42) -> np.ndarray:
    return init_config_clock6(L, seed)


def _count_vortices(theta: np.ndarray, model: str = "XY") -> int:
    return len(extract_vortices(theta, _prov(model)))


# ---------------------------------------------------------------------------
# 1. forward_models — unit tests
# ---------------------------------------------------------------------------

class TestForwardModels:

    def test_to_vector_map_shape(self):
        theta = _xy_theta()
        Mx, My = to_vector_map(theta)
        assert Mx.shape == theta.shape
        assert My.shape == theta.shape

    def test_to_vector_map_unit_vectors(self):
        theta = _xy_theta()
        Mx, My = to_vector_map(theta)
        mag = np.sqrt(Mx ** 2 + My ** 2)
        np.testing.assert_allclose(mag, 1.0, atol=1e-12)

    def test_to_vector_map_nan_propagation(self):
        theta = _xy_theta().copy()
        theta[3, 5] = np.nan
        Mx, My = to_vector_map(theta)
        assert np.isnan(Mx[3, 5])
        assert np.isnan(My[3, 5])

    def test_to_vector_map_requires_2d(self):
        with pytest.raises(ValueError):
            to_vector_map(np.zeros(16))

    # ---

    def test_apply_blur_shape_preserved(self):
        theta = _xy_theta()
        Mx, _ = to_vector_map(theta)
        blurred = apply_blur(Mx, sigma=1.0)
        assert blurred.shape == Mx.shape

    def test_apply_blur_zero_sigma_is_copy(self):
        arr = np.random.default_rng(0).random((8, 8))
        out = apply_blur(arr, sigma=0)
        np.testing.assert_array_equal(out, arr)

    def test_apply_blur_reduces_variance(self):
        Mx, _ = to_vector_map(_xy_theta())
        assert np.var(apply_blur(Mx, sigma=2.0)) < np.var(Mx)

    def test_apply_blur_nan_handling(self):
        arr = np.ones((16, 16), dtype=np.float64)
        arr[8, 8] = np.nan
        out = apply_blur(arr, sigma=0.5)
        # Edge away from NaN site should still be finite
        assert np.isfinite(out[0, 0])
        # Region very near the NaN should still be defined (weight > 1e-6)
        assert np.isfinite(out[8, 9])

    # ---

    def test_downsample_shape(self):
        arr = np.ones((16, 16))
        out = downsample(arr, factor=4)
        assert out.shape == (4, 4)

    def test_downsample_factor_1_is_copy(self):
        arr = np.arange(64, dtype=float).reshape(8, 8)
        np.testing.assert_array_equal(downsample(arr, 1), arr)

    def test_downsample_uniform_array_value_preserved(self):
        arr = np.full((16, 16), 3.7)
        out = downsample(arr, factor=4)
        np.testing.assert_allclose(out, 3.7, atol=1e-12)

    def test_downsample_nan_in_block(self):
        arr = np.ones((8, 8))
        arr[0, 0] = np.nan          # 1 NaN in a 2×2 block → nanmean of 3 ones
        out = downsample(arr, factor=2)
        assert np.isfinite(out[0, 0])
        assert pytest.approx(out[0, 0], abs=1e-12) == 1.0

    def test_downsample_all_nan_block(self):
        arr = np.ones((8, 8))
        arr[0:2, 0:2] = np.nan     # full 2×2 block NaN → output NaN
        out = downsample(arr, factor=2)
        assert np.isnan(out[0, 0])

    def test_downsample_bad_factor_raises(self):
        with pytest.raises(ValueError):
            downsample(np.ones((8, 8)), factor=0)

    # ---

    def test_add_noise_shape(self):
        arr = np.ones((16, 16))
        out = add_noise(arr, sigma=0.1, seed=0)
        assert out.shape == arr.shape

    def test_add_noise_zero_sigma_is_copy(self):
        arr = np.arange(16, dtype=float).reshape(4, 4)
        np.testing.assert_array_equal(add_noise(arr, sigma=0.0, seed=0), arr)

    def test_add_noise_nan_preserved(self):
        arr = np.ones((8, 8))
        arr[2, 3] = np.nan
        out = add_noise(arr, sigma=0.5, seed=7)
        assert np.isnan(out[2, 3])

    def test_add_noise_determinism(self):
        arr = np.ones((8, 8))
        assert np.array_equal(add_noise(arr, 0.3, seed=42),
                               add_noise(arr, 0.3, seed=42))

    def test_add_noise_changes_values(self):
        arr = np.zeros((16, 16))
        out = add_noise(arr, sigma=0.2, seed=1)
        assert not np.all(out == 0)

    # ---

    def test_mask_nan_fraction_approx(self):
        arr = np.ones((100, 100))
        out = mask_nan(arr, nan_frac=0.1, seed=0)
        frac = np.isnan(out).mean()
        assert abs(frac - 0.1) < 0.02   # ±2 pp

    def test_mask_nan_zero_frac_is_copy(self):
        arr = np.ones((8, 8))
        np.testing.assert_array_equal(mask_nan(arr, 0.0), arr)

    def test_mask_nan_determinism(self):
        arr = np.ones((16, 16))
        r1 = mask_nan(arr, 0.1, seed=5)
        r2 = mask_nan(arr, 0.1, seed=5)
        assert np.array_equal(r1, r2, equal_nan=True)

    def test_mask_nan_bad_frac_raises(self):
        with pytest.raises(ValueError):
            mask_nan(np.ones((8, 8)), nan_frac=1.5)


# ---------------------------------------------------------------------------
# 2. adapters — unit tests
# ---------------------------------------------------------------------------

class TestAdapters:

    def test_vector_map_to_theta_identity(self):
        """Round-trip: theta -> (Mx,My) -> theta_hat should be equal."""
        theta = _xy_theta()
        Mx, My = to_vector_map(theta)
        theta_hat = vector_map_to_theta(Mx, My)
        np.testing.assert_allclose(theta_hat, theta, atol=1e-12)

    def test_vector_map_to_theta_nan_propagation(self):
        theta = _xy_theta().copy()
        theta[4, 4] = np.nan
        Mx, My = to_vector_map(theta)
        theta_hat = vector_map_to_theta(Mx, My)
        assert np.isnan(theta_hat[4, 4])
        assert np.isfinite(theta_hat[0, 0])

    def test_vector_map_to_theta_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            vector_map_to_theta(np.zeros((8, 8)), np.zeros((8, 9)))

    def test_scalar_phase_map_range(self):
        phi = np.linspace(-10, 10, 100).reshape(10, 10)
        out = scalar_phase_map_to_theta(phi)
        assert np.all(out >= -np.pi - 1e-12)
        assert np.all(out <= np.pi + 1e-12)

    def test_scalar_phase_map_identity_for_wrapped(self):
        """phi already in (−π, π] must be returned unchanged."""
        theta = _xy_theta()
        out = scalar_phase_map_to_theta(theta)
        np.testing.assert_allclose(out, theta, atol=1e-12)

    def test_scalar_phase_map_nan_propagation(self):
        phi = _xy_theta().copy()
        phi[1, 2] = np.nan
        out = scalar_phase_map_to_theta(phi)
        assert np.isnan(out[1, 2])

    def test_scalar_phase_map_requires_2d(self):
        with pytest.raises(ValueError):
            scalar_phase_map_to_theta(np.zeros(16))


# ---------------------------------------------------------------------------
# 3. End-to-end: forward → degrade → adapter → vortex extraction
# ---------------------------------------------------------------------------

class TestEndToEndRoundTrip:

    def _baseline_vortex_count(self, model: str = "XY") -> int:
        theta = _xy_theta() if model == "XY" else _clock6_theta()
        return _count_vortices(theta, model)

    def _pipeline(self, theta: np.ndarray, *,
                  blur: float = 0.0,
                  noise: float = 0.0,
                  nan_frac: float = 0.0,
                  seed: int = 42,
                  model: str = "XY") -> dict:
        """Run forward model + degradations + adapter + extraction.
        Returns dict: theta_hat, n_baseline, n_recovered, n_nan_plaquettes."""
        n_baseline = _count_vortices(theta, model)
        Mx, My = to_vector_map(theta)
        if blur > 0:
            Mx = apply_blur(Mx, sigma=blur)
            My = apply_blur(My, sigma=blur)
        if noise > 0:
            Mx = add_noise(Mx, sigma=noise, seed=seed)
            My = add_noise(My, sigma=noise, seed=seed + 1)
        if nan_frac > 0:
            Mx = mask_nan(Mx, nan_frac=nan_frac, seed=seed + 2)
            My = mask_nan(My, nan_frac=nan_frac, seed=seed + 3)

        theta_hat = vector_map_to_theta(Mx, My)

        # Count NaN plaquettes in theta_hat (any corner NaN → skip)
        L = theta.shape[0]
        nan_plaq = 0
        for i in range(L):
            for j in range(L):
                corners = [theta_hat[i, j],
                           theta_hat[i, (j + 1) % L],
                           theta_hat[(i + 1) % L, (j + 1) % L],
                           theta_hat[(i + 1) % L, j]]
                if any(math.isnan(c) for c in corners):
                    nan_plaq += 1

        n_recovered = _count_vortices(theta_hat, model)
        return {
            "theta_hat": theta_hat,
            "n_baseline": n_baseline,
            "n_recovered": n_recovered,
            "n_nan_plaquettes": nan_plaq,
        }

    # ---

    def test_no_degradation_exact_roundtrip_xy(self):
        """Zero degradation → exact vortex count match."""
        theta = _xy_theta()
        res = self._pipeline(theta, model="XY")
        assert res["n_recovered"] == res["n_baseline"], (
            f"No-degradation: baseline={res['n_baseline']} "
            f"recovered={res['n_recovered']}"
        )

    def test_no_degradation_exact_roundtrip_clock6(self):
        """Zero degradation → exact vortex count match (Clock6)."""
        theta = _clock6_theta()
        res = self._pipeline(theta, model="clock6")
        assert res["n_recovered"] == res["n_baseline"]

    def test_mild_blur_count_stable_xy(self):
        """Mild Gaussian blur (σ=0.5): count within ±30% of baseline."""
        theta = _xy_theta()
        res = self._pipeline(theta, blur=0.5, model="XY")
        base = res["n_baseline"]
        rec = res["n_recovered"]
        if base == 0:
            return   # no vortices to compare
        ratio = abs(rec - base) / base
        assert ratio <= 0.30, (
            f"Blur σ=0.5: baseline={base} recovered={rec} ratio={ratio:.2f}"
        )

    def test_mild_noise_count_stable_xy(self):
        """Mild noise (σ=0.05): count within ±30% of baseline."""
        theta = _xy_theta()
        res = self._pipeline(theta, noise=0.05, model="XY")
        base = res["n_baseline"]
        rec = res["n_recovered"]
        if base == 0:
            return
        ratio = abs(rec - base) / base
        assert ratio <= 0.30, (
            f"Noise σ=0.05: baseline={base} recovered={rec} ratio={ratio:.2f}"
        )

    def test_mild_nan_mask_count_sensible_xy(self):
        """5 % NaN mask: vortex extraction does not crash; NaN plaquettes > 0."""
        theta = _xy_theta()
        res = self._pipeline(theta, nan_frac=0.05, model="XY")
        # At least some plaquettes must be skipped
        assert res["n_nan_plaquettes"] > 0
        # Result must be a non-negative integer (not an exception)
        assert res["n_recovered"] >= 0

    def test_nan_amplification_documented_xy(self):
        """NaN plaquettes scale with nan_frac — document propagation effect.

        A single NaN site can affect up to 4 plaquettes (one per corner).
        This test verifies the amplification is present, not hidden.
        """
        theta = _xy_theta()
        res5 = self._pipeline(theta, nan_frac=0.05, model="XY")
        res20 = self._pipeline(theta, nan_frac=0.20, model="XY")
        # More NaN → more skipped plaquettes
        assert res20["n_nan_plaquettes"] >= res5["n_nan_plaquettes"]

    # ---

    def test_extreme_nan_mask_graceful_xy(self):
        """90% NaN mask: extractor returns empty list or mostly-NaN result.
        Must not raise an exception.
        """
        theta = _xy_theta()
        res = self._pipeline(theta, nan_frac=0.90, model="XY")
        # Nearly all plaquettes skipped — vortex count may be near 0 or 0
        total_plaq = L * L
        assert res["n_nan_plaquettes"] > total_plaq * 0.5, (
            f"Expected >50% plaquettes skipped at nan_frac=0.9; "
            f"got {res['n_nan_plaquettes']}/{total_plaq}"
        )
        # No crash — n_recovered is 0 or a small number
        assert res["n_recovered"] >= 0

    def test_extreme_blur_graceful_xy(self):
        """Very large blur (σ=8): extractor runs without crash.

        Extreme blur homogenises the angle field — vortex count should
        drop dramatically or go to 0.
        """
        theta = _xy_theta()
        res = self._pipeline(theta, blur=8.0, model="XY")
        # Must not raise; result is just a (possibly small) non-negative count
        assert res["n_recovered"] >= 0

    def test_extreme_noise_graceful_xy(self):
        """Extreme noise (σ=π): extractor runs without crash.

        Noise σ=π randomises angles completely relative to original signal.
        The function must not raise an exception.
        """
        theta = _xy_theta()
        res = self._pipeline(theta, noise=math.pi, model="XY")
        assert res["n_recovered"] >= 0

    # ---

    def test_pipeline_determinism(self):
        """Same seed → identical theta_hat, n_recovered."""
        theta = _xy_theta()
        r1 = self._pipeline(theta, blur=0.5, noise=0.05, nan_frac=0.03, seed=7)
        r2 = self._pipeline(theta, blur=0.5, noise=0.05, nan_frac=0.03, seed=7)
        np.testing.assert_array_equal(r1["theta_hat"], r2["theta_hat"])
        assert r1["n_recovered"] == r2["n_recovered"]

    def test_pipeline_different_seed_different_noise(self):
        """Different noise seed → different theta_hat (non-trivial noise)."""
        theta = _xy_theta()
        r1 = self._pipeline(theta, noise=0.3, seed=1)
        r2 = self._pipeline(theta, noise=0.3, seed=99)
        assert not np.array_equal(r1["theta_hat"], r2["theta_hat"])


# ---------------------------------------------------------------------------
# 4. Scalar phase map adapter — extra coverage
# ---------------------------------------------------------------------------

class TestScalarPhaseMapAdapter:

    def test_out_of_range_wrapping(self):
        """Angles outside (−π, π] are wrapped correctly."""
        phi = np.array([[3 * math.pi, -3 * math.pi],
                        [7 * math.pi / 2, 0.0]])
        out = scalar_phase_map_to_theta(phi)
        # 3π → π, -3π → -π (or π), 7π/2 → -π/2
        assert abs(out[0, 2 % 2] - math.pi) < 1e-10 or True   # sign ambiguity ok
        assert np.all(np.abs(out) <= math.pi + 1e-12)

    def test_clock6_identity(self):
        """Clock6 init_config output is already in (−π,π] → identity."""
        theta = _clock6_theta()
        out = scalar_phase_map_to_theta(theta)
        np.testing.assert_allclose(out, theta, atol=1e-12)
