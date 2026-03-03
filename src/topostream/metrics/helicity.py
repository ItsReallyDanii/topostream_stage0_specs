"""
src/topostream/metrics/helicity.py
====================================
Helicity modulus Υ(L, T) estimator for the XY / clock model.

Implements: agents/04_clock_metrics.md  —  helicity.py section
Spec refs:   SPEC_FORMULAE.md §4, SPEC_METRICS.md §1.1

Formula (SPEC_FORMULAE §4):

    Υ = (1/L²) × [ ⟨Σ_x cos(θᵢ − θⱼ)⟩ − (1/T) × ⟨(Σ_x sin(θᵢ − θⱼ))²⟩ ]

where Σ_x runs over ALL horizontal bonds only (twist in x-direction).

For a single configuration (no ensemble averaging), the ⟨⟩ simply evaluates
on that configuration.  When computing over *multiple* configs (jackknife),
each config contributes one sample of Σ_x cos(…) and Σ_x sin(…).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_helicity(
    theta: np.ndarray,
    T: float,
    J: float = 1.0,
) -> tuple[float, float]:
    """Helicity modulus Υ for a single spin configuration.

    Parameters
    ----------
    theta:
        Shape ``(L, L)`` angle field, float64, radians, PBC assumed.
    T:
        Temperature (must be > 0).
    J:
        Coupling constant (default 1.0).

    Returns
    -------
    (Upsilon, Upsilon_err)
        For a single config the jackknife error is zero; callers that
        aggregate over seeds should use ``compute_helicity_ensemble``
        instead.
    """
    theta = np.asarray(theta, dtype=np.float64)
    L = theta.shape[0]
    if theta.ndim != 2 or theta.shape[1] != L:
        raise ValueError(f"theta must be (L, L); got {theta.shape}")
    if T <= 0:
        raise ValueError(f"T must be > 0; got {T}")

    upsilon = _helicity_single(theta, T, J)
    return (float(upsilon), 0.0)


def compute_helicity_ensemble(
    configs: list[np.ndarray],
    T: float,
    J: float = 1.0,
) -> tuple[float, float]:
    """Helicity modulus Υ averaged over multiple configurations with
    jackknife error estimate.

    Parameters
    ----------
    configs:
        List of shape ``(L, L)`` angle fields.
    T:
        Temperature.
    J:
        Coupling constant.

    Returns
    -------
    (Upsilon_mean, Upsilon_jackknife_err)
    """
    if len(configs) == 0:
        raise ValueError("Need at least one configuration.")

    samples = np.array([_helicity_single(c, T, J) for c in configs])
    mean_val = float(np.mean(samples))

    if len(configs) < 2:
        return (mean_val, 0.0)

    # Jackknife error
    n = len(samples)
    jk_estimates = np.empty(n)
    for k in range(n):
        jk_estimates[k] = np.mean(np.delete(samples, k))
    jk_var = ((n - 1) / n) * np.sum((jk_estimates - np.mean(jk_estimates)) ** 2)
    jk_err = float(np.sqrt(jk_var))

    return (mean_val, jk_err)


# ---------------------------------------------------------------------------
# Internals — fully vectorised, no Python per-site loops
# ---------------------------------------------------------------------------

def _helicity_single(theta: np.ndarray, T: float, J: float) -> float:
    """Compute Υ for one configuration (SPEC_FORMULAE §4).

    Υ = (1/L²) × [ Σ_x cos(Δθ) − (1/T) × (Σ_x sin(Δθ))² ]

    Σ_x sums over all L² horizontal bonds (site (i,j) → (i, (j+1)%L)).
    """
    L = theta.shape[0]

    # Horizontal bond angle differences: Δθ_ij = θ[i, (j+1)%L] − θ[i, j]
    # Vectorised via np.roll
    dtheta_x = theta - np.roll(theta, -1, axis=1)  # θ[i,j] − θ[i,j+1]
    # Note: cos is symmetric so cos(a-b) = cos(b-a); sign doesn't matter.
    # But for sin we need sign consistent with the formula.
    # Formula uses cos(θᵢ − θⱼ) and sin(θᵢ − θⱼ) where j is the
    # neighbour in +x direction, so Δθ = θ[i,j] − θ[i, j+1].

    sum_cos = float(np.sum(np.cos(dtheta_x)))
    sum_sin = float(np.sum(np.sin(dtheta_x)))

    upsilon = (J / (L * L)) * (sum_cos - (1.0 / T) * sum_sin * sum_sin)
    return upsilon
