"""
src/topostream/metrics/clock.py
=================================
Sixfold order parameter ψ₆ and angle histogram for clock-model analysis.

Implements: agents/04_clock_metrics.md  —  clock.py section
Spec refs:   SPEC_FORMULAE.md §3, SPEC_METRICS.md §1.5

ψ₆ formula (SPEC_FORMULAE §3):

    ψ₆ = (1/N_sites) Σ_{x,y} exp(6i·θ(x,y))

    |ψ₆| → 1       in perfectly six-state clock-ordered phase (T < T₁)
    |ψ₆| → small    in QLRO phase (T₁ < T < T₂)
    |ψ₆| → 0        in disordered phase (T > T₂)

IMPORTANT: QLRO ≠ clock_ordered.  Never conflate them.

Angle histogram (SPEC_FORMULAE §3):

    Bin θ(x,y) into 36 bins over [−π, π).
    Six peaks at multiples of π/3 in the clock-ordered phase.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_psi6(theta: np.ndarray) -> complex:
    """Sixfold order parameter ψ₆ (SPEC_FORMULAE §3).

    Parameters
    ----------
    theta:
        Shape ``(L, L)`` angle field, float64, radians.

    Returns
    -------
    complex
        ψ₆ = (1/N) Σ exp(6i·θ).  Take ``abs()`` for the magnitude.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError(f"theta must be (L, L); got {theta.shape}")

    # Fully vectorised — no Python loops
    psi6 = np.mean(np.exp(6j * theta))
    return complex(psi6)


def compute_angle_histogram(
    theta: np.ndarray,
    n_bins: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Angle histogram over [−π, π) (SPEC_FORMULAE §3).

    Parameters
    ----------
    theta:
        Shape ``(L, L)`` angle field, float64, radians.
    n_bins:
        Number of bins (default 36 per spec).

    Returns
    -------
    (bin_centers, counts)
        ``bin_centers`` is float64 array of length ``n_bins``.
        ``counts`` is int64 array of length ``n_bins``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError(f"theta must be (L, L); got {theta.shape}")

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(theta.ravel(), bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return (bin_centers.astype(np.float64), counts.astype(np.int64))


def compute_clock6_order(theta: np.ndarray) -> float:
    """Population-concentration order parameter for the q=6 clock model.

    Counts the fraction of spins in each of the 6 canonical clock states
    (k × π/3 for k = 0..5, normalised via arctan2 wrapping) and returns
    the maximum fraction:

        p_k   = fraction of spins whose angle matches state k (within tol)
        clock6_order = max_k p_k

    Range and interpretation:
        → 1.0  : all spins in a single clock state (perfect single-domain order)
        → 1/6 ≈ 0.1667 : uniform population across all 6 states (fully disordered)

    This metric meaningfully discriminates temperature for the clock6 simulator
    (unlike |ψ₆|, which equals 1 for every discrete clock config by algebra).

    Parameters
    ----------
    theta : np.ndarray
        Shape ``(L, L)`` angle field, float64, radians.  Values should be
        elements of the 6-state allowed set produced by clock6 simulator,
        but the function tolerates arbitrary continuous angles by snapping
        each site to its nearest clock state.

    Returns
    -------
    float
        Scalar in [1/6, 1.0] (for a non-empty array with at least one state
        present).

    Raises
    ------
    ValueError
        If ``theta`` is not a 2-D square array.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError(f"theta must be (L, L); got {theta.shape}")

    # 6 canonical clock states, wrapped into (−π, π]
    _allowed = np.array(
        [np.arctan2(np.sin(k * np.pi / 3.0), np.cos(k * np.pi / 3.0))
         for k in range(6)],
        dtype=np.float64,
    )

    flat = theta.ravel()
    n = len(flat)
    if n == 0:
        return 0.0

    # For each site, find the nearest clock state (by angular distance)
    # Shape: (n_sites, 6)
    diffs = np.abs(np.arctan2(
        np.sin(flat[:, np.newaxis] - _allowed[np.newaxis, :]),
        np.cos(flat[:, np.newaxis] - _allowed[np.newaxis, :]),
    ))
    nearest = np.argmin(diffs, axis=1)   # index of nearest clock state per site

    # Count population per state and return maximum fraction
    counts = np.bincount(nearest, minlength=6)
    return float(np.max(counts)) / n


def label_regime(
    T: float,
    T1: Optional[float] = None,
    T2: Optional[float] = None,
) -> str:
    """Classify temperature into one of the three q=6 clock-model regimes.

    Regime definitions (SPEC_METRICS §0):
        T > T₂          → ``"disordered"``
        T₁ < T < T₂     → ``"QLRO"``       (quasi-long-range order)
        T < T₁           → ``"clock_ordered"``

    If T₁ or T₂ are not supplied, returns ``"unknown"``.

    IMPORTANT: QLRO ≠ clock_ordered.  Never conflate them.
    """
    if T1 is None or T2 is None:
        return "unknown"
    if T > T2:
        return "disordered"
    if T > T1:
        return "QLRO"
    return "clock_ordered"
