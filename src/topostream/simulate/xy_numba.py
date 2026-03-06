"""
src/topostream/simulate/xy_numba.py
=====================================
Numba-jitted Metropolis Monte Carlo for the 2-D XY model on a square
lattice with periodic boundary conditions (PBC).

Implements: agents/01_sim_xy.md
Spec refs:   SPEC_ALGORITHMS.md §3, SPEC_FORMULAE.md §1 & §4,
             SPEC_UQ.md §4-5, SPEC_INPUTS.md §1.

CPU rule (agents/00_repo_rules.md):
    The per-site inner loop is @numba.njit.  No pure-Python per-spin loops
    exist in this module.

Determinism:
    With a fixed seed, output is exactly reproducible.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numba
import numpy as np

from topostream.metrics.helicity import compute_helicity_ensemble

logger = logging.getLogger(__name__)

# Step size per SPEC_ALGORITHMS §3
_STEP_SIZE = math.pi / 3.0


# ===================================================================
# Numba-jitted core (NO pure-Python per-spin loops)
# ===================================================================

@numba.njit
def _wrap(dtheta: float) -> float:
    """Normative angle wrap: arctan2(sin(Δθ), cos(Δθ)) — SPEC_FORMULAE §1."""
    return math.atan2(math.sin(dtheta), math.cos(dtheta))


@numba.njit
def _metropolis_sweep(
    theta: np.ndarray,
    L: int,
    T: float,
    J: float,
    rng_state: np.ndarray,
    step_size: float,
) -> float:
    """One full Metropolis sweep over all L² sites (random order).

    Parameters
    ----------
    theta : (L, L) float64 — mutated in place.
    L : lattice side.
    T : temperature (> 0).
    J : coupling constant.
    rng_state : 1-element uint64 array used as xorshift64 state.
    step_size : max angular proposal (π/3 per spec).

    Returns
    -------
    energy_per_spin : float — total energy / L² after the sweep.
    """
    n_sites = L * L

    # Generate a random order for site visits
    order = np.arange(n_sites)
    for k in range(n_sites - 1, 0, -1):
        j_idx = _xorshift_int(rng_state, k + 1)
        order[k], order[j_idx] = order[j_idx], order[k]

    for idx in range(n_sites):
        site = order[idx]
        i = site // L
        j = site % L

        # Nearest neighbours with PBC
        ip = (i + 1) % L
        im = (i - 1) % L
        jp = (j + 1) % L
        jm = (j - 1) % L

        theta_old = theta[i, j]

        # Propose: dtheta = step_size × (2 × rand − 1)
        r1 = _xorshift_uniform(rng_state)
        dtheta_prop = step_size * (2.0 * r1 - 1.0)
        theta_new = _wrap(theta_old + dtheta_prop)

        # Energy change: dE = −J Σ_nn [cos(θ_new − θ_k) − cos(θ_old − θ_k)]
        dE = 0.0
        for theta_k in (theta[ip, j], theta[im, j], theta[i, jp], theta[i, jm]):
            dE += math.cos(theta_old - theta_k) - math.cos(theta_new - theta_k)
        dE *= -J

        # Metropolis accept / reject
        if dE < 0.0:
            theta[i, j] = theta_new
        else:
            r2 = _xorshift_uniform(rng_state)
            if r2 < math.exp(-dE / T):
                theta[i, j] = theta_new

    # Compute total energy for this configuration
    energy = 0.0
    for i in range(L):
        for j in range(L):
            # Only count right and down neighbours to avoid double counting
            jp = (j + 1) % L
            ip = (i + 1) % L
            energy -= J * math.cos(theta[i, j] - theta[i, jp])
            energy -= J * math.cos(theta[i, j] - theta[ip, j])
    return energy / n_sites


# ===================================================================
# Xorshift64 PRNG (Numba-compatible, deterministic)
# ===================================================================

@numba.njit
def _xorshift_next(state: np.ndarray) -> np.uint64:
    """Advance xorshift64 state and return raw 64-bit value."""
    x = state[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    state[0] = x
    return x


@numba.njit
def _xorshift_uniform(state: np.ndarray) -> float:
    """Uniform float in [0, 1)."""
    v = _xorshift_next(state)
    return (v >> np.uint64(11)) / 9007199254740992.0  # / 2^53


@numba.njit
def _xorshift_int(state: np.ndarray, n: int) -> int:
    """Uniform integer in [0, n)."""
    v = _xorshift_next(state)
    return int(v % np.uint64(n))


# ===================================================================
# Public API
# ===================================================================

def run_xy(
    L: int,
    T: float,
    J: float = 1.0,
    N_equil: int = 10_000,
    N_meas: int = 50_000,
    N_thin: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Metropolis MC for the 2-D XY model.

    Parameters
    ----------
    L : int        Lattice side length (≥ 8).
    T : float      Temperature (> 0).
    J : float      Coupling constant (default 1.0).
    N_equil : int  Number of equilibration sweeps (discarded).
    N_meas : int   Number of measurement sweeps.
    N_thin : int   Keep every N_thin-th config.
    seed : int     RNG seed (deterministic).

    Returns
    -------
    dict with keys:
        configs          : list[np.ndarray]  — thinned measurement configs
        helicity         : float             — Υ mean over configs
        helicity_err     : float             — Υ jackknife error
        energy_per_spin  : list[float]       — energy trace (every sweep)
        provenance       : dict              — run metadata
    """
    if L < 8:
        raise ValueError(f"L must be ≥ 8; got {L}")
    if T <= 0:
        raise ValueError(f"T must be > 0; got {T}")

    # --- Initialise ---
    theta = init_config(L, seed)
    rng_state = _make_rng_state(seed)

    # --- Equilibration ---
    for _ in range(N_equil):
        _metropolis_sweep(theta, L, T, J, rng_state, _STEP_SIZE)

    # --- Measurement ---
    configs: list[np.ndarray] = []
    energy_trace: list[float] = []

    for sweep_idx in range(N_meas):
        e = _metropolis_sweep(theta, L, T, J, rng_state, _STEP_SIZE)
        energy_trace.append(e)
        if sweep_idx % N_thin == 0:
            configs.append(theta.copy())

    # --- Thermalization check (SPEC_UQ §4) ---
    _check_thermalization(energy_trace)

    # --- Helicity modulus (SPEC_FORMULAE §4) ---
    helicity, helicity_err = compute_helicity_ensemble(configs, T, J)

    provenance = {
        "model": "XY",
        "L": L,
        "T": T,
        "J": J,
        "seed": seed,
        "N_equil": N_equil,
        "N_meas": N_meas,
        "N_thin": N_thin,
        "schema_version": "1.1.0",
    }

    return {
        "configs": configs,
        "helicity": helicity,
        "helicity_err": helicity_err,
        "energy_per_spin": energy_trace,
        "provenance": provenance,
    }


def init_config(L: int, seed: int) -> np.ndarray:
    """Initialise θ[L, L] ~ Uniform[−π, π) with explicit seed.

    (SPEC_ALGORITHMS §3: "Initialize: theta[L,L] ~ Uniform[−π, π)")
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(L, L)).astype(np.float64)


def metropolis_sweep(
    theta: np.ndarray,
    T: float,
    J: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Convenience wrapper: one Metropolis sweep (public, testable).

    Returns (theta_updated, energy_per_spin).
    theta is mutated in-place AND returned.
    """
    L = theta.shape[0]
    rng_state = _make_rng_state(seed)
    e = _metropolis_sweep(theta, L, T, J, rng_state, _STEP_SIZE)
    return theta, float(e)


# ===================================================================
# Internals
# ===================================================================

def _make_rng_state(seed: int) -> np.ndarray:
    """Create xorshift64 state from seed (must be nonzero)."""
    s = np.uint64(seed if seed != 0 else 1)
    return np.array([s], dtype=np.uint64)


def _check_thermalization(energy_trace: list[float]) -> None:
    """SPEC_UQ §4 thermalization check.

    E_early = mean energy/spin over first 25% of measurement sweeps.
    E_late  = mean energy/spin over last 25% of measurement sweeps.
    Assert: |E_late − E_early| / |E_late| < 0.01
    """
    n = len(energy_trace)
    if n < 4:
        logger.warning("Too few measurement sweeps for thermalization check.")
        return

    quarter = n // 4
    E_early = float(np.mean(energy_trace[:quarter]))
    E_late = float(np.mean(energy_trace[-quarter:]))

    if abs(E_late) < 1e-15:
        logger.warning("E_late ≈ 0; skipping thermalization ratio check.")
        return

    ratio = abs(E_late - E_early) / abs(E_late)
    if ratio >= 0.01:
        logger.warning(
            "Thermalization check FAILED: |E_late − E_early|/|E_late| = %.4f "
            "(threshold 0.01). E_early=%.6f, E_late=%.6f. "
            "Consider doubling N_equil.",
            ratio, E_early, E_late,
        )
