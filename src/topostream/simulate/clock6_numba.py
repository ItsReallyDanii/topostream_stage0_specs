"""
src/topostream/simulate/clock6_numba.py
=========================================
Numba-jitted Metropolis Monte Carlo for the 2-D q=6 Clock model on a square
lattice with periodic boundary conditions (PBC).

Spec refs:   SPEC_ALGORITHMS.md §4, SPEC_FORMULAE.md §1 & §4,
             agents/00_repo_rules.md §PHYSICS, §CPU, §REPRODUCIBILITY.

Algorithm (SPEC_ALGORITHMS §4):
    Identical to XY MC with one modification:
    After proposing theta_new, project to nearest q=6 state:
        allowed = [k × π/3 for k in range(6)]
        theta_new = allowed[argmin_k |wrap(theta_new − allowed[k])|]

CPU rule (agents/00_repo_rules.md):
    The per-site inner loop is @numba.njit.  No pure-Python per-spin loops
    exist in this module.

Determinism:
    With a fixed seed, output is exactly reproducible.

Allowed angles: {0, π/3, 2π/3, π, 4π/3, 5π/3}  (in [0, 2π) before wrapping)
After normative wrap to (−π, π]:
    0, π/3, 2π/3, π (= −π after wrap), −2π/3, −π/3
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numba
import numpy as np

from topostream.metrics.helicity import compute_helicity_ensemble

logger = logging.getLogger(__name__)

# Allowed angles for q=6 clock model (stored as float64 array for Numba)
# These are k*pi/3 for k in 0..5, wrapped into (−π, π] using arctan2.
_Q = 6
_STEP_SIZE = math.pi / 3.0  # proposal step size per SPEC_ALGORITHMS §3

# Pre-compute allowed angles at module level for use in @njit functions.
# Numba can capture module-level numpy arrays as compile-time constants.
_ALLOWED_ANGLES: np.ndarray = np.array(
    [math.atan2(math.sin(k * math.pi / 3.0), math.cos(k * math.pi / 3.0))
     for k in range(_Q)],
    dtype=np.float64,
)


# ===================================================================
# Numba-jitted helpers (shared PRNG, identical to xy_numba.py)
# ===================================================================

@numba.njit
def _wrap(dtheta: float) -> float:
    """Normative angle wrap: arctan2(sin(Δθ), cos(Δθ)) — SPEC_FORMULAE §1."""
    return math.atan2(math.sin(dtheta), math.cos(dtheta))


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
# Clock6 projection: snap to nearest of 6 allowed angles
# ===================================================================

@numba.njit
def _project_clock6(theta_prop: float, allowed: np.ndarray) -> float:
    """Project proposed angle to nearest q=6 clock state.

    Uses normative wrap for angular distance (SPEC_FORMULAE §1 & SPEC_ALGORITHMS §4).

    Parameters
    ----------
    theta_prop : proposed continuous angle (radians)
    allowed    : array of 6 allowed angles in (−π, π]

    Returns
    -------
    The nearest allowed angle.
    """
    best_ang = allowed[0]
    best_dist = math.fabs(_wrap(theta_prop - allowed[0]))
    for k in range(1, 6):
        dist = math.fabs(_wrap(theta_prop - allowed[k]))
        if dist < best_dist:
            best_dist = dist
            best_ang = allowed[k]
    return best_ang


# ===================================================================
# Core Numba-jitted Metropolis sweep — Clock6
# ===================================================================

@numba.njit
def _metropolis_sweep_clock6(
    theta: np.ndarray,
    L: int,
    T: float,
    J: float,
    rng_state: np.ndarray,
    step_size: float,
    allowed: np.ndarray,
) -> float:
    """One full Metropolis sweep over all L² sites for the q=6 Clock model.

    SPEC_ALGORITHMS §4:
      - Identical to XY sweep, except proposed angle is projected to nearest
        allowed clock state before the Metropolis acceptance test.

    Parameters
    ----------
    theta      : (L, L) float64 — mutated in place; values in allowed set.
    L          : lattice side.
    T          : temperature (> 0).
    J          : coupling constant.
    rng_state  : 1-element uint64 array used as xorshift64 state.
    step_size  : max angular proposal (π/3 per spec).
    allowed    : 1-D array of 6 allowed angles.

    Returns
    -------
    energy_per_spin : float — total energy / L² after the sweep.
    """
    n_sites = L * L

    # Shuffle site visit order (Fisher-Yates in place)
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

        # Propose continuous move, then project to nearest clock state
        r1 = _xorshift_uniform(rng_state)
        dtheta_prop = step_size * (2.0 * r1 - 1.0)
        theta_cont = _wrap(theta_old + dtheta_prop)
        theta_new = _project_clock6(theta_cont, allowed)

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
            jp = (j + 1) % L
            ip = (i + 1) % L
            energy -= J * math.cos(theta[i, j] - theta[i, jp])
            energy -= J * math.cos(theta[i, j] - theta[ip, j])
    return energy / n_sites


# ===================================================================
# Internal helpers
# ===================================================================

def _make_rng_state(seed: int) -> np.ndarray:
    """Create xorshift64 state from seed (must be nonzero)."""
    s = np.uint64(seed if seed != 0 else 1)
    return np.array([s], dtype=np.uint64)


def _check_thermalization(energy_trace: list[float]) -> None:
    """SPEC_UQ §4 thermalization check (mirrored from xy_numba)."""
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


# ===================================================================
# Public API
# ===================================================================

def init_config_clock6(L: int, seed: int) -> np.ndarray:
    """Initialise θ[L, L] with each site randomly chosen from the 6 clock states.

    Allowed angles are {k × π/3 for k in 0..5} wrapped into (−π, π].
    This is the normative clock6 initialisation.

    Parameters
    ----------
    L    : lattice side length (≥ 8).
    seed : RNG seed (deterministic).

    Returns
    -------
    theta : (L, L) float64 array with values in the 6-element allowed set.
    """
    if L < 8:
        raise ValueError(f"L must be ≥ 8; got {L}")
    rng = np.random.default_rng(seed)
    # Draw random indices 0..5 for each site
    idx = rng.integers(0, _Q, size=(L, L))
    return _ALLOWED_ANGLES[idx].astype(np.float64)


def metropolis_sweep_clock6(
    theta: np.ndarray,
    T: float,
    J: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Convenience wrapper: one Metropolis sweep for Clock6 (public, testable).

    Parameters
    ----------
    theta : (L, L) float64 — mutated in place.
    T     : temperature (> 0).
    J     : coupling (default 1.0).
    seed  : RNG seed for this single sweep.

    Returns
    -------
    (theta, energy_per_spin) — theta is mutated in-place AND returned.
    """
    L = theta.shape[0]
    rng_state = _make_rng_state(seed)
    e = _metropolis_sweep_clock6(theta, L, T, J, rng_state, _STEP_SIZE, _ALLOWED_ANGLES)
    return theta, float(e)


def run_clock6(
    L: int,
    T: float,
    J: float = 1.0,
    N_equil: int = 10_000,
    N_meas: int = 50_000,
    N_thin: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Metropolis MC for the 2-D q=6 Clock model.

    Parameters
    ----------
    L       : int   Lattice side length (≥ 8).
    T       : float Temperature (> 0).
    J       : float Coupling constant (default 1.0).
    N_equil : int   Number of equilibration sweeps (discarded).
    N_meas  : int   Number of measurement sweeps.
    N_thin  : int   Keep every N_thin-th config.
    seed    : int   RNG seed (deterministic).

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
    theta = init_config_clock6(L, seed)
    rng_state = _make_rng_state(seed)

    # --- Equilibration ---
    for _ in range(N_equil):
        _metropolis_sweep_clock6(theta, L, T, J, rng_state, _STEP_SIZE, _ALLOWED_ANGLES)

    # --- Measurement ---
    configs: list[np.ndarray] = []
    energy_trace: list[float] = []

    for sweep_idx in range(N_meas):
        e = _metropolis_sweep_clock6(theta, L, T, J, rng_state, _STEP_SIZE, _ALLOWED_ANGLES)
        energy_trace.append(e)
        if sweep_idx % N_thin == 0:
            configs.append(theta.copy())

    # --- Thermalization check ---
    _check_thermalization(energy_trace)

    # --- Helicity modulus (SPEC_FORMULAE §4; same estimator for XY and Clock6) ---
    helicity, helicity_err = compute_helicity_ensemble(configs, T, J)

    provenance = {
        "model": "clock6",
        "L": L,
        "T": T,
        "J": J,
        "seed": seed,
        "N_equil": N_equil,
        "N_meas": N_meas,
        "N_thin": N_thin,
        "schema_version": "1.0.0",
    }

    return {
        "configs": configs,
        "helicity": helicity,
        "helicity_err": helicity_err,
        "energy_per_spin": energy_trace,
        "provenance": provenance,
    }
