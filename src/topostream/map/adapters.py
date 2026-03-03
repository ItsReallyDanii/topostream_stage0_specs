"""
src/topostream/map/adapters.py
================================
Explicit map → angle-field adapters.

"Adapter" means a SPECIFIC, documented inverse of a SPECIFIC forward model.
There is no generic inversion here.  Each adapter has a named forward
counterpart and a documented domain of validity.

Public API
----------
vector_map_to_theta(Mx, My)             arctan2 inversion of to_vector_map()
scalar_phase_map_to_theta(phi)          identity adapter for pre-computed angle maps

NaN handling
------------
All adapters propagate NaNs: if ANY input component at site (i,j) is NaN,
the output at (i,j) is NaN.  This is intentional — the caller decides how to
handle missing data (skip in vortex extraction, interpolate, flag, etc.).
NaNs are NEVER silently filled.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Vector map → angle field  (inverse of to_vector_map)
# ---------------------------------------------------------------------------

def vector_map_to_theta(Mx: np.ndarray, My: np.ndarray) -> np.ndarray:
    """Recover spin angles from a 2-component vector map via arctan2.

    This is the exact inverse of ``to_vector_map`` for noise-free data:

        theta_hat = arctan2(My, Mx)

    For noisy / blurred / downsampled Mx, My the recovery is approximate.
    The output theta_hat can be passed directly to ``extract_vortices`` — the
    vortex extractor uses the same normative arctan2 wrapping, so no
    additional wrapping step is needed here.

    NaN propagation
    ~~~~~~~~~~~~~~~
    If Mx[i,j] is NaN **or** My[i,j] is NaN, theta_hat[i,j] = NaN.
    The vortex extractor skips plaquettes that contain any NaN corner
    (documented in extract/vortices.py).

    Parameters
    ----------
    Mx, My : np.ndarray
        Both shape ``(H, W)`` float64.  The x- and y-components of the
        measured spin vector (e.g. from to_vector_map or a real AFM/MFM map).
        Values need NOT be unit-normalised — arctan2 is scale-invariant.
        Both arrays must have the same shape.

    Returns
    -------
    theta_hat : np.ndarray
        Shape ``(H, W)`` float64, values in ``(−π, π]``.

    Raises
    ------
    ValueError
        If ``Mx`` and ``My`` have different shapes.
    """
    Mx = np.asarray(Mx, dtype=np.float64)
    My = np.asarray(My, dtype=np.float64)
    if Mx.shape != My.shape:
        raise ValueError(
            f"Mx and My must have the same shape; got {Mx.shape} vs {My.shape}"
        )
    # np.arctan2 propagates NaN naturally when either argument is NaN
    return np.arctan2(My, Mx)


# ---------------------------------------------------------------------------
# 2. Scalar phase map → angle field  (identity adapter)
# ---------------------------------------------------------------------------

def scalar_phase_map_to_theta(phi: np.ndarray) -> np.ndarray:
    """Identity adapter: returns the input array wrapped into (−π, π].

    **Domain of validity**: ``phi`` must already represent spin angles in
    radians, produced by instruments or simulation outputs that natively
    output an angle field (e.g. phase maps from off-axis electron holography,
    or direct output of a clock-model simulator).  This adapter does NOT
    perform phase unwrapping; it only applies the normative wrap:

        theta_hat = arctan2(sin(phi), cos(phi))

    so that values outside ``(−π, π]`` are correctly folded.

    NaN propagation
    ~~~~~~~~~~~~~~~
    NaN inputs produce NaN outputs.  No filling is performed.

    Parameters
    ----------
    phi : np.ndarray
        Shape ``(H, W)`` float64 angle field in radians.  Values may be
        outside ``(−π, π]`` — they will be wrapped.

    Returns
    -------
    theta_hat : np.ndarray
        Same shape, float64, values in ``(−π, π]``.

    Raises
    ------
    ValueError
        If ``phi`` is not a 2-D array.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D; got shape {phi.shape}")
    # Normative wrap — consistent with SPEC_FORMULAE §1
    return np.arctan2(np.sin(phi), np.cos(phi))
