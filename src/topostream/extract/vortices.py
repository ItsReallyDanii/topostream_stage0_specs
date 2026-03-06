"""
src/topostream/extract/vortices.py
===================================
Vortex extraction from a 2-D angle field.

Implements: agents/02_extract_vortices.md
Spec refs:   SPEC_FORMULAE.md §1-2, SPEC_ALGORITHMS.md §1, SPEC_VALIDATION.md §1

All angle differences use the normative wrap:
    wrap(Δθ) = arctan2(sin(Δθ), cos(Δθ))

No other wrapping operator is permitted anywhere in this module.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * np.pi
_CHARGE_TOL = 0.01  # |W_raw − round(W_raw)| threshold from SPEC_ALGORITHMS §1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_vortices(theta: np.ndarray, provenance: dict[str, Any]) -> list[dict]:
    """Extract vortex tokens from a spin-angle configuration.

    Parameters
    ----------
    theta:
        Shape ``(L, L)`` float64 array of angles in radians, with periodic
        boundary conditions assumed.  Values should lie in ``[−π, π)``.
    provenance:
        Dict with *at least* the keys required by the schema:
        ``model``, ``L``, ``T``, ``seed``, ``sweep_index``,
        ``schema_version``.  Any extra keys are passed through unchanged.

    Returns
    -------
    list[dict]
        One dict per detected vortex / antivortex, each conforming to
        ``schemas/topology_event_stream.schema.json`` with
        ``token_type == "vortex"``.

    Raises
    ------
    ValueError
        If ``theta`` is not a 2-D square array with side length L ≥ 8.
    """
    theta = np.asarray(theta, dtype=np.float64)
    _validate_theta(theta)

    L = theta.shape[0]
    tokens: list[dict] = []
    nan_skips = 0
    bad_winding = 0

    for i in range(L):
        for j in range(L):
            # Corners (counter-clockwise order per SPEC_FORMULAE §2)
            A = theta[i,           j          ]
            B = theta[i,          (j + 1) % L ]
            C = theta[(i + 1) % L, (j + 1) % L]
            D = theta[(i + 1) % L, j          ]

            # Map-mode: skip plaquettes containing NaN corners
            if np.isnan(A) or np.isnan(B) or np.isnan(C) or np.isnan(D):
                nan_skips += 1
                continue

            # Normative wrap for ALL angle differences — SPEC_FORMULAE §1
            dAB = _wrap(B - A)
            dBC = _wrap(C - B)
            dCD = _wrap(D - C)
            dDA = _wrap(A - D)

            W_raw = (dAB + dBC + dCD + dDA) / _TWO_PI
            charge = int(round(W_raw))

            if abs(W_raw - charge) > _CHARGE_TOL:
                logger.warning(
                    "Non-integer winding %.4f at plaquette (%d, %d); skipping.",
                    W_raw, i, j,
                )
                bad_winding += 1
                continue

            if charge == 0:
                continue

            # Plaquette centre (SPEC_ALGORITHMS §1)
            x_pos = j + 0.5
            y_pos = i + 0.5

            token = _make_vortex_token(
                x=x_pos,
                y=y_pos,
                charge=charge,
                strength=abs(W_raw),
                provenance=provenance,
                row=i,
                col=j,
            )
            tokens.append(token)

    if nan_skips > 0:
        logger.info("Skipped %d NaN-corner plaquettes.", nan_skips)
    if bad_winding > 0:
        logger.info("Skipped %d non-integer winding plaquettes.", bad_winding)

    logger.debug(
        "extract_vortices: L=%d, found %d vortices (%d+, %d-).",
        L,
        len(tokens),
        sum(1 for t in tokens if t["vortex"]["charge"] == +1),
        sum(1 for t in tokens if t["vortex"]["charge"] == -1),
    )

    return tokens


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wrap(dtheta: float | np.ndarray) -> float | np.ndarray:
    """Normative angle-difference operator (SPEC_FORMULAE §1).

    wrap(Δθ) = arctan2(sin(Δθ), cos(Δθ))

    Maps any real Δθ into (−π, π].  This is the ONLY permitted wrapping
    function in this codebase.
    """
    return np.arctan2(np.sin(dtheta), np.cos(dtheta))


def _validate_theta(theta: np.ndarray) -> None:
    """Raise ValueError if theta is not a valid L×L angle field."""
    if theta.ndim != 2:
        raise ValueError(
            f"theta must be a 2-D array; got shape {theta.shape}."
        )
    L_rows, L_cols = theta.shape
    if L_rows != L_cols:
        raise ValueError(
            f"theta must be square; got {L_rows}×{L_cols}."
        )
    if L_rows < 8:
        raise ValueError(
            f"L must be ≥ 8 (SPEC_ALGORITHMS §1); got L={L_rows}."
        )


def _make_vortex_token(
    *,
    x: float,
    y: float,
    charge: int,
    strength: float,
    provenance: dict[str, Any],
    row: int = 0,
    col: int = 0,
) -> dict:
    """Build a schema-compliant vortex token.

    ``confidence`` defaults to 1.0 for single-run extraction; callers that
    aggregate over multiple seeds should overwrite this field with the value
    computed per SPEC_UQ §3.

    IDs are deterministic based on charge sign and plaquette position
    (required by 00_repo_rules.md REPRODUCIBILITY RULES).
    """
    sign_char = "p" if charge > 0 else "m"
    vortex_id = f"v_{sign_char}_r{row:03d}_c{col:03d}"
    return {
        "schema_version": "1.1.0",
        "token_type": "vortex",
        "provenance": provenance,
        "vortex": {
            "id": vortex_id,
            "x": float(x),
            "y": float(y),
            "charge": int(charge),
            "strength": float(strength),
            "confidence": 1.0,  # single-run default; see SPEC_UQ §3
        },
    }
