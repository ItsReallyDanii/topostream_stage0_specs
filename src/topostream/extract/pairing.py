"""
src/topostream/extract/pairing.py
===================================
Hungarian min-cost vortex–antivortex pairing on a periodic lattice.

Implements: agents/03_pairing.md
Spec refs:   SPEC_ALGORITHMS.md §2 (normative algorithm),
             SPEC_FORMULAE.md  §5 (minimum-image separation),
             SPEC_METRICS.md   §1.3 (pairing fraction).

This module is the ONLY permitted pairing algorithm.  No nearest-
neighbour greedy, no Delaunay triangulation, no heuristic shortcuts.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Sentinel used to mask entries beyond r_max (SPEC_ALGORITHMS §2 step 2).
_INF_COST = 1e9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pair_vortices(
    vortices: list[dict],
    antivortices: list[dict],
    L: int,
    r_max: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict:
    """Match vortices to antivortices via Hungarian min-cost assignment.

    Parameters
    ----------
    vortices :
        List of vortex token dicts.  Each **must** contain at minimum a
        ``vortex`` sub-dict with keys ``id``, ``x``, ``y``, ``charge``
        (charge == +1).
    antivortices :
        Same structure, but with charge == −1.
    L :
        Lattice side length (integer ≥ 8).  Used for minimum-image PBC
        distances.
    r_max :
        Maximum pairing separation in lattice units.  Default ``L / 4``
        per SPEC_ALGORITHMS §2.
    provenance :
        Provenance dict passed through to emitted pair tokens.  If
        ``None``, a minimal stub is used.

    Returns
    -------
    dict
        ``{"pairs": [...], "unmatched_vortex_ids": [...],
          "unmatched_antivortex_ids": [...], "f_paired": float}``

        Each item in ``pairs`` is a schema-compliant pair token.
    """
    if r_max is None:
        r_max = L / 4.0

    if provenance is None:
        provenance = {}

    n_v = len(vortices)
    n_a = len(antivortices)

    # ----- trivial cases -----
    if n_v == 0 or n_a == 0:
        return _empty_result(vortices, antivortices, r_max)

    # ----- step 1: build cost matrix (SPEC_ALGORITHMS §2.1) -----
    cost = np.empty((n_v, n_a), dtype=np.float64)
    for i, v_tok in enumerate(vortices):
        v = v_tok["vortex"]
        for j, a_tok in enumerate(antivortices):
            a = a_tok["vortex"]
            cost[i, j] = _minimum_image_distance(
                v["x"], v["y"], a["x"], a["y"], L,
            )

    # ----- step 2: mask beyond r_max (SPEC_ALGORITHMS §2.2) -----
    cost_masked = cost.copy()
    cost_masked[cost > r_max] = _INF_COST

    # ----- step 3: solve (SPEC_ALGORITHMS §2.3) -----
    row_ind, col_ind = linear_sum_assignment(cost_masked)

    # ----- step 4–5: classify matches vs unmatched -----
    matched_v: set[int] = set()
    matched_a: set[int] = set()
    pairs: list[dict] = []

    for i, j in zip(row_ind, col_ind):
        if cost_masked[i, j] >= _INF_COST:
            # Assigned but penalised → treat as unmatched (SPEC_ALGORITHMS §2.4)
            continue
        v_id = vortices[i]["vortex"]["id"]
        a_id = antivortices[j]["vortex"]["id"]
        pair_token = _make_pair_token(
            vortex_id=v_id,
            antivortex_id=a_id,
            separation_r=float(cost[i, j]),   # true distance, not masked
            r_max_used=float(r_max),
            provenance=provenance,
        )
        pairs.append(pair_token)
        matched_v.add(i)
        matched_a.add(j)

    unmatched_v_ids = [
        vortices[i]["vortex"]["id"] for i in range(n_v) if i not in matched_v
    ]
    unmatched_a_ids = [
        antivortices[j]["vortex"]["id"] for j in range(n_a) if j not in matched_a
    ]

    if unmatched_v_ids:
        logger.info("Unmatched vortex ids: %s", unmatched_v_ids)
    if unmatched_a_ids:
        logger.info("Unmatched antivortex ids: %s", unmatched_a_ids)

    # ----- pairing fraction (SPEC_ALGORITHMS §2, SPEC_METRICS §1.3) -----
    total_defects = n_v + n_a
    f_paired = (2.0 * len(pairs)) / total_defects if total_defects > 0 else 0.0

    logger.debug(
        "pair_vortices: %d vortices, %d antivortices → %d pairs "
        "(f_paired=%.3f, r_max=%.2f).",
        n_v, n_a, len(pairs), f_paired, r_max,
    )

    return {
        "pairs": pairs,
        "unmatched_vortex_ids": unmatched_v_ids,
        "unmatched_antivortex_ids": unmatched_a_ids,
        "f_paired": f_paired,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _minimum_image_distance(
    px: float, py: float, qx: float, qy: float, L: int,
) -> float:
    """Minimum-image pair separation on an L×L PBC lattice (SPEC_FORMULAE §5).

        r_x = min(|px − qx|, L − |px − qx|)
        r_y = min(|py − qy|, L − |py − qy|)
        r   = sqrt(r_x² + r_y²)
    """
    dx = abs(px - qx)
    dy = abs(py - qy)
    rx = min(dx, L - dx)
    ry = min(dy, L - dy)
    return math.sqrt(rx * rx + ry * ry)


def _make_pair_token(
    *,
    vortex_id: str,
    antivortex_id: str,
    separation_r: float,
    r_max_used: float,
    provenance: dict[str, Any],
) -> dict:
    """Build a schema-compliant pair token.

    pair_id convention (from agents/03_pairing.md):
        ``"pair_{vortex_id}_{antivortex_id}"``
    """
    pair_id = f"pair_{vortex_id}_{antivortex_id}"
    return {
        "schema_version": "1.1.0",
        "token_type": "pair",
        "provenance": provenance,
        "pair": {
            "pair_id": pair_id,
            "vortex_id": vortex_id,
            "antivortex_id": antivortex_id,
            "separation_r": separation_r,
            "r_max_used": r_max_used,
        },
    }


def _empty_result(
    vortices: list[dict],
    antivortices: list[dict],
    r_max: float,
) -> dict:
    """Return value when one or both defect lists are empty."""
    return {
        "pairs": [],
        "unmatched_vortex_ids": [v["vortex"]["id"] for v in vortices],
        "unmatched_antivortex_ids": [a["vortex"]["id"] for a in antivortices],
        "f_paired": 0.0,
    }
