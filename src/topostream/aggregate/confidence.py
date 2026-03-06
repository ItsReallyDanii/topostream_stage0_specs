"""
src/topostream/aggregate/confidence.py
========================================
Multi-seed aggregation for vortex confidence and condition-level stability.

Implements SPEC_UQ §3:
    confidence = clip(1 − σ_count / max(μ_count, ε), 0, 1)

Two outputs:
  A) Condition-level global_detection_stability — scalar per (model, L, T).
  B) Per-vortex confidence — fraction of seeds in which a same-charge vortex
     is re-detected at the same location (within tolerance, minimum-image PBC).

No speculative tracker.  No heavy redesign.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default match tolerance in plaquette (lattice) units.
DEFAULT_MATCH_TOLERANCE: float = 1.0

# Epsilon to avoid division by zero in stability formula.
_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_results_dir(
    results_dir: str | Path,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> dict[tuple[str, int, float], dict]:
    """Aggregate all per-seed token files in *results_dir*.

    Scans for ``tokens_*.jsonl`` files, groups vortex tokens by
    (model, L, T), computes condition-level stability and per-vortex
    confidence, and writes aggregate artifacts back to *results_dir*.

    Returns
    -------
    dict[(model, L, T)] -> condition_summary dict
    """
    results_dir = Path(results_dir)
    token_files = sorted(results_dir.glob("tokens_*.jsonl"))
    if not token_files:
        logger.warning("No token files found in %s", results_dir)
        return {}

    # Collect vortex tokens grouped by condition key.
    condition_seeds: dict[tuple[str, int, float], dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for tf in token_files:
        with tf.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tok = json.loads(line)
                if tok.get("token_type") != "vortex":
                    continue
                prov = tok["provenance"]
                key = (prov["model"], prov["L"], prov["T"])
                seed = prov["seed"]
                condition_seeds[key][seed].append(tok)

    results: dict[tuple[str, int, float], dict] = {}

    for cond_key, seeds_dict in sorted(condition_seeds.items()):
        model, L, T = cond_key
        summary = compute_condition_aggregate(
            seeds_dict, model=model, L=L, T=T,
            match_tolerance=match_tolerance,
        )
        results[cond_key] = summary

        # Write condition-level aggregate artifact.
        agg_file = results_dir / f"aggregate_{model}_{L}x{L}_T{T:.4f}.json"
        with agg_file.open("w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        logger.info("Wrote aggregate: %s", agg_file.name)

        # Update per-seed token files with real confidence values.
        _update_token_files(results_dir, cond_key, summary, seeds_dict)

    return results


def compute_condition_aggregate(
    seeds_dict: dict[int, list[dict]],
    *,
    model: str,
    L: int,
    T: float,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> dict[str, Any]:
    """Compute condition-level stability and per-vortex confidence.

    Parameters
    ----------
    seeds_dict :
        Mapping from seed -> list of vortex tokens for that seed.
    model, L, T :
        Condition identifiers.
    match_tolerance :
        Maximum minimum-image distance (plaquette units) for two
        same-charge vortices to be considered the same detection.

    Returns
    -------
    dict with keys: model, L, T, N_seeds, mu_vortex_count,
    sigma_vortex_count, global_detection_stability, n_consensus_clusters,
    consensus_vortices (list of consensus vortex dicts with confidence).
    """
    seeds = sorted(seeds_dict.keys())
    N_seeds = len(seeds)

    # --- A) Condition-level global detection stability ---
    counts = np.array([len(seeds_dict[s]) for s in seeds], dtype=np.float64)
    mu_count = float(np.mean(counts))
    sigma_count = float(np.std(counts, ddof=0))  # population std, matches spec
    global_detection_stability = float(
        np.clip(1.0 - sigma_count / max(mu_count, _EPS), 0.0, 1.0)
    )

    # --- B) Per-vortex confidence via consensus clustering ---
    # Collect all vortex positions across seeds, separated by charge.
    all_vortices: list[dict] = []
    for seed in seeds:
        for tok in seeds_dict[seed]:
            v = tok["vortex"]
            all_vortices.append({
                "x": v["x"],
                "y": v["y"],
                "charge": v["charge"],
                "strength": v["strength"],
                "seed": seed,
                "id": v["id"],
            })

    consensus = _build_consensus_clusters(
        all_vortices, L=L, match_tolerance=match_tolerance, N_seeds=N_seeds,
    )

    return {
        "model": model,
        "L": L,
        "T": T,
        "N_seeds": N_seeds,
        "mu_vortex_count": mu_count,
        "sigma_vortex_count": sigma_count,
        "global_detection_stability": global_detection_stability,
        "n_consensus_clusters": len(consensus),
        "consensus_vortices": consensus,
    }


def compute_per_vortex_confidence(
    seeds_dict: dict[int, list[dict]],
    *,
    L: int,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> list[dict]:
    """Compute per-vortex confidence only (no condition summary).

    Returns list of consensus vortex dicts, each with a ``confidence``
    field = fraction of seeds that detected this vortex.
    """
    seeds = sorted(seeds_dict.keys())
    N_seeds = len(seeds)

    all_vortices: list[dict] = []
    for seed in seeds:
        for tok in seeds_dict[seed]:
            v = tok["vortex"]
            all_vortices.append({
                "x": v["x"],
                "y": v["y"],
                "charge": v["charge"],
                "strength": v["strength"],
                "seed": seed,
                "id": v["id"],
            })

    return _build_consensus_clusters(
        all_vortices, L=L, match_tolerance=match_tolerance, N_seeds=N_seeds,
    )


def minimum_image_distance(
    px: float, py: float, qx: float, qy: float, L: int,
) -> float:
    """Minimum-image pair separation on an L×L PBC lattice (SPEC_FORMULAE §5).

        r_x = min(|px − qx|, L − |px − qx|)
        r_y = min(|py − qy|, L − |py − qy|)
        r   = sqrt(r_x² + r_y²)

    Public so tests can verify PBC wrap logic directly.
    """
    dx = abs(px - qx)
    dy = abs(py - qy)
    rx = min(dx, L - dx)
    ry = min(dy, L - dy)
    return math.sqrt(rx * rx + ry * ry)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_consensus_clusters(
    all_vortices: list[dict],
    *,
    L: int,
    match_tolerance: float,
    N_seeds: int,
) -> list[dict]:
    """Greedy consensus clustering of vortex detections across seeds.

    Algorithm:
    1. Separate vortices by charge (same-charge matching only).
    2. For each charge group, greedily cluster:
       - Take the first unassigned vortex as a seed for a new cluster.
       - Find all unassigned vortices from OTHER seeds that are within
         match_tolerance (minimum-image distance) of the cluster centroid.
       - Each seed contributes at most one member to a cluster.
       - Confidence = n_members / N_seeds.
    3. Return cluster centroids with confidence values.

    Conservative: simple greedy, no iterative EM.
    """
    clusters: list[dict] = []

    for charge_val in [+1, -1]:
        candidates = [
            v for v in all_vortices if v["charge"] == charge_val
        ]
        if not candidates:
            continue

        assigned = [False] * len(candidates)

        for i, cand in enumerate(candidates):
            if assigned[i]:
                continue

            # Start a new cluster with this vortex.
            cluster_members = [cand]
            assigned[i] = True
            member_seeds = {cand["seed"]}

            # Cluster centroid (using first member as reference).
            cx, cy = cand["x"], cand["y"]

            # Find matches from other seeds.
            for j in range(i + 1, len(candidates)):
                if assigned[j]:
                    continue
                other = candidates[j]
                if other["seed"] in member_seeds:
                    # Same seed — do not merge (each seed contributes once).
                    continue
                dist = minimum_image_distance(cx, cy, other["x"], other["y"], L)
                if dist <= match_tolerance:
                    cluster_members.append(other)
                    assigned[j] = True
                    member_seeds.add(other["seed"])
                    # Update centroid as running mean.
                    n = len(cluster_members)
                    cx = cx + (other["x"] - cx) / n
                    cy = cy + (other["y"] - cy) / n

            confidence = len(cluster_members) / N_seeds
            mean_strength = float(np.mean([m["strength"] for m in cluster_members]))

            clusters.append({
                "x": cx,
                "y": cy,
                "charge": charge_val,
                "strength": mean_strength,
                "confidence": confidence,
                "n_detections": len(cluster_members),
                "detection_seeds": sorted(member_seeds),
            })

    return clusters


def _update_token_files(
    results_dir: Path,
    cond_key: tuple[str, int, float],
    summary: dict,
    seeds_dict: dict[int, list[dict]],
) -> None:
    """Rewrite per-seed token files with updated vortex.confidence values.

    For each vortex token in each seed's file, find the best-matching
    consensus cluster and set vortex.confidence to the cluster's confidence.
    Non-vortex tokens (pairs, sweep_deltas) are passed through unchanged.
    """
    model, L, T = cond_key
    consensus = summary["consensus_vortices"]

    for seed, _ in seeds_dict.items():
        token_file = results_dir / f"tokens_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
        if not token_file.exists():
            continue

        updated_lines: list[str] = []
        with token_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tok = json.loads(line)
                if tok.get("token_type") == "vortex":
                    v = tok["vortex"]
                    best_conf = _find_best_confidence(
                        v["x"], v["y"], v["charge"], consensus, L,
                    )
                    tok["vortex"]["confidence"] = best_conf
                updated_lines.append(json.dumps(tok, default=_json_default))

        with token_file.open("w") as f:
            for uline in updated_lines:
                f.write(uline + "\n")


def _find_best_confidence(
    x: float, y: float, charge: int,
    consensus: list[dict], L: int,
) -> float:
    """Find the consensus cluster closest to (x, y) with matching charge.

    Returns the cluster's confidence value, or 0.0 if no match found.
    """
    best_dist = float("inf")
    best_conf = 0.0

    for c in consensus:
        if c["charge"] != charge:
            continue
        dist = minimum_image_distance(x, y, c["x"], c["y"], L)
        if dist < best_dist:
            best_dist = dist
            best_conf = c["confidence"]

    return best_conf


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
