"""
src/topostream/analysis/token_benchmark.py
=============================================
Downstream token-only cross-pipeline defect-stability benchmark.

This module is the core claim of the tokenisation abstraction:
a single consumer can compare outputs from heterogeneous producers
(simulation, synthetic map-mode, future experimental adapters)
WITHOUT reading raw spin fields, vector maps, simulator internals,
or any producer-specific data format.

Public API
----------
load_tokens(jsonl_path)             → list[dict]
extract_vortex_set(tokens)          → list[VortexRecord]
extract_pair_set(tokens)            → list[PairRecord]
match_vortex_sets(ref, cand, L, tol)  → MatchResult
compare_token_streams(ref_path, cand_path, L) → ComparisonResult

ALL inputs are JSONL file paths or lists of schema-valid tokens.
No raw arrays, no theta fields, no producer-specific knowledge.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VortexRecord:
    """Minimal vortex representation extracted from a token dict."""
    id: str
    x: float
    y: float
    charge: int
    strength: float
    confidence: float


@dataclass(frozen=True)
class PairRecord:
    """Minimal pair representation extracted from a token dict."""
    pair_id: str
    vortex_id: str
    antivortex_id: str
    separation_r: float


@dataclass
class MatchResult:
    """Result of spatial matching between reference and candidate vortex sets."""
    n_ref: int
    n_cand: int
    n_matched: int
    recall: float          # n_matched / n_ref (if n_ref > 0)
    precision: float       # n_matched / n_cand (if n_cand > 0)
    false_positive_rate: float  # (n_cand - n_matched) / n_cand if n_cand > 0
    matched_distances: list[float] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Full comparison output between a reference and candidate token stream."""
    # Identification
    ref_path: str
    cand_path: str
    condition_label: str

    # Vortex matching
    match: MatchResult

    # Pairing metrics
    paired_fraction_ref: float
    paired_fraction_cand: float
    paired_fraction_error: float  # abs(cand - ref)

    # Separation metrics
    mean_separation_ref: float    # mean pair.separation_r for reference
    mean_separation_cand: float
    mean_separation_error: float  # abs(cand - ref)

    # Confidence summary (if available)
    mean_confidence_ref: float | None
    mean_confidence_cand: float | None


# ---------------------------------------------------------------------------
# Token loading (ONLY reads JSONL — no raw arrays)
# ---------------------------------------------------------------------------

def load_tokens(jsonl_path: str | Path) -> list[dict]:
    """Load all tokens from a JSONL file.

    Each line must be a valid JSON object conforming to the
    topology_event_stream schema.  No filtering is applied.
    """
    path = Path(jsonl_path)
    tokens: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(json.loads(line))
    return tokens


# ---------------------------------------------------------------------------
# Token extraction (schema-aware, producer-agnostic)
# ---------------------------------------------------------------------------

def extract_vortex_set(tokens: list[dict]) -> list[VortexRecord]:
    """Extract VortexRecords from schema-valid tokens.

    Reads only ``token_type == "vortex"`` entries.  Uses only the
    schema-defined fields: id, x, y, charge, strength, confidence.
    """
    vortices: list[VortexRecord] = []
    for tok in tokens:
        if tok.get("token_type") != "vortex":
            continue
        v = tok["vortex"]
        vortices.append(VortexRecord(
            id=v["id"],
            x=v["x"],
            y=v["y"],
            charge=v["charge"],
            strength=v["strength"],
            confidence=v["confidence"],
        ))
    return vortices


def extract_pair_set(tokens: list[dict]) -> list[PairRecord]:
    """Extract PairRecords from schema-valid tokens.

    Reads only ``token_type == "pair"`` entries.
    """
    pairs: list[PairRecord] = []
    for tok in tokens:
        if tok.get("token_type") != "pair":
            continue
        p = tok["pair"]
        pairs.append(PairRecord(
            pair_id=p["pair_id"],
            vortex_id=p["vortex_id"],
            antivortex_id=p["antivortex_id"],
            separation_r=p["separation_r"],
        ))
    return pairs


# ---------------------------------------------------------------------------
# Spatial matching (minimum-image distance, same-charge only)
# ---------------------------------------------------------------------------

def _minimum_image_distance(
    x1: float, y1: float, x2: float, y2: float, L: int,
) -> float:
    """Minimum-image PBC distance on an L×L lattice."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    rx = min(dx, L - dx)
    ry = min(dy, L - dy)
    return math.sqrt(rx * rx + ry * ry)


def match_vortex_sets(
    ref: list[VortexRecord],
    cand: list[VortexRecord],
    L: int,
    tol: float = 1.5,
) -> MatchResult:
    """Match reference vortices to candidate vortices by charge and proximity.

    Algorithm:
    1. Group vortices by charge (+1 and -1 separately).
    2. For each reference vortex, find the nearest unmatched candidate vortex
       of the same charge within ``tol`` lattice units (minimum-image distance).
    3. Greedy nearest-first matching (simple, conservative, documented).

    Parameters
    ----------
    ref : list[VortexRecord]
        Reference vortex set (from clean simulation).
    cand : list[VortexRecord]
        Candidate vortex set (from degraded pipeline).
    L : int
        Lattice side length for PBC distance.
    tol : float
        Maximum distance for a match, in lattice units.  Default 1.5
        (just over the diagonal of one plaquette — allows for small
        position shifts from degradation but rejects spurious far-field
        matches).

    Returns
    -------
    MatchResult
    """
    matched_distances: list[float] = []
    total_matched = 0

    for charge_val in (+1, -1):
        ref_c = [v for v in ref if v.charge == charge_val]
        cand_c = [v for v in cand if v.charge == charge_val]
        used_cand: set[int] = set()

        # Build all distances, sort by distance, assign greedily.
        pairs: list[tuple[float, int, int]] = []
        for i, rv in enumerate(ref_c):
            for j, cv in enumerate(cand_c):
                d = _minimum_image_distance(rv.x, rv.y, cv.x, cv.y, L)
                if d <= tol:
                    pairs.append((d, i, j))

        pairs.sort()  # nearest first

        used_ref: set[int] = set()
        for d, i, j in pairs:
            if i in used_ref or j in used_cand:
                continue
            used_ref.add(i)
            used_cand.add(j)
            matched_distances.append(d)
            total_matched += 1

    n_ref = len(ref)
    n_cand = len(cand)

    return MatchResult(
        n_ref=n_ref,
        n_cand=n_cand,
        n_matched=total_matched,
        recall=total_matched / n_ref if n_ref > 0 else 0.0,
        precision=total_matched / n_cand if n_cand > 0 else 0.0,
        false_positive_rate=(
            (n_cand - total_matched) / n_cand if n_cand > 0 else 0.0
        ),
        matched_distances=matched_distances,
    )


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------

def compare_token_streams(
    ref_path: str | Path,
    cand_path: str | Path,
    L: int,
    condition_label: str = "",
    match_tol: float = 1.5,
) -> ComparisonResult:
    """Compare two token JSONL files: a reference and a candidate.

    This is the main entry point for downstream benchmarking.
    Reads ONLY token JSONL files.  No raw arrays, no theta fields.

    Parameters
    ----------
    ref_path : path
        JSONL file from the reference producer (e.g. clean simulation).
    cand_path : path
        JSONL file from the candidate producer (e.g. degraded map-mode).
    L : int
        Lattice side length for PBC distance.
    condition_label : str
        Human-readable label for the degradation condition.
    match_tol : float
        Vortex matching tolerance in lattice units.

    Returns
    -------
    ComparisonResult
    """
    ref_tokens = load_tokens(ref_path)
    cand_tokens = load_tokens(cand_path)

    ref_vortices = extract_vortex_set(ref_tokens)
    cand_vortices = extract_vortex_set(cand_tokens)

    ref_pairs = extract_pair_set(ref_tokens)
    cand_pairs = extract_pair_set(cand_tokens)

    # Vortex matching.
    match = match_vortex_sets(ref_vortices, cand_vortices, L, tol=match_tol)

    # Paired fraction.
    n_ref_defects = len(ref_vortices)
    n_cand_defects = len(cand_vortices)

    paired_frac_ref = (
        (2 * len(ref_pairs)) / n_ref_defects if n_ref_defects > 0 else 0.0
    )
    paired_frac_cand = (
        (2 * len(cand_pairs)) / n_cand_defects if n_cand_defects > 0 else 0.0
    )

    # Mean pair separation.
    mean_sep_ref = (
        sum(p.separation_r for p in ref_pairs) / len(ref_pairs)
        if ref_pairs else 0.0
    )
    mean_sep_cand = (
        sum(p.separation_r for p in cand_pairs) / len(cand_pairs)
        if cand_pairs else 0.0
    )

    # Confidence summary (may not be present for all producers).
    ref_confs = [v.confidence for v in ref_vortices if v.confidence is not None]
    cand_confs = [v.confidence for v in cand_vortices if v.confidence is not None]

    return ComparisonResult(
        ref_path=str(ref_path),
        cand_path=str(cand_path),
        condition_label=condition_label,
        match=match,
        paired_fraction_ref=paired_frac_ref,
        paired_fraction_cand=paired_frac_cand,
        paired_fraction_error=abs(paired_frac_cand - paired_frac_ref),
        mean_separation_ref=mean_sep_ref,
        mean_separation_cand=mean_sep_cand,
        mean_separation_error=abs(mean_sep_cand - mean_sep_ref),
        mean_confidence_ref=(
            sum(ref_confs) / len(ref_confs) if ref_confs else None
        ),
        mean_confidence_cand=(
            sum(cand_confs) / len(cand_confs) if cand_confs else None
        ),
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def result_to_dict(r: ComparisonResult) -> dict[str, Any]:
    """Convert a ComparisonResult to a JSON-serialisable dict."""
    return {
        "condition_label": r.condition_label,
        "ref_vortex_count": r.match.n_ref,
        "cand_vortex_count": r.match.n_cand,
        "matched_vortex_count": r.match.n_matched,
        "recall": round(r.match.recall, 6),
        "precision": round(r.match.precision, 6),
        "false_positive_rate": round(r.match.false_positive_rate, 6),
        "paired_fraction_ref": round(r.paired_fraction_ref, 6),
        "paired_fraction_cand": round(r.paired_fraction_cand, 6),
        "paired_fraction_error": round(r.paired_fraction_error, 6),
        "mean_separation_ref": round(r.mean_separation_ref, 4),
        "mean_separation_cand": round(r.mean_separation_cand, 4),
        "mean_separation_error": round(r.mean_separation_error, 4),
        "mean_confidence_ref": (
            round(r.mean_confidence_ref, 4) if r.mean_confidence_ref is not None else None
        ),
        "mean_confidence_cand": (
            round(r.mean_confidence_cand, 4) if r.mean_confidence_cand is not None else None
        ),
        "match_tolerance_used": 1.5,
    }


def format_results_table(results: list[ComparisonResult]) -> str:
    """Format a list of ComparisonResults as a human-readable table."""
    header = (
        f"{'Condition':<25} {'Ref':>4} {'Cand':>4} {'Match':>5} "
        f"{'Recall':>7} {'Prec':>7} {'FPR':>7} "
        f"{'PF_ref':>7} {'PF_cnd':>7} {'PF_err':>7} "
        f"{'Sep_r':>6} {'Sep_c':>6} {'Sep_e':>6}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        m = r.match
        lines.append(
            f"{r.condition_label:<25} {m.n_ref:>4} {m.n_cand:>4} {m.n_matched:>5} "
            f"{m.recall:>7.3f} {m.precision:>7.3f} {m.false_positive_rate:>7.3f} "
            f"{r.paired_fraction_ref:>7.3f} {r.paired_fraction_cand:>7.3f} "
            f"{r.paired_fraction_error:>7.3f} "
            f"{r.mean_separation_ref:>6.2f} {r.mean_separation_cand:>6.2f} "
            f"{r.mean_separation_error:>6.2f}"
        )
    return "\n".join(lines)
