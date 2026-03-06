"""
tests/test_token_benchmark.py
================================
Tests for the downstream token-only analysis module.

Validates:
1. Token loading and extraction produce correct VortexRecord/PairRecord sets.
2. Spatial matching:
   - exact match gives recall=1.0, precision=1.0
   - no-overlap gives recall=0.0
   - charge-mismatch is not spuriously matched
3. PBC distance is correctly computed.
4. compare_token_streams reads only JSONL files and produces valid results.
5. The consumer is producer-agnostic: identical tokens from different
   provenance ("XY" vs "map_mode") produce identical comparison results.

These tests do NOT touch raw theta fields, vector maps, or simulator internals.
They directly construct token dicts and JSONL files to test the consumer.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from topostream.analysis.token_benchmark import (
    ComparisonResult,
    MatchResult,
    PairRecord,
    VortexRecord,
    _minimum_image_distance,
    compare_token_streams,
    extract_pair_set,
    extract_vortex_set,
    load_tokens,
    match_vortex_sets,
    result_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vortex_token(x, y, charge, *, vid=None, confidence=1.0,
                       model="XY", L=16, T=0.9, seed=42):
    sign = "p" if charge > 0 else "m"
    vid = vid or f"v_{sign}_{x:.0f}_{y:.0f}"
    return {
        "schema_version": "1.1.0",
        "token_type": "vortex",
        "provenance": {
            "model": model, "L": L, "T": T, "seed": seed,
            "sweep_index": 0, "schema_version": "1.1.0",
        },
        "vortex": {
            "id": vid, "x": x, "y": y,
            "charge": charge, "strength": 1.0,
            "confidence": confidence,
        },
    }


def _make_pair_token(v_id, a_id, sep_r, *, model="XY", L=16, T=0.9, seed=42):
    return {
        "schema_version": "1.1.0",
        "token_type": "pair",
        "provenance": {
            "model": model, "L": L, "T": T, "seed": seed,
            "sweep_index": 0, "schema_version": "1.1.0",
        },
        "pair": {
            "pair_id": f"pair_{v_id}_{a_id}",
            "vortex_id": v_id,
            "antivortex_id": a_id,
            "separation_r": sep_r,
            "r_max_used": 4.0,
        },
    }


def _write_jsonl(tokens, path):
    with open(path, "w") as f:
        for tok in tokens:
            f.write(json.dumps(tok, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Tests: token loading
# ---------------------------------------------------------------------------

class TestTokenLoading:

    def test_load_tokens_reads_all_lines(self, tmp_path):
        tokens = [
            _make_vortex_token(0.5, 0.5, +1),
            _make_vortex_token(1.5, 1.5, -1),
        ]
        p = tmp_path / "tokens.jsonl"
        _write_jsonl(tokens, p)
        loaded = load_tokens(p)
        assert len(loaded) == 2
        assert loaded[0]["vortex"]["charge"] == +1

    def test_extract_vortex_set(self):
        tokens = [
            _make_vortex_token(0.5, 0.5, +1),
            _make_vortex_token(3.5, 3.5, -1),
            _make_pair_token("v1", "v2", 2.0),
        ]
        vortices = extract_vortex_set(tokens)
        assert len(vortices) == 2
        assert vortices[0].charge == +1
        assert vortices[1].charge == -1

    def test_extract_pair_set(self):
        tokens = [
            _make_vortex_token(0.5, 0.5, +1),
            _make_pair_token("vp", "vm", 1.5),
        ]
        pairs = extract_pair_set(tokens)
        assert len(pairs) == 1
        assert pairs[0].separation_r == 1.5


# ---------------------------------------------------------------------------
# Tests: spatial matching
# ---------------------------------------------------------------------------

class TestSpatialMatching:

    def test_pbc_distance_simple(self):
        d = _minimum_image_distance(0.5, 0.5, 1.5, 0.5, 16)
        assert abs(d - 1.0) < 1e-10

    def test_pbc_distance_wraps(self):
        # Distance across PBC boundary: should be 1.0, not 15.0
        d = _minimum_image_distance(0.5, 0.5, 15.5, 0.5, 16)
        assert abs(d - 1.0) < 1e-10

    def test_exact_match_recall_precision(self):
        """Identical vortex sets → recall=1.0, precision=1.0."""
        vortices = [
            VortexRecord("v1", 0.5, 0.5, +1, 1.0, 1.0),
            VortexRecord("v2", 4.5, 4.5, -1, 1.0, 1.0),
        ]
        result = match_vortex_sets(vortices, vortices, L=16, tol=1.5)
        assert result.recall == 1.0
        assert result.precision == 1.0
        assert result.n_matched == 2

    def test_no_overlap(self):
        """Non-overlapping vortex sets → recall=0, precision=0."""
        ref = [VortexRecord("v1", 0.5, 0.5, +1, 1.0, 1.0)]
        cand = [VortexRecord("v2", 8.5, 8.5, +1, 1.0, 1.0)]
        result = match_vortex_sets(ref, cand, L=16, tol=1.5)
        assert result.recall == 0.0
        assert result.precision == 0.0
        assert result.n_matched == 0

    def test_charge_mismatch_not_matched(self):
        """Same position but different charge → no match."""
        ref = [VortexRecord("v1", 0.5, 0.5, +1, 1.0, 1.0)]
        cand = [VortexRecord("v2", 0.5, 0.5, -1, 1.0, 1.0)]
        result = match_vortex_sets(ref, cand, L=16, tol=1.5)
        assert result.n_matched == 0
        assert result.recall == 0.0

    def test_partial_match(self):
        """Ref has 2 vortices, cand has 1 matching + 1 extra → recall=0.5."""
        ref = [
            VortexRecord("r1", 0.5, 0.5, +1, 1.0, 1.0),
            VortexRecord("r2", 8.5, 8.5, +1, 1.0, 1.0),
        ]
        cand = [
            VortexRecord("c1", 0.5, 0.5, +1, 1.0, 1.0),
            VortexRecord("c2", 5.5, 5.5, +1, 1.0, 1.0),  # not near any ref
        ]
        result = match_vortex_sets(ref, cand, L=16, tol=1.5)
        assert result.n_matched == 1
        assert result.recall == 0.5
        assert result.precision == 0.5

    def test_empty_sets(self):
        result = match_vortex_sets([], [], L=16, tol=1.5)
        assert result.n_matched == 0
        assert result.recall == 0.0
        assert result.precision == 0.0


# ---------------------------------------------------------------------------
# Tests: compare_token_streams
# ---------------------------------------------------------------------------

class TestCompareTokenStreams:

    def test_identical_streams_perfect_scores(self, tmp_path):
        """Same JSONL for ref and cand → recall=1, precision=1, PF_err=0."""
        tokens = [
            _make_vortex_token(0.5, 0.5, +1),
            _make_vortex_token(2.5, 2.5, -1),
            _make_pair_token("v_p_0_0", "v_m_2_2", 2.83),
        ]
        ref_path = tmp_path / "ref.jsonl"
        cand_path = tmp_path / "cand.jsonl"
        _write_jsonl(tokens, ref_path)
        _write_jsonl(tokens, cand_path)

        cr = compare_token_streams(ref_path, cand_path, L=16, condition_label="test")
        assert cr.match.recall == 1.0
        assert cr.match.precision == 1.0
        assert abs(cr.paired_fraction_error) < 1e-10
        assert abs(cr.mean_separation_error) < 1e-10

    def test_producer_agnostic(self, tmp_path):
        """Same vortex positions with different provenance model names
        produce identical comparison results.

        This is the core tokenisation claim: the consumer does not care
        which producer emitted the tokens.
        """
        ref_tokens = [
            _make_vortex_token(0.5, 0.5, +1, model="XY"),
            _make_vortex_token(4.5, 4.5, -1, model="XY"),
        ]
        cand_tokens_xy = [
            _make_vortex_token(0.5, 0.5, +1, model="XY"),
            _make_vortex_token(4.5, 4.5, -1, model="XY"),
        ]
        cand_tokens_map = [
            _make_vortex_token(0.5, 0.5, +1, model="map_mode"),
            _make_vortex_token(4.5, 4.5, -1, model="map_mode"),
        ]

        ref_path = tmp_path / "ref.jsonl"
        cand_xy = tmp_path / "cand_xy.jsonl"
        cand_map = tmp_path / "cand_map.jsonl"
        _write_jsonl(ref_tokens, ref_path)
        _write_jsonl(cand_tokens_xy, cand_xy)
        _write_jsonl(cand_tokens_map, cand_map)

        cr_xy = compare_token_streams(ref_path, cand_xy, L=16)
        cr_map = compare_token_streams(ref_path, cand_map, L=16)

        assert cr_xy.match.recall == cr_map.match.recall
        assert cr_xy.match.precision == cr_map.match.precision
        assert cr_xy.match.n_matched == cr_map.match.n_matched


# ---------------------------------------------------------------------------
# Tests: result formatting
# ---------------------------------------------------------------------------

class TestResultFormatting:

    def test_result_to_dict_keys(self):
        mr = MatchResult(
            n_ref=10, n_cand=12, n_matched=9,
            recall=0.9, precision=0.75,
            false_positive_rate=0.25,
        )
        cr = ComparisonResult(
            ref_path="ref.jsonl", cand_path="cand.jsonl",
            condition_label="test",
            match=mr,
            paired_fraction_ref=0.8,
            paired_fraction_cand=0.7,
            paired_fraction_error=0.1,
            mean_separation_ref=2.0,
            mean_separation_cand=2.3,
            mean_separation_error=0.3,
            mean_confidence_ref=1.0,
            mean_confidence_cand=1.0,
        )
        d = result_to_dict(cr)
        required_keys = {
            "condition_label", "ref_vortex_count", "cand_vortex_count",
            "matched_vortex_count", "recall", "precision",
            "false_positive_rate", "paired_fraction_ref",
            "paired_fraction_cand", "paired_fraction_error",
            "mean_separation_ref", "mean_separation_cand",
            "mean_separation_error", "mean_confidence_ref",
            "mean_confidence_cand", "match_tolerance_used",
        }
        assert required_keys.issubset(d.keys())
