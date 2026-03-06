"""
tests/test_aggregate_confidence.py
====================================
Tests for multi-seed vortex aggregation (SPEC_UQ §3).

Covers:
  - Perfect repeatability across seeds → confidence 1.0
  - One missed detection across 4 seeds → confidence 0.75
  - Opposite charges do not merge
  - Periodic wrap matching works across boundaries
  - global_detection_stability matches expected formula
  - No schema breakage for updated token outputs
  - Edge cases: single seed, zero vortices
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from topostream.aggregate.confidence import (
    compute_condition_aggregate,
    compute_per_vortex_confidence,
    minimum_image_distance,
    aggregate_results_dir,
    DEFAULT_MATCH_TOLERANCE,
)

# ---------------------------------------------------------------------------
# Schema fixture
# ---------------------------------------------------------------------------

SCHEMA_PATH = (
    Path(__file__).parent.parent / "schemas" / "topology_event_stream.schema.json"
)

with SCHEMA_PATH.open() as _f:
    _SCHEMA = json.load(_f)


def _prov(
    model: str = "XY", L: int = 16, T: float = 0.9, seed: int = 42,
    sweep_index: int = 0,
) -> dict:
    return {
        "model": model,
        "L": L,
        "T": T,
        "seed": seed,
        "sweep_index": sweep_index,
        "schema_version": "1.1.0",
    }


def _make_vortex_token(
    x: float, y: float, charge: int, seed: int,
    model: str = "XY", L: int = 16, T: float = 0.9,
    strength: float = 1.0, confidence: float = 1.0,
) -> dict:
    """Build a minimal schema-compliant vortex token."""
    sign_char = "p" if charge > 0 else "m"
    row = int(y - 0.5) if y >= 0.5 else 0
    col = int(x - 0.5) if x >= 0.5 else 0
    vortex_id = f"v_{sign_char}_r{row:03d}_c{col:03d}"
    return {
        "schema_version": "1.1.0",
        "token_type": "vortex",
        "provenance": _prov(model=model, L=L, T=T, seed=seed),
        "vortex": {
            "id": vortex_id,
            "x": float(x),
            "y": float(y),
            "charge": int(charge),
            "strength": float(strength),
            "confidence": float(confidence),
        },
    }


# ---------------------------------------------------------------------------
# Tests: minimum_image_distance (PBC)
# ---------------------------------------------------------------------------

class TestMinimumImageDistance:
    """PBC distance function used for vortex matching."""

    def test_same_point(self):
        assert minimum_image_distance(5.5, 5.5, 5.5, 5.5, 16) == 0.0

    def test_simple_distance(self):
        d = minimum_image_distance(1.5, 1.5, 3.5, 1.5, 16)
        assert abs(d - 2.0) < 1e-12

    def test_periodic_wrap_x(self):
        """Point near right edge and point near left edge should be close."""
        L = 16
        d = minimum_image_distance(0.5, 5.5, 15.5, 5.5, L)
        # Direct distance: 15.0. Wrapped: 16 - 15 = 1.0
        assert abs(d - 1.0) < 1e-12

    def test_periodic_wrap_y(self):
        """Point near top edge and point near bottom edge should be close."""
        L = 16
        d = minimum_image_distance(5.5, 0.5, 5.5, 15.5, L)
        assert abs(d - 1.0) < 1e-12

    def test_periodic_wrap_diagonal(self):
        """Both axes wrap simultaneously."""
        L = 16
        d = minimum_image_distance(0.5, 0.5, 15.5, 15.5, L)
        expected = math.sqrt(1.0 + 1.0)  # sqrt(2)
        assert abs(d - expected) < 1e-12

    def test_midpoint_not_wrapped(self):
        """Distance within L/2 should not be wrapped."""
        L = 16
        d = minimum_image_distance(3.5, 3.5, 7.5, 3.5, L)
        assert abs(d - 4.0) < 1e-12


# ---------------------------------------------------------------------------
# Tests: Perfect repeatability → confidence 1.0
# ---------------------------------------------------------------------------

class TestPerfectRepeatability:
    """When all seeds detect the same vortex at the same location,
    confidence must be 1.0 and stability must be 1.0."""

    def test_all_seeds_identical(self):
        L = 16
        # 4 seeds, each detects exactly one vortex at (5.5, 5.5)
        seeds_dict = {}
        for seed in [42, 43, 44, 45]:
            seeds_dict[seed] = [
                _make_vortex_token(5.5, 5.5, +1, seed, L=L),
            ]

        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=L, T=0.9,
        )

        assert summary["N_seeds"] == 4
        assert summary["mu_vortex_count"] == 1.0
        assert summary["sigma_vortex_count"] == 0.0
        assert summary["global_detection_stability"] == 1.0
        assert summary["n_consensus_clusters"] == 1

        # The single consensus vortex should have confidence = 1.0
        cv = summary["consensus_vortices"]
        assert len(cv) == 1
        assert cv[0]["confidence"] == 1.0
        assert cv[0]["charge"] == +1

    def test_multiple_vortices_identical(self):
        """Multiple vortices, all detected identically across seeds."""
        L = 16
        seeds_dict = {}
        for seed in [42, 43, 44, 45]:
            seeds_dict[seed] = [
                _make_vortex_token(3.5, 3.5, +1, seed, L=L),
                _make_vortex_token(10.5, 10.5, -1, seed, L=L),
            ]

        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=L, T=0.9,
        )

        assert summary["mu_vortex_count"] == 2.0
        assert summary["sigma_vortex_count"] == 0.0
        assert summary["global_detection_stability"] == 1.0
        assert summary["n_consensus_clusters"] == 2

        for cv in summary["consensus_vortices"]:
            assert cv["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Tests: One missed detection → confidence 0.75
# ---------------------------------------------------------------------------

class TestPartialDetection:
    """When one of 4 seeds misses a vortex, confidence = 3/4 = 0.75."""

    def test_one_missed_of_four(self):
        L = 16
        seeds_dict = {
            42: [_make_vortex_token(5.5, 5.5, +1, 42, L=L)],
            43: [_make_vortex_token(5.5, 5.5, +1, 43, L=L)],
            44: [_make_vortex_token(5.5, 5.5, +1, 44, L=L)],
            45: [],  # seed 45 misses the vortex
        }

        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=L, T=0.9,
        )

        assert summary["N_seeds"] == 4
        # mu = (1+1+1+0)/4 = 0.75, sigma = std([1,1,1,0]) = sqrt(3/16)
        assert summary["n_consensus_clusters"] == 1

        cv = summary["consensus_vortices"]
        assert len(cv) == 1
        assert cv[0]["confidence"] == 0.75
        assert cv[0]["n_detections"] == 3

    def test_two_missed_of_four(self):
        """Two of four seeds miss → confidence = 0.5."""
        L = 16
        seeds_dict = {
            42: [_make_vortex_token(5.5, 5.5, +1, 42, L=L)],
            43: [_make_vortex_token(5.5, 5.5, +1, 43, L=L)],
            44: [],
            45: [],
        }

        consensus = compute_per_vortex_confidence(
            seeds_dict, L=L,
        )

        assert len(consensus) == 1
        assert consensus[0]["confidence"] == 0.5


# ---------------------------------------------------------------------------
# Tests: Opposite charges do not merge
# ---------------------------------------------------------------------------

class TestChargeIsolation:
    """Vortex (+1) and antivortex (-1) at the same location must NOT merge."""

    def test_same_position_different_charge(self):
        L = 16
        seeds_dict = {
            42: [
                _make_vortex_token(5.5, 5.5, +1, 42, L=L),
                _make_vortex_token(5.5, 5.5, -1, 42, L=L),
            ],
            43: [
                _make_vortex_token(5.5, 5.5, +1, 43, L=L),
                _make_vortex_token(5.5, 5.5, -1, 43, L=L),
            ],
        }

        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=L, T=0.9,
        )

        # Must get 2 clusters: one for +1, one for -1.
        assert summary["n_consensus_clusters"] == 2
        charges = sorted(c["charge"] for c in summary["consensus_vortices"])
        assert charges == [-1, +1]

        for cv in summary["consensus_vortices"]:
            assert cv["confidence"] == 1.0

    def test_only_positive_present_in_some_seeds(self):
        """Mix: seed 42 has +1 and -1; seed 43 has only +1.
        The +1 cluster: confidence 1.0 (both seeds).
        The -1 cluster: confidence 0.5 (only seed 42)."""
        L = 16
        seeds_dict = {
            42: [
                _make_vortex_token(5.5, 5.5, +1, 42, L=L),
                _make_vortex_token(5.5, 5.5, -1, 42, L=L),
            ],
            43: [
                _make_vortex_token(5.5, 5.5, +1, 43, L=L),
            ],
        }

        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=L, T=0.9,
        )

        assert summary["n_consensus_clusters"] == 2
        by_charge = {c["charge"]: c for c in summary["consensus_vortices"]}
        assert by_charge[+1]["confidence"] == 1.0
        assert by_charge[-1]["confidence"] == 0.5


# ---------------------------------------------------------------------------
# Tests: Periodic boundary wrap matching
# ---------------------------------------------------------------------------

class TestPBCWrapMatching:
    """Vortices near opposite edges should match across boundaries."""

    def test_vortex_match_across_x_boundary(self):
        L = 16
        # Seed 42: vortex at x=0.5, seed 43: vortex at x=15.5
        # These are 1.0 apart under PBC, within default tolerance.
        seeds_dict = {
            42: [_make_vortex_token(0.5, 5.5, +1, 42, L=L)],
            43: [_make_vortex_token(15.5, 5.5, +1, 43, L=L)],
        }

        consensus = compute_per_vortex_confidence(
            seeds_dict, L=L, match_tolerance=1.0,
        )

        # Should merge into one cluster with confidence 1.0
        assert len(consensus) == 1
        assert consensus[0]["confidence"] == 1.0

    def test_vortex_match_across_y_boundary(self):
        L = 16
        seeds_dict = {
            42: [_make_vortex_token(5.5, 0.5, -1, 42, L=L)],
            43: [_make_vortex_token(5.5, 15.5, -1, 43, L=L)],
        }

        consensus = compute_per_vortex_confidence(
            seeds_dict, L=L, match_tolerance=1.0,
        )

        assert len(consensus) == 1
        assert consensus[0]["confidence"] == 1.0

    def test_vortex_no_match_when_too_far(self):
        """Vortices that are more than tolerance apart should not match."""
        L = 16
        seeds_dict = {
            42: [_make_vortex_token(0.5, 5.5, +1, 42, L=L)],
            43: [_make_vortex_token(5.5, 5.5, +1, 43, L=L)],
        }

        consensus = compute_per_vortex_confidence(
            seeds_dict, L=L, match_tolerance=1.0,
        )

        # 5.0 apart — should NOT merge. Each has confidence 0.5.
        assert len(consensus) == 2
        for c in consensus:
            assert c["confidence"] == 0.5


# ---------------------------------------------------------------------------
# Tests: global_detection_stability formula
# ---------------------------------------------------------------------------

class TestGlobalDetectionStability:
    """Verify the SPEC_UQ §3 formula:
    global_detection_stability = clip(1 − σ_count / max(μ_count, ε), 0, 1)
    """

    def test_all_identical_counts(self):
        """σ = 0 → stability = 1.0."""
        seeds_dict = {
            s: [_make_vortex_token(5.5, 5.5, +1, s)]
            for s in [42, 43, 44, 45]
        }
        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=16, T=0.9,
        )
        assert summary["global_detection_stability"] == 1.0

    def test_high_variance(self):
        """Counts = [0, 0, 0, 4] → μ=1, σ=sqrt(3) ≈ 1.732 → stability = clip(1-1.732, 0, 1) = 0."""
        seeds_dict = {
            42: [],
            43: [],
            44: [],
            45: [_make_vortex_token(i+0.5, 0.5, +1, 45) for i in range(4)],
        }
        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=16, T=0.9,
        )
        # counts = [0, 0, 0, 4], mu=1.0, sigma=sqrt(3) ≈ 1.732
        # stability = clip(1 - 1.732/1.0, 0, 1) = 0.0
        assert summary["global_detection_stability"] == 0.0

    def test_moderate_variance(self):
        """Counts = [2, 2, 2, 0] → μ=1.5, σ=std(...) → check formula."""
        L = 16
        seeds_dict = {
            42: [_make_vortex_token(1.5, 1.5, +1, 42, L=L),
                 _make_vortex_token(3.5, 3.5, -1, 42, L=L)],
            43: [_make_vortex_token(1.5, 1.5, +1, 43, L=L),
                 _make_vortex_token(3.5, 3.5, -1, 43, L=L)],
            44: [_make_vortex_token(1.5, 1.5, +1, 44, L=L),
                 _make_vortex_token(3.5, 3.5, -1, 44, L=L)],
            45: [],  # seed 45 detects nothing
        }
        counts = np.array([2, 2, 2, 0], dtype=np.float64)
        mu = float(np.mean(counts))
        sigma = float(np.std(counts, ddof=0))
        expected = float(np.clip(1.0 - sigma / max(mu, 1e-12), 0.0, 1.0))

        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=L, T=0.9,
        )

        assert abs(summary["mu_vortex_count"] - mu) < 1e-12
        assert abs(summary["sigma_vortex_count"] - sigma) < 1e-12
        assert abs(summary["global_detection_stability"] - expected) < 1e-12

    def test_zero_vortices_all_seeds(self):
        """All seeds produce 0 vortices → μ=0, σ=0 → stability = 1.0
        (1 - 0/eps = 1 - 0 = 1, clipped to [0,1])."""
        seeds_dict = {42: [], 43: [], 44: [], 45: []}
        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=16, T=0.9,
        )
        assert summary["mu_vortex_count"] == 0.0
        assert summary["sigma_vortex_count"] == 0.0
        assert summary["global_detection_stability"] == 1.0
        assert summary["n_consensus_clusters"] == 0


# ---------------------------------------------------------------------------
# Tests: Schema compliance after aggregation
# ---------------------------------------------------------------------------

class TestSchemaAfterAggregation:
    """Updated tokens must still validate against the schema."""

    def test_updated_tokens_pass_schema(self):
        """Write token files to a temp dir, run aggregation, then validate."""
        L = 16
        T = 0.9
        model = "XY"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write per-seed token files.
            for seed in [42, 43, 44, 45]:
                tokens = [
                    _make_vortex_token(5.5, 5.5, +1, seed, model=model, L=L, T=T),
                    _make_vortex_token(10.5, 10.5, -1, seed, model=model, L=L, T=T),
                ]
                fname = f"tokens_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
                with (tmpdir / fname).open("w") as f:
                    for tok in tokens:
                        f.write(json.dumps(tok) + "\n")

            # Run aggregation.
            results = aggregate_results_dir(tmpdir)
            assert len(results) == 1

            # Re-read aggregated token files and validate against schema.
            for seed in [42, 43, 44, 45]:
                agg_fname = f"tokens_aggregated_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
                assert (tmpdir / agg_fname).exists(), f"Missing {agg_fname}"
                with (tmpdir / agg_fname).open() as f:
                    for line in f:
                        tok = json.loads(line.strip())
                        jsonschema.validate(tok, _SCHEMA)
                        if tok["token_type"] == "vortex":
                            # Confidence should now be 1.0 (all seeds identical).
                            assert tok["vortex"]["confidence"] == 1.0

            # Originals must still exist unchanged.
            for seed in [42, 43, 44, 45]:
                orig_fname = f"tokens_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
                assert (tmpdir / orig_fname).exists()

    def test_aggregate_artifact_written(self):
        """Verify the aggregate JSON artifact is created."""
        L = 16
        T = 0.9
        model = "XY"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for seed in [42, 43]:
                tokens = [
                    _make_vortex_token(5.5, 5.5, +1, seed, model=model, L=L, T=T),
                ]
                fname = f"tokens_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
                with (tmpdir / fname).open("w") as f:
                    for tok in tokens:
                        f.write(json.dumps(tok) + "\n")

            aggregate_results_dir(tmpdir)

            agg_file = tmpdir / f"aggregate_{model}_{L}x{L}_T{T:.4f}.json"
            assert agg_file.exists()

            with agg_file.open() as f:
                agg = json.load(f)

            # Check required fields.
            assert agg["model"] == model
            assert agg["L"] == L
            assert agg["T"] == T
            assert agg["N_seeds"] == 2
            assert "mu_vortex_count" in agg
            assert "sigma_vortex_count" in agg
            assert "global_detection_stability" in agg
            assert "n_consensus_clusters" in agg


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_seed(self):
        """With one seed, confidence = 1.0 and stability = 1.0.
        This is a DEGENERATE case: N_seeds=1 provides no cross-seed
        evidence, so confidence=1.0 means 'not contradicted', not
        'confirmed reliable'."""
        L = 16
        seeds_dict = {
            42: [_make_vortex_token(5.5, 5.5, +1, 42, L=L)],
        }
        summary = compute_condition_aggregate(
            seeds_dict, model="XY", L=L, T=0.9,
        )
        assert summary["N_seeds"] == 1
        assert summary["global_detection_stability"] == 1.0
        assert summary["n_consensus_clusters"] == 1
        assert summary["consensus_vortices"][0]["confidence"] == 1.0

    def test_slightly_offset_positions_within_tolerance(self):
        """Vortices at slightly different positions (< tolerance) should merge."""
        L = 16
        seeds_dict = {
            42: [_make_vortex_token(5.5, 5.5, +1, 42, L=L)],
            43: [_make_vortex_token(5.8, 5.3, +1, 43, L=L)],
            44: [_make_vortex_token(5.3, 5.7, +1, 44, L=L)],
        }

        consensus = compute_per_vortex_confidence(
            seeds_dict, L=L, match_tolerance=1.0,
        )

        # All within tolerance of each other → single cluster, confidence 1.0.
        assert len(consensus) == 1
        assert consensus[0]["confidence"] == 1.0

    def test_no_token_files(self):
        """Empty directory → empty result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = aggregate_results_dir(tmpdir)
            assert results == {}

    def test_non_vortex_tokens_preserved(self):
        """Pair tokens in the aggregated file should be passed through unchanged."""
        L = 16
        T = 0.9
        model = "XY"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            prov = _prov(model=model, L=L, T=T, seed=42)
            vortex_tok = _make_vortex_token(5.5, 5.5, +1, 42, model=model, L=L, T=T)
            pair_tok = {
                "schema_version": "1.1.0",
                "token_type": "pair",
                "provenance": prov,
                "pair": {
                    "pair_id": "pair_v_p_r005_c005_v_m_r010_c010",
                    "vortex_id": "v_p_r005_c005",
                    "antivortex_id": "v_m_r010_c010",
                    "separation_r": 5.0,
                    "r_max_used": 4.0,
                },
            }

            fname = f"tokens_{model}_{L}x{L}_T{T:.4f}_seed0042.jsonl"
            with (tmpdir / fname).open("w") as f:
                f.write(json.dumps(vortex_tok) + "\n")
                f.write(json.dumps(pair_tok) + "\n")

            aggregate_results_dir(tmpdir)

            # Re-read AGGREGATED file: pair token should be unchanged.
            agg_fname = f"tokens_aggregated_{model}_{L}x{L}_T{T:.4f}_seed0042.jsonl"
            with (tmpdir / agg_fname).open() as f:
                lines = [json.loads(l.strip()) for l in f if l.strip()]

            pair_lines = [l for l in lines if l["token_type"] == "pair"]
            assert len(pair_lines) == 1
            assert pair_lines[0]["pair"]["pair_id"] == pair_tok["pair"]["pair_id"]

            # Original file must still exist.
            assert (tmpdir / fname).exists()


# ---------------------------------------------------------------------------
# Tests: Order invariance
# ---------------------------------------------------------------------------

class TestOrderInvariance:
    """Confidence results must not depend on the order seeds are processed."""

    def test_permuted_seed_order_gives_same_confidence(self):
        """Build seeds_dict in two different key orders.
        Both must produce identical confidence and cluster count."""
        L = 16
        base = {
            42: [_make_vortex_token(5.5, 5.5, +1, 42, L=L)],
            43: [_make_vortex_token(5.5, 5.5, +1, 43, L=L)],
            44: [_make_vortex_token(5.8, 5.3, +1, 44, L=L)],
            45: [],
        }

        # Forward order.
        summary_fwd = compute_condition_aggregate(
            dict(sorted(base.items())),
            model="XY", L=L, T=0.9,
        )

        # Reverse order.
        summary_rev = compute_condition_aggregate(
            dict(sorted(base.items(), reverse=True)),
            model="XY", L=L, T=0.9,
        )

        assert summary_fwd["n_consensus_clusters"] == summary_rev["n_consensus_clusters"]
        assert summary_fwd["global_detection_stability"] == summary_rev["global_detection_stability"]

        for cv_f, cv_r in zip(
            sorted(summary_fwd["consensus_vortices"], key=lambda c: (c["charge"], c["x"])),
            sorted(summary_rev["consensus_vortices"], key=lambda c: (c["charge"], c["x"])),
        ):
            assert cv_f["confidence"] == cv_r["confidence"]
            assert cv_f["n_detections"] == cv_r["n_detections"]
            assert cv_f["charge"] == cv_r["charge"]
            assert abs(cv_f["x"] - cv_r["x"]) < 1e-12
            assert abs(cv_f["y"] - cv_r["y"]) < 1e-12

    def test_permuted_with_offset_positions(self):
        """Slightly offset positions, permuted seed order."""
        L = 16
        base = {
            42: [_make_vortex_token(5.5, 5.5, +1, 42, L=L)],
            43: [_make_vortex_token(5.7, 5.3, +1, 43, L=L)],
            44: [_make_vortex_token(5.3, 5.8, +1, 44, L=L)],
        }

        import itertools
        results = []
        for perm in itertools.permutations(base.keys()):
            d = {k: base[k] for k in perm}
            s = compute_condition_aggregate(d, model="XY", L=L, T=0.9)
            results.append(s)

        # All permutations must give the same confidence values.
        for s in results[1:]:
            assert s["n_consensus_clusters"] == results[0]["n_consensus_clusters"]
            for cv_a, cv_b in zip(
                sorted(s["consensus_vortices"], key=lambda c: c["charge"]),
                sorted(results[0]["consensus_vortices"], key=lambda c: c["charge"]),
            ):
                assert cv_a["confidence"] == cv_b["confidence"]


# ---------------------------------------------------------------------------
# Tests: Idempotence
# ---------------------------------------------------------------------------

class TestIdempotence:
    """Running aggregation twice must produce identical outputs."""

    def test_double_aggregation_idempotent(self):
        L = 16
        T = 0.9
        model = "XY"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write per-seed token files.
            for seed in [42, 43, 44, 45]:
                tokens = [
                    _make_vortex_token(5.5, 5.5, +1, seed, model=model, L=L, T=T),
                    _make_vortex_token(10.5, 10.5, -1, seed, model=model, L=L, T=T),
                ]
                if seed == 45:
                    tokens = [tokens[0]]  # seed 45 misses the antivortex
                fname = f"tokens_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
                with (tmpdir / fname).open("w") as f:
                    for tok in tokens:
                        f.write(json.dumps(tok) + "\n")

            # First aggregation.
            results_1 = aggregate_results_dir(tmpdir)

            # Read all outputs after first run.
            agg_file = tmpdir / f"aggregate_{model}_{L}x{L}_T{T:.4f}.json"
            with agg_file.open() as f:
                agg_1 = json.load(f)

            agg_tokens_1 = {}
            for seed in [42, 43, 44, 45]:
                agg_fname = f"tokens_aggregated_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
                with (tmpdir / agg_fname).open() as f:
                    agg_tokens_1[seed] = [json.loads(l.strip()) for l in f if l.strip()]

            # Second aggregation.
            results_2 = aggregate_results_dir(tmpdir)

            # Read all outputs after second run.
            with agg_file.open() as f:
                agg_2 = json.load(f)

            agg_tokens_2 = {}
            for seed in [42, 43, 44, 45]:
                agg_fname = f"tokens_aggregated_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
                with (tmpdir / agg_fname).open() as f:
                    agg_tokens_2[seed] = [json.loads(l.strip()) for l in f if l.strip()]

            # Aggregate JSON must be identical.
            assert agg_1 == agg_2

            # Aggregated token files must be identical.
            for seed in [42, 43, 44, 45]:
                assert agg_tokens_1[seed] == agg_tokens_2[seed]

            # Confidence values must match.
            key = (model, L, T)
            for cv_1, cv_2 in zip(
                sorted(results_1[key]["consensus_vortices"], key=lambda c: (c["charge"], c["x"])),
                sorted(results_2[key]["consensus_vortices"], key=lambda c: (c["charge"], c["x"])),
            ):
                assert cv_1["confidence"] == cv_2["confidence"]
                assert cv_1["n_detections"] == cv_2["n_detections"]
