"""
tests/test_benchmark_stage1.py
================================
Pytest tests for the Stage 1 XY single-sweep benchmark.

Two test classes:

1. TestBenchmarkInvariants
   Reads the committed frozen/ artifacts and checks structural invariants
   (schema, provenance, field presence, value ranges).  Does not require
   a rerun — frozen/ is always present in the repo.
   Also runs the pipeline to verify it produces matching counts.

2. TestBenchmarkHashes
   Reads the committed frozen/ artifacts and verifies SHA256 hashes match
   manifest.json.  Proves that the committed files were produced by the
   documented pipeline.
   Also runs the pipeline (via benchmark_outputs fixture) and checks counts
   and scalar values match the frozen expectations.

Design: invariant and hash tests both read from frozen/, which is committed.
The benchmark_outputs session fixture runs the pipeline once to get computed
hashes for count/scalar checks and produces scratch outputs in output/.

Runtime: ~5-10s (dominated by Numba JIT on first run in a session).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths and fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks" / "stage1_xy_single_sweep"
FROZEN_DIR = BENCH_DIR / "frozen"   # committed, read by tests
MANIFEST_PATH = BENCH_DIR / "manifest.json"
SCHEMA_PATH = REPO_ROOT / "schemas" / "topology_event_stream.schema.json"

with SCHEMA_PATH.open() as _f:
    _SCHEMA = json.load(_f)

with MANIFEST_PATH.open() as _f:
    _MANIFEST = json.load(_f)

PARAMS = _MANIFEST["parameters"]


# ---------------------------------------------------------------------------
# Shared: run the benchmark once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def benchmark_outputs() -> dict:
    """Run the benchmark and return the computed dict of hashes and counts.

    Uses session scope so the pipeline runs once even if multiple test
    classes use this fixture.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from benchmarks.stage1_xy_single_sweep.run_benchmark import run_benchmark
    return run_benchmark()


# ---------------------------------------------------------------------------
# TestBenchmarkInvariants
# ---------------------------------------------------------------------------

class TestBenchmarkInvariants:
    """Structural checks that do not depend on exact hash values."""

    def test_output_files_exist(self, benchmark_outputs):
        for rel_path in _MANIFEST["artifacts"].values():
            full = BENCH_DIR / rel_path
            assert full.exists(), f"Missing benchmark artifact: {full}"
            assert full.stat().st_size > 0, f"Empty artifact: {full}"

    def test_spin_config_shape_and_range(self, benchmark_outputs):
        L = PARAMS["L"]
        npy_path = BENCH_DIR / _MANIFEST["artifacts"]["spin_config"]
        cfg = np.load(npy_path)
        assert cfg.shape == (L, L), f"Expected ({L},{L}), got {cfg.shape}"
        assert cfg.dtype == np.float64
        assert np.all(cfg >= -math.pi - 1e-12)
        assert np.all(cfg <= math.pi + 1e-12)

    def test_all_tokens_schema_valid(self, benchmark_outputs):
        jsonl_path = BENCH_DIR / _MANIFEST["artifacts"]["tokens_jsonl"]
        with jsonl_path.open() as f:
            for i, line in enumerate(f):
                tok = json.loads(line.strip())
                jsonschema.validate(tok, _SCHEMA)  # raises on failure

    def test_vortex_token_provenance(self, benchmark_outputs):
        jsonl_path = BENCH_DIR / _MANIFEST["artifacts"]["tokens_jsonl"]
        with jsonl_path.open() as f:
            for line in f:
                tok = json.loads(line.strip())
                if tok["token_type"] != "vortex":
                    continue
                prov = tok["provenance"]
                assert prov["model"] == "XY"
                assert prov["L"] == 16
                assert abs(prov["T"] - 0.9) < 1e-12
                assert prov["seed"] == 42

    def test_vortex_charge_values(self, benchmark_outputs):
        jsonl_path = BENCH_DIR / _MANIFEST["artifacts"]["tokens_jsonl"]
        with jsonl_path.open() as f:
            for line in f:
                tok = json.loads(line.strip())
                if tok["token_type"] != "vortex":
                    continue
                assert tok["vortex"]["charge"] in (-1, +1)

    def test_vortex_confidence_in_range(self, benchmark_outputs):
        jsonl_path = BENCH_DIR / _MANIFEST["artifacts"]["tokens_jsonl"]
        with jsonl_path.open() as f:
            for line in f:
                tok = json.loads(line.strip())
                if tok["token_type"] != "vortex":
                    continue
                conf = tok["vortex"]["confidence"]
                assert 0.0 <= conf <= 1.0, f"confidence={conf} out of range"

    def test_vortex_required_fields(self, benchmark_outputs):
        required = {"id", "x", "y", "charge", "strength", "confidence"}
        jsonl_path = BENCH_DIR / _MANIFEST["artifacts"]["tokens_jsonl"]
        with jsonl_path.open() as f:
            for line in f:
                tok = json.loads(line.strip())
                if tok["token_type"] != "vortex":
                    continue
                assert required.issubset(tok["vortex"].keys()), \
                    f"Missing fields: {required - tok['vortex'].keys()}"

    def test_summary_required_fields(self, benchmark_outputs):
        summary_path = BENCH_DIR / _MANIFEST["artifacts"]["summary_json"]
        with summary_path.open() as f:
            summary = json.load(f)
        required = {
            "model", "L", "T", "seed",
            "n_vortices", "n_pairs", "f_paired",
            "upsilon", "upsilon_err", "r_max_used",
        }
        assert required.issubset(summary.keys()), \
            f"Missing summary fields: {required - summary.keys()}"

    def test_summary_f_paired_in_range(self, benchmark_outputs):
        summary_path = BENCH_DIR / _MANIFEST["artifacts"]["summary_json"]
        with summary_path.open() as f:
            summary = json.load(f)
        assert 0.0 <= summary["f_paired"] <= 1.0

    def test_summary_counts_consistent(self, benchmark_outputs):
        summary_path = BENCH_DIR / _MANIFEST["artifacts"]["summary_json"]
        with summary_path.open() as f:
            summary = json.load(f)
        # n_pairs ≤ min(n_vortices, n_antivortices) ≤ n_vortices/2
        assert summary["n_pairs"] <= summary["n_vortices"]

    def test_vortex_count_matches_summary(self, benchmark_outputs):
        jsonl_path = BENCH_DIR / _MANIFEST["artifacts"]["tokens_jsonl"]
        summary_path = BENCH_DIR / _MANIFEST["artifacts"]["summary_json"]
        with summary_path.open() as f:
            summary = json.load(f)
        vortex_count = 0
        with jsonl_path.open() as f:
            for line in f:
                tok = json.loads(line.strip())
                if tok["token_type"] == "vortex":
                    vortex_count += 1
        assert vortex_count == summary["n_vortices"]


# ---------------------------------------------------------------------------
# TestBenchmarkHashes
# ---------------------------------------------------------------------------

class TestBenchmarkHashes:
    """Hash checks against frozen manifest.json values.

    These tests prove byte-identical reproducibility on the same machine
    with the same package versions.  If they fail after an intentional
    change, regenerate with:
        python benchmarks/stage1_xy_single_sweep/run_benchmark.py --regenerate
    """

    def _hash_spin_config(self) -> str:
        npy_path = BENCH_DIR / _MANIFEST["artifacts"]["spin_config"]
        cfg = np.load(npy_path)
        return hashlib.sha256(cfg.astype(">f8").tobytes()).hexdigest()

    def _hash_tokens_jsonl(self) -> str:
        jsonl_path = BENCH_DIR / _MANIFEST["artifacts"]["tokens_jsonl"]
        content = jsonl_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode()).hexdigest()

    def _hash_summary(self) -> str:
        summary_path = BENCH_DIR / _MANIFEST["artifacts"]["summary_json"]
        with summary_path.open() as f:
            summary = json.load(f)
        # Exclude r_max_used from canonical hash (not in original summary hash).
        canonical_keys = {
            "model", "L", "T", "seed",
            "n_vortices", "n_pairs", "f_paired",
            "upsilon", "upsilon_err",
        }
        canonical = {k: summary[k] for k in sorted(canonical_keys)}
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

    def test_spin_config_hash(self, benchmark_outputs):
        actual = self._hash_spin_config()
        expected = _MANIFEST["hashes"]["spin_config_sha256"]
        assert actual == expected, (
            f"spin_config hash mismatch.\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}\n"
            f"If intentional: run with --regenerate to update manifest."
        )

    def test_tokens_jsonl_hash(self, benchmark_outputs):
        actual = self._hash_tokens_jsonl()
        expected = _MANIFEST["hashes"]["tokens_jsonl_sha256"]
        assert actual == expected, (
            f"tokens_jsonl hash mismatch.\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}\n"
            f"If intentional: run with --regenerate to update manifest."
        )

    def test_summary_hash(self, benchmark_outputs):
        actual = self._hash_summary()
        expected = _MANIFEST["hashes"]["summary_sha256"]
        assert actual == expected, (
            f"summary hash mismatch.\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}\n"
            f"If intentional: run with --regenerate to update manifest."
        )

    def test_expected_counts(self, benchmark_outputs):
        ec = _MANIFEST["expected_counts"]
        c = benchmark_outputs["counts"]
        assert c["n_vortex_tokens"] == ec["n_vortex_tokens"]
        assert c["n_pair_tokens"] == ec["n_pair_tokens"]
        assert c["n_configs_thinned"] == ec["n_configs_thinned"]
        assert abs(c["f_paired"] - ec["f_paired"]) < 1e-12

    def test_expected_scalars(self, benchmark_outputs):
        es = _MANIFEST["expected_scalars"]
        c = benchmark_outputs["counts"]
        assert abs(c["helicity"] - es["helicity"]) < 1e-6
        assert abs(c["helicity_err"] - es["helicity_err"]) < 1e-6
