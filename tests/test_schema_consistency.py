"""
tests/test_schema_consistency.py
==================================
SPEC_VALIDATION §5 — Schema consistency test.

Runs the real pipeline (simulator → vortex extraction → pairing) and
validates every emitted token against schemas/topology_event_stream.schema.json.

Checks:
  - Vortex tokens validate.
  - Pair tokens validate.
  - Provenance fields (seed, L, T, schema_version) present and correct.

Deterministic: all seeds fixed.

Run with:
    python -m pytest tests/test_schema_consistency.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from topostream.simulate.xy_numba import run_xy
from topostream.extract.vortices import extract_vortices
from topostream.extract.pairing import pair_vortices

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_PATH = (
    Path(__file__).parent.parent / "schemas" / "topology_event_stream.schema.json"
)

with SCHEMA_PATH.open() as _f:
    _SCHEMA = json.load(_f)


def _validate(token: dict) -> None:
    jsonschema.validate(token, _SCHEMA)


def _prov(L: int, T: float, seed: int = 42, sweep_index: int = 0) -> dict:
    return {
        "model": "XY",
        "L": L,
        "T": T,
        "seed": seed,
        "sweep_index": sweep_index,
        "schema_version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# §5 Schema consistency — full pipeline
# ---------------------------------------------------------------------------

class TestSchemaConsistencyFullPipeline:
    """Run XY pipeline end-to-end, validate every token against schema."""

    L = 16
    T = 0.8

    @pytest.fixture(scope="class")
    def pipeline_output(self):
        """Run MC → extract → pair once for the class."""
        result = run_xy(L=self.L, T=self.T, N_equil=100, N_meas=200,
                        N_thin=50, seed=42)
        cfg = result["configs"][-1]
        prov = _prov(L=self.L, T=self.T)

        vortex_tokens = extract_vortices(cfg, prov)

        vortices = [t for t in vortex_tokens if t["vortex"]["charge"] == +1]
        antivortices = [t for t in vortex_tokens if t["vortex"]["charge"] == -1]

        pair_result = pair_vortices(vortices, antivortices, self.L, provenance=prov)

        return {
            "vortex_tokens": vortex_tokens,
            "pair_tokens": pair_result["pairs"],
            "provenance": prov,
        }

    def test_all_vortex_tokens_valid(self, pipeline_output):
        """Every vortex token validates against schema."""
        for tok in pipeline_output["vortex_tokens"]:
            _validate(tok)

    def test_all_pair_tokens_valid(self, pipeline_output):
        """Every pair token validates against schema."""
        for tok in pipeline_output["pair_tokens"]:
            _validate(tok)

    def test_vortex_provenance_fields(self, pipeline_output):
        """Provenance in vortex tokens has required fields."""
        for tok in pipeline_output["vortex_tokens"]:
            prov = tok["provenance"]
            for key in ("model", "L", "T", "seed", "sweep_index", "schema_version"):
                assert key in prov, f"Missing provenance field: {key}"
            assert prov["L"] == self.L
            assert prov["T"] == self.T

    def test_pair_provenance_fields(self, pipeline_output):
        """Provenance in pair tokens has required fields."""
        for tok in pipeline_output["pair_tokens"]:
            prov = tok["provenance"]
            for key in ("model", "L", "T", "seed", "sweep_index", "schema_version"):
                assert key in prov, f"Missing provenance field: {key}"

    def test_schema_version_format(self, pipeline_output):
        """schema_version follows semver pattern."""
        import re
        semver = re.compile(r"^\d+\.\d+\.\d+$")
        for tok in pipeline_output["vortex_tokens"]:
            assert semver.match(tok["schema_version"]), (
                f"Bad schema_version: {tok['schema_version']}"
            )
        for tok in pipeline_output["pair_tokens"]:
            assert semver.match(tok["schema_version"])

    def test_token_type_correct(self, pipeline_output):
        for tok in pipeline_output["vortex_tokens"]:
            assert tok["token_type"] == "vortex"
        for tok in pipeline_output["pair_tokens"]:
            assert tok["token_type"] == "pair"


# ---------------------------------------------------------------------------
# §5 — Hot config tokens also validate
# ---------------------------------------------------------------------------

class TestSchemaHotConfig:
    """Tokens from a hot (disordered) config must also pass schema."""

    def test_hot_vortex_tokens_valid(self):
        L = 16
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, size=(L, L))
        prov = _prov(L=L, T=5.0)
        tokens = extract_vortices(theta, prov)
        assert len(tokens) > 0, "Expected tokens for hot config"
        for tok in tokens:
            _validate(tok)

    def test_hot_pair_tokens_valid(self):
        L = 16
        rng = np.random.default_rng(43)
        theta = rng.uniform(-np.pi, np.pi, size=(L, L))
        prov = _prov(L=L, T=5.0)
        tokens = extract_vortices(theta, prov)
        vortices = [t for t in tokens if t["vortex"]["charge"] == +1]
        antivortices = [t for t in tokens if t["vortex"]["charge"] == -1]
        if vortices and antivortices:
            result = pair_vortices(vortices, antivortices, L, provenance=prov)
            for ptok in result["pairs"]:
                _validate(ptok)
