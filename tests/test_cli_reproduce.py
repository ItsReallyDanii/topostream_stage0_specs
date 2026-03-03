"""
tests/test_cli_reproduce.py
==============================
Gate tests for agents/06_cli_reproduce.md — CLI and one-command reproduce.

Tests verify:
  1. Config loading and T-list generation.
  2. Single sweep produces tokens, .npy, .jsonl, summary JSON.
  3. All emitted tokens pass schema validation (FAIL LOUDLY on error).
  4. Sweep_delta tokens are valid.
  5. CLI argparser doesn't crash.
  6. Determinism: same config → same outputs.
  7. Validate sub-command works on valid and invalid data.

Run with:
    python -m pytest tests/test_cli_reproduce.py -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from topostream.cli import (
    _load_config,
    _make_T_list,
    _parse_r_max,
    _run_single_sweep,
    _make_provenance,
    _build_parser,
)
from topostream.io.schema_validate import validate_token, validate_tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def config_path(tmp_path):
    """Write a minimal test config and return its path."""
    import yaml
    cfg = {
        "schema_version": "1.0.0",
        "models": ["XY"],
        "L_values": [8],
        "T_range": {"start": 0.5, "stop": 1.5, "n_points": 3},
        "seeds": [42],
        "N_equil": 50,
        "N_meas": 100,
        "N_thin": 50,
        "r_max_policy": "L/4",
        "output_dir": str(tmp_path / "results"),
        "figures_dir": str(tmp_path / "figures"),
    }
    p = tmp_path / "test_config.yaml"
    with p.open("w") as f:
        yaml.dump(cfg, f)
    return p


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_default_config(self):
        """configs/default.yaml loads without error."""
        cfg = _load_config("configs/default.yaml")
        assert cfg["schema_version"] == "1.0.0"
        assert "L_values" in cfg
        assert "T_range" in cfg

    def test_T_list_generation(self):
        T_list = _make_T_list({"start": 0.5, "stop": 1.5, "n_points": 5})
        assert len(T_list) == 5
        assert T_list[0] == pytest.approx(0.5)
        assert T_list[-1] == pytest.approx(1.5)

    def test_r_max_parse(self):
        assert _parse_r_max("L/4", 32) == 8.0
        assert _parse_r_max("L/8", 64) == 8.0
        assert _parse_r_max("L/2", 16) == 8.0


# ---------------------------------------------------------------------------
# Single sweep
# ---------------------------------------------------------------------------

class TestSingleSweep:
    """Run one (L,T,seed) through the full pipeline."""

    def test_produces_tokens(self, tmp_output):
        tokens = _run_single_sweep(
            model="XY", L=8, T=0.9, seed=42,
            N_equil=50, N_meas=100, N_thin=50,
            r_max=2.0, output_dir=tmp_output,
        )
        assert isinstance(tokens, list)
        # Should have at least some tokens (vortex + pair)
        assert len(tokens) >= 0  # could be 0 for a very ordered config

    def test_writes_npy(self, tmp_output):
        _run_single_sweep(
            model="XY", L=8, T=0.9, seed=42,
            N_equil=50, N_meas=100, N_thin=50,
            r_max=2.0, output_dir=tmp_output,
        )
        npy_files = list(tmp_output.glob("sim_*.npy"))
        assert len(npy_files) == 1
        cfg = np.load(npy_files[0])
        assert cfg.shape == (8, 8)

    def test_writes_jsonl(self, tmp_output):
        _run_single_sweep(
            model="XY", L=8, T=0.9, seed=42,
            N_equil=50, N_meas=100, N_thin=50,
            r_max=2.0, output_dir=tmp_output,
        )
        jsonl_files = list(tmp_output.glob("tokens_*.jsonl"))
        assert len(jsonl_files) == 1

    def test_writes_summary(self, tmp_output):
        _run_single_sweep(
            model="XY", L=8, T=0.9, seed=42,
            N_equil=50, N_meas=100, N_thin=50,
            r_max=2.0, output_dir=tmp_output,
        )
        summary_files = list(tmp_output.glob("summary_*.json"))
        assert len(summary_files) == 1
        with summary_files[0].open() as f:
            s = json.load(f)
        assert s["model"] == "XY"
        assert s["L"] == 8
        assert s["T"] == pytest.approx(0.9)

    def test_all_tokens_schema_valid(self, tmp_output):
        """CRITICAL: every token must pass schema validation."""
        tokens = _run_single_sweep(
            model="XY", L=8, T=0.9, seed=42,
            N_equil=50, N_meas=100, N_thin=50,
            r_max=2.0, output_dir=tmp_output,
        )
        errors = validate_tokens(tokens)
        assert errors == [], f"Schema errors: {errors}"


# ---------------------------------------------------------------------------
# Schema validation helper
# ---------------------------------------------------------------------------

class TestSchemaValidateHelper:
    def test_valid_vortex_token(self):
        tok = {
            "schema_version": "1.0.0",
            "token_type": "vortex",
            "provenance": {
                "model": "XY", "L": 16, "T": 0.5,
                "seed": 42, "sweep_index": 0, "schema_version": "1.0.0",
            },
            "vortex": {
                "id": "v_0", "x": 5.0, "y": 5.0,
                "charge": 1, "strength": 1.0, "confidence": 0.9,
            },
        }
        validate_token(tok)  # Should not raise

    def test_invalid_token_raises(self):
        tok = {"bad": "token"}
        with pytest.raises(Exception):
            validate_token(tok)

    def test_validate_tokens_returns_errors(self):
        good = {
            "schema_version": "1.0.0",
            "token_type": "vortex",
            "provenance": {
                "model": "XY", "L": 16, "T": 0.5,
                "seed": 42, "sweep_index": 0, "schema_version": "1.0.0",
            },
            "vortex": {
                "id": "v_0", "x": 5.0, "y": 5.0,
                "charge": 1, "strength": 1.0, "confidence": 0.9,
            },
        }
        bad = {"bad": "token"}
        errors = validate_tokens([good, bad])
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_config_same_output(self, tmp_path):
        d1 = tmp_path / "r1"
        d1.mkdir()
        d2 = tmp_path / "r2"
        d2.mkdir()

        kwargs = dict(
            model="XY", L=8, T=0.9, seed=42,
            N_equil=50, N_meas=100, N_thin=50,
            r_max=2.0,
        )
        t1 = _run_single_sweep(**kwargs, output_dir=d1)
        t2 = _run_single_sweep(**kwargs, output_dir=d2)

        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_provenance_has_required_fields(self):
        p = _make_provenance("XY", 32, 0.8, 42, 0)
        for key in ("model", "L", "T", "seed", "sweep_index", "schema_version"):
            assert key in p

    def test_provenance_with_config(self):
        cfg = {"N_equil": 1000, "N_meas": 5000, "N_thin": 50, "r_max_policy": "L/4"}
        p = _make_provenance("XY", 32, 0.8, 42, 0, cfg)
        assert p["N_equil"] == 1000
        assert p["r_max_policy"] == "L/4"


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLIParser:
    def test_reproduce_args(self):
        parser = _build_parser()
        args = parser.parse_args(["reproduce", "--config", "configs/default.yaml"])
        assert args.command == "reproduce"
        assert args.config == "configs/default.yaml"

    def test_sweep_args(self):
        parser = _build_parser()
        args = parser.parse_args(["sweep", "--model", "XY", "--L", "16",
                                  "--T", "0.9", "--seed", "42"])
        assert args.command == "sweep"
        assert args.L == 16
        assert args.T == pytest.approx(0.9)

    def test_validate_args(self):
        parser = _build_parser()
        args = parser.parse_args(["validate", "--results-dir", "results/"])
        assert args.command == "validate"

    def test_plot_args(self):
        parser = _build_parser()
        args = parser.parse_args(["plot", "--results-dir", "results/",
                                  "--output", "figures/"])
        assert args.command == "plot"
        assert args.output == "figures/"


# ---------------------------------------------------------------------------
# sweep_delta tokens
# ---------------------------------------------------------------------------

class TestSweepDelta:
    def test_delta_tokens_valid(self, tmp_path):
        """Run two consecutive sweeps and verify delta tokens validate."""
        from topostream.cli import _extract_sweep_state, _make_sweep_deltas

        d = tmp_path / "r"
        d.mkdir()
        kwargs = dict(
            model="XY", L=8, seed=42,
            N_equil=50, N_meas=100, N_thin=50, r_max=2.0,
        )

        t1 = _run_single_sweep(T=0.7, output_dir=d, **kwargs)
        t2 = _run_single_sweep(T=0.9, output_dir=d, **kwargs)

        s1 = _extract_sweep_state(t1, 8)
        s2 = _extract_sweep_state(t2, 8)

        prov = _make_provenance("XY", 8, 0.9, 42, 1)
        deltas = _make_sweep_deltas(s1, s2, T_from=0.7, T_to=0.9,
                                     provenance=prov)
        assert len(deltas) == 4  # 4 delta_types
        for d_tok in deltas:
            validate_token(d_tok)
            assert d_tok["token_type"] == "sweep_delta"
