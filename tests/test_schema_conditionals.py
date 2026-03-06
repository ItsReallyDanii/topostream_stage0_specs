"""
tests/test_schema_conditionals.py
====================================
Gate tests: token_type conditionals must be enforced by the schema.

Each token_type (vortex / pair / sweep_delta) MUST require its corresponding
payload object.  A token that declares token_type="vortex" but omits the
"vortex" key must FAIL schema validation, not silently pass.

If any of these tests fail it means the schema is missing if/then conditionals
and the schema file MUST be patched (see agents/00_repo_rules.md §SCHEMA).
"""

from __future__ import annotations

import pytest

from topostream.io.schema_validate import validate_token


def _base_prov() -> dict:
    return {
        "model": "XY",
        "L": 16,
        "T": 1.0,
        "seed": 42,
        "sweep_index": 0,
        "schema_version": "1.1.0",
    }


# ---------------------------------------------------------------------------
# Helper — build intentionally-invalid tokens
# ---------------------------------------------------------------------------

def _invalid_vortex_token() -> dict:
    """token_type='vortex' but the 'vortex' payload key is absent."""
    return {
        "schema_version": "1.1.0",
        "token_type": "vortex",
        "provenance": _base_prov(),
        # 'vortex' key intentionally missing
    }


def _invalid_pair_token() -> dict:
    """token_type='pair' but the 'pair' payload key is absent."""
    return {
        "schema_version": "1.1.0",
        "token_type": "pair",
        "provenance": _base_prov(),
        # 'pair' key intentionally missing
    }


def _invalid_sweep_delta_token() -> dict:
    """token_type='sweep_delta' but the 'sweep_delta' payload key is absent."""
    return {
        "schema_version": "1.1.0",
        "token_type": "sweep_delta",
        "provenance": _base_prov(),
        # 'sweep_delta' key intentionally missing
    }


# ---------------------------------------------------------------------------
# Tests — each invalid token MUST raise (schema validation fails loudly)
# ---------------------------------------------------------------------------

class TestSchemaConditionals:
    """Schema MUST enforce: token_type → required payload key present."""

    def test_vortex_token_type_requires_vortex_key(self):
        """token_type='vortex' without 'vortex' payload must fail validation."""
        token = _invalid_vortex_token()
        with pytest.raises(Exception, match=r"(?i)(valid|schema|required|vortex)"):
            validate_token(token)

    def test_pair_token_type_requires_pair_key(self):
        """token_type='pair' without 'pair' payload must fail validation."""
        token = _invalid_pair_token()
        with pytest.raises(Exception, match=r"(?i)(valid|schema|required|pair)"):
            validate_token(token)

    def test_sweep_delta_token_type_requires_sweep_delta_key(self):
        """token_type='sweep_delta' without 'sweep_delta' payload must fail validation."""
        token = _invalid_sweep_delta_token()
        with pytest.raises(Exception, match=r"(?i)(valid|schema|required|sweep_delta)"):
            validate_token(token)


# ---------------------------------------------------------------------------
# Positive controls — ensure valid tokens still pass
# ---------------------------------------------------------------------------

class TestSchemaConditionalsPositiveControl:
    """Valid tokens must continue to pass validation (regression guard)."""

    def test_valid_vortex_token_passes(self):
        token = {
            "schema_version": "1.1.0",
            "token_type": "vortex",
            "provenance": _base_prov(),
            "vortex": {
                "id": "v_p_r004_c004",
                "x": 4.5,
                "y": 4.5,
                "charge": 1,
                "strength": 1.0,
                "confidence": 1.0,
            },
        }
        validate_token(token)   # must not raise

    def test_valid_pair_token_passes(self):
        token = {
            "schema_version": "1.1.0",
            "token_type": "pair",
            "provenance": _base_prov(),
            "pair": {
                "pair_id": "pair_001",
                "vortex_id": "v_p_r004_c004",
                "antivortex_id": "v_m_r006_c006",
                "separation_r": 2.83,
                "r_max_used": 8.0,
            },
        }
        validate_token(token)   # must not raise

    def test_valid_sweep_delta_token_passes(self):
        token = {
            "schema_version": "1.1.0",
            "token_type": "sweep_delta",
            "provenance": _base_prov(),
            "sweep_delta": {
                "delta_type": "vortex_density_change",
                "T_from": 0.9,
                "T_to": 1.2,
                "delta_value": 0.05,
            },
        }
        validate_token(token)   # must not raise
