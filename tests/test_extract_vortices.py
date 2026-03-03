"""
tests/test_extract_vortices.py
================================
Gate tests for agents/02_extract_vortices.md.

Covers ALL toy configurations from SPEC_VALIDATION.md §1 plus schema
validation.  This test suite must pass 100 % before the handoff is marked
done.

Run with:
    python -m pytest tests/test_extract_vortices.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from topostream.extract.vortices import extract_vortices, _wrap

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SCHEMA_PATH = (
    Path(__file__).parent.parent / "schemas" / "topology_event_stream.schema.json"
)

with SCHEMA_PATH.open() as _f:
    _SCHEMA = json.load(_f)


def _prov(L: int = 32, T: float = 0.5, seed: int = 42, sweep_index: int = 0) -> dict:
    """Minimal schema-compliant provenance block."""
    return {
        "model": "XY",
        "L": L,
        "T": T,
        "seed": seed,
        "sweep_index": sweep_index,
        "schema_version": "1.0.0",
    }


def _validate_tokens(tokens: list[dict]) -> None:
    """Assert every token validates against the canonical schema."""
    for tok in tokens:
        jsonschema.validate(tok, _SCHEMA)


def _make_single_vortex_field(L: int = 32) -> np.ndarray:
    """Lattice vortex field with vortex at plaquette centre (L/2-0.5, L/2-0.5).

    Implements SPEC_VALIDATION §1.1: the spec's construction
    ``θ(x,y) = arctan2(y − L/2, x − L/2)`` placed at the centre of the
    lattice.  We use the minimum-image convention to locate the nearest
    image of the vortex core, which keeps the branch cut localised.

    Note on PBC topology: on a flat torus the net Pontryagin charge must
    equal zero, so a single vortex always comes with boundary-image
    antivortices.  The spec assertion "exactly 1 vortex" refers to the
    central / dominant defect, not to the total token count.
    """
    theta = np.zeros((L, L), dtype=np.float64)
    # Place core at plaquette centre so only one plaquette has full winding.
    cy = L / 2 - 0.5
    cx = L / 2 - 0.5
    for i in range(L):
        for j in range(L):
            dy = float(i) - cy
            dx = float(j) - cx
            # Minimum-image wrapping — keeps branch cut to one side of lattice.
            dy -= L * round(dy / L)
            dx -= L * round(dx / L)
            theta[i, j] = math.atan2(dy, dx)
    return theta


def _make_single_antivortex_field(L: int = 32) -> np.ndarray:
    """θ(x,y) = −arctan2(y − L/2, x − L/2)  — SPEC_VALIDATION §1.2."""
    return -_make_single_vortex_field(L)


def _central_defects(
    tokens: list[dict], L: int, charge: int, radius: float = 4.0
) -> list[dict]:
    """Filter tokens to those near the lattice centre with the given charge.

    On a PBC lattice, the branch cut of an arctan2 vortex field produces
    boundary-image defects.  This helper isolates the intended central
    defect for the SPEC_VALIDATION §1 assertions.
    """
    cx, cy = L / 2, L / 2
    return [
        t for t in tokens
        if t["vortex"]["charge"] == charge
        and math.hypot(t["vortex"]["x"] - cx, t["vortex"]["y"] - cy) < radius
    ]


def _make_bound_pair_field(L: int = 32, d: float = 4.0) -> np.ndarray:
    """Superposition of one vortex and one antivortex separated by d=4.

    yv = ya = L/2;  xv = L/2 − 2;  xa = L/2 + 2  — SPEC_VALIDATION §1.3.
    """
    theta = np.zeros((L, L), dtype=np.float64)
    cy = L / 2
    xv = L / 2 - d / 2   # = L/2 − 2
    xa = L / 2 + d / 2   # = L/2 + 2
    for i in range(L):
        for j in range(L):
            theta[i, j] = math.atan2(i - cy, j - xv) - math.atan2(i - cy, j - xa)
    return theta


# ---------------------------------------------------------------------------
# §1.1 Single vortex
# ---------------------------------------------------------------------------

class TestSingleVortex:
    """SPEC_VALIDATION §1.1 — Single vortex field.

    On a PBC lattice the net Pontryagin charge is zero, so the arctan2
    single-vortex field also generates boundary-image antivortices to
    compensate.  The spec's assertion "exactly 1 vortex, charge=+1,
    position ≈ (L/2, L/2)" refers to the *central* defect, which our
    extractor must detect correctly.
    """

    def test_central_vortex_found(self):
        """Exactly one +1 defect must exist near the lattice centre."""
        L = 32
        theta = _make_single_vortex_field(L)
        tokens = extract_vortices(theta, _prov(L=L))
        central = _central_defects(tokens, L, charge=+1, radius=4.0)
        assert len(central) == 1, (
            f"Expected exactly 1 central vortex (charge=+1), "
            f"found {len(central)}.  All tokens: "
            + str([(t['vortex']['charge'], t['vortex']['x'], t['vortex']['y'])
                   for t in tokens])
        )

    def test_central_charge_is_positive(self):
        L = 32
        theta = _make_single_vortex_field(L)
        tokens = extract_vortices(theta, _prov(L=L))
        central = _central_defects(tokens, L, charge=+1, radius=4.0)
        assert central[0]["vortex"]["charge"] == +1

    def test_position_near_centre(self):
        L = 32
        theta = _make_single_vortex_field(L)
        tokens = extract_vortices(theta, _prov(L=L))
        central = _central_defects(tokens, L, charge=+1, radius=4.0)
        assert len(central) >= 1
        v = central[0]["vortex"]
        # Position should be within 1 plaquette of the centre
        assert abs(v["x"] - L / 2) <= 1.5, f"x={v['x']} not near {L/2}"
        assert abs(v["y"] - L / 2) <= 1.5, f"y={v['y']} not near {L/2}"

    def test_schema_valid(self):
        theta = _make_single_vortex_field(32)
        tokens = extract_vortices(theta, _prov())
        _validate_tokens(tokens)

    def test_strength_near_one(self):
        L = 32
        theta = _make_single_vortex_field(L)
        tokens = extract_vortices(theta, _prov(L=L))
        central = _central_defects(tokens, L, charge=+1, radius=4.0)
        assert len(central) >= 1
        assert 0.9 <= central[0]["vortex"]["strength"] <= 1.1


# ---------------------------------------------------------------------------
# §1.2 Single antivortex
# ---------------------------------------------------------------------------

class TestSingleAntivortex:
    """SPEC_VALIDATION §1.2 — Single antivortex field.

    Mirror of §1.1: checks the central antivortex is found with charge=-1.
    """

    def test_central_antivortex_found(self):
        """Exactly one -1 defect must exist near the lattice centre."""
        L = 32
        theta = _make_single_antivortex_field(L)
        tokens = extract_vortices(theta, _prov(L=L))
        central = _central_defects(tokens, L, charge=-1, radius=4.0)
        assert len(central) == 1, (
            f"Expected exactly 1 central antivortex (charge=-1), "
            f"found {len(central)}.  All tokens: "
            + str([(t['vortex']['charge'], t['vortex']['x'], t['vortex']['y'])
                   for t in tokens])
        )
        assert central[0]["vortex"]["charge"] == -1

    def test_schema_valid(self):
        theta = _make_single_antivortex_field(32)
        tokens = extract_vortices(theta, _prov())
        _validate_tokens(tokens)


# ---------------------------------------------------------------------------
# §1.3 Bound pair d=4
# ---------------------------------------------------------------------------

class TestBoundPair:
    def test_token_count(self):
        L = 32
        theta = _make_bound_pair_field(L, d=4.0)
        tokens = extract_vortices(theta, _prov(L=L))
        assert len(tokens) == 2, f"Expected 2 tokens, got {len(tokens)}"

    def test_charges_pm1(self):
        L = 32
        theta = _make_bound_pair_field(L, d=4.0)
        tokens = extract_vortices(theta, _prov(L=L))
        charges = sorted(t["vortex"]["charge"] for t in tokens)
        assert charges == [-1, +1]

    def test_schema_valid(self):
        L = 32
        theta = _make_bound_pair_field(L, d=4.0)
        tokens = extract_vortices(theta, _prov(L=L))
        _validate_tokens(tokens)

    def test_separation_in_range(self):
        """Pair separation should be in [3.5, 4.5] per SPEC_VALIDATION §1.3."""
        L = 32
        theta = _make_bound_pair_field(L, d=4.0)
        tokens = extract_vortices(theta, _prov(L=L))
        # Two tokens; compute their separation
        pos = [(t["vortex"]["x"], t["vortex"]["y"]) for t in tokens]
        dx = pos[0][0] - pos[1][0]
        dy = pos[0][1] - pos[1][1]
        # Minimum-image distance on L×L lattice (SPEC_FORMULAE §5)
        rx = min(abs(dx), L - abs(dx))
        ry = min(abs(dy), L - abs(dy))
        r = math.sqrt(rx**2 + ry**2)
        assert 3.5 <= r <= 4.5, f"Pair separation {r:.3f} out of [3.5, 4.5]"


# ---------------------------------------------------------------------------
# §1.4 Uniform field (vortex-free)
# ---------------------------------------------------------------------------

class TestUniformField:
    def test_no_vortices(self):
        L = 32
        theta = np.zeros((L, L), dtype=np.float64)
        tokens = extract_vortices(theta, _prov(L=L))
        assert len(tokens) == 0, f"Expected 0 tokens, got {len(tokens)}"

    def test_constant_nonzero_no_vortices(self):
        """Any constant angle field should be vortex-free."""
        L = 16
        theta = np.full((L, L), math.pi / 4, dtype=np.float64)
        tokens = extract_vortices(theta, _prov(L=L))
        assert len(tokens) == 0

    def test_schema_valid(self):
        # Empty list is trivially valid; no tokens to check
        theta = np.zeros((32, 32), dtype=np.float64)
        tokens = extract_vortices(theta, _prov())
        assert tokens == []


# ---------------------------------------------------------------------------
# Wrap function unit tests
# ---------------------------------------------------------------------------

class TestWrap:
    """Normative wrap(Δθ) = arctan2(sin(Δθ), cos(Δθ))  — SPEC_FORMULAE §1."""

    def test_identity_small(self):
        for val in [0.0, 0.5, -0.5, math.pi / 4]:
            assert abs(_wrap(val) - val) < 1e-12

    def test_maps_beyond_pi(self):
        result = _wrap(math.pi + 0.1)
        assert -math.pi <= result <= math.pi

    def test_maps_below_neg_pi(self):
        result = _wrap(-math.pi - 0.1)
        assert -math.pi <= result <= math.pi

    def test_two_pi_maps_to_zero(self):
        assert abs(_wrap(2 * math.pi)) < 1e-12

    def test_vectorised(self):
        arr = np.array([0.0, math.pi, -math.pi, 2 * math.pi])
        result = _wrap(arr)
        assert result.shape == arr.shape
        assert np.all(result >= -math.pi - 1e-12)
        assert np.all(result <= math.pi + 1e-12)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_l_too_small_raises(self):
        theta = np.zeros((4, 4))
        with pytest.raises(ValueError, match="L must be"):
            extract_vortices(theta, _prov(L=4))

    def test_non_square_raises(self):
        theta = np.zeros((16, 32))
        with pytest.raises(ValueError, match="square"):
            extract_vortices(theta, _prov(L=16))

    def test_3d_raises(self):
        theta = np.zeros((16, 16, 2))
        with pytest.raises(ValueError, match="2-D"):
            extract_vortices(theta, _prov(L=16))


# ---------------------------------------------------------------------------
# NaN-corner map-mode plaquettes are skipped (SPEC_ALGORITHMS §1)
# ---------------------------------------------------------------------------

class TestNanHandling:
    def test_nan_corner_plaquette_skipped(self):
        """A single NaN at (0,0) causes up to 4 plaquettes to be skipped;
        the extractor should not raise and should log the skip count."""
        L = 16
        theta = np.zeros((L, L), dtype=np.float64)
        theta[0, 0] = np.nan
        # Should not raise; a uniform field with some NaN → 0 vortex tokens
        tokens = extract_vortices(theta, _prov(L=L))
        assert isinstance(tokens, list)

    def test_all_nan_returns_empty(self):
        L = 16
        theta = np.full((L, L), np.nan)
        tokens = extract_vortices(theta, _prov(L=L))
        assert tokens == []


# ---------------------------------------------------------------------------
# Schema field completeness
# ---------------------------------------------------------------------------

class TestTokenFields:
    """Every token must carry the exact fields the schema requires."""

    def test_required_top_level_keys(self):
        theta = _make_single_vortex_field(16)
        tokens = extract_vortices(theta, _prov(L=16))
        for tok in tokens:
            assert "schema_version" in tok
            assert "token_type" in tok
            assert "provenance" in tok
            assert tok["token_type"] == "vortex"

    def test_required_vortex_keys(self):
        theta = _make_single_vortex_field(16)
        tokens = extract_vortices(theta, _prov(L=16))
        for tok in tokens:
            v = tok["vortex"]
            for key in ("id", "x", "y", "charge", "strength", "confidence"):
                assert key in v, f"Missing vortex field: {key}"

    def test_provenance_passthrough(self):
        """Provenance dict supplied by caller must appear unmodified in token."""
        prov = _prov(L=16, T=0.75, seed=99, sweep_index=7)
        theta = _make_single_vortex_field(16)
        tokens = extract_vortices(theta, prov)
        for tok in tokens:
            assert tok["provenance"]["seed"] == 99
            assert tok["provenance"]["sweep_index"] == 7
            assert tok["provenance"]["T"] == 0.75
