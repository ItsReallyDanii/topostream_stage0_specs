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


# ---------------------------------------------------------------------------
# Central Defect Detection Gate (official gate rule for §1.1 / §1.2)
# ---------------------------------------------------------------------------
# On a PBC torus the net Pontryagin charge is always zero, so any single-
# vortex arctan2 field also generates boundary-image anti-defects to
# compensate.  The gate therefore checks the *central* defect only:
#
#   Gate rule: within radius=4.0 lattice units of (L/2, L/2)
#              exactly ONE defect of the expected charge must exist,
#              its position must be within 1.5 units of (L/2, L/2),
#              and its strength must lie in [0.9, 1.1].
# ---------------------------------------------------------------------------

GATE_RADIUS: float = 4.0       # search radius for central defect (lattice units)
GATE_POS_TOL: float = 1.5      # max |x - L/2| and |y - L/2| (lattice units)
GATE_STR_LO: float = 0.9       # minimum accepted vortex strength
GATE_STR_HI: float = 1.1       # maximum accepted vortex strength


def _gate_central_defect(
    tokens: list[dict],
    L: int,
    charge: int,
    *,
    radius: float = GATE_RADIUS,
    pos_tol: float = GATE_POS_TOL,
    str_lo: float = GATE_STR_LO,
    str_hi: float = GATE_STR_HI,
) -> dict:
    """Central Defect Detection Gate.

    Asserts that exactly one token with the given *charge* exists within
    *radius* lattice units of the lattice centre (L/2, L/2), that its
    position is within *pos_tol* of (L/2, L/2) on both axes, and that
    its winding-number strength is within [str_lo, str_hi].

    Returns the single matching vortex token on success; raises
    AssertionError with a diagnostic message on any failure.

    Gate parameters (defaults match SPEC_VALIDATION §1):
        radius  = 4.0  lattice units   (search window around centre)
        pos_tol = 1.5  lattice units   (max allowed offset from L/2)
        str_lo  = 0.9, str_hi = 1.1   (strength ≈ 1 for clean vortex)
    """
    cx, cy = L / 2, L / 2
    candidates = [
        t for t in tokens
        if t["vortex"]["charge"] == charge
        and math.hypot(t["vortex"]["x"] - cx, t["vortex"]["y"] - cy) < radius
    ]
    all_summary = [
        (t["vortex"]["charge"], t["vortex"]["x"], t["vortex"]["y"])
        for t in tokens
    ]
    assert len(candidates) == 1, (
        f"Central Defect Gate: expected exactly 1 defect with charge={charge:+d} "
        f"within radius={radius} of ({cx},{cy}); "
        f"found {len(candidates)}.  All tokens={all_summary}"
    )
    tok = candidates[0]
    v = tok["vortex"]
    assert abs(v["x"] - cx) <= pos_tol, (
        f"Central Defect Gate: x={v['x']:.2f} not within {pos_tol} of {cx}"
    )
    assert abs(v["y"] - cy) <= pos_tol, (
        f"Central Defect Gate: y={v['y']:.2f} not within {pos_tol} of {cy}"
    )
    assert str_lo <= v["strength"] <= str_hi, (
        f"Central Defect Gate: strength={v['strength']:.4f} not in "
        f"[{str_lo}, {str_hi}]"
    )
    return tok


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

    Gate rule (Central Defect Detection Gate):
      - Exactly ONE token with charge=+1 within radius=4.0 of (L/2, L/2).
      - Its position must be within 1.5 lattice units of (L/2, L/2).
      - Its strength must be in [0.9, 1.1] (well-resolved winding ≈ 1).

    Note on PBC topology: on a flat torus the net Pontryagin charge is
    always zero, so the arctan2 single-vortex field generates boundary-image
    antivortices to compensate.  The gate checks the *central* defect only.
    """

    def test_central_vortex_gate(self):
        """Central Defect Detection Gate: charge=+1, pos within 1.5, strength in [0.9,1.1]."""
        L = 32
        theta = _make_single_vortex_field(L)
        tokens = extract_vortices(theta, _prov(L=L))
        # _gate_central_defect raises AssertionError with diagnostics on any failure.
        _gate_central_defect(tokens, L, charge=+1)

    def test_schema_valid(self):
        theta = _make_single_vortex_field(32)
        tokens = extract_vortices(theta, _prov())
        _validate_tokens(tokens)


# ---------------------------------------------------------------------------
# §1.2 Single antivortex
# ---------------------------------------------------------------------------

class TestSingleAntivortex:
    """SPEC_VALIDATION §1.2 — Single antivortex field.

    Gate rule (Central Defect Detection Gate, mirrored from §1.1):
      - Exactly ONE token with charge=-1 within radius=4.0 of (L/2, L/2).
      - Its position must be within 1.5 lattice units of (L/2, L/2).
      - Its strength must be in [0.9, 1.1].

    Same PBC caveat as §1.1: boundary-image vortices are present but
    irrelevant; the gate checks the central defect only.
    """

    def test_central_antivortex_gate(self):
        """Central Defect Detection Gate: charge=-1, pos within 1.5, strength in [0.9,1.1]."""
        L = 32
        theta = _make_single_antivortex_field(L)
        tokens = extract_vortices(theta, _prov(L=L))
        _gate_central_defect(tokens, L, charge=-1)

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
# High-temperature (random field) sanity check
# ---------------------------------------------------------------------------

class TestRandomField:
    """Sanity checks on a fully disordered (high-T) angle field.

    SPEC_VALIDATION §2.1: random iid θ ~ Uniform[-π, π) on an L=32 lattice
    should yield vortex density ≈ 0.5, i.e. roughly L²/2 ≈ 512 vortices.
    We do NOT require an exact count — we only assert the extractor is not
    broken by checking two weak bounds:
      1. At least one token is produced (not suspiciously empty).
      2. All produced tokens validate against the schema.
    """

    def test_not_empty_for_hot_field(self):
        """A fully random angle field must produce at least one vortex token."""
        rng = np.random.default_rng(seed=42)
        L = 32
        theta = rng.uniform(-math.pi, math.pi, size=(L, L))
        tokens = extract_vortices(theta, _prov(L=L, T=5.0))
        assert len(tokens) > 0, (
            "Expected at least one vortex token for a hot random field; "
            f"got {len(tokens)}.  This indicates the extractor is broken."
        )

    def test_hot_field_schema_valid(self):
        """All tokens from a random field must pass schema validation."""
        rng = np.random.default_rng(seed=43)
        L = 32
        theta = rng.uniform(-math.pi, math.pi, size=(L, L))
        tokens = extract_vortices(theta, _prov(L=L, T=5.0))
        _validate_tokens(tokens)

    def test_hot_field_density_plausible(self):
        """Vortex density for hot random field should be in rough range [0.3, 0.7]."""
        rng = np.random.default_rng(seed=44)
        L = 32
        theta = rng.uniform(-math.pi, math.pi, size=(L, L))
        tokens = extract_vortices(theta, _prov(L=L, T=5.0))
        rho = len(tokens) / (L * L)
        assert 0.3 <= rho <= 0.7, (
            f"Hot-field vortex density rho={rho:.3f} outside expected [0.3, 0.7]; "
            "check extractor logic."
        )


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
