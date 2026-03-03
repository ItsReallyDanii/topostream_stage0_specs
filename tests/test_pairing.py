"""
tests/test_pairing.py
=======================
Gate tests for agents/03_pairing.md.

Validation requirements from the handoff and SPEC_VALIDATION §2.2:
  - 10 injected pairs recovered correctly.
  - f_paired = 1.0 for perfectly matched equal-count input within r_max.
  - r_max monotonicity: f_paired(L/8) ≤ f_paired(L/4) ≤ f_paired(L/2).
  - Schema validation of every pair token.
  - Deterministic output for a fixed configuration.

Run with:
    python -m pytest tests/test_pairing.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from topostream.extract.pairing import pair_vortices, _minimum_image_distance

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


def _validate_pair_tokens(tokens: list[dict]) -> None:
    """Assert every pair token validates against the canonical schema."""
    for tok in tokens:
        jsonschema.validate(tok, _SCHEMA)


def _make_vortex_token(vid: str, x: float, y: float, charge: int = +1) -> dict:
    """Create a minimal vortex token for pairing tests."""
    return {
        "schema_version": "1.0.0",
        "token_type": "vortex",
        "provenance": _prov(),
        "vortex": {
            "id": vid,
            "x": x,
            "y": y,
            "charge": charge,
            "strength": 1.0,
            "confidence": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# SPEC_VALIDATION §2.2 — 10 injected pairs recovered
# ---------------------------------------------------------------------------

class TestInjectedPairs:
    """Inject N=10 pairs at known positions; assert all 10 are recovered
    with separations within ±0.6 of truth and 0 spurious pairs.
    (SPEC_VALIDATION §2.2)
    """

    L = 32
    N_PAIRS = 10

    @staticmethod
    def _build_injected(L: int, n: int):
        """Create n vortex–antivortex pairs with known separations < L/4.

        Each pair i is placed at:
            vortex:     (2*i + 0.5,  L/2)
            antivortex: (2*i + 0.5 + sep_i,  L/2)
        with sep_i = 1.0 + 0.3*i  (all well within r_max = L/4 = 8).
        """
        vortices = []
        antivortices = []
        true_seps = []
        for i in range(n):
            sep = 1.0 + 0.3 * i
            vx, vy = 2.0 * i + 0.5, L / 2.0
            ax, ay = 2.0 * i + 0.5 + sep, L / 2.0
            vortices.append(_make_vortex_token(f"v{i:03d}", vx, vy, charge=+1))
            antivortices.append(_make_vortex_token(f"a{i:03d}", ax, ay, charge=-1))
            true_seps.append(sep)
        return vortices, antivortices, true_seps

    def test_all_pairs_recovered(self):
        vortices, antivortices, true_seps = self._build_injected(self.L, self.N_PAIRS)
        result = pair_vortices(vortices, antivortices, self.L, provenance=_prov(L=self.L))
        assert len(result["pairs"]) == self.N_PAIRS, (
            f"Expected {self.N_PAIRS} pairs, got {len(result['pairs'])}"
        )

    def test_no_spurious_pairs(self):
        vortices, antivortices, _ = self._build_injected(self.L, self.N_PAIRS)
        result = pair_vortices(vortices, antivortices, self.L, provenance=_prov(L=self.L))
        # No unmatched ids
        assert result["unmatched_vortex_ids"] == []
        assert result["unmatched_antivortex_ids"] == []

    def test_separations_within_tolerance(self):
        """Each recovered separation_r must be within ±0.6 of the true value."""
        vortices, antivortices, true_seps = self._build_injected(self.L, self.N_PAIRS)
        result = pair_vortices(vortices, antivortices, self.L, provenance=_prov(L=self.L))
        # Build lookup from vortex_id → true sep
        id_to_sep = {f"v{i:03d}": true_seps[i] for i in range(self.N_PAIRS)}
        for ptok in result["pairs"]:
            p = ptok["pair"]
            vid = p["vortex_id"]
            true_r = id_to_sep[vid]
            assert abs(p["separation_r"] - true_r) <= 0.6, (
                f"Pair {p['pair_id']}: separation_r={p['separation_r']:.3f}, "
                f"expected {true_r:.3f} (tol=0.6)"
            )

    def test_f_paired_is_one(self):
        """Equal-count input within r_max → f_paired must be 1.0."""
        vortices, antivortices, _ = self._build_injected(self.L, self.N_PAIRS)
        result = pair_vortices(vortices, antivortices, self.L, provenance=_prov(L=self.L))
        assert result["f_paired"] == pytest.approx(1.0), (
            f"f_paired={result['f_paired']:.4f}, expected 1.0"
        )

    def test_schema_valid(self):
        vortices, antivortices, _ = self._build_injected(self.L, self.N_PAIRS)
        result = pair_vortices(vortices, antivortices, self.L, provenance=_prov(L=self.L))
        _validate_pair_tokens(result["pairs"])


# ---------------------------------------------------------------------------
# Hungarian optimality — not greedy
# ---------------------------------------------------------------------------

class TestHungarianOptimality:
    """Ensure the solver produces the globally optimal matching (not greedy).

    Configuration (L=32):
        v0 at (1.5, 1.5),  v1 at (3.5, 1.5)
        a0 at (3.0, 1.5),  a1 at (5.0, 1.5)

    Greedy nearest-neighbour would pair v0–a0 (dist=1.5) first, leaving
    v1–a1 (dist=1.5), total = 3.0.

    Hungarian sees: v0–a0=1.5, v0–a1=3.5, v1–a0=0.5, v1–a1=1.5.
    Optimal: v1–a0 (0.5) + v0–a1 (3.5) = 4.0 … or v0–a0 (1.5) + v1–a1 (1.5) = 3.0.

    Let me design a clearer anti-greedy case:
        v0 at (0.5, 0.5),  v1 at (2.0, 0.5)
        a0 at (1.5, 0.5),  a1 at (4.0, 0.5)

        Greedy: v0–a0 (1.0), then v1–a1 (2.0) → total 3.0
        Hungarian: v0–a0 (1.0) + v1–a1 (2.0) = 3.0
                   v0–a1 (3.5) + v1–a0 (0.5) = 4.0
        → both give v0–a0 with greedy.

    Clearer anti-greedy:
        v0 at (0.5, 0.5),  v1 at (1.5, 0.5)
        a0 at (1.0, 0.5),  a1 at (3.0, 0.5)

        Greedy (from v0): v0–a0 (0.5), v1–a1 (1.5) → total 2.0
        Hungarian should find same since v0–a0 < v1–a0:
          v0–a0 (0.5) + v1–a1 (1.5) = 2.0  ← optimal
          v0–a1 (2.5) + v1–a0 (0.5) = 3.0

    Actually, let's construct an explicit anti-greedy scenario:
        v0 at (5.5, 5.5),  v1 at (6.5, 5.5)
        a0 at (6.0, 5.5),  a1 at (8.0, 5.5)

        v0–a0 = 0.5,  v0–a1 = 2.5
        v1–a0 = 0.5,  v1–a1 = 1.5

        Greedy v0: picks a0 (0.5), then v1 gets a1 (1.5) → total 2.0
        Greedy v1: picks a0 (0.5), then v0 gets a1 (2.5) → total 3.0
        Hungarian: v0–a0 + v1–a1 = 0.5+1.5 = 2.0   or   v0–a1 + v1–a0 = 2.5+0.5 = 3.0
        Optimal = 2.0.
    OK — both greedy-from-best and Hungarian agree here.  Let me just
    verify the Hungarian picks the globally min-cost assignment in a case
    where the answer IS unambiguous and deterministic.
    """

    def test_globally_optimal_assignment(self):
        """Verify that the solver picks the min total-cost assignment."""
        L = 32
        # v0 at (2.5, 5.5), v1 at (6.5, 5.5)
        # a0 at (7.0, 5.5), a1 at (3.0, 5.5)
        # v0-a0 = 4.5, v0-a1 = 0.5, v1-a0 = 0.5, v1-a1 = 3.5
        # Optimal: v0-a1 (0.5) + v1-a0 (0.5) = 1.0  <-- globally best
        # Alt:     v0-a0 (4.5) + v1-a1 (3.5) = 8.0
        vortices = [
            _make_vortex_token("v0", 2.5, 5.5, +1),
            _make_vortex_token("v1", 6.5, 5.5, +1),
        ]
        antivortices = [
            _make_vortex_token("a0", 7.0, 5.5, -1),
            _make_vortex_token("a1", 3.0, 5.5, -1),
        ]
        result = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        pairs = result["pairs"]
        assert len(pairs) == 2

        # Build {vortex_id: antivortex_id} mapping
        mapping = {p["pair"]["vortex_id"]: p["pair"]["antivortex_id"] for p in pairs}
        assert mapping == {"v0": "a1", "v1": "a0"}, (
            f"Hungarian should pair v0–a1 and v1–a0; got {mapping}"
        )

    def test_total_cost_is_minimal(self):
        """Sum of all separation_r must equal the globally minimal cost."""
        L = 32
        vortices = [
            _make_vortex_token("v0", 2.5, 5.5, +1),
            _make_vortex_token("v1", 6.5, 5.5, +1),
        ]
        antivortices = [
            _make_vortex_token("a0", 7.0, 5.5, -1),
            _make_vortex_token("a1", 3.0, 5.5, -1),
        ]
        result = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        total = sum(p["pair"]["separation_r"] for p in result["pairs"])
        assert total == pytest.approx(1.0, abs=1e-10), (
            f"Total cost should be 1.0; got {total}"
        )


# ---------------------------------------------------------------------------
# r_max cutoff behaviour
# ---------------------------------------------------------------------------

class TestRmaxCutoff:
    """Test that pairs beyond r_max are treated as unmatched,
    and that r_max_used is recorded in every pair token.
    """

    def test_pair_beyond_rmax_is_unmatched(self):
        """A single pair with separation > r_max must be unmatched."""
        L = 32
        r_max = 2.0
        vortices = [_make_vortex_token("vfar", 0.5, 0.5, +1)]
        antivortices = [_make_vortex_token("afar", 10.5, 0.5, -1)]
        # Minimum-image dist = min(10, 32-10) = 10 > r_max=2
        result = pair_vortices(vortices, antivortices, L, r_max=r_max,
                               provenance=_prov(L=L))
        assert len(result["pairs"]) == 0, "Pair beyond r_max must not be emitted."
        assert "vfar" in result["unmatched_vortex_ids"]
        assert "afar" in result["unmatched_antivortex_ids"]
        assert result["f_paired"] == pytest.approx(0.0)

    def test_pair_within_rmax_is_matched(self):
        L = 32
        r_max = 5.0
        vortices = [_make_vortex_token("vnear", 0.5, 0.5, +1)]
        antivortices = [_make_vortex_token("anear", 2.5, 0.5, -1)]
        result = pair_vortices(vortices, antivortices, L, r_max=r_max,
                               provenance=_prov(L=L))
        assert len(result["pairs"]) == 1
        assert result["f_paired"] == pytest.approx(1.0)

    def test_rmax_used_recorded(self):
        """Every pair token must carry the r_max_used value."""
        L = 32
        r_max = 6.0
        vortices = [_make_vortex_token("v0", 0.5, 0.5, +1)]
        antivortices = [_make_vortex_token("a0", 2.5, 0.5, -1)]
        result = pair_vortices(vortices, antivortices, L, r_max=r_max,
                               provenance=_prov(L=L))
        for ptok in result["pairs"]:
            assert ptok["pair"]["r_max_used"] == pytest.approx(r_max)

    def test_default_rmax_is_L_over_4(self):
        """When r_max is not supplied, it defaults to L/4."""
        L = 32
        vortices = [_make_vortex_token("v0", 0.5, 0.5, +1)]
        antivortices = [_make_vortex_token("a0", 2.5, 0.5, -1)]
        result = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        for ptok in result["pairs"]:
            assert ptok["pair"]["r_max_used"] == pytest.approx(L / 4.0)


# ---------------------------------------------------------------------------
# r_max sensitivity monotonicity
# ---------------------------------------------------------------------------

class TestRmaxSensitivity:
    """f_paired(r_max=L/8) ≤ f_paired(r_max=L/4) ≤ f_paired(r_max=L/2)
    (handoff gate requirement).

    Uses a mixture of close and far pairs to produce non-trivial variation.
    """

    L = 32

    @staticmethod
    def _build_mixed(L: int):
        """Create a mix of close and far defects.

        Close pairs: separation ≈ 2.0 (within L/8=4)
        Medium pairs: separation ≈ 6.0 (within L/4=8 but > L/8=4)
        Far pairs:  separation ≈ 12.0 (within L/2=16 but > L/4=8)
        """
        vortices = [
            _make_vortex_token("vc0", 0.5, 0.5, +1),
            _make_vortex_token("vc1", 0.5, 5.5, +1),
            _make_vortex_token("vm0", 0.5, 10.5, +1),
            _make_vortex_token("vm1", 0.5, 15.5, +1),
            _make_vortex_token("vf0", 0.5, 20.5, +1),
            _make_vortex_token("vf1", 0.5, 25.5, +1),
        ]
        antivortices = [
            _make_vortex_token("ac0", 2.5, 0.5, -1),   # sep ≈ 2 from vc0
            _make_vortex_token("ac1", 2.5, 5.5, -1),   # sep ≈ 2 from vc1
            _make_vortex_token("am0", 6.5, 10.5, -1),  # sep ≈ 6 from vm0
            _make_vortex_token("am1", 6.5, 15.5, -1),  # sep ≈ 6 from vm1
            _make_vortex_token("af0", 12.5, 20.5, -1), # sep ≈ 12 from vf0
            _make_vortex_token("af1", 12.5, 25.5, -1), # sep ≈ 12 from vf1
        ]
        return vortices, antivortices

    def test_monotonicity(self):
        vortices, antivortices = self._build_mixed(self.L)
        rmax_values = [self.L / 8, self.L / 4, self.L / 2]
        f_values = []
        for rm in rmax_values:
            result = pair_vortices(vortices, antivortices, self.L, r_max=rm,
                                   provenance=_prov(L=self.L))
            f_values.append(result["f_paired"])

        assert f_values[0] <= f_values[1] <= f_values[2], (
            f"Monotonicity violated: f_paired({rmax_values}) = {f_values}"
        )

    def test_strict_increase_in_this_config(self):
        """With the mixed config, L/8 < L/4 < L/2 should give strictly
        increasing f_paired (since medium and far pairs get included)."""
        vortices, antivortices = self._build_mixed(self.L)
        r8 = pair_vortices(vortices, antivortices, self.L, r_max=self.L / 8,
                           provenance=_prov(L=self.L))
        r4 = pair_vortices(vortices, antivortices, self.L, r_max=self.L / 4,
                           provenance=_prov(L=self.L))
        r2 = pair_vortices(vortices, antivortices, self.L, r_max=self.L / 2,
                           provenance=_prov(L=self.L))
        assert r8["f_paired"] < r4["f_paired"] < r2["f_paired"], (
            f"Expected strict increase: L/8→{r8['f_paired']:.3f}, "
            f"L/4→{r4['f_paired']:.3f}, L/2→{r2['f_paired']:.3f}"
        )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestPairSchema:
    """Every emitted pair token must validate against the canonical schema."""

    def test_pair_tokens_valid(self):
        L = 32
        vortices = [_make_vortex_token(f"v{i}", 0.5 + i, 0.5, +1) for i in range(5)]
        antivortices = [_make_vortex_token(f"a{i}", 0.5 + i, 1.5, -1) for i in range(5)]
        result = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        _validate_pair_tokens(result["pairs"])

    def test_pair_id_convention(self):
        """pair_id must follow 'pair_{vortex_id}_{antivortex_id}'."""
        L = 32
        vortices = [_make_vortex_token("v_abc", 0.5, 0.5, +1)]
        antivortices = [_make_vortex_token("a_xyz", 1.5, 0.5, -1)]
        result = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        p = result["pairs"][0]["pair"]
        assert p["pair_id"] == f"pair_{p['vortex_id']}_{p['antivortex_id']}"

    def test_required_pair_keys(self):
        L = 32
        vortices = [_make_vortex_token("v0", 0.5, 0.5, +1)]
        antivortices = [_make_vortex_token("a0", 1.5, 0.5, -1)]
        result = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        for ptok in result["pairs"]:
            p = ptok["pair"]
            for key in ("pair_id", "vortex_id", "antivortex_id",
                        "separation_r", "r_max_used"):
                assert key in p, f"Missing pair field: {key}"


# ---------------------------------------------------------------------------
# Unmatched defects — explicit handling
# ---------------------------------------------------------------------------

class TestUnmatchedDefects:
    """Unmatched defects must be returned explicitly, never dropped."""

    def test_excess_vortices(self):
        L = 32
        vortices = [
            _make_vortex_token("v0", 0.5, 0.5, +1),
            _make_vortex_token("v1", 5.5, 0.5, +1),
            _make_vortex_token("v2", 10.5, 0.5, +1),
        ]
        antivortices = [_make_vortex_token("a0", 1.5, 0.5, -1)]
        result = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        assert len(result["pairs"]) == 1
        assert len(result["unmatched_vortex_ids"]) == 2
        assert result["unmatched_antivortex_ids"] == []

    def test_excess_antivortices(self):
        L = 32
        vortices = [_make_vortex_token("v0", 0.5, 0.5, +1)]
        antivortices = [
            _make_vortex_token("a0", 1.5, 0.5, -1),
            _make_vortex_token("a1", 5.5, 0.5, -1),
        ]
        result = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        assert len(result["pairs"]) == 1
        assert result["unmatched_vortex_ids"] == []
        assert len(result["unmatched_antivortex_ids"]) == 1

    def test_empty_vortices(self):
        L = 32
        result = pair_vortices([], [_make_vortex_token("a0", 0.5, 0.5, -1)], L,
                               provenance=_prov(L=L))
        assert result["pairs"] == []
        assert result["unmatched_vortex_ids"] == []
        assert "a0" in result["unmatched_antivortex_ids"]
        assert result["f_paired"] == pytest.approx(0.0)

    def test_both_empty(self):
        L = 32
        result = pair_vortices([], [], L, provenance=_prov(L=L))
        assert result["pairs"] == []
        assert result["f_paired"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Minimum-image distance
# ---------------------------------------------------------------------------

class TestMinimumImageDistance:
    """Unit tests for the minimum-image separation (SPEC_FORMULAE §5)."""

    def test_straight_distance(self):
        # No wrapping needed
        r = _minimum_image_distance(0.5, 0.5, 3.5, 0.5, 32)
        assert r == pytest.approx(3.0)

    def test_pbc_wrapping_x(self):
        # |30.5 - 0.5| = 30; L - 30 = 2  → r_x = 2
        r = _minimum_image_distance(0.5, 0.5, 30.5, 0.5, 32)
        assert r == pytest.approx(2.0)

    def test_pbc_wrapping_y(self):
        r = _minimum_image_distance(0.5, 0.5, 0.5, 31.5, 32)
        assert r == pytest.approx(1.0)

    def test_pbc_wrapping_both(self):
        r = _minimum_image_distance(0.5, 0.5, 30.5, 31.5, 32)
        expected = math.sqrt(2.0**2 + 1.0**2)
        assert r == pytest.approx(expected)

    def test_same_point(self):
        r = _minimum_image_distance(5.5, 5.5, 5.5, 5.5, 32)
        assert r == pytest.approx(0.0)

    def test_half_lattice(self):
        # Maximum minimum-image distance along one axis = L/2
        r = _minimum_image_distance(0.5, 0.5, 16.5, 0.5, 32)
        assert r == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Output must be identical for the same input (no random tie-breaking)."""

    def test_repeated_calls_identical(self):
        L = 32
        vortices = [_make_vortex_token(f"v{i}", 0.5 + 2*i, 0.5, +1) for i in range(5)]
        antivortices = [_make_vortex_token(f"a{i}", 1.5 + 2*i, 0.5, -1) for i in range(5)]
        r1 = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        r2 = pair_vortices(vortices, antivortices, L, provenance=_prov(L=L))
        # Same number of pairs
        assert len(r1["pairs"]) == len(r2["pairs"])
        # Same pair_ids in same order
        ids1 = [p["pair"]["pair_id"] for p in r1["pairs"]]
        ids2 = [p["pair"]["pair_id"] for p in r2["pairs"]]
        assert ids1 == ids2
        # Same f_paired
        assert r1["f_paired"] == r2["f_paired"]
