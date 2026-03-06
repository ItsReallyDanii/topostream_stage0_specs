"""
scripts/run_token_benchmark.py
================================
Cross-pipeline defect-stability benchmark via token-only comparison.

Generates token JSONL from two producer paths:
  1. clean simulation    (run_xy → extract → pair → JSONL)
  2. degraded map-mode   (run_xy → forward_model degrade → adapter → extract → pair → JSONL)

Then feeds both into the downstream consumer (token_benchmark.py) which
reads ONLY the JSONL files — never raw spin fields — and produces a
comparison table + machine-readable JSON results.

Usage:
    python scripts/run_token_benchmark.py [--results-dir DIR]

Degradation ladder (small, real, reproducible):
    clean          no degradation (simulation → forward model → adapter round-trip)
    blur_s1        Gaussian blur σ=1.0 on vector map before adapter inversion
    blur_s2        Gaussian blur σ=2.0
    noise_s03      additive Gaussian noise σ=0.3 on vector map
    noise_s05      additive Gaussian noise σ=0.5
    mask_10        10% random NaN masking on vector map
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from topostream.analysis.token_benchmark import (
    compare_token_streams,
    format_results_table,
    result_to_dict,
)
from topostream.extract.pairing import pair_vortices
from topostream.extract.vortices import extract_vortices
from topostream.io.schema_validate import validate_token
from topostream.map.adapters import vector_map_to_theta
from topostream.map.forward_models import add_noise, apply_blur, mask_nan, to_vector_map
from topostream.simulate.xy_numba import run_xy


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

L = 16
T = 0.9
SEED = 42
J = 1.0
N_EQUIL = 1000
N_MEAS = 2000
N_THIN = 50
R_MAX = L / 4.0

# Degradation ladder: (label, degrade_fn)
# Each degrade_fn takes (Mx, My) and returns (Mx_degraded, My_degraded).

def _no_degrade(Mx, My):
    return Mx.copy(), My.copy()

def _blur_s1(Mx, My):
    return apply_blur(Mx, 1.0), apply_blur(My, 1.0)

def _blur_s2(Mx, My):
    return apply_blur(Mx, 2.0), apply_blur(My, 2.0)

def _noise_s03(Mx, My):
    return add_noise(Mx, 0.3, seed=100), add_noise(My, 0.3, seed=101)

def _noise_s05(Mx, My):
    return add_noise(Mx, 0.5, seed=100), add_noise(My, 0.5, seed=101)

def _mask_10(Mx, My):
    m = mask_nan(np.ones_like(Mx), 0.10, seed=200)
    nan_sites = np.isnan(m)
    Mx_out, My_out = Mx.copy(), My.copy()
    Mx_out[nan_sites] = np.nan
    My_out[nan_sites] = np.nan
    return Mx_out, My_out


DEGRADATION_LADDER = [
    ("clean", _no_degrade),
    ("blur_s1", _blur_s1),
    ("blur_s2", _blur_s2),
    ("noise_s03", _noise_s03),
    ("noise_s05", _noise_s05),
    ("mask_10", _mask_10),
]


# ---------------------------------------------------------------------------
# Token JSONL generation
# ---------------------------------------------------------------------------

def _make_provenance(model: str) -> dict:
    return {
        "model": model,
        "L": L,
        "T": T,
        "seed": SEED,
        "sweep_index": 0,
        "schema_version": "1.1.0",
        "N_equil": N_EQUIL,
        "N_meas": N_MEAS,
        "N_thin": N_THIN,
    }


def _write_tokens(tokens: list[dict], path: Path) -> None:
    """Write schema-valid tokens to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(json.dumps(tok, sort_keys=True) + "\n")


def _generate_pipeline_tokens(
    theta: np.ndarray,
    provenance: dict,
    r_max: float,
) -> list[dict]:
    """Run extraction + pairing on a theta field and return all tokens."""
    vortex_tokens = extract_vortices(theta, provenance)
    vortices = [t for t in vortex_tokens if t["vortex"]["charge"] == +1]
    antivortices = [t for t in vortex_tokens if t["vortex"]["charge"] == -1]
    pair_result = pair_vortices(
        vortices, antivortices, provenance["L"],
        r_max=r_max, provenance=provenance,
    )
    all_tokens = vortex_tokens + pair_result["pairs"]
    for tok in all_tokens:
        validate_token(tok)
    return all_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Cross-pipeline defect-stability token benchmark"
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=str(REPO_ROOT / "results" / "token_benchmark"),
        help="Directory for output files",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Run simulation ----
    print(f"Running XY simulation: L={L}, T={T}, seed={SEED}")
    result = run_xy(L=L, T=T, J=J, N_equil=N_EQUIL, N_meas=N_MEAS,
                    N_thin=N_THIN, seed=SEED)
    theta_clean = result["configs"][-1]

    # ---- Step 2: Produce reference tokens (clean simulation, direct path) ----
    print("Generating reference tokens (clean simulation)...")
    prov_sim = _make_provenance("XY")
    ref_tokens = _generate_pipeline_tokens(theta_clean, prov_sim, R_MAX)
    ref_path = results_dir / "tokens_ref_simulation.jsonl"
    _write_tokens(ref_tokens, ref_path)
    print(f"  Reference: {len([t for t in ref_tokens if t['token_type'] == 'vortex'])} vortices, "
          f"{len([t for t in ref_tokens if t['token_type'] == 'pair'])} pairs → {ref_path}")

    # ---- Step 3: Produce map-mode candidate tokens for each degradation ----
    #   Path: theta → to_vector_map → degrade → vector_map_to_theta → extract → pair → JSONL
    #
    #   The schema enum has "map_mode" as a valid model.
    #   Provenance explicitly marks this as map-mode with preprocessing_params.
    Mx_clean, My_clean = to_vector_map(theta_clean)

    comparison_results = []
    for label, degrade_fn in DEGRADATION_LADDER:
        print(f"Generating candidate tokens: {label}...")
        Mx_deg, My_deg = degrade_fn(Mx_clean, My_clean)
        theta_recovered = vector_map_to_theta(Mx_deg, My_deg)

        prov_map = _make_provenance("map_mode")
        prov_map["preprocessing_params"] = {"degradation": label}

        cand_tokens = _generate_pipeline_tokens(theta_recovered, prov_map, R_MAX)
        cand_path = results_dir / f"tokens_cand_{label}.jsonl"
        _write_tokens(cand_tokens, cand_path)

        n_v = len([t for t in cand_tokens if t["token_type"] == "vortex"])
        n_p = len([t for t in cand_tokens if t["token_type"] == "pair"])
        print(f"  {label}: {n_v} vortices, {n_p} pairs → {cand_path}")

    # ---- Step 4: Run downstream comparison (token-only) ----
    print("\n" + "=" * 72)
    print("DOWNSTREAM TOKEN-ONLY COMPARISON")
    print("=" * 72)
    print(f"Matching tolerance: 1.5 lattice units (minimum-image PBC)")
    print(f"Matching rule: same-charge, greedy nearest-first")
    print()

    for label, _ in DEGRADATION_LADDER:
        cand_path = results_dir / f"tokens_cand_{label}.jsonl"
        cr = compare_token_streams(
            ref_path=ref_path,
            cand_path=cand_path,
            L=L,
            condition_label=label,
        )
        comparison_results.append(cr)

    # Print table.
    print(format_results_table(comparison_results))
    print()

    # ---- Step 5: Write machine-readable results ----
    results_json = [result_to_dict(r) for r in comparison_results]
    json_path = results_dir / "benchmark_results.json"
    with json_path.open("w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Machine-readable results written to: {json_path}")

    # ---- Confidence note ----
    has_conf = any(
        r.mean_confidence_cand is not None for r in comparison_results
    )
    if not has_conf:
        print("\nNote: Confidence calibration is not available for single-run "
              "map-mode outputs. Multi-seed aggregation would be required "
              "to produce meaningful confidence values for map-mode tokens.")
    else:
        print("\nConfidence values are present (single-run defaults).")
        print("Calibration comparison should use multi-seed aggregated values.")


if __name__ == "__main__":
    main()
