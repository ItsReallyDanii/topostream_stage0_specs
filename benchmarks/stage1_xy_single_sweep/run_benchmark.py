"""
benchmarks/stage1_xy_single_sweep/run_benchmark.py
=====================================================
Run the Stage 1 XY single-sweep benchmark.

Directory layout
----------------
  frozen/   Committed frozen outputs. These are the inspectable benchmark
            artifacts stored in the repo. Hash tests compare against these.
  output/   Scratch directory for reruns. Gitignored. Overwritten on each run.

Usage:
    python benchmarks/stage1_xy_single_sweep/run_benchmark.py
        Run pipeline and write scratch outputs to output/.
        Prints hashes but does NOT compare.

    python benchmarks/stage1_xy_single_sweep/run_benchmark.py --check
        Run pipeline, write scratch outputs to output/, then compare
        SHA256 hashes against manifest.json (which matches frozen/).
        Exits with code 1 on any mismatch.

    python benchmarks/stage1_xy_single_sweep/run_benchmark.py --regenerate
        Run pipeline, overwrite frozen/ with new outputs, rewrite
        manifest.json hashes with new values.
        Use this after an intentional parameter or code change.
        Commit the updated frozen/ files and manifest.json together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
FROZEN_DIR = BENCH_DIR / "frozen"   # committed, inspectable
OUTPUT_DIR = BENCH_DIR / "output"   # scratch, gitignored
MANIFEST_PATH = BENCH_DIR / "manifest.json"

# Add src/ to path so this script works when run directly without install.
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Parameters (must match manifest.json)
# ---------------------------------------------------------------------------

PARAMS = {
    "model": "XY",
    "L": 16,
    "T": 0.9,
    "seed": 42,
    "J": 1.0,
    "N_equil": 1000,
    "N_meas": 2000,
    "N_thin": 50,
    "r_max": 4.0,
}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_benchmark() -> dict:
    """Execute pipeline and return dict of computed hashes + counts."""
    from topostream.extract.pairing import pair_vortices
    from topostream.extract.vortices import extract_vortices
    from topostream.io.schema_validate import validate_token
    from topostream.metrics.clock import compute_psi6
    from topostream.simulate.xy_numba import run_xy

    p = PARAMS
    prov = {
        "model": p["model"], "L": p["L"], "T": p["T"], "seed": p["seed"],
        "sweep_index": 0, "schema_version": "1.1.0",
        "N_equil": p["N_equil"], "N_meas": p["N_meas"], "N_thin": p["N_thin"],
    }

    print(f"Running XY benchmark: L={p['L']}, T={p['T']}, seed={p['seed']}, "
          f"N_equil={p['N_equil']}, N_meas={p['N_meas']}, N_thin={p['N_thin']}")

    result = run_xy(
        L=p["L"], T=p["T"], J=p["J"],
        N_equil=p["N_equil"], N_meas=p["N_meas"], N_thin=p["N_thin"],
        seed=p["seed"],
    )
    cfg = result["configs"][-1]

    vortex_tokens = extract_vortices(cfg, prov)
    vortices = [t for t in vortex_tokens if t["vortex"]["charge"] == +1]
    antivortices = [t for t in vortex_tokens if t["vortex"]["charge"] == -1]
    pair_result = pair_vortices(vortices, antivortices, p["L"],
                                r_max=p["r_max"], provenance=prov)
    all_tokens = vortex_tokens + pair_result["pairs"]

    # Schema-validate every token.
    for i, tok in enumerate(all_tokens):
        validate_token(tok)

    # Write artifacts.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    L = p["L"]
    T = p["T"]
    seed = p["seed"]
    model = p["model"]
    base = f"{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}"

    npy_path = OUTPUT_DIR / f"sim_{base}.npy"
    jsonl_path = OUTPUT_DIR / f"tokens_{base}.jsonl"
    summary_path = OUTPUT_DIR / f"summary_{base}.json"

    np.save(npy_path, cfg)

    with jsonl_path.open("w") as f:
        for tok in all_tokens:
            f.write(json.dumps(tok, sort_keys=True) + "\n")

    summary = {
        "model": model, "L": L, "T": T, "seed": seed,
        "n_vortices": len(vortex_tokens),
        "n_pairs": len(pair_result["pairs"]),
        "f_paired": pair_result["f_paired"],
        "upsilon": result["helicity"],
        "upsilon_err": result["helicity_err"],
        "r_max_used": p["r_max"],
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Compute hashes.
    cfg_bytes = cfg.astype(">f8").tobytes()
    cfg_sha256 = hashlib.sha256(cfg_bytes).hexdigest()

    jsonl_content = jsonl_path.read_text(encoding="utf-8")
    tok_sha256 = hashlib.sha256(jsonl_content.encode()).hexdigest()

    summary_canonical = json.dumps(
        {k: summary[k] for k in sorted(summary)},
        sort_keys=True,
    )
    sum_sha256 = hashlib.sha256(summary_canonical.encode()).hexdigest()

    counts = {
        "n_vortex_tokens": len(vortex_tokens),
        "n_pair_tokens": len(pair_result["pairs"]),
        "n_configs_thinned": len(result["configs"]),
        "f_paired": pair_result["f_paired"],
        "helicity": result["helicity"],
        "helicity_err": result["helicity_err"],
    }

    print(f"  n_vortex_tokens  : {counts['n_vortex_tokens']}")
    print(f"  n_pair_tokens    : {counts['n_pair_tokens']}")
    print(f"  f_paired         : {counts['f_paired']:.6f}")
    print(f"  helicity         : {counts['helicity']:.8f}")
    print(f"  helicity_err     : {counts['helicity_err']:.8f}")
    print(f"  spin_config_sha256  : {cfg_sha256}")
    print(f"  tokens_jsonl_sha256 : {tok_sha256}")
    print(f"  summary_sha256      : {sum_sha256}")
    print(f"Scratch artifacts written to: {OUTPUT_DIR}")

    return {
        "spin_config_sha256": cfg_sha256,
        "tokens_jsonl_sha256": tok_sha256,
        "summary_sha256": sum_sha256,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def check_against_manifest(actual: dict) -> bool:
    """Compare actual hashes against manifest.json. Returns True if all pass."""
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    frozen = manifest["hashes"]
    expected_counts = manifest["expected_counts"]
    expected_scalars = manifest["expected_scalars"]

    failures: list[str] = []

    for key in ("spin_config_sha256", "tokens_jsonl_sha256", "summary_sha256"):
        if actual[key] != frozen[key]:
            failures.append(
                f"  HASH MISMATCH {key}:\n"
                f"    expected: {frozen[key]}\n"
                f"    actual  : {actual[key]}"
            )

    # Count checks.
    c = actual["counts"]
    for field in ("n_vortex_tokens", "n_pair_tokens", "n_configs_thinned"):
        if c[field] != expected_counts[field]:
            failures.append(
                f"  COUNT MISMATCH {field}: "
                f"expected={expected_counts[field]} actual={c[field]}"
            )

    # Scalar checks with tolerance.
    for field in ("helicity", "helicity_err"):
        exp = expected_scalars[field]
        got = c[field]
        if abs(got - exp) > 1e-6:
            failures.append(
                f"  SCALAR MISMATCH {field}: expected={exp:.8f} actual={got:.8f}"
            )

    if failures:
        print("BENCHMARK CHECK FAILED:")
        for f in failures:
            print(f)
        return False

    print("BENCHMARK CHECK PASSED: all hashes, counts, and scalars match manifest.")
    return True


# ---------------------------------------------------------------------------
# Regenerate manifest hashes
# ---------------------------------------------------------------------------

def regenerate_manifest(actual: dict) -> None:
    """Rewrite manifest.json hashes and overwrite frozen/ with current outputs."""
    import shutil

    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    manifest["hashes"]["spin_config_sha256"] = actual["spin_config_sha256"]
    manifest["hashes"]["tokens_jsonl_sha256"] = actual["tokens_jsonl_sha256"]
    manifest["hashes"]["summary_sha256"] = actual["summary_sha256"]

    c = actual["counts"]
    manifest["expected_counts"]["n_vortex_tokens"] = c["n_vortex_tokens"]
    manifest["expected_counts"]["n_pair_tokens"] = c["n_pair_tokens"]
    manifest["expected_counts"]["n_configs_thinned"] = c["n_configs_thinned"]
    manifest["expected_counts"]["f_paired"] = c["f_paired"]

    manifest["expected_scalars"]["helicity"] = c["helicity"]
    manifest["expected_scalars"]["helicity_err"] = c["helicity_err"]

    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest regenerated: {MANIFEST_PATH}")

    # Copy scratch outputs into frozen/ so they match the new manifest.
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    for src_file in OUTPUT_DIR.iterdir():
        dst = FROZEN_DIR / src_file.name
        shutil.copy2(src_file, dst)
        print(f"  frozen/{src_file.name} updated")
    print(f"Commit frozen/ and manifest.json together.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 XY single-sweep benchmark"
    )
    parser.add_argument("--check", action="store_true",
                        help="Compare outputs to frozen manifest hashes")
    parser.add_argument("--regenerate", action="store_true",
                        help="Rewrite manifest hashes with current run values")
    args = parser.parse_args()

    actual = run_benchmark()

    if args.check:
        ok = check_against_manifest(actual)
        sys.exit(0 if ok else 1)
    elif args.regenerate:
        regenerate_manifest(actual)


if __name__ == "__main__":
    main()
