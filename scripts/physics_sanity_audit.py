"""
scripts/physics_sanity_audit.py
================================
Diagnostic runner that checks basic equilibrium trends for both models
(XY and Clock6) WITHOUT modifying any source under src/.

Outputs:
    - Console summary table
    - results/diagnostics/physics_sanity_XY.json
    - results/diagnostics/physics_sanity_clock6.json

Usage:
    python scripts/physics_sanity_audit.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project imports (read-only usage — nothing in src/ is modified)
# ---------------------------------------------------------------------------
from topostream.simulate.xy_numba import run_xy
from topostream.simulate.clock6_numba import run_clock6
from topostream.metrics.helicity import compute_helicity
from topostream.metrics.clock import compute_psi6, compute_clock6_order
from topostream.extract.vortices import extract_vortices

logging.basicConfig(
    level=logging.ERROR,  # suppress routine INFO/WARNING from simulators
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("physics_sanity_audit")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEMPERATURES = [0.3, 0.6, 0.9, 1.2, 2.0]
SEED = 42
L_VALUES = [16]

# Short MC parameters — enough for diagnostics, fast on a laptop
N_EQUIL = 2_000
N_MEAS = 2_000
N_THIN = 50


# ---------------------------------------------------------------------------
# Acceptance-rate estimator (script-only; does NOT touch src/)
# ---------------------------------------------------------------------------
def _estimate_acceptance_rate(
    model: str, L: int, T: float, seed: int, n_sweeps: int = 200,
) -> float:
    """Estimate Metropolis acceptance rate by comparing configs pre/post sweep.

    We measure the fraction of sites that CHANGED after each sweep —
    a proxy for acceptance rate.  This avoids instrumenting the core sim.
    """
    if model == "XY":
        from topostream.simulate.xy_numba import (
            init_config, _metropolis_sweep, _STEP_SIZE,
        )
        theta = init_config(L, seed)
    elif model == "clock6":
        from topostream.simulate.clock6_numba import (
            init_config_clock6, _metropolis_sweep_clock6,
            _ALLOWED_ANGLES, _STEP_SIZE,
        )
        theta = init_config_clock6(L, seed)
    else:
        return float("nan")

    # Build an RNG state matching what the sim uses internally
    rng_state = np.array([np.uint64(seed if seed != 0 else 1)], dtype=np.uint64)

    n_sites = L * L
    total_changed = 0
    total_sites = 0

    for _ in range(n_sweeps):
        before = theta.copy()
        if model == "XY":
            _metropolis_sweep(theta, L, T, 1.0, rng_state, _STEP_SIZE)
        else:
            _metropolis_sweep_clock6(
                theta, L, T, 1.0, rng_state, _STEP_SIZE, _ALLOWED_ANGLES,
            )
        changed = int(np.sum(before != theta))
        total_changed += changed
        total_sites += n_sites

    return total_changed / total_sites if total_sites > 0 else 0.0


# ---------------------------------------------------------------------------
# Single (model, L, T) diagnostic
# ---------------------------------------------------------------------------
def _diagnose_single(
    model: str, L: int, T: float, seed: int,
) -> dict[str, Any]:
    """Run one (model, L, T, seed) and return a diagnostic dict."""
    # --- Simulate ---
    if model == "XY":
        result = run_xy(L=L, T=T, J=1.0,
                        N_equil=N_EQUIL, N_meas=N_MEAS, N_thin=N_THIN,
                        seed=seed)
    else:
        result = run_clock6(L=L, T=T, J=1.0,
                            N_equil=N_EQUIL, N_meas=N_MEAS, N_thin=N_THIN,
                            seed=seed)

    cfg = result["configs"][-1]
    energy_trace = result["energy_per_spin"]

    # --- Metrics ---
    mean_energy = float(np.mean(energy_trace))
    helicity = result["helicity"]
    helicity_err = result["helicity_err"]
    psi6_mag = float(abs(compute_psi6(cfg)))

    # --- Clock6 population-concentration order (meaningful for discrete models) ---
    clock6_order = float(compute_clock6_order(cfg))

    # --- Vortex density ---
    prov = {"model": model, "L": L, "T": T, "seed": seed,
            "sweep_index": 0, "schema_version": "1.0.0"}
    vortex_tokens = extract_vortices(cfg, prov)
    n_vortices = len(vortex_tokens)
    rho = n_vortices / (L * L)

    # --- Acceptance rate (script-only proxy) ---
    accept_rate = _estimate_acceptance_rate(model, L, T, seed, n_sweeps=100)

    return {
        "model": model,
        "L": L,
        "T": T,
        "seed": seed,
        "mean_energy_per_spin": mean_energy,
        "helicity": helicity,
        "helicity_err": helicity_err,
        "psi6_mag": psi6_mag,
        "n_vortices": n_vortices,
        "vortex_density_rho": rho,
        "acceptance_rate_proxy": accept_rate,
        "clock6_order": clock6_order,
        "energy_std": float(np.std(energy_trace)),
        "N_equil": N_EQUIL,
        "N_meas": N_MEAS,
        "N_thin": N_THIN,
    }


# ---------------------------------------------------------------------------
# Full audit for one model
# ---------------------------------------------------------------------------
def audit_model(model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for L in L_VALUES:
        for T in TEMPERATURES:
            row = _diagnose_single(model, L, T, SEED)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------
def _print_table(model: str, rows: list[dict]) -> None:
    header = (
        f"{'T':>5s}  {'E/spin':>9s}  {'\u03a5':>9s}  {'|\u03c8\u2086|':>7s}  "
        f"{'\u03c1':>8s}  {'n_vort':>6s}  {'accept':>7s}  {'c6_ord':>7s}  {'E_std':>8s}"
    )
    print(f"\n===== {model}  L={rows[0]['L']}  seed={rows[0]['seed']}  "
          f"N_eq={N_EQUIL} N_m={N_MEAS} =====")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['T']:5.2f}  {r['mean_energy_per_spin']:9.4f}  "
            f"{r['helicity']:9.4f}  {r['psi6_mag']:7.4f}  "
            f"{r['vortex_density_rho']:8.5f}  {r['n_vortices']:6d}  "
            f"{r['acceptance_rate_proxy']:7.4f}  {r['clock6_order']:7.4f}  "
            f"{r['energy_std']:8.5f}"
        )


# ---------------------------------------------------------------------------
# JSON serialiser helper
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    out_dir = Path("results/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in ("XY", "clock6"):
        rows = audit_model(model)
        _print_table(model, rows)

        # Save JSON
        out_file = out_dir / f"physics_sanity_{model}.json"
        with out_file.open("w") as f:
            json.dump(rows, f, indent=2, default=_json_default)
        print(f"\nSaved: {out_file}")

    # --- Quick inline assertions (print PASS / WARN, never crash) ---
    print("\n--- Inline checks ---")

    for model in ("XY", "clock6"):
        path = out_dir / f"physics_sanity_{model}.json"
        with path.open() as f:
            data = json.load(f)

        # 1. No NaN / inf in energies
        all_ok = all(math.isfinite(r["mean_energy_per_spin"]) for r in data)
        _report(f"{model}: energies finite", all_ok)

        # 2. Υ(T=0.6) > Υ(T=1.2) for XY
        if model == "XY":
            h06 = next(r["helicity"] for r in data if abs(r["T"] - 0.6) < 0.01)
            h12 = next(r["helicity"] for r in data if abs(r["T"] - 1.2) < 0.01)
            _report(f"XY: Υ(0.6)={h06:.4f} > Υ(1.2)={h12:.4f}", h06 > h12)

        # 3. ρ(2.0) > ρ(0.3) — or WARN if frozen
        rho_hi = next(r["vortex_density_rho"] for r in data if abs(r["T"] - 2.0) < 0.01)
        rho_lo = next(r["vortex_density_rho"] for r in data if abs(r["T"] - 0.3) < 0.01)
        if rho_hi > rho_lo:
            _report(f"{model}: ρ(2.0)={rho_hi:.5f} > ρ(0.3)={rho_lo:.5f}", True)
        else:
            print(f"  WARN  {model}: ρ(2.0)={rho_hi:.5f} ≤ ρ(0.3)={rho_lo:.5f} "
                  "(model may be frozen/trapped at low T)")


def _report(label: str, ok: bool) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {label}")


# ---------------------------------------------------------------------------
# Map-mode synthetic diagnostics block
# ---------------------------------------------------------------------------

def _run_map_mode_diagnostics() -> None:
    """Optional map-mode synthetic block (Stage 3 groundwork).

    For each model, runs the forward-model pipeline at several degradation
    levels and prints: baseline vortex count, recovered vortex count, and
    fraction of NaN plaquettes skipped by the extractor.
    """
    import math as _math
    from topostream.simulate.xy_numba import init_config as _init_xy
    from topostream.simulate.clock6_numba import init_config_clock6 as _init_c6
    from topostream.map.forward_models import (
        to_vector_map, apply_blur, add_noise, mask_nan,
    )
    from topostream.map.adapters import vector_map_to_theta

    print("\n--- Map-mode synthetic diagnostics ---")
    _L = 16
    _SEED = 42
    _prov = lambda m: {"model": m, "L": _L, "T": 1.0, "seed": _SEED,
                        "sweep_index": 0, "schema_version": "1.0.0"}

    profiles = [
        ("clean",        dict(blur=0.0, noise=0.00, nan_f=0.00)),
        ("mild blur",    dict(blur=0.5, noise=0.00, nan_f=0.00)),
        ("mild noise",   dict(blur=0.0, noise=0.05, nan_f=0.00)),
        ("5% NaN",       dict(blur=0.0, noise=0.00, nan_f=0.05)),
        ("combined",     dict(blur=0.5, noise=0.05, nan_f=0.05)),
        ("extreme NaN",  dict(blur=0.0, noise=0.00, nan_f=0.70)),
    ]

    for model in ("XY", "clock6"):
        theta0 = _init_xy(_L, _SEED) if model == "XY" else _init_c6(_L, _SEED)
        n_base = len(extract_vortices(theta0, _prov(model)))
        print(f"\n  {model}  L={_L}  baseline_vortices={n_base}")
        print(f"  {'Degradation':<14s}  {'recovered':>9s}  {'nan_plaq%':>9s}  {'Δ':>6s}")
        print("  " + "-" * 45)

        for label, cfg in profiles:
            Mx, My = to_vector_map(theta0)
            if cfg["blur"] > 0:
                Mx = apply_blur(Mx, sigma=cfg["blur"])
                My = apply_blur(My, sigma=cfg["blur"])
            if cfg["noise"] > 0:
                Mx = add_noise(Mx, sigma=cfg["noise"], seed=_SEED)
                My = add_noise(My, sigma=cfg["noise"], seed=_SEED + 1)
            if cfg["nan_f"] > 0:
                Mx = mask_nan(Mx, nan_frac=cfg["nan_f"], seed=_SEED + 2)
                My = mask_nan(My, nan_frac=cfg["nan_f"], seed=_SEED + 3)

            theta_hat = vector_map_to_theta(Mx, My)

            # Count NaN plaquettes
            nan_plaq = 0
            for i in range(_L):
                for j in range(_L):
                    corners = [theta_hat[i, j],
                               theta_hat[i, (j + 1) % _L],
                               theta_hat[(i + 1) % _L, (j + 1) % _L],
                               theta_hat[(i + 1) % _L, j]]
                    if any(_math.isnan(c) for c in corners):
                        nan_plaq += 1

            n_rec = len(extract_vortices(theta_hat, _prov(model)))
            nan_pct = 100.0 * nan_plaq / (_L * _L)
            delta = n_rec - n_base
            delta_str = f"{delta:+d}" if n_base > 0 else "n/a"
            print(f"  {label:<14s}  {n_rec:>9d}  {nan_pct:>8.1f}%  {delta_str:>6s}")


if __name__ == "__main__":
    main()
    _run_map_mode_diagnostics()
