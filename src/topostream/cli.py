"""
src/topostream/cli.py
=======================
CLI entrypoint for the topostream pipeline.

Supports four sub-commands per agents/06_cli_reproduce.md:
    reproduce  — full sweep over L/T grid, extract, pair, emit tokens
    sweep      — single temperature run
    validate   — validate all tokens in results/ against JSON schema
    plot       — generate metric plots into figures/

Usage:
    python -m topostream.cli reproduce --config configs/default.yaml
    python -m topostream.cli sweep --model XY --L 16 --T 0.9 --seed 42
    python -m topostream.cli validate --results-dir results/
    python -m topostream.cli plot --results-dir results/ --output figures/
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("topostream.cli")


# ===================================================================
# Config loader
# ===================================================================

def _load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config file.  Requires PyYAML (declared in pyproject.toml)."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ERROR: PyYAML is required for config loading but is not installed.\n"
            "Install it with:  pip install pyyaml\n"
            "Or install the full package:  pip install topostream"
        ) from exc
    with open(path) as f:
        return yaml.safe_load(f)


def _make_T_list(t_range: dict) -> list[float]:
    """Build monotone temperature list from config range dict."""
    return list(np.linspace(
        t_range["start"], t_range["stop"], t_range["n_points"],
    ))


def _parse_r_max(policy: str, L: int) -> float:
    """Evaluate r_max policy string like 'L/4'."""
    # Only support L/N pattern for safety
    if policy.startswith("L/"):
        denom = int(policy[2:])
        return L / denom
    return float(policy)


# ===================================================================
# Provenance factory
# ===================================================================

def _make_provenance(
    model: str, L: int, T: float, seed: int,
    sweep_index: int, config: dict | None = None,
) -> dict:
    prov: dict[str, Any] = {
        "model": model,
        "L": L,
        "T": T,
        "seed": seed,
        "sweep_index": sweep_index,
        "schema_version": "1.1.0",
    }
    if config:
        for k in ("N_equil", "N_meas", "N_thin", "r_max_policy"):
            if k in config:
                prov[k] = config[k]
    return prov


# ===================================================================
# Single-sweep pipeline
# ===================================================================

def _run_single_sweep(
    model: str,
    L: int,
    T: float,
    seed: int,
    N_equil: int,
    N_meas: int,
    N_thin: int,
    r_max: float,
    output_dir: Path,
    sweep_index: int = 0,
    config: dict | None = None,
) -> list[dict]:
    """Run simulation → extraction → pairing → emit tokens for one (L,T,seed)."""
    from topostream.extract.vortices import extract_vortices
    from topostream.extract.pairing import pair_vortices
    from topostream.metrics.clock import compute_psi6, compute_clock6_order
    from topostream.io.schema_validate import validate_token

    logger.info("sweep: model=%s L=%d T=%.4f seed=%d", model, L, T, seed)

    # 1. Simulate — dispatch on model
    if model == "XY":
        from topostream.simulate.xy_numba import run_xy
        result = run_xy(
            L=L, T=T, J=1.0,
            N_equil=N_equil, N_meas=N_meas, N_thin=N_thin,
            seed=seed,
        )
    elif model == "clock6":
        from topostream.simulate.clock6_numba import run_clock6
        result = run_clock6(
            L=L, T=T, J=1.0,
            N_equil=N_equil, N_meas=N_meas, N_thin=N_thin,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model!r}. Supported: 'XY', 'clock6'.")
    cfg = result["configs"][-1]  # Last thinned config

    # Save spin config as .npy (SPEC_INPUTS §3)
    fname = f"sim_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.npy"
    np.save(output_dir / fname, cfg)

    prov = _make_provenance(model, L, T, seed, sweep_index, config)

    # 2. Extract vortices
    vortex_tokens = extract_vortices(cfg, prov)

    # 3. Pair
    vortices = [t for t in vortex_tokens if t["vortex"]["charge"] == +1]
    antivortices = [t for t in vortex_tokens if t["vortex"]["charge"] == -1]
    pair_result = pair_vortices(
        vortices, antivortices, L, r_max=r_max, provenance=prov,
    )

    # 4. Collect all tokens
    all_tokens: list[dict] = []
    all_tokens.extend(vortex_tokens)
    all_tokens.extend(pair_result["pairs"])

    # 5. Schema-validate every token — FAIL LOUDLY on first error
    for i, tok in enumerate(all_tokens):
        try:
            validate_token(tok)
        except Exception as e:
            logger.error("Schema validation FAILED on token %d: %s", i, e)
            logger.error("Token: %s", json.dumps(tok, indent=2, default=str))
            raise SystemExit(
                f"FATAL: token {i} failed schema validation: {e}"
            ) from e

    # 6. Write tokens to JSONL
    tokens_file = output_dir / f"tokens_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
    with tokens_file.open("w") as f:
        for tok in all_tokens:
            f.write(json.dumps(tok, default=_json_default) + "\n")

    # 7. Write summary JSON (for metric aggregation)
    upsilon = result["helicity"]
    upsilon_err = result["helicity_err"]
    rho = len(vortex_tokens) / (L * L)
    f_paired = pair_result["f_paired"]
    psi6_val = float(abs(compute_psi6(cfg)))

    # Model-aware primary order parameter.
    # For clock6: clock6_order is the meaningful discriminant; |ψ₆| is
    # algebraically near-1 for all discrete clock configs and is kept only
    # as a diagnostic.
    # For XY: |ψ₆| is a continuous observable and may be used diagnostically;
    # no single-site order parameter is the standard primary metric here.
    if model == "clock6":
        primary_order_name = "clock6_order"
        primary_order_value = compute_clock6_order(cfg)
    else:  # XY and any future continuous model
        primary_order_name = "psi6_mag"
        primary_order_value = psi6_val

    summary = {
        "model": model, "L": L, "T": T, "seed": seed,
        "n_vortices": len(vortex_tokens),
        "n_pairs": len(pair_result["pairs"]),
        "rho": rho, "f_paired": f_paired,
        "upsilon": upsilon, "upsilon_err": upsilon_err,
        "psi6_mag": psi6_val,           # diagnostic for all models
        "primary_order_name": primary_order_name,
        "primary_order_value": primary_order_value,
        "r_max_used": r_max,
    }
    # For clock6 also store clock6_order explicitly so it is introspectable
    # without having to read primary_order_name first.
    if model == "clock6":
        summary["clock6_order"] = primary_order_value

    summary_file = output_dir / f"summary_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    logger.info(
        "  → %d vortex tokens, %d pair tokens, Υ=%.4f, %s=%.4f",
        len(vortex_tokens), len(pair_result["pairs"]), upsilon,
        primary_order_name, primary_order_value,
    )

    return all_tokens


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ===================================================================
# Sub-commands
# ===================================================================

def cmd_reproduce(args: argparse.Namespace) -> None:
    """Full reproduce run over L/T grid."""
    config = _load_config(args.config)
    output_dir = Path(config.get("output_dir", "results/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    models = config.get("models", ["XY"])
    L_values = config.get("L_values", [16])
    T_list = _make_T_list(config["T_range"])
    seeds = config.get("seeds", [42])
    N_equil = config.get("N_equil", 10000)
    N_meas = config.get("N_meas", 50000)
    N_thin = config.get("N_thin", 50)
    r_max_policy = config.get("r_max_policy", "L/4")

    total_runs = len(models) * len(L_values) * len(T_list) * len(seeds)
    logger.info("reproduce: %d total sweeps", total_runs)

    all_tokens: list[dict] = []
    sweep_idx = 0

    # --- sweep_delta tracking (per model/L/seed) ---
    prev_state: dict[str, dict] = {}

    for model in models:
        for L in L_values:
            r_max = _parse_r_max(r_max_policy, L)
            for seed in seeds:
                prev_key = f"{model}_{L}_{seed}"
                for T in T_list:
                    tokens = _run_single_sweep(
                        model=model, L=L, T=T, seed=seed,
                        N_equil=N_equil, N_meas=N_meas, N_thin=N_thin,
                        r_max=r_max, output_dir=output_dir,
                        sweep_index=sweep_idx, config=config,
                    )
                    all_tokens.extend(tokens)

                    # --- Emit sweep_delta tokens ---
                    current_state = _extract_sweep_state(tokens, L)
                    if prev_key in prev_state:
                        deltas = _make_sweep_deltas(
                            prev_state[prev_key], current_state,
                            T_from=prev_state[prev_key]["T"], T_to=T,
                            provenance=_make_provenance(
                                model, L, T, seed, sweep_idx, config,
                            ),
                        )
                        all_tokens.extend(deltas)
                        # Write deltas to their own JSONL
                        delta_file = output_dir / (
                            f"deltas_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.jsonl"
                        )
                        with delta_file.open("w") as f:
                            for d in deltas:
                                f.write(json.dumps(d, default=_json_default) + "\n")

                    current_state["T"] = T
                    prev_state[prev_key] = current_state
                    sweep_idx += 1

    logger.info("reproduce complete: %d total tokens emitted.", len(all_tokens))

    # --- Run multi-seed aggregation ---
    from topostream.aggregate.confidence import aggregate_results_dir
    agg_results = aggregate_results_dir(output_dir)
    logger.info(
        "aggregation complete: %d condition groups processed.",
        len(agg_results),
    )


def cmd_sweep(args: argparse.Namespace) -> None:
    """Single temperature sweep."""
    output_dir = Path("results/")
    output_dir.mkdir(parents=True, exist_ok=True)
    r_max = args.L / 4.0
    _run_single_sweep(
        model=args.model, L=args.L, T=args.T, seed=args.seed,
        N_equil=args.N_equil, N_meas=args.N_meas, N_thin=args.N_thin,
        r_max=r_max, output_dir=output_dir,
    )


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate all JSONL tokens in results dir against schema."""
    from topostream.io.schema_validate import validate_token

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory does not exist: %s", results_dir)
        raise SystemExit(1)

    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("No .jsonl files found in %s", results_dir)
        raise SystemExit(1)

    total = 0
    errors = 0
    for jf in jsonl_files:
        with jf.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                tok = json.loads(line)
                total += 1
                try:
                    validate_token(tok)
                except Exception as e:
                    errors += 1
                    logger.error(
                        "FAIL: %s line %d: %s", jf.name, line_num, e,
                    )
                    if errors == 1:
                        logger.error("First failing token: %s",
                                     json.dumps(tok, indent=2, default=str))

    if errors:
        logger.error("%d / %d tokens FAILED validation.", errors, total)
        raise SystemExit(1)

    logger.info("validate: %d tokens from %d files — ALL PASSED.",
                total, len(jsonl_files))


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate metric plots from results summaries."""
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(results_dir.glob("summary_*.json"))
    if not summary_files:
        logger.error("No summary files in %s", results_dir)
        raise SystemExit(1)

    summaries: list[dict] = []
    for sf in summary_files:
        with sf.open() as f:
            summaries.append(json.load(f))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ERROR: matplotlib is required for plotting but is not installed.\n"
            "Install it with:  pip install matplotlib\n"
            "Or install the plot extra:  pip install 'topostream[plot]'"
        ) from exc

    # Group by (model, L)
    from collections import defaultdict
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for s in summaries:
        grouped[(s["model"], s["L"])].append(s)

    # Plot Υ vs T
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, L), data in sorted(grouped.items()):
        data.sort(key=lambda x: x["T"])
        Ts = [d["T"] for d in data]
        Ys = [d["upsilon"] for d in data]
        ax.plot(Ts, Ys, "o-", markersize=3, label=f"{model} L={L}")
    ax.axvline(0.893, color="gray", linestyle="--", alpha=0.5, label="T_BKT ≈ 0.893")
    ax.set_xlabel("T")
    ax.set_ylabel("Υ(L,T)")
    ax.set_title("Helicity modulus vs Temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "helicity_vs_T.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "helicity_vs_T.png")

    # Plot ρ vs T
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, L), data in sorted(grouped.items()):
        data.sort(key=lambda x: x["T"])
        Ts = [d["T"] for d in data]
        rhos = [d["rho"] for d in data]
        ax.plot(Ts, rhos, "o-", markersize=3, label=f"{model} L={L}")
    ax.set_xlabel("T")
    ax.set_ylabel("ρ(T)")
    ax.set_title("Vortex density vs Temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "vortex_density_vs_T.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "vortex_density_vs_T.png")

    # Plot model-aware primary order parameter vs T.
    # Uses summary["primary_order_value"] which is set per-model:
    #   clock6 → clock6_order (the meaningful discriminant)
    #   XY     → psi6_mag     (kept as the XY diagnostic default)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, L), data in sorted(grouped.items()):
        data.sort(key=lambda x: x["T"])
        Ts = [d["T"] for d in data]
        ys = [d.get("primary_order_value", d.get("psi6_mag", 0)) for d in data]
        op_name = data[0].get("primary_order_name", "psi6_mag") if data else "psi6_mag"
        ax.plot(Ts, ys, "o-", markersize=3, label=f"{model} L={L} ({op_name})")
    ax.set_xlabel("T")
    ax.set_ylabel("primary order parameter")
    ax.set_title("Order parameter vs Temperature (model-aware)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "order_parameter_vs_T.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "order_parameter_vs_T.png")

    # Plot |ψ₆| vs T — diagnostic only.
    # NOTE: For clock6, |ψ₆| ≈ 1 by algebra for all discrete configs and
    # does NOT discriminate temperature.  This plot is kept as a diagnostic
    # cross-check, not as a primary observable.
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, L), data in sorted(grouped.items()):
        data.sort(key=lambda x: x["T"])
        Ts = [d["T"] for d in data]
        psi6s = [d.get("psi6_mag", 0) for d in data]
        ax.plot(Ts, psi6s, "o-", markersize=3, label=f"{model} L={L}")
    ax.set_xlabel("T")
    ax.set_ylabel("|ψ₆|(T)")
    ax.set_title("|ψ₆| vs Temperature — DIAGNOSTIC (not primary for clock6)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "psi6_vs_T_diagnostic.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "psi6_vs_T_diagnostic.png")

    logger.info("plot: %d figures saved to %s", 4, output_dir)


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Run multi-seed aggregation on an existing results directory."""
    from topostream.aggregate.confidence import aggregate_results_dir

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory does not exist: %s", results_dir)
        raise SystemExit(1)

    match_tolerance = getattr(args, "match_tolerance", 1.0)
    agg_results = aggregate_results_dir(results_dir, match_tolerance=match_tolerance)

    if not agg_results:
        logger.warning("No conditions found to aggregate.")
        raise SystemExit(1)

    for key, summary in sorted(agg_results.items()):
        model, L, T = key
        logger.info(
            "  %s L=%d T=%.4f: stability=%.3f, %d consensus clusters, %d seeds",
            model, L, T,
            summary["global_detection_stability"],
            summary["n_consensus_clusters"],
            summary["N_seeds"],
        )

    logger.info("aggregate: %d condition groups processed.", len(agg_results))


# ===================================================================
# sweep_delta helpers
# ===================================================================

def _extract_sweep_state(tokens: list[dict], L: int) -> dict:
    """Extract summary state from tokens for sweep_delta computation."""
    n_v = sum(1 for t in tokens if t["token_type"] == "vortex"
              and t["vortex"]["charge"] == +1)
    n_a = sum(1 for t in tokens if t["token_type"] == "vortex"
              and t["vortex"]["charge"] == -1)
    n_pairs = sum(1 for t in tokens if t["token_type"] == "pair")
    total_defects = n_v + n_a
    rho = total_defects / (L * L) if L > 0 else 0.0
    f_paired = (2 * n_pairs) / total_defects if total_defects > 0 else 0.0
    return {
        "rho": rho,
        "f_paired": f_paired,
        "n_pairs": n_pairs,
        "n_free": total_defects - 2 * n_pairs,
    }


def _make_sweep_deltas(
    prev: dict, curr: dict,
    T_from: float, T_to: float,
    provenance: dict,
) -> list[dict]:
    """Generate sweep_delta tokens between consecutive temperature snapshots."""
    from topostream.io.schema_validate import validate_token

    delta_specs = [
        ("vortex_density_change", curr["rho"] - prev["rho"]),
        ("pairing_fraction_change", curr["f_paired"] - prev["f_paired"]),
        ("pair_count_change", curr["n_pairs"] - prev["n_pairs"]),
        ("free_vortex_count_change", curr["n_free"] - prev["n_free"]),
    ]

    deltas: list[dict] = []
    for delta_type, delta_value in delta_specs:
        tok = {
            "schema_version": "1.1.0",
            "token_type": "sweep_delta",
            "provenance": provenance,
            "sweep_delta": {
                "delta_type": delta_type,
                "T_from": T_from,
                "T_to": T_to,
                "delta_value": float(delta_value),
            },
        }
        validate_token(tok)
        deltas.append(tok)
    return deltas


# ===================================================================
# Argument parser
# ===================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topostream",
        description="TopoStream: topological defect analysis pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # reproduce
    p_rep = sub.add_parser("reproduce", help="Full reproduce run")
    p_rep.add_argument("--config", required=True, help="Path to YAML config")

    # sweep
    p_sw = sub.add_parser("sweep", help="Single temperature sweep")
    p_sw.add_argument("--model", default="XY", choices=["XY", "clock6"])
    p_sw.add_argument("--L", type=int, required=True)
    p_sw.add_argument("--T", type=float, required=True)
    p_sw.add_argument("--seed", type=int, default=42)
    p_sw.add_argument("--N-equil", type=int, default=10000)
    p_sw.add_argument("--N-meas", type=int, default=50000)
    p_sw.add_argument("--N-thin", type=int, default=50)

    # validate
    p_val = sub.add_parser("validate", help="Validate output tokens")
    p_val.add_argument("--results-dir", required=True,
                       help="Directory containing .jsonl token files")

    # plot
    p_plt = sub.add_parser("plot", help="Generate metric plots")
    p_plt.add_argument("--results-dir", required=True)
    p_plt.add_argument("--output", required=True, help="Output figures directory")

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Run multi-seed aggregation")
    p_agg.add_argument("--results-dir", required=True,
                       help="Directory containing per-seed .jsonl token files")
    p_agg.add_argument("--match-tolerance", type=float, default=1.0,
                       help="Match tolerance in plaquette units (default: 1.0)")

    return parser


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "reproduce": cmd_reproduce,
        "sweep": cmd_sweep,
        "validate": cmd_validate,
        "plot": cmd_plot,
        "aggregate": cmd_aggregate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
