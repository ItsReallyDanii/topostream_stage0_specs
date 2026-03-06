# Stage 1 Benchmark: XY Single Sweep

Regression anchor for the core pipeline: simulate → extract → pair → schema-validate.
Not a publication benchmark. Claims reproducibility and structural integrity only.

## Directory layout

```
benchmarks/stage1_xy_single_sweep/
  frozen/              Committed frozen outputs — inspectable in-repo.
    sim_XY_16x16_T0.9000_seed0042.npy
    tokens_XY_16x16_T0.9000_seed0042.jsonl
    summary_XY_16x16_T0.9000_seed0042.json
  output/              Scratch rerun directory. Gitignored. Overwritten on each check run.
  manifest.json        Frozen SHA256 hashes + expected counts/scalars.
  expected_invariants.json  Structural invariant specification.
  run_benchmark.py     Runner with --check and --regenerate modes.
  README.md            This file.
```

---

## Exact parameters

| Parameter   | Value |
|-------------|-------|
| model       | XY    |
| L           | 16    |
| T           | 0.9   |
| seed        | 42    |
| N_equil     | 1000  |
| N_meas      | 2000  |
| N_thin      | 50    |
| r_max       | L/4 = 4.0 |

---

## Exact command

Validate rerun against frozen manifest hashes:

```
python benchmarks/stage1_xy_single_sweep/run_benchmark.py --check
```

Or via make:

```
make benchmark-check
```

Run and update both frozen outputs and manifest after intentional changes:

```
python benchmarks/stage1_xy_single_sweep/run_benchmark.py --regenerate
# then commit frozen/ + manifest.json together
```

---

## Expected artifact filenames

All relative to `benchmarks/stage1_xy_single_sweep/frozen/` (committed):

| File | Description |
|------|-------------|
| `sim_XY_16x16_T0.9000_seed0042.npy` | Spin configuration (numpy float64, big-endian on disk) |
| `tokens_XY_16x16_T0.9000_seed0042.jsonl` | All vortex + pair tokens, sort_keys canonical JSON |
| `summary_XY_16x16_T0.9000_seed0042.json` | Run summary (counts, helicity, etc.) |

These files are committed and directly inspectable. They are identical to what
`run_benchmark.py --check` produces in `output/` when the run is reproducible.

---

## What is being checked (invariants.json)

1. All three artifact files exist and are non-empty.
2. Every token in the JSONL validates against `schemas/topology_event_stream.schema.json`.
3. Every vortex token has `provenance.seed == 42`, `provenance.L == 16`, `provenance.T == 0.9`.
4. Every vortex token has `vortex.confidence` in [0.0, 1.0].
5. `vortex.charge` ∈ {-1, +1} for all vortex tokens.
6. Summary JSON contains keys: model, L, T, seed, n_vortices, n_pairs, f_paired, upsilon.
7. SHA256 hashes of spin config, token JSONL, and summary match frozen values.

---

## What is not being claimed

- Physics correctness (no comparison to known T_BKT).
- Statistical convergence (N_meas=2000 is too short for publication).
- Cross-platform byte identity (xorshift64 is deterministic; float64 arithmetic
  may differ across CPU architectures for very small rounding differences,
  but in practice is stable across x86-64 runs with the same NumPy/Numba versions).

---

## Byte identity on rerun

The xorshift64 PRNG is fully deterministic given the seed. On the same
machine with the same package versions, byte-identical output is expected.
The frozen hashes in `manifest.json` were generated on the development machine
and must be reproduced exactly by `python benchmarks/stage1_xy_single_sweep/run_benchmark.py --check`.

If hash checking fails on a different machine or after a package upgrade, the
manifest must be regenerated with `--regenerate` and the new hashes committed.
The test (`tests/test_benchmark_stage1.py`) checks hashes only when the frozen
output directory exists; otherwise it runs structural-only invariant checks.
