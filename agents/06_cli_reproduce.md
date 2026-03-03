# agents/06_cli_reproduce.md — CLI and One-Command Reproduce Agent
Implements: src/topostream/cli.py, configs/default.yaml, Makefile

Read before starting: SPEC_INPUTS.md §3, SPEC_UQ.md §1, SPEC_METRICS.md §3

---

## Your deliverable
A CLI that supports:

```bash
# Full reproduce run (generates all tokens + metrics + plots into results/)
python -m topostream.cli reproduce --config configs/default.yaml

# Single temperature sweep
python -m topostream.cli sweep --model XY --L 64 --T 0.9 --seed 42

# Validate all output tokens
python -m topostream.cli validate --results-dir results/

# Plot metrics
python -m topostream.cli plot --results-dir results/ --output figures/
```

## configs/default.yaml structure
```yaml
schema_version: "1.0.0"
models: ["XY", "clock6"]
L_values: [16, 32, 64]
T_range: {start: 0.5, stop: 1.5, n_points: 25}
seeds: [42, 43, 44, 45]
N_equil: 10000
N_meas: 50000
N_thin: 50
r_max_policy: "L/4"
output_dir: "results/"
figures_dir: "figures/"
```

## Makefile targets
```makefile
reproduce:
    python -m topostream.cli reproduce --config configs/default.yaml

validate:
    python -m topostream.cli validate --results-dir results/

test:
    python -m pytest tests/ -v

clean:
    rm -rf results/ figures/
```

## Do NOT
- Hardcode any path, L value, or temperature in the CLI logic
- Allow reproduce to succeed if schema validation fails
- Generate plots without saving them to figures/ with deterministic filenames