# agents/00_repo_rules.md — Global Repo Constraints (READ FIRST)
schema_version: 1.0.0

These rules apply to ALL agents. Violating any rule requires human review before merge.

---

## PHYSICS RULES
- The q=6 clock model has THREE phases and TWO BKT transitions (T₂ > T₁). Never conflate QLRO with clock-ordered.
- The helicity modulus Υ(L,T) is a required metric in every simulation run. It is not optional.
- All angle differences use wrap(Δθ) = arctan2(sin(Δθ), cos(Δθ)). No other wrapping is permitted.
- Vortex pairing uses Hungarian min-cost matching (scipy.optimize.linear_sum_assignment) only.
- "Events" are temperature-sweep snapshot deltas, NOT physical time dynamics. Use token_type="sweep_delta".

## CPU RULES
- All inner MC loops (per-site spin update) MUST be Numba-jitted (@numba.njit) in production code.
- Pure Python per-spin update loops are forbidden in any file under src/topostream/simulate/.
- Job-level parallelism is allowed (one subprocess per seed/temperature).
- Target: L=64 sweep in < 5 minutes on a 4-core laptop.

## SCHEMA RULES
- Every output token must validate against schemas/topology_event_stream.schema.json.
- All provenance fields (seed, L, T, sweep_index, schema_version) must be populated.
- Schema version bumps require updating ALL spec docs and a changelog entry.
- Do not add fields to tokens without a corresponding spec doc update.

## REPRODUCIBILITY RULES
- All seeds must be explicit integers from the list [42,43,44,45,46,47,48,49].
- All file outputs follow naming convention in SPEC_INPUTS.md §3.
- One-command reproduce: `python -m topostream.cli reproduce --config configs/default.yaml`
- No hardcoded paths; all paths from config file.

## CODE QUALITY RULES
- No agent writes code that another agent's spec contradicts. Conflicts escalate to human.
- All modules have corresponding test files in tests/.
- No print() statements in library code; use Python logging module.
- Type hints required on all public functions.