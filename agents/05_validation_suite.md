# agents/05_validation_suite.md — Validation Suite Agent
Implements: tests/test_*.py, scripts/run_validation.py

Read before starting: SPEC_VALIDATION.md (entire document)

---

## Your deliverable
A complete pytest test suite that covers every test in SPEC_VALIDATION.md:

1. tests/test_extract_vortices.py — toy configs §1
2. tests/test_pairing.py — pairing algorithm §2
3. tests/test_sim_xy.py — finite-size scaling §3
4. tests/test_noise_robustness.py — noise/perturbation §4
5. tests/test_schema_consistency.py — schema validation §5
6. tests/test_r_max_sensitivity.py — r_max sweep §6

## Running
```bash
python -m pytest tests/ -v --tb=short
```
All 6 files must pass with 0 failures and 0 errors before Stage 2 is closed.

## Schema validation helper
```python
import jsonschema, json
with open("schemas/topology_event_stream.schema.json") as f:
    SCHEMA = json.load(f)
def validate_token(token: dict):
    jsonschema.validate(token, SCHEMA)  # raises on failure
```

## Do NOT
- Write tests that pass by hardcoding expected values from a specific run
- Skip thermalization checks (they must actually run the MC)
- Use relative imports that break when run from repo root