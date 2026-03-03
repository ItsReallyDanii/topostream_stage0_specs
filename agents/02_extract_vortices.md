# agents/02_extract_vortices.md — Vortex Extraction Agent
Implements: src/topostream/extract/vortices.py

Read before starting: SPEC_FORMULAE.md §1–2, SPEC_ALGORITHMS.md §1, SPEC_VALIDATION.md §1

---

## Your deliverable
A vortex extractor that:
1. Accepts a spin configuration array (L×L float64 in radians) + provenance dict
2. Returns a list of vortex tokens conforming to the topology_event_stream schema
3. Uses wrap(Δθ) = arctan2(sin(Δθ), cos(Δθ)) for ALL angle differences — no exceptions

## Required function (public API)
```python
def extract_vortices(theta: np.ndarray, provenance: dict) -> list[dict]:
    # Returns list of vortex token dicts, each validating against schema
```

## Critical implementation note
The winding number loop must use:
```python
dAB = np.arctan2(np.sin(B - A), np.cos(B - A))
```
NOT:
```python
dAB = (B - A) % (2*np.pi)   # WRONG — do not use
```

## Validation gate (must pass before handoff)
tests/test_extract_vortices.py must pass all toy configs from SPEC_VALIDATION.md §1:
- Single vortex: 1 token, charge=+1
- Single antivortex: 1 token, charge=−1
- Bound pair d=4: 2 tokens, charges ±1
- Uniform: 0 tokens

## Do NOT
- Implement your own pairing logic here (that is agent 03's job)
- Silently drop non-integer winding numbers without logging
- Use any angle difference operator other than arctan2(sin, cos)