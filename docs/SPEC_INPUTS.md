# SPEC_INPUTS.md — Input Specification
schema_version: 1.0.0

---

## 1. Simulation Mode (primary)

### 1.1 Accepted spin-field forms
| Field type | Array shape | Dtype | Notes |
|---|---|---|---|
| Angle field | `(L, L)` float64 | radians | θ(x,y) ∈ [−π, π) |
| Vector field | `(L, L, 2)` float64 | unitless | (Sx, Sy); must satisfy Sx²+Sy² = 1 ± 1e-6 |

Conversion rule: if vector field supplied, `θ = arctan2(Sy, Sx)`.
All inputs are validated on load; non-unit vectors raise `ValueError`.

### 1.2 Lattice conventions
- Square lattice, periodic boundary conditions (PBC) in both x and y.
- Coordinates: `(i, j)` where i indexes rows (y-axis), j indexes columns (x-axis).
- Plaquette (i,j) is defined by corners (i,j), (i+1,j), (i+1,j+1), (i,j+1) with PBC wrapping.
- Origin (0,0) is top-left corner.

### 1.3 Temperature sweep
- Provide a monotone sequence `T_list` (ascending or descending; label direction in provenance).
- Each temperature yields exactly one spin configuration (post-equilibration snapshot).
- Multiple seeds produce independent sweeps; all are stored separately.

---

## 2. Map Mode (Stage 3, adapter-explicit)

### 2.1 Supported map families
Map mode is NOT probe-agnostic. Only the following families are supported.
Each has an explicit inversion assumption. Any other input must raise `UNSUPPORTED`.

| Family ID | Description | Inversion to angle field | Caveats |
|---|---|---|---|
| `SCALAR_REAL` | Re[ψ(x,y)] | Requires paired `SCALAR_IMAG`; θ = arctan2(Im,Re) | Both channels must come from same source |
| `SCALAR_IMAG` | Im[ψ(x,y)] | As above | As above |
| `COMPLEX_FIELD` | Complex 2D array ψ(x,y) | θ = arctan2(Im(ψ), Re(ψ)) | Amplitude discarded with logged warning |
| `VECTOR_2D` | Two-channel (Vx,Vy) | θ = arctan2(Vy,Vx) after normalization | Zero-magnitude pixels → masked NaN |
| `PHASE_IMAGE` | Direct phase map in [−π, π) | Identity (no conversion) | Caller asserts this is already an angle field |

### 2.2 Required metadata for map mode
Every map-mode input MUST supply a sidecar JSON:
```json
{
  "map_family": "SCALAR_REAL",
  "source_description": "what physical quantity this map represents",
  "pixel_to_lattice_scale": 1.0,
  "preprocessing_applied": ["blur_sigma_1.0", "downsample_2x"],
  "schema_version": "1.0.0"
}
```

### 2.3 Masked pixels
Pixels where inversion is undefined are marked NaN.
Vortex extraction skips any plaquette containing ≥1 NaN corner. Log count.

---

## 3. File naming convention
`{mode}_{model}_{L}x{L}_T{T:.4f}_seed{seed:04d}.npy`
All spin configs: NumPy `.npy` float64. Map inputs: `.npy` + sidecar `.meta.json`.