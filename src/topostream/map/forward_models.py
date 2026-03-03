"""
src/topostream/map/forward_models.py
======================================
Synthetic forward models: spin-angle field → observable map.

Stage 3 synthetic-first design — all functions operate on NumPy arrays only
and have NO dependency on any measurement apparatus.  They exist solely to
generate degraded maps from known ground-truth spin configurations so that
adapters (see adapters.py) can be validated against a known answer.

Public API
----------
to_vector_map(theta)           -> (Mx, My)          cos/sin projection
apply_blur(arr, sigma)         -> blurred array      separable Gaussian via scipy
downsample(arr, factor)        -> smaller array      integer block-average
add_noise(arr, sigma, seed)    -> noisy array        i.i.d. Gaussian additive
mask_nan(arr, nan_frac, seed)  -> array with NaNs    random site masking

All functions accept and return float64 NumPy arrays.
NaN values in inputs are propagated where noted; never silently dropped.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# 1. Spin-field → vector map
# ---------------------------------------------------------------------------

def to_vector_map(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a spin-angle field into a 2-component vector observable map.

    Maps each site angle θ → (cos θ, sin θ), which is the standard
    representation for a 2D XY or clock spin as a unit vector.

    Parameters
    ----------
    theta : np.ndarray
        Shape ``(L, L)`` float64 angle field in radians.  NaN entries are
        propagated: NaN → (NaN, NaN).

    Returns
    -------
    Mx, My : tuple[np.ndarray, np.ndarray]
        Both shape ``(L, L)`` float64.  Values lie in ``[−1, 1]`` for
        non-NaN sites.

    Raises
    ------
    ValueError
        If ``theta`` is not 2-D.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError(f"theta must be 2-D; got shape {theta.shape}")
    return np.cos(theta), np.sin(theta)


# ---------------------------------------------------------------------------
# 2. Gaussian blur
# ---------------------------------------------------------------------------

def apply_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian blur to an array.

    NaN values are handled via a weighted normalisation trick: NaN sites are
    replaced by 0 before blurring, and a corresponding weight mask is blurred
    separately; the ratio re-introduces the correct normalisation.  Regions
    whose weight falls below ``1e-6`` are set to NaN in the output.

    Parameters
    ----------
    arr : np.ndarray
        Input array of any shape, float64.
    sigma : float
        Gaussian standard deviation in pixels (isotropic).  ``sigma=0`` is
        a no-op (returns a copy).

    Returns
    -------
    np.ndarray
        Same shape as ``arr``, float64.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if sigma <= 0.0:
        return arr.copy()

    nan_mask = np.isnan(arr)
    filled = np.where(nan_mask, 0.0, arr)
    weight = np.where(nan_mask, 0.0, 1.0)

    blurred_data = gaussian_filter(filled, sigma=sigma, mode="wrap")
    blurred_weight = gaussian_filter(weight, sigma=sigma, mode="wrap")

    out = np.full_like(arr, np.nan)
    good = blurred_weight > 1e-6
    out[good] = blurred_data[good] / blurred_weight[good]
    return out


# ---------------------------------------------------------------------------
# 3. Integer downsampling (block average)
# ---------------------------------------------------------------------------

def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2-D array by integer block-averaging.

    Each ``factor × factor`` block is reduced to a single pixel via
    nanmean, so isolated NaN sites inside a block are gracefully handled.
    If ALL sites in a block are NaN the output pixel is NaN.

    Parameters
    ----------
    arr : np.ndarray
        Shape ``(H, W)`` float64.
    factor : int
        Integer downsampling factor ≥ 1.  Rows and columns that do not fit
        cleanly into a multiple of ``factor`` are cropped.

    Returns
    -------
    np.ndarray
        Shape ``(H // factor, W // factor)`` float64.

    Raises
    ------
    ValueError
        If ``factor < 1`` or ``arr`` is not 2-D.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"arr must be 2-D; got shape {arr.shape}")
    if factor < 1:
        raise ValueError(f"factor must be ≥ 1; got {factor}")
    if factor == 1:
        return arr.copy()

    H, W = arr.shape
    H_new = H // factor
    W_new = W // factor
    clipped = arr[:H_new * factor, :W_new * factor]
    # Reshape into blocks and take nanmean
    blocks = clipped.reshape(H_new, factor, W_new, factor)
    with np.errstate(all="ignore"):
        return np.nanmean(blocks, axis=(1, 3))


# ---------------------------------------------------------------------------
# 4. Additive Gaussian noise
# ---------------------------------------------------------------------------

def add_noise(arr: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Add i.i.d. Gaussian noise to an array.

    NaN sites in the input remain NaN in the output (noise is NOT added to
    masked regions — callers that need noise-then-mask should call
    add_noise before mask_nan).

    Parameters
    ----------
    arr : np.ndarray
        Input array, float64.
    sigma : float
        Standard deviation of the additive noise (in the same units as arr).
        ``sigma=0`` returns a copy unchanged.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Same shape, float64, NaN-preserving.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if sigma == 0.0:
        return arr.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=arr.shape)
    out = arr + noise
    out[np.isnan(arr)] = np.nan  # re-stamp NaN mask after addition
    return out


# ---------------------------------------------------------------------------
# 5. Random NaN masking
# ---------------------------------------------------------------------------

def mask_nan(
    arr: np.ndarray,
    nan_frac: float,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly set a fraction of array sites to NaN.

    Simulates missing or invalid measurement pixels (e.g. detector dead
    pixels, out-of-FOV regions, saturated values).

    Parameters
    ----------
    arr : np.ndarray
        Input array, float64.  Already-NaN sites are kept as NaN regardless
        of whether they are "chosen" by the masking draw.
    nan_frac : float
        Fraction of sites to set to NaN, in ``[0, 1]``.  ``nan_frac=0``
        returns a copy unchanged.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Same shape as ``arr``, float64, with additional NaN sites.

    Raises
    ------
    ValueError
        If ``nan_frac`` is outside ``[0, 1]``.
    """
    if not 0.0 <= nan_frac <= 1.0:
        raise ValueError(f"nan_frac must be in [0, 1]; got {nan_frac}")
    arr = np.asarray(arr, dtype=np.float64)
    if nan_frac == 0.0:
        return arr.copy()
    rng = np.random.default_rng(seed)
    out = arr.copy()
    mask = rng.random(size=arr.shape) < nan_frac
    out[mask] = np.nan
    return out
