"""Smoothness estimation and FWER threshold computation.

Smoothness formulas match FSL ``smoothest`` and the R pTFCE ``smoothest()``
function.  Along each axis the spatial autocorrelation is the *scale-invariant*
ratio ``corr_i = SS_i / S2_i`` with ``SS_i = sum(z_a * z_b)`` and
``S2_i = sum((z_a**2 + z_b**2) / 2)`` over neighbouring voxel pairs, i.e.
``corr_i = 1 - var_deriv_i / (2 * mean_sq_i)`` with ``mean_sq_i = S2_i / N``.
The former ``1 - var_deriv_i / 2`` silently assumes ``mean(Z**2) == 1`` and
under-estimates smoothness on signal-dominated Z-maps.  With the FSL width
parameter ``sigmasq_i = -1 / (4 * ln(corr_i))``::

    dLh    = prod(sigmasq_i)^{-1/2} * 8^{-1/2}     (= det(Lambda)^{1/2}, FSL DLH)
    FWHM_i = sqrt(8 * ln(2) * sigmasq_i)

The FWER threshold solves the GRF Euler-characteristic equation (not Bonferroni).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats


def estimate_smoothness(Z: np.ndarray, mask: np.ndarray) -> dict:
    """Estimate image smoothness from a 3-D Z-score map.

    Uses the scale-invariant lag-1 autocorrelation of *Z* within *mask* (FSL / R pTFCE).
    Returns a dict with V, Rd, dLh, fwhm_voxels, resel_size, n_resels.
    """
    mask = mask.astype(bool)
    V = int(mask.sum())

    # Scale-invariant lag-1 autocorrelation exactly as FSL / R pTFCE smoothest(): voxel
    # pairs (centre, backward neighbour) over the "core" where the centre and all three
    # backward neighbours lie inside the mask.
    Z = np.asarray(Z, dtype=np.float64)
    core = (mask[1:, 1:, 1:] & mask[:-1, 1:, 1:]
            & mask[1:, :-1, 1:] & mask[1:, 1:, :-1])
    n_core = max(int(core.sum()), 1)
    centre = Z[1:, 1:, 1:]
    neighbours = (Z[:-1, 1:, 1:], Z[1:, :-1, 1:], Z[1:, 1:, :-1])
    ss = np.zeros(3, dtype=np.float64)  # sum z_a * z_b
    s2 = np.zeros(3, dtype=np.float64)  # sum (z_a^2 + z_b^2) / 2
    for axis, nb in enumerate(neighbours):
        ss[axis] = float((centre * nb)[core].sum())
        s2[axis] = float((0.5 * (centre * centre + nb * nb))[core].sum())
    var_deriv = 2.0 * (s2 - ss) / n_core
    mean_sq = np.maximum(s2 / n_core, 1e-300)

    dLh, fwhm_per_axis = _var_deriv_to_smoothness(var_deriv, mean_sq)

    Rd = V * dLh
    resel_size = float(np.prod(fwhm_per_axis))
    n_resels = V / max(resel_size, 1e-10)

    return {
        "V": V,
        "Rd": float(Rd),
        "dLh": float(dLh),
        "var_deriv": var_deriv.tolist(),
        "fwhm_voxels": fwhm_per_axis.tolist(),
        "resel_size": resel_size,
        "n_resels": n_resels,
    }


def estimate_smoothness_from_residuals(
    Y: np.ndarray,
    X: np.ndarray,
    mask: np.ndarray,
    n_sample: int = 20,
) -> dict:
    """Estimate image smoothness from GLM residuals (preferred over Z-map).

    Standard neuroimaging practice (FSL smoothest, SPM):  estimate spatial
    smoothness from the MSE-normalised residuals rather than the stat map.
    This avoids the signal contamination that biases Z-map-based estimates
    — especially severe for spatially rough effects (scanner, site).

    Parameters
    ----------
    Y : (n_subjects, V_masked)  flattened observations inside the brain mask.
    X : (n_subjects, p)  design matrix.
    mask : 3-D boolean brain mask.
    n_sample : number of subject residuals to average over (≤ n_subjects).

    Returns dict with the same keys as :func:`estimate_smoothness`.
    """
    n, p = X.shape
    df_resid = n - p
    spatial = mask.shape
    mask_flat = mask.ravel().astype(bool)

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)

    # MSE per voxel (pool across all subjects)
    predicted = X @ beta
    ssq_resid = np.einsum("ij,ij->j", Y - predicted, Y - predicted)
    del predicted
    mse = ssq_resid / max(df_resid, 1)
    del ssq_resid
    inv_sqrt_mse = 1.0 / np.sqrt(np.clip(mse, 1e-10, None))

    # Accumulate first-difference variance over a sample of normalised residuals
    n_use = min(n_sample, n)
    ssq = np.zeros(3, dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)

    for i in range(n_use):
        r_norm = (Y[i] - X[i] @ beta) * inv_sqrt_mse
        r_3d = np.zeros(int(np.prod(spatial)), dtype=np.float64)
        r_3d[mask_flat] = r_norm
        r_3d = r_3d.reshape(spatial)

        for axis in range(3):
            sa = [slice(None)] * 3
            sb = [slice(None)] * 3
            sa[axis] = slice(1, None)
            sb[axis] = slice(None, -1)
            mb = mask[tuple(sa)] & mask[tuple(sb)]
            diff = r_3d[tuple(sa)] - r_3d[tuple(sb)]
            ssq[axis] += float((diff[mb] ** 2).sum())
            counts[axis] += int(mb.sum())

    var_deriv = ssq / np.maximum(counts, 1)

    V = int(mask.sum())
    dLh, fwhm_per_axis = _var_deriv_to_smoothness(var_deriv)
    Rd = V * dLh
    resel_size = float(np.prod(fwhm_per_axis))
    n_resels = V / max(resel_size, 1e-10)

    return {
        "V": V,
        "Rd": float(Rd),
        "dLh": float(dLh),
        "var_deriv": var_deriv.tolist(),
        "fwhm_voxels": fwhm_per_axis.tolist(),
        "resel_size": resel_size,
        "n_resels": n_resels,
        "source": "residuals",
        "n_sample": n_use,
    }


def _var_deriv_to_smoothness(
    var_deriv: np.ndarray,
    mean_sq: np.ndarray | float = 1.0,
) -> tuple[float, np.ndarray]:
    """Convert mean-squared first-differences to dLh and FWHM.

    Matches R pTFCE ``smoothest()`` and FSL ``smoothest``::

        corr_i    = 1 - var_deriv_i / (2 * mean_sq_i)   (= SS_i / S2_i)
        sigmasq_i = -1 / (4 * ln(corr_i))               (FSL "sigmasq")
        dLh       = prod(sigmasq_i)^{-1/2} * 8^{-1/2}   (det(Lambda)^{1/2})
        FWHM_i    = sqrt(8 * ln(2) * sigmasq_i)

    ``mean_sq`` is the mean of ``z**2`` over the same voxel pairs.  Leave it at 1.0
    only for fields that are unit-variance by construction (MSE-normalised residuals).
    """
    corr = np.clip(1.0 - var_deriv / (2.0 * mean_sq), 1e-10, 1.0 - 1e-10)
    W = -1.0 / (4.0 * np.log(corr))
    dLh = float(np.prod(W) ** (-0.5) * 8.0 ** (-0.5))
    fwhm_per_axis = np.sqrt(8.0 * np.log(2.0) * W)
    return dLh, fwhm_per_axis


_EC3_COEFF = (4.0 * math.log(2.0)) ** 1.5 / (2.0 * math.pi) ** 2


def fwer_z_threshold(n_resels: float, alpha: float = 0.05) -> float:
    """GRF Euler-characteristic FWER Z-threshold (matches R pTFCE).

    Solves  R₃ · c₃ · (z² − 1) · exp(−z²/2) = α  for z, where
    c₃ = (4 ln 2)^{3/2} / (2π)².

    The function g(z) = (z²−1)·exp(−z²/2) is positive for z > 1, peaks at
    z = √3, and decays to 0.  We bracket the descending-side root.
    """
    R3 = max(n_resels, 1.0)
    target = alpha / (R3 * _EC3_COEFF)

    peak_val = 2.0 * math.exp(-1.5)  # g(√3) ≈ 0.4463
    if target >= peak_val:
        return 1.0

    def _ec_residual(z: float) -> float:
        return (z * z - 1.0) * math.exp(-0.5 * z * z) - target

    try:
        return float(optimize.brentq(_ec_residual, math.sqrt(3.0), 38.0))
    except ValueError:
        return float(stats.norm.isf(alpha / R3))
