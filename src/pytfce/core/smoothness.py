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
    core = _core(mask)
    ss, s2 = _pair_moments(np.asarray(Z, dtype=np.float64), core)
    return _smoothness(int(mask.sum()), ss, s2, max(int(core.sum()), 1))


def _core(mask: np.ndarray) -> np.ndarray:
    """Voxels whose three backward neighbours also lie in *mask* (FSL / R pTFCE smoothest),
    indexed as ``field[1:, 1:, 1:]``; every axis uses this one core."""
    return mask[1:, 1:, 1:] & mask[:-1, 1:, 1:] & mask[1:, :-1, 1:] & mask[1:, 1:, :-1]


def _pair_moments(field: np.ndarray, core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per axis, ``SS = sum(z_a * z_b)`` and ``S2 = sum((z_a**2 + z_b**2) / 2)`` over the core's
    (centre, backward neighbour) pairs."""
    centre = field[1:, 1:, 1:]
    neighbours = (field[:-1, 1:, 1:], field[1:, :-1, 1:], field[1:, 1:, :-1])
    ss = np.array([float((centre * nb)[core].sum()) for nb in neighbours])
    s2 = np.array(
        [float((0.5 * (centre * centre + nb * nb))[core].sum()) for nb in neighbours]
    )
    return ss, s2


def _smoothness(
    V: int, ss: np.ndarray, s2: np.ndarray, n_pairs: int, dof_factor: float = 1.0
) -> dict:
    """The smoothness dict from pooled pair moments (``corr_i = SS_i / S2_i``, scale-invariant)."""
    var_deriv = 2.0 * (s2 - ss) / n_pairs
    mean_sq = np.maximum(s2 / n_pairs, 1e-300)
    dLh, fwhm_per_axis = _var_deriv_to_smoothness(var_deriv, mean_sq)
    dLh *= dof_factor
    resel_size = float(np.prod(fwhm_per_axis))
    return {
        "V": V,
        "Rd": float(V * dLh),
        "dLh": float(dLh),
        "var_deriv": var_deriv.tolist(),
        "fwhm_voxels": fwhm_per_axis.tolist(),
        "resel_size": resel_size,
        "n_resels": V / max(resel_size, 1e-10),
    }


# FSL smoothest's temporal-DOF correction of dLh for residual input (fsl-cluster smoothest.cc,
# SMOOTHEST::Interpolate): squared factors by degrees of freedom.
_FSL_DOF_LUT = {
    5: 1.5423138,
    6: 1.3757105,
    7: 1.2842680,
    8: 1.2272151,
    9: 1.1885232,
    10: 1.1606988,
    11: 1.1398000,
    12: 1.1235677,
    13: 1.1106196,
    14: 1.1000651,
    15: 1.0913060,
    16: 1.0839261,
    17: 1.0776276,
    18: 1.0721920,
    19: 1.0674553,
    20: 1.0632924,
    25: 1.0483053,
    30: 1.0390117,
    40: 1.0281339,
    50: 1.0219834,
    60: 1.0180339,
    70: 1.0152850,
    80: 1.0132621,
    90: 1.0117115,
    100: 1.0104851,
    150: 1.0068808,
    200: 1.0051200,
    300: 1.0033865,
    500: 1.0020191,
}


def _fsl_dof_factor(v: float) -> float:
    """FSL's dLh multiplier for *v* residual degrees of freedom, reproduced as FSL computes it:
    from the first table key >= int(v) and the key below it, the line through them is evaluated
    at *v* but offset from the upper key's value (``... * (v - lower) + upper``), as in smoothest.cc."""
    if v < 6:
        return 1.1  # FSL's own constant below 6 dof, applied without the square root
    keys = sorted(_FSL_DOF_LUT)
    upper = next((k for k in keys if k >= int(v)), None)
    if upper is None:
        squared = 1.0321 / v + 1
    else:
        lower = keys[keys.index(upper) - 1]
        slope = (_FSL_DOF_LUT[upper] - _FSL_DOF_LUT[lower]) / (upper - lower)
        squared = slope * (v - lower) + _FSL_DOF_LUT[upper]
    return math.sqrt(squared)


def estimate_smoothness_from_residuals(
    Y: np.ndarray,
    X: np.ndarray,
    mask: np.ndarray,
    n_sample: int | None = None,
) -> dict:
    """Estimate image smoothness from GLM residuals (preferred over Z-map).

    Standard neuroimaging practice (FSL smoothest, SPM):  estimate spatial
    smoothness from the standardised residuals rather than the stat map.
    This avoids the signal contamination that biases Z-map-based estimates
    — especially severe for spatially rough effects (scanner, site).

    Matches FSL ``smoothest -r res4d -d dof``: each voxel's residuals are centred and
    divided by their sample SD, the pair moments of all residuals are pooled over one
    common core (so the field's overall scale cancels), and dLh takes FSL's temporal-DOF
    correction for ``dof = n - rank(X)``.  FWHM and n_resels carry no DOF correction,
    as in FSL.  (Before 0.1.3 the residuals were scaled by ``sqrt(SSE/dof)``, whose mean
    square is ``dof/n``, and converted as if it were 1: n_resels came out ~7% low.)

    Parameters
    ----------
    Y : (n_subjects, V_masked)  flattened observations inside the brain mask.
    X : (n_subjects, p)  design matrix.
    mask : 3-D boolean brain mask.
    n_sample : use only the first *n_sample* residuals (default: all, as FSL does).

    Returns dict with the same keys as :func:`estimate_smoothness`.
    """
    n = X.shape[0]
    dof = n - int(np.linalg.matrix_rank(X))
    mask = mask.astype(bool)
    mask_flat = mask.ravel()

    resid = Y - X @ (np.linalg.pinv(X.T @ X) @ (X.T @ Y))
    resid = resid - resid.mean(axis=0)
    sd = resid.std(axis=0, ddof=1)
    valid = np.zeros(mask.size, dtype=bool)
    valid[mask_flat] = sd > 0  # FSL drops voxels without variance from the pairs
    core = _core(valid.reshape(mask.shape))
    resid[:, sd > 0] /= sd[sd > 0]

    n_use = n if n_sample is None else min(n_sample, n)
    ss, s2 = np.zeros(3), np.zeros(3)
    field = np.zeros(mask.size, dtype=np.float64)
    for i in range(n_use):
        field[mask_flat] = resid[i]
        a, b = _pair_moments(field.reshape(mask.shape), core)
        ss += a
        s2 += b
    info = _smoothness(
        int(mask.sum()),
        ss,
        s2,
        max(int(core.sum()), 1) * n_use,
        dof_factor=_fsl_dof_factor(dof) if n_use > 1 else 1.0,
    )
    return {**info, "source": "residuals", "n_sample": n_use}


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
        return float(
            optimize.root_scalar(
                _ec_residual, bracket=(math.sqrt(3.0), 38.0), method="brentq"
            ).root
        )
    except ValueError:
        return float(stats.norm.isf(alpha / R3))
