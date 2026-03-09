"""Synthetic phantom data for pTFCE validation experiments.

Generates group-level Z-score maps from Gaussian noise with embedded
ellipsoidal signal blobs, suitable for evaluating statistical enhancement
methods (pTFCE, TFCE, cluster-extent, etc.) against a known ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, stats


# ---------------------------------------------------------------------------
# predefined blob configurations
# ---------------------------------------------------------------------------

BLOBS_MULTI: list[dict] = [
    {"c": (22, 32, 32), "r": (5, 5, 5)},
    {"c": (42, 32, 32), "r": (3, 3, 3)},
    {"c": (32, 22, 32), "r": (4, 4, 4)},
]
"""Three spatially separated spherical blobs of varying radius."""

BLOBS_LARGE: list[dict] = [
    {"c": (32, 32, 32), "r": (8, 8, 8)},
]
"""Single large spherical blob (radius 8 voxels)."""

BLOBS_SMALL: list[dict] = [
    {"c": (32, 32, 32), "r": (4, 4, 4)},
]
"""Single small spherical blob (radius 4 voxels)."""

BLOBS_ELONG: list[dict] = [
    {"c": (32, 32, 32), "r": (3, 3, 12)},
]
"""Single elongated ellipsoidal blob stretching along the Z axis."""


# ---------------------------------------------------------------------------
# primitive shapes
# ---------------------------------------------------------------------------

def _sphere(
    center: tuple[int, int, int],
    radius: float,
    shape: tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """Boolean mask of a sphere.

    Parameters
    ----------
    center : (x, y, z)
        Voxel coordinates of the sphere centre.
    radius : float
        Sphere radius in voxels.
    shape : (X, Y, Z)
        Volume dimensions.

    Returns
    -------
    ndarray of bool
        ``True`` inside the sphere (inclusive of the boundary).
    """
    coords = np.ogrid[
        0:shape[0],
        0:shape[1],
        0:shape[2],
    ]
    dist_sq = sum(
        (c - float(ctr)) ** 2 for c, ctr in zip(coords, center)
    )
    return dist_sq <= radius ** 2


def _ellipsoid(
    center: tuple[int, int, int],
    radii: tuple[float, float, float],
    shape: tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """Boolean mask of an axis-aligned ellipsoid.

    Parameters
    ----------
    center : (x, y, z)
        Voxel coordinates of the ellipsoid centre.
    radii : (rx, ry, rz)
        Semi-axis lengths in voxels.
    shape : (X, Y, Z)
        Volume dimensions.

    Returns
    -------
    ndarray of bool
        ``True`` inside the ellipsoid (inclusive of the boundary).
    """
    coords = np.ogrid[
        0:shape[0],
        0:shape[1],
        0:shape[2],
    ]
    normalised = sum(
        ((c - float(ctr)) / max(float(r), 1e-10)) ** 2
        for c, ctr, r in zip(coords, center, radii)
    )
    return normalised <= 1.0


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------

def generate_data(
    signal_blobs: list[dict],
    amplitude: float,
    n_subjects: int = 80,
    smooth_sigma: float = 1.5,
    shape: tuple[int, int, int] = (64, 64, 64),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic group-level Z-score map.

    For each of *n_subjects*, the pipeline:

    1. Draws an i.i.d. Gaussian noise volume.
    2. Applies isotropic Gaussian smoothing (``smooth_sigma``).
    3. Adds the ground-truth signal (scaled by *amplitude*) within the brain
       mask.

    A one-sample t-test across subjects is converted to a Z-score map.

    Parameters
    ----------
    signal_blobs : list of dict
        Each dict must have ``"c"`` (centre, 3-tuple) and ``"r"`` (semi-axis
        radii, 3-tuple).  Blobs are combined with logical OR.
    amplitude : float
        Signal amplitude added to each subject within the ground-truth region.
    n_subjects : int
        Number of synthetic subjects.
    smooth_sigma : float
        Gaussian smoothing kernel sigma (voxels).
    shape : (X, Y, Z)
        Volume dimensions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    subjects_4d : ndarray, shape (*shape, n_subjects)
        Smoothed + signal individual volumes.
    Z_map : ndarray, shape *shape*
        Group-level Z-score map (one-sample t → Z).
    brain_mask : ndarray of bool, shape *shape*
        Spherical brain mask (radius 28, centred in the volume).
    ground_truth : ndarray of bool, shape *shape*
        True signal region (union of all blobs inside the brain mask).
    """
    rng = np.random.default_rng(seed)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2

    # brain mask — large centred sphere
    brain_mask = _sphere((cx, cy, cz), radius=28, shape=shape)

    # ground truth — union of ellipsoidal blobs clipped to the brain
    ground_truth = np.zeros(shape, dtype=bool)
    for blob in signal_blobs:
        ground_truth |= _ellipsoid(blob["c"], blob["r"], shape=shape)
    ground_truth &= brain_mask

    # individual subjects
    subjects_4d = np.empty((*shape, n_subjects), dtype=np.float64)
    for s in range(n_subjects):
        vol = rng.standard_normal(shape)
        if smooth_sigma > 0:
            vol = ndimage.gaussian_filter(vol, sigma=smooth_sigma)
        vol[ground_truth] += amplitude
        subjects_4d[..., s] = vol

    # group-level Z via one-sample t-test (vectorised)
    mean = subjects_4d.mean(axis=-1)
    std = subjects_4d.std(axis=-1, ddof=1)
    std[std < 1e-12] = 1e-12
    t_map = mean / (std / np.sqrt(n_subjects))

    df = n_subjects - 1
    p_map = stats.t.sf(t_map, df=df)
    p_map = np.clip(p_map, 1e-300, 1.0 - 1e-15)
    Z_map = stats.norm.isf(p_map)

    Z_map[~brain_mask] = 0.0
    Z_map = np.where(np.isfinite(Z_map), Z_map, 0.0)

    return subjects_4d, Z_map, brain_mask, ground_truth
