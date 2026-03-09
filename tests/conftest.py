"""Shared fixtures for pytfce test suite.

All fixtures use small volumes (32^3) and few subjects (20) to keep
the test suite fast while still exercising the statistical machinery.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage, stats

from pytfce.utils.phantoms import generate_data

# ── constants ──────────────────────────────────────────────────────
SHAPE = (32, 32, 32)
N_SUBJECTS = 20
AMPLITUDE = 0.8
SMOOTH_SIGMA = 1.5
SEED = 99

BLOBS_SMALL_VOL = [
    {"c": (16, 16, 16), "r": (4, 4, 4)},
]
"""Single centred blob sized for a 32^3 volume."""


@pytest.fixture(scope="session")
def phantom_data():
    """Generate a small phantom dataset once per session.

    Returns (subjects_4d, Z_map, brain_mask, ground_truth).
    """
    return generate_data(
        signal_blobs=BLOBS_SMALL_VOL,
        amplitude=AMPLITUDE,
        n_subjects=N_SUBJECTS,
        smooth_sigma=SMOOTH_SIGMA,
        shape=SHAPE,
        seed=SEED,
    )


@pytest.fixture(scope="session")
def z_map(phantom_data):
    """Group-level Z-score map from the phantom."""
    return phantom_data[1]


@pytest.fixture(scope="session")
def brain_mask(phantom_data):
    """Boolean brain mask from the phantom."""
    return phantom_data[2]


@pytest.fixture(scope="session")
def ground_truth(phantom_data):
    """Ground-truth signal region from the phantom."""
    return phantom_data[3]


@pytest.fixture(scope="session")
def noise_z_map():
    """Pure-noise Z-map (no embedded signal) for false-positive checks."""
    rng = np.random.default_rng(123)
    noise_4d = np.empty((*SHAPE, N_SUBJECTS), dtype=np.float64)
    for s in range(N_SUBJECTS):
        vol = rng.standard_normal(SHAPE)
        noise_4d[..., s] = ndimage.gaussian_filter(vol, sigma=SMOOTH_SIGMA)

    mean = noise_4d.mean(axis=-1)
    std = noise_4d.std(axis=-1, ddof=1)
    std[std < 1e-12] = 1e-12
    t_map = mean / (std / np.sqrt(N_SUBJECTS))

    p_map = stats.t.sf(t_map, df=N_SUBJECTS - 1)
    p_map = np.clip(p_map, 1e-300, 1.0 - 1e-15)
    Z = stats.norm.isf(p_map)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z


@pytest.fixture(scope="session")
def noise_mask():
    """Brain mask for the noise volume — full cuboid interior."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[2:-2, 2:-2, 2:-2] = True
    return mask
