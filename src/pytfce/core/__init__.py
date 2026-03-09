"""Core pTFCE algorithms and GRF computations."""

import numpy as np


def _empty_result(Z: np.ndarray) -> dict:
    """Fallback result when there are no supra-threshold voxels."""
    return {
        "p": np.ones_like(Z), "logp": np.zeros_like(Z),
        "Z_enhanced": np.zeros_like(Z), "smoothness": {},
        "fwer_z_thresh": np.inf, "diagnostics": {},
    }
