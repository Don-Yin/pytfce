"""pTFCE: probabilistic Threshold-Free Cluster Enhancement.

Pure-Python reimplementation of pTFCE (Spisák et al. 2019),
with the hybrid eTFCE–GRF variant (union-find + analytical GRF p-values).
"""

from importlib import metadata as _metadata

from pytfce.core.grf import aggregate_logpvals_vec, pvox_clust
from pytfce.core.ptfce_baseline import ptfce as ptfce_baseline
from pytfce.core.ptfce_exact import ptfce_exact
from pytfce.core.smoothness import (
    estimate_smoothness,
    estimate_smoothness_from_residuals,
    fwer_z_threshold,
)

try:
    __version__ = _metadata.version("pytfce")
except _metadata.PackageNotFoundError:
    # A source checkout may be imported without installed distribution metadata.
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "aggregate_logpvals_vec",
    "estimate_smoothness",
    "estimate_smoothness_from_residuals",
    "fwer_z_threshold",
    "ptfce_baseline",
    "ptfce_exact",
    "pvox_clust",
]
