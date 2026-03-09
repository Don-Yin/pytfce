"""pTFCE: probabilistic Threshold-Free Cluster Enhancement.

Pure-Python reimplementation of pTFCE (Spisák et al. 2019),
with the hybrid eTFCE–GRF variant (union-find + analytical GRF p-values).
"""

from pytfce.core.ptfce_baseline import ptfce as ptfce_baseline
from pytfce.core.ptfce_exact import ptfce_exact
from pytfce.core.smoothness import estimate_smoothness, estimate_smoothness_from_residuals, fwer_z_threshold
from pytfce.core.grf import pvox_clust, aggregate_logpvals_vec

__all__ = [
    "ptfce_baseline",
    "ptfce_exact",
    "estimate_smoothness",
    "estimate_smoothness_from_residuals",
    "fwer_z_threshold",
    "pvox_clust",
    "aggregate_logpvals_vec",
]
