"""Standard TFCE transform (Smith & Nichols 2009).

Self-contained — no project imports outside numpy/scipy.
"""

from __future__ import annotations

import cc3d
import numpy as np
from scipy import ndimage


def tfce_transform(
    stat_map: np.ndarray,
    E: float = 0.5,
    H: float = 2.0,
    dh: float = 0.1,
    connectivity: int = 6,
) -> np.ndarray:
    """Apply the TFCE transform to a 3-D statistic map (non-negative).

    Args:
        stat_map: 3-D statistic map (non-negative).
        E: Extent exponent (default 0.5).
        H: Height exponent (default 2.0).
        dh: Height step (default 0.1).
        connectivity: 6 for face-adjacent (diamond, matches R TFCE) or 26 for
            full neighborhood. Default 6.
    """
    safe = np.clip(np.where(np.isfinite(stat_map), stat_map, 0.0), 0, None)
    out = np.zeros_like(safe, dtype=np.float64)
    mx = float(safe.max())
    if mx <= 0:
        return out
    _use_cc3d = (connectivity == 6)
    if not _use_cc3d:
        structure = ndimage.generate_binary_structure(3, 3)
    for h in np.arange(dh, mx + dh, dh):
        binary = safe >= h
        if _use_cc3d:
            labelled = cc3d.connected_components(binary.view(np.uint8), connectivity=6)
            nc = int(labelled.max())
        else:
            labelled, nc = ndimage.label(binary, structure=structure)
        if nc == 0:
            continue
        counts = np.bincount(labelled.ravel(), minlength=nc + 1)
        sizes = np.concatenate(
            [[0.0], counts[1 : nc + 1].astype(np.float64)]
        )
        out += (sizes[labelled] ** E) * (h ** H) * dh
    return out
