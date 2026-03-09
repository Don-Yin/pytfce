"""Hybrid eTFCE–GRF: union-find cluster retrieval + analytical GRF p-values.

Our contribution — combines:
  - eTFCE's union-find (Chen & Nichols 2026) for O(1) incremental
    cluster retrieval (single sweep, no repeated CCL)
  - pTFCE's analytical GRF p-values (Spisák et al. 2019; no permutations)
  - Dense equi-spaced threshold grid (Nh=500 by default, 5x finer than
    standard pTFCE) at minimal extra cost thanks to union-find

Not to be confused with eTFCE itself, which still requires permutation
testing for inference.

Note: the grid remains equi-spaced on the -log(p) scale (required by
the Q-function aggregation). The union-find simply makes denser grids
affordable.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy import stats

from .grf import aggregate_logpvals_vec, pvox_clust
from .ptfce_baseline import get_or_build_lut, pvox_clust_lut
from .smoothness import estimate_smoothness, fwer_z_threshold

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def ptfce_exact(
    Z: np.ndarray,
    mask: np.ndarray,
    z_est_thr: float = 1.3,
    lut_density: int = 120,
    use_lut: bool = True,
    max_thresholds: int = 500,
    verbose: bool = True,
    progress_queue=None,
    smooth_override: dict | None = None,
) -> dict:
    """Run hybrid eTFCE–GRF: union-find clusters + analytical GRF p-values.

    Parameters
    ----------
    max_thresholds : int
        Number of equi-spaced thresholds on the -log(p) scale.
        Default 500 (5x finer than standard pTFCE's 100).
        Union-find makes denser grids cheap.
    smooth_override : dict or None
        Pre-computed smoothness dict (from residuals).  When provided,
        skips the Z-map-based smoothness estimation.
    """
    Z = np.asarray(Z, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    Z_clean = np.where(np.isfinite(Z) & mask, Z, 0.0)
    z_max = float(Z_clean.max())

    if z_max <= 0:
        return _empty_result(Z)

    t0_total = time.perf_counter()
    if smooth_override is not None:
        smooth_info = smooth_override
    else:
        smooth_info = estimate_smoothness(Z_clean, mask)
    Rd = smooth_info["Rd"]
    V = smooth_info["V"]
    n_resels = smooth_info["n_resels"]
    fwer_thresh = fwer_z_threshold(n_resels)

    lut = None
    if use_lut:
        h_max_lut = max(z_max + 1.0, 12.0)
        lut = get_or_build_lut(V, Rd, z_est_thr, lut_density, h_max_lut,
                               verbose=verbose and progress_queue is None)
    t_lut = time.perf_counter() - t0_total

    # ── Build DENSE equi-spaced threshold grid ─────────────────────
    # Key: use the same equi-spaced -log(p) grid as standard pTFCE,
    # but with many more points (Nh=500). The union-find makes this
    # cheap: no repeated CCL, just incremental merges.
    flat_Z = Z_clean.ravel()
    pos_mask = (flat_Z > 0) & mask.ravel()
    pos_indices = np.flatnonzero(pos_mask)
    pos_values = flat_Z[pos_indices]
    N = len(pos_indices)

    if N == 0:
        return _empty_result(Z)

    Nh = max_thresholds

    logp_max = -stats.norm.logsf(z_max)
    logp_grid = np.linspace(0, logp_max, Nh)
    delta = logp_grid[1] - logp_grid[0] if Nh > 1 else logp_max
    h_grid = stats.norm.isf(np.exp(-logp_grid))
    h_grid[0] = -1e10

    if verbose and progress_queue is None:
        print(f"  [hybrid eTFCE–GRF] Rd={Rd:.1f}, V={V}, Nh={Nh}, "
              f"n_resels={n_resels:.0f}, FWER-Z={fwer_thresh:.3f}")

    # ── Union-find sweep ─────────────────────────────────────────────
    order = np.argsort(-pos_values)
    sorted_flat = pos_indices[order]
    sorted_vals = pos_values[order]

    flat_to_rank = np.full(flat_Z.shape[0], -1, dtype=np.intp)
    flat_to_rank[sorted_flat] = np.arange(N)

    parent = np.arange(N, dtype=np.intp)
    uf_rank = np.zeros(N, dtype=np.intp)
    uf_size = np.ones(N, dtype=np.intp)
    active = np.zeros(N, dtype=bool)

    sx, sy, sz = Z.shape
    offsets_3d = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=np.intp)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if uf_rank[ra] < uf_rank[rb]:
            parent[ra] = rb
            uf_size[rb] += uf_size[ra]
        elif uf_rank[ra] > uf_rank[rb]:
            parent[rb] = ra
            uf_size[ra] += uf_size[rb]
        else:
            parent[rb] = ra
            uf_size[ra] += uf_size[rb]
            uf_rank[ra] += 1

    t0_sweep = time.perf_counter()

    cursor = 0
    accumulator = np.zeros(N, dtype=np.float64)
    pvox_cache: dict[tuple[int, float], float] = {}

    for step, j in enumerate(range(Nh - 1, -1, -1)):
        h = float(h_grid[j])
        h_rounded = round(h, 6)

        if progress_queue and step % 50 == 0:
            progress_queue.put({"name": "hybrid-etfce-grf",
                                "current": step, "total": Nh,
                                "phase": "thresholds"})

        while cursor < N and sorted_vals[cursor] >= h:
            i = cursor
            active[i] = True
            fi = int(sorted_flat[i])
            xi = fi // (sy * sz)
            yi = (fi % (sy * sz)) // sz
            zi = fi % sz

            for d in range(6):
                nx = xi + int(offsets_3d[d, 0])
                ny = yi + int(offsets_3d[d, 1])
                nz = zi + int(offsets_3d[d, 2])
                if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                    nf = nx * sy * sz + ny * sz + nz
                    ni = flat_to_rank[nf]
                    if ni >= 0 and active[ni]:
                        union(i, ni)
            cursor += 1

        if cursor == 0:
            continue

        seen_sizes: set[int] = set()
        for i in range(cursor):
            r = find(i)
            c = int(uf_size[r])
            seen_sizes.add(c)

        for c in seen_sizes:
            key = (c, h_rounded)
            if key not in pvox_cache:
                if h < z_est_thr:
                    pval = 0.5 * math.erfc(h * _INV_SQRT_2)
                elif lut is not None:
                    pval = pvox_clust_lut(lut, c, h, z_est_thr)
                else:
                    pval = pvox_clust(V, Rd, c, h, z_est_thr)
                pvox_cache[key] = pval

        for i in range(cursor):
            r = find(i)
            c = int(uf_size[r])
            key = (c, h_rounded)
            accumulator[i] += -math.log(max(pvox_cache[key], 1e-300))

    t_sweep = time.perf_counter() - t0_sweep

    # ── Aggregate ────────────────────────────────────────────────────
    brain_acc = np.zeros(flat_Z.shape[0], dtype=np.float64)
    for i in range(N):
        brain_acc[sorted_flat[i]] = accumulator[i]
    brain_acc = brain_acc.reshape(Z.shape)

    active_mask = mask & (brain_acc > 0)
    enhanced_logp = np.zeros_like(Z_clean)
    enhanced_logp[active_mask] = aggregate_logpvals_vec(
        brain_acc[active_mask], delta)

    p_enhanced = np.ones_like(Z_clean)
    p_enhanced[active_mask] = np.exp(-enhanced_logp[active_mask])
    p_enhanced = np.clip(p_enhanced,
                         np.finfo(float).tiny, 1.0 - np.finfo(float).eps)

    Z_enhanced = np.zeros_like(Z_clean)
    Z_enhanced[active_mask] = stats.norm.isf(p_enhanced[active_mask])

    if progress_queue:
        progress_queue.put({"name": "hybrid-etfce-grf", "current": Nh,
                            "total": Nh, "phase": "done"})

    t_total = time.perf_counter() - t0_total

    return {
        "p": p_enhanced,
        "logp": enhanced_logp,
        "Z_enhanced": Z_enhanced,
        "smoothness": smooth_info,
        "fwer_z_thresh": fwer_thresh,
        "diagnostics": {
            "sweep_time": round(t_sweep, 4),
            "lut_time": round(t_lut, 4),
            "total_time": round(t_total, 4),
            "n_positive_voxels": N,
            "n_thresholds_used": Nh,
            "n_pvox_cache_entries": len(pvox_cache),
        },
    }


from pytfce.core import _empty_result  # noqa: E402
