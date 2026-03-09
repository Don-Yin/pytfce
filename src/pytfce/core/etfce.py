"""Pure eTFCE: exact TFCE scores via union-find cluster retrieval.

Reimplements the score computation from:
    Chen, X., Weeda, W., Nichols, T.E., & Goeman, J.J. (2026).
    eTFCE: Exact Threshold-Free Cluster Enhancement via Fast Cluster Retrieval.
    arXiv:2603.03004.

Produces TFCE scores (not p-values). Inference still requires permutation
testing. The advantage over standard discretized TFCE is:
  (1) Exact integration (no dh discretization error)
  (2) Single sweep replaces Nh separate CCL passes

This implementation uses a dense equi-spaced Z-grid with union-find
cluster retrieval. The grid can be made arbitrarily fine at minimal
cost because union-find avoids repeated CCL.
"""

from __future__ import annotations

import time

import numpy as np


def etfce(
    stat_map: np.ndarray,
    mask: np.ndarray | None = None,
    E: float = 0.5,
    H: float = 2.0,
    dh: float = 0.01,
    verbose: bool = True,
    progress_queue=None,
) -> dict:
    """Compute TFCE scores using union-find cluster retrieval.

    Parameters
    ----------
    stat_map : 3D array
        Z-statistic map (only positive values are enhanced).
    mask : 3D bool array, optional
        Brain mask. If None, uses stat_map > 0.
    E, H : float
        TFCE extent and height exponents (default 0.5, 2.0).
    dh : float
        Threshold step size in Z-space (default 0.01, 10x finer than FSL).
    """
    stat_map = np.asarray(stat_map, dtype=np.float64)
    if mask is None:
        mask = stat_map > 0
    else:
        mask = np.asarray(mask, dtype=bool)

    safe = np.where(np.isfinite(stat_map) & mask, stat_map, 0.0)
    safe = np.clip(safe, 0, None)
    z_max = float(safe.max())

    if z_max <= 0:
        return {"tfce": np.zeros_like(safe), "diagnostics": {}}

    t0 = time.perf_counter()

    flat = safe.ravel()
    pos_mask = flat > 0
    pos_indices = np.flatnonzero(pos_mask)
    pos_values = flat[pos_indices]
    N = len(pos_indices)

    order = np.argsort(-pos_values)
    sorted_flat = pos_indices[order]
    sorted_vals = pos_values[order]

    flat_to_rank = np.full(flat.shape[0], -1, dtype=np.intp)
    flat_to_rank[sorted_flat] = np.arange(N)

    parent = np.arange(N, dtype=np.intp)
    uf_rank = np.zeros(N, dtype=np.intp)
    uf_size = np.ones(N, dtype=np.intp)
    active = np.zeros(N, dtype=bool)

    sx, sy, sz = stat_map.shape
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

    h_grid = np.arange(dh, z_max + dh, dh)[::-1]
    Nh = len(h_grid)

    if verbose and progress_queue is None:
        print(f"  [eTFCE] z_max={z_max:.1f}, Nh={Nh}, dh={dh}, "
              f"N_pos={N}, E={E}, H={H}")

    cursor = 0
    accumulator = np.zeros(N, dtype=np.float64)

    for step, h in enumerate(h_grid):
        if progress_queue and step % max(1, Nh // 20) == 0:
            progress_queue.put({"name": "eTFCE",
                                "current": step, "total": Nh,
                                "phase": "sweep"})

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

        active_idx = np.arange(cursor)
        roots = parent[active_idx].copy()
        for _ in range(5):
            new_roots = parent[roots]
            if np.array_equal(new_roots, roots):
                break
            roots = new_roots
        parent[active_idx] = roots

        sizes = uf_size[roots]
        accumulator[:cursor] += (sizes.astype(np.float64) ** E) * (h ** H) * dh

    tfce_out = np.zeros(flat.shape[0], dtype=np.float64)
    for i in range(N):
        tfce_out[sorted_flat[i]] = accumulator[i]
    tfce_out = tfce_out.reshape(stat_map.shape)

    t_total = time.perf_counter() - t0

    if progress_queue:
        progress_queue.put({"name": "eTFCE",
                            "current": Nh, "total": Nh, "phase": "done"})

    if verbose and progress_queue is None:
        print(f"  [eTFCE] done in {t_total:.2f}s")

    return {
        "tfce": tfce_out,
        "diagnostics": {
            "total_time": round(t_total, 4),
            "n_thresholds": Nh,
            "dh": dh,
            "n_positive_voxels": N,
        },
    }
