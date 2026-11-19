"""Optional integer union-find kernels for the hybrid eTFCE-GRF sweep.

Imported lazily by ``ptfce_exact(uf_backend="numba"|"auto")``. No GRF,
interpolation, threshold-grid or Q-function arithmetic is compiled here.
No fastmath or parallel reduction: each accumulator receives the original
Python-computed log-p weight, in the original descending-threshold order.
"""

from __future__ import annotations

from numba import njit


@njit(cache=True)
def _find(parent, a):
    root = a
    while parent[root] != root:
        root = parent[root]
    while parent[a] != root:
        previous = parent[a]
        parent[a] = root
        a = previous
    return root


@njit(cache=True)
def _union(parent, uf_rank, uf_size, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
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


@njit(cache=True)
def activate(
    sorted_vals, sorted_flat, flat_to_rank, parent, uf_rank,
    uf_size, active, offsets_3d, cursor, h, sx, sy, sz,
):
    """Activate new voxels, using the unchanged six-neighbour/union order."""
    n = len(sorted_vals)
    while cursor < n and sorted_vals[cursor] >= h:
        i = cursor
        active[i] = True
        fi = sorted_flat[i]
        xi = fi // (sy * sz)
        yi = (fi % (sy * sz)) // sz
        zi = fi % sz
        for d in range(6):
            nx = xi + offsets_3d[d, 0]
            ny = yi + offsets_3d[d, 1]
            nz = zi + offsets_3d[d, 2]
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                nf = nx * sy * sz + ny * sz + nz
                ni = flat_to_rank[nf]
                if ni >= 0 and active[ni]:
                    _union(parent, uf_rank, uf_size, i, ni)
        cursor += 1
    return cursor


@njit(cache=True)
def collect_sizes(parent, uf_size, cursor, seen, unique):
    """Return a scratch view of distinct component sizes, resetting flags.

    A byte flag array avoids per-voxel Python set/dict operations and sorting.
    Order differs from Python's set order, but values and cache keys do not.
    """
    n_unique = 0
    for i in range(cursor):
        r = _find(parent, i)
        c = uf_size[r]
        if not seen[c]:
            seen[c] = True
            unique[n_unique] = c
            n_unique += 1
    for k in range(n_unique):
        seen[unique[k]] = False
    return unique[:n_unique]


@njit(cache=True)
def accumulate(parent, uf_size, cursor, logp_by_size, accumulator):
    """Add exactly one precomputed scalar weight to each active voxel."""
    for i in range(cursor):
        r = _find(parent, i)
        c = uf_size[r]
        accumulator[i] += logp_by_size[c]
