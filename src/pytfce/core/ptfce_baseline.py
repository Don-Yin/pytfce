"""Baseline pTFCE with LUT (lookup-table) acceleration.

Modular implementation that delegates GRF computations to ``grf`` and
smoothness estimation to ``smoothness``.  The LUT replaces thousands of
individual ``scipy.integrate.quad`` calls with a single 2-D linear
interpolation lookup, yielding a ~5–10× wall-clock speedup.

References
----------
Spisak, T. et al. (2019). Probabilistic TFCE: a generalized combination of
cluster size and voxel intensity to increase statistical power. *NeuroImage*,
185, 12-26. doi:10.1016/j.neuroimage.2018.09.078
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import cc3d
import numpy as np
from scipy import ndimage, stats
from scipy.interpolate import RegularGridInterpolator

from .grf import pvox_clust, aggregate_logpvals_vec
from .smoothness import estimate_smoothness, fwer_z_threshold


# ---------------------------------------------------------------------------
# progress bar (no tqdm dependency)
# ---------------------------------------------------------------------------

def _bar(
    current: int,
    total: int,
    width: int = 40,
    prefix: str = "",
    suffix: str = "",
    t0: float | None = None,
) -> None:
    """Write a Unicode progress bar to *stderr*.

    Parameters
    ----------
    current : int
        Current step (1-indexed).
    total : int
        Total number of steps.
    width : int
        Character width of the bar.
    prefix, suffix : str
        Optional text flanking the bar.
    t0 : float or None
        ``time.perf_counter()`` at the start; enables elapsed/ETA display.
    """
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = ""
    if t0 is not None:
        dt = time.perf_counter() - t0
        if frac > 0.05:
            eta = dt / frac * (1 - frac)
            elapsed = f" [{dt:.0f}s<{eta:.0f}s]"
        else:
            elapsed = f" [{dt:.0f}s]"
    sys.stderr.write(
        f"\r  {prefix} |{bar}| {current}/{total} {suffix}{elapsed}"
    )
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# LUT cache — avoids rebuilding the expensive quad-based table
# Key: (V, Rd_rounded, z_est_thr, density, h_max_rounded)
# ---------------------------------------------------------------------------

_LUT_CACHE: dict[tuple, RegularGridInterpolator] = {}


def _disk_cache_dir() -> Path:
    """Return the disk cache directory, creating it if needed.

    Resolution order:
    1. $RESULTS_DIR/tfce/.lut_cache  (SLURM -- RESULTS_DIR set by slurm-common.sh)
    2. $PYTFCE_CACHE_DIR             (explicit override)
    3. <cwd>/.results/tfce/.lut_cache (local -- experiments always run from project root)
    """
    import os
    results = os.environ.get("RESULTS_DIR")
    if results:
        cache_dir = Path(results) / "tfce" / ".lut_cache"
    elif os.environ.get("PYTFCE_CACHE_DIR"):
        cache_dir = Path(os.environ["PYTFCE_CACHE_DIR"])
    else:
        cache_dir = Path.cwd() / ".results" / "tfce" / ".lut_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _disk_cache_path(key: tuple) -> Path:
    """Return the disk cache file path for the given key."""
    key_str = repr(key)
    h = hashlib.sha256(key_str.encode()).hexdigest()
    return _disk_cache_dir() / f"lut_{h}.npz"


def get_or_build_lut(
    V: int, Rd: float, z_est_thr: float = 1.3,
    density: int = 120, h_max: float = 12.0,
    verbose: bool = False,
) -> RegularGridInterpolator:
    """Return a cached LUT, building it only if no match exists.

    Checks in-memory cache first, then disk cache, then builds via
    build_pvox_lut and saves to disk.
    """
    key = (V, round(Rd, 1), z_est_thr, density, round(h_max))
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    cache_path = _disk_cache_path(key)
    if cache_path.exists():
        data = np.load(cache_path)
        log_c_grid = data["log_c_grid"]
        h_grid = data["h_grid"]
        table = data["table"]
        lut = RegularGridInterpolator(
            (log_c_grid, h_grid),
            table,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        _LUT_CACHE[key] = lut
        return lut
    lut = build_pvox_lut(V, Rd, z_est_thr, density, density,
                         h_max=h_max, verbose=verbose)
    _LUT_CACHE[key] = lut
    log_c_grid, h_grid = lut.grid
    np.savez(
        cache_path,
        log_c_grid=log_c_grid,
        h_grid=h_grid,
        table=lut.values,
        key=np.array(key),
    )
    return lut


# ---------------------------------------------------------------------------
# LUT builder & lookup
# ---------------------------------------------------------------------------

def build_pvox_lut(
    V: int,
    Rd: float,
    z_est_thr: float = 1.3,
    n_c: int = 120,
    n_h: int = 120,
    c_max: int | None = None,
    h_max: float = 12.0,
    verbose: bool = True,
) -> RegularGridInterpolator:
    """Pre-compute ``pvox_clust`` on a ``(log_c, h)`` grid.

    Returns a :class:`~scipy.interpolate.RegularGridInterpolator` that can
    replace individual ``quad()`` evaluations with fast bilinear lookups.
    """
    if c_max is None:
        c_max = V
    log_c_grid = np.linspace(0, np.log1p(c_max), n_c)
    h_grid = np.linspace(z_est_thr, h_max, n_h)
    table = np.ones((n_c, n_h), dtype=np.float64)

    t0 = time.perf_counter()
    for ci, lc in enumerate(log_c_grid):
        c = np.expm1(lc)
        if c < 1:
            continue
        for hi, h in enumerate(h_grid):
            table[ci, hi] = pvox_clust(V, Rd, c, h, z_est_thr)
        if verbose:
            _bar(ci + 1, n_c, prefix="LUT build", t0=t0)

    return RegularGridInterpolator(
        (log_c_grid, h_grid),
        table,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def pvox_clust_lut(
    lut: RegularGridInterpolator,
    c: float,
    h: float,
    z_est_thr: float = 1.3,
) -> float:
    """Fast ``pvox_clust`` via LUT bilinear interpolation.

    Falls back to the marginal survival function for degenerate inputs
    (``c ≤ 0`` or ``h`` below the estimation threshold).

    Parameters
    ----------
    lut : RegularGridInterpolator
        Table built by :func:`build_pvox_lut`.
    c : float
        Cluster size (voxels).
    h : float
        Z-score threshold.
    z_est_thr : float
        Lower estimation threshold (must match the LUT).

    Returns
    -------
    float
        Posterior probability, clipped to ``[1e-300, 1.0]``.
    """
    if c <= 0 or h < z_est_thr:
        return float(stats.norm.sf(h)) if np.isfinite(h) else 1.0
    val = float(lut(np.array([[np.log1p(c), h]]))[0])
    return float(np.clip(val, 1e-300, 1.0))


# ---------------------------------------------------------------------------
# main pTFCE function
# ---------------------------------------------------------------------------

def ptfce(
    Z: np.ndarray,
    mask: np.ndarray,
    Rd: float | None = None,
    V: int | None = None,
    Nh: int = 100,
    z_est_thr: float = 1.3,
    connectivity: int = 6,
    use_lut: bool = True,
    lut_density: int = 120,
    prebuilt_lut: RegularGridInterpolator | None = None,
    verbose: bool = True,
    progress_queue: Any = None,
    smooth_override: dict | None = None,
) -> dict:
    """Run probabilistic TFCE on a 3-D Z-score image.

    Parameters
    ----------
    Z : ndarray, shape (X, Y, Z)
        Voxel-wise Z-score map.
    mask : ndarray, shape (X, Y, Z)
        Boolean brain mask.
    Rd : float or None
        RESEL count; estimated from *Z* and *mask* when ``None``.
    V : int or None
        Number of in-mask voxels; estimated when ``None``.
    Nh : int
        Number of equispaced thresholds on the -log-p scale.
    z_est_thr : float
        Lower integration bound for the Bayesian posterior.
    connectivity : int
        Voxel connectivity for ``ndimage.label``: **6** (face) or **26** (full).
    use_lut : bool
        If ``True``, build a 2-D interpolation LUT for ``pvox_clust``.
    lut_density : int
        Grid density for the LUT (``n_c == n_h == lut_density``).
    verbose : bool
        Print progress and summary lines to *stderr* / *stdout*.
    progress_queue : multiprocessing.Queue or None
        If supplied, progress dicts are ``.put()`` into this queue instead of
        printing to stderr.  Each dict has keys ``name``, ``current``,
        ``total``, and ``phase``.
    smooth_override : dict or None
        Pre-computed smoothness dict (from residuals).  When provided,
        skips the Z-map-based smoothness estimation.

    Returns
    -------
    dict
        ``p``             – enhanced p-values (same shape as *Z*)
        ``logp``          – enhanced -log(p)
        ``Z_enhanced``    – Z-scores from the enhanced p-values
        ``smoothness``    – dict from :func:`estimate_smoothness`
        ``fwer_z_thresh`` – GRF-based FWER Z-threshold
        ``diagnostics``   – timing, cache stats, sampled threshold info
    """
    t_start = time.perf_counter()

    Z = np.asarray(Z, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    Z_clean = np.where(np.isfinite(Z) & mask, Z, 0.0)
    z_max = float(Z_clean.max())

    empty: dict[str, Any] = {
        "p": np.ones_like(Z),
        "logp": np.zeros_like(Z),
        "Z_enhanced": np.zeros_like(Z),
        "smoothness": {},
        "fwer_z_thresh": np.inf,
        "diagnostics": {},
    }
    if z_max <= 0:
        return empty

    # ---- smoothness --------------------------------------------------
    if smooth_override is not None:
        smooth_info = smooth_override
    else:
        smooth_info = estimate_smoothness(Z_clean, mask)
    if Rd is None:
        Rd = smooth_info["Rd"]
    if V is None:
        V = smooth_info["V"]

    show = verbose and progress_queue is None

    if show:
        print(
            f"  smoothness: Rd={Rd:.1f}, V={V}, "
            f"n_resels={smooth_info['n_resels']:.0f}, "
            f"FWHM={[f'{f:.2f}' for f in smooth_info['fwhm_voxels']]}"
        )

    fwer_thresh = fwer_z_threshold(smooth_info["n_resels"])
    if show:
        voxel_bonf = float(stats.norm.isf(0.05 / V))
        print(
            f"  FWER Z-threshold: {fwer_thresh:.3f} (GRF-EC) "
            f"vs {voxel_bonf:.3f} (voxel-Bonf)"
        )

    # ---- LUT build (reuse prebuilt when available) ─────────────────────
    lut = prebuilt_lut
    if lut is None and use_lut:
        h_max_lut = max(z_max + 1.0, 12.0)
        if show:
            print(f"  building LUT ({lut_density}×{lut_density}) ...")
        lut = get_or_build_lut(V, Rd, z_est_thr, lut_density, h_max_lut,
                               verbose=show)
    t_lut = time.perf_counter()

    # ---- threshold grid (equi-spaced in -logp) -----------------------
    logp_max = -stats.norm.logsf(z_max)
    logp_grid = np.linspace(0, logp_max, Nh)
    delta = logp_grid[1] - logp_grid[0] if Nh > 1 else logp_max
    h_grid = stats.norm.isf(np.exp(-logp_grid))
    h_grid[0] = -1e10

    # ---- connectivity structure --------------------------------------
    _use_cc3d = (connectivity == 6)
    if not _use_cc3d:
        structure = ndimage.generate_binary_structure(3, 3)

    # ---- threshold sweep & accumulation ------------------------------
    accumulator = np.zeros_like(Z_clean)
    pvox_cache: dict[tuple[int, float], float] = {}

    diag_thresholds: list[dict] = []
    t0_loop = time.perf_counter()

    for i, h in enumerate(h_grid):
        h = float(h)
        # progress reporting
        if progress_queue is not None:
            progress_queue.put({
                "name": "ptfce",
                "current": i,
                "total": Nh,
                "phase": "thresholds",
            })
        elif show:
            _bar(i + 1, Nh, prefix="thresholds", t0=t0_loop)

        binary = Z_clean >= h
        if _use_cc3d:
            labelled = cc3d.connected_components(binary.view(np.uint8), connectivity=6)
            n_clusters = int(labelled.max())
        else:
            labelled, n_clusters = ndimage.label(binary, structure=structure)
        if n_clusters == 0:
            diag_thresholds.append({"h": float(h), "n_clusters": 0, "max_size": 0})
            continue

        counts = np.bincount(labelled.ravel(), minlength=n_clusters + 1)
        cluster_sizes = counts[1:n_clusters + 1].astype(np.float64)
        max_cs = int(np.max(cluster_sizes))

        diag_thresholds.append({
            "h": round(float(h), 4),
            "n_clusters": n_clusters,
            "max_size": max_cs,
            "mean_size": round(float(np.mean(cluster_sizes)), 1),
        })

        h_rounded = round(h, 6)
        unique_sizes = np.unique(cluster_sizes)
        unique_sizes = unique_sizes[unique_sizes > 0]
        for uc in unique_sizes:
            c = int(uc)
            key = (c, h_rounded)
            if key not in pvox_cache:
                if h < z_est_thr:
                    pval = 0.5 * math.erfc(h * (1.0 / math.sqrt(2.0)))
                elif lut is not None:
                    pval = pvox_clust_lut(lut, c, h, z_est_thr)
                else:
                    pval = pvox_clust(V, Rd, c, h, z_est_thr)
                pvox_cache[key] = pval

        size_map = np.zeros(n_clusters + 1)
        for label_id in range(1, n_clusters + 1):
            c = int(cluster_sizes[label_id - 1])
            if c > 0:
                size_map[label_id] = -math.log(max(pvox_cache[(c, h_rounded)], 1e-300))
        accumulator += size_map[labelled]

    t_loop = time.perf_counter()

    # ---- aggregation (vectorised Q-function) -------------------------
    brain = mask & (accumulator > 0)
    enhanced_logp = np.zeros_like(Z_clean)
    enhanced_logp[brain] = aggregate_logpvals_vec(accumulator[brain], delta)

    p_enhanced = np.ones_like(Z_clean)
    p_enhanced[brain] = np.exp(-enhanced_logp[brain])
    p_enhanced = np.clip(
        p_enhanced, np.finfo(float).tiny, 1.0 - np.finfo(float).eps
    )

    Z_enhanced = np.zeros_like(Z_clean)
    Z_enhanced[brain] = stats.norm.isf(p_enhanced[brain])

    t_end = time.perf_counter()

    # final progress notification
    if progress_queue is not None:
        progress_queue.put({
            "name": "ptfce",
            "current": Nh,
            "total": Nh,
            "phase": "done",
        })

    return {
        "p": p_enhanced,
        "logp": enhanced_logp,
        "Z_enhanced": Z_enhanced,
        "smoothness": smooth_info,
        "fwer_z_thresh": fwer_thresh,
        "diagnostics": {
            "z_max_input": z_max,
            "delta": float(delta),
            "Nh": Nh,
            "n_cache_entries": len(pvox_cache),
            "n_enhanced_voxels": int(brain.sum()),
            "Z_enhanced_max": float(Z_enhanced.max()),
            "Z_enhanced_p95": (
                float(np.percentile(Z_enhanced[brain], 95))
                if brain.any() else 0.0
            ),
            "thresholds_sample": diag_thresholds[::10],
            "timing": {
                "lut_build_s": round(t_lut - t_start, 3),
                "threshold_loop_s": round(t_loop - t0_loop, 3),
                "total_s": round(t_end - t_start, 3),
            },
            "cache_stats": {
                "unique_entries": len(pvox_cache),
                "lut_used": lut is not None,
            },
        },
    }


# ---------------------------------------------------------------------------
# shared threshold sweep (extracted helper for threshold iteration + aggregation)
# ---------------------------------------------------------------------------

def _run_threshold_sweep(
    Z_clean: np.ndarray,
    mask: np.ndarray,
    smooth_info: dict,
    Rd: float,
    V: int,
    Nh: int,
    z_est_thr: float,
    connectivity: int,
    lut: RegularGridInterpolator,
    fwer_thresh: float,
    verbose: bool,
    progress_queue: Any,
    prefix: str = "ptfce",
) -> dict:
    """Threshold sweep + aggregation with a pre-built LUT."""
    z_max = float(Z_clean.max())

    logp_max = -stats.norm.logsf(z_max)
    logp_grid = np.linspace(0, logp_max, Nh)
    delta = logp_grid[1] - logp_grid[0] if Nh > 1 else logp_max
    h_grid = stats.norm.isf(np.exp(-logp_grid))
    h_grid[0] = -1e10

    _use_cc3d = (connectivity == 6)
    if not _use_cc3d:
        structure = ndimage.generate_binary_structure(3, 3)
    accumulator = np.zeros_like(Z_clean)
    pvox_cache: dict[tuple[int, float], float] = {}

    t0_loop = time.perf_counter()
    for i, h in enumerate(h_grid):
        h = float(h)
        if progress_queue is not None:
            progress_queue.put({"name": prefix, "current": i + 1,
                                "total": Nh, "phase": "thresholds"})
        elif verbose:
            _bar(i + 1, Nh, prefix=f"{prefix}-sweep", t0=t0_loop)

        binary = Z_clean >= h
        if _use_cc3d:
            labelled = cc3d.connected_components(binary.view(np.uint8), connectivity=6)
            n_clusters = int(labelled.max())
        else:
            labelled, n_clusters = ndimage.label(binary, structure=structure)
        if n_clusters == 0:
            continue

        counts = np.bincount(labelled.ravel(), minlength=n_clusters + 1)
        cluster_sizes = counts[1:n_clusters + 1].astype(np.float64)

        h_rounded = round(h, 6)
        unique_sizes = np.unique(cluster_sizes)
        unique_sizes = unique_sizes[unique_sizes > 0]
        for uc in unique_sizes:
            c = int(uc)
            key = (c, h_rounded)
            if key not in pvox_cache:
                if h < z_est_thr:
                    pval = 0.5 * math.erfc(h * (1.0 / math.sqrt(2.0)))
                else:
                    pval = pvox_clust_lut(lut, c, h, z_est_thr)
                pvox_cache[key] = pval

        size_map = np.zeros(n_clusters + 1)
        for label_id in range(1, n_clusters + 1):
            c = int(cluster_sizes[label_id - 1])
            if c > 0:
                size_map[label_id] = -math.log(max(pvox_cache[(c, h_rounded)], 1e-300))
        accumulator += size_map[labelled]
    t_loop = time.perf_counter() - t0_loop

    brain = mask & (accumulator > 0)
    enhanced_logp = np.zeros_like(Z_clean)
    enhanced_logp[brain] = aggregate_logpvals_vec(accumulator[brain], delta)

    p_enhanced = np.ones_like(Z_clean)
    p_enhanced[brain] = np.exp(-enhanced_logp[brain])
    p_enhanced = np.clip(p_enhanced, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)

    Z_enhanced = np.zeros_like(Z_clean)
    Z_enhanced[brain] = stats.norm.isf(p_enhanced[brain])

    return {
        "p": p_enhanced, "logp": enhanced_logp, "Z_enhanced": Z_enhanced,
        "smoothness": smooth_info, "fwer_z_thresh": fwer_thresh,
        "diagnostics": {
            "z_max_input": z_max, "delta": float(delta), "Nh": Nh,
            "n_cache_entries": len(pvox_cache),
            "n_enhanced_voxels": int(brain.sum()),
            "Z_enhanced_max": float(Z_enhanced.max()),
            "timing_loop": t_loop,
        },
    }
