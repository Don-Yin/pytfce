"""Regression tests for the inverse-normal threshold-grid endpoint."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("variant", ["baseline", "prebuilt", "exact", "etfce"])
@pytest.mark.parametrize("top_direction", [0, 1, -1])
def test_top_threshold_never_excludes_the_peak(monkeypatch, variant, top_direction):
    """An upward one-ulp round trip must equal the exact-endpoint sweep."""
    module_name = {"baseline": "ptfce_baseline", "prebuilt": "ptfce_baseline",
                   "exact": "ptfce_exact", "etfce": "ptfce_etfce"}[variant]
    module = importlib.import_module(f"pytfce.core.{module_name}")
    nh = 500 if variant == "exact" else 100
    z_max = 6.0
    z = np.zeros((3, 3, 3))
    z[1, 1, 1] = z_max
    mask = np.ones(z.shape, dtype=bool)
    smooth = {"Rd": 2.0, "V": int(mask.sum()), "n_resels": 2.0}
    monkeypatch.setattr(module, "estimate_smoothness", lambda *_: smooth)
    # Isolate threshold membership from posterior integration and LUT error.
    calls = []

    def posterior(*args):
        h = float(args[-2])
        calls.append(h)
        return np.exp(-1.0)

    monkeypatch.setattr(module, "pvox_clust", posterior, raising=False)
    monkeypatch.setattr(module, "pvox_clust_lut", posterior, raising=False)
    original_isf = module.stats.norm.isf

    def boundary_isf(p):
        result = original_isf(p)
        if np.shape(p) == (nh,) and p[0] == 1.0:
            result[-1] = (z_max if top_direction == 0 else
                          np.nextafter(z_max, np.inf if top_direction > 0 else -np.inf))
        return result

    monkeypatch.setattr(module.stats.norm, "isf", boundary_isf)
    if variant == "baseline":
        result = module.ptfce(z, mask, Nh=nh, use_lut=False,
                              smooth_override=smooth, verbose=False)
    elif variant == "prebuilt":
        result = module._run_threshold_sweep(
            z, mask, smooth, 2.0, int(mask.sum()), nh, 1.3, 6,
            object(), 4.0, False, None)
    elif variant == "exact":
        result = module.ptfce_exact(z, mask, max_thresholds=nh, use_lut=False,
                                    smooth_override=smooth, verbose=False)
    else:
        result = module.ptfce_etfce(z, mask, Nh=nh, use_lut=False, verbose=False)

    expected_top = z_max if top_direction >= 0 else np.nextafter(z_max, -np.inf)
    assert max(calls) == expected_top
    # Every non-sentinel level includes the peak; clipping must not move a
    # downward-rounded endpoint upward or alter the log-p integration spacing.
    grid = np.linspace(0, -module.stats.norm.logsf(z_max), nh)
    heights = original_isf(np.exp(-grid))
    expected_sum = np.where(heights[1:] < 1.3,
                            -module.stats.norm.logsf(heights[1:]), 1.0).sum()
    expected = module.aggregate_logpvals_vec(np.array([expected_sum]), grid[1])[0]
    np.testing.assert_allclose(result["logp"][1, 1, 1], expected, rtol=1e-14)
