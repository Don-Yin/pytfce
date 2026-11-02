"""Regression tests for the Python-vs-R pTFCE reconciliation (reports/05-py-r-diagnosis)."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy import ndimage

from pytfce import estimate_smoothness, pvox_clust


def _smooth_field(seed: int = 0, shape=(24, 24, 24), sigma: float = 1.5) -> np.ndarray:
    return ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal(shape), sigma)


class TestSmoothnessMatchesFslR:

    def test_scale_invariant(self):
        """R / FSL smoothest is invariant to the scale of the map; so must we be."""
        z = _smooth_field()
        mask = np.ones(z.shape, dtype=bool)
        a = estimate_smoothness(z, mask)
        b = estimate_smoothness(37.0 * z, mask)
        assert b["n_resels"] == pytest.approx(a["n_resels"], rel=1e-9)
        assert b["Rd"] == pytest.approx(a["Rd"], rel=1e-9)

    def test_rd_over_n_resels_is_fsl_constant(self):
        """FSL: DLH = det(Lambda)^(1/2), RESELS = prod(FWHM)  =>  Rd / n_resels = (4 ln 2)^(3/2)."""
        z = _smooth_field(1)
        info = estimate_smoothness(z, np.ones(z.shape, dtype=bool))
        assert info["Rd"] / info["n_resels"] == pytest.approx((4.0 * math.log(2.0)) ** 1.5, rel=1e-9)


class TestPvoxClustGiantCluster:
    V, RD, H = 2_075_183, 271_349.2, 2.468  # saved ADNI run

    def test_saturates_instead_of_marginal(self):
        marginal = 0.5 * math.erfc(self.H / math.sqrt(2.0))
        p = pvox_clust(self.V, self.RD, 1_200_000, self.H)
        assert p < 1e-100 < marginal

    def test_monotone_through_the_old_underflow_cliff(self):
        ps = [pvox_clust(self.V, self.RD, c, self.H) for c in (1e5, 5e5, 1.0e6, 1.2e6, 2.0e6)]
        assert all(b <= a * (1 + 1e-12) for a, b in pairwise(ps))

    @pytest.mark.parametrize("h", [12.0, 25.0, 32.0, 36.5])
    def test_no_overflow_in_the_far_tail(self, h):
        for c in (1.0, 5e2, 1e4, 1e6):
            p = pvox_clust(self.V, self.RD, c, h)
            assert 0.0 < p <= 1.0 and math.isfinite(p)


class TestLutSingleVoxelClusters:
    def test_lut_matches_exact_for_one_voxel_clusters(self):
        """c = 1 sits between two LUT nodes; nodes with c < 1 used to hold log p = 0 (p = 1)."""
        from pytfce.core.ptfce_baseline import build_pvox_lut, pvox_clust_lut

        V, Rd = 91_965, 5_557.5
        lut = build_pvox_lut(V, Rd, 1.3, 120, 30, h_max=8.0, verbose=False)
        for h in (2.0, 3.0, 5.0):
            exact = -math.log(pvox_clust(V, Rd, 1, h))
            approx = -math.log(pvox_clust_lut(lut, 1, h))
            assert approx == pytest.approx(exact, rel=0.05)
