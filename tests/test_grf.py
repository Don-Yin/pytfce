"""Unit tests for GRF (Gaussian Random Field) functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from pytfce.core.grf import (
    aggregate_logpvals,
    aggregate_logpvals_vec,
    cluster_pdf,
    expected_cluster_size,
    pvox_clust,
)


# ── expected_cluster_size ──────────────────────────────────────────

class TestExpectedClusterSize:

    def test_positive_output(self):
        h = np.array([2.0, 3.0, 4.0])
        ec = expected_cluster_size(h, V=10_000, Rd=50.0)
        assert np.all(ec > 0), "expected cluster size must be positive"

    def test_minimum_clipped_to_one(self):
        """EC is clipped to >= 1 (a cluster has at least 1 voxel)."""
        h = np.array([8.0, 10.0])
        ec = expected_cluster_size(h, V=1000, Rd=10.0)
        assert np.all(ec >= 1.0)

    def test_monotonically_decreasing(self):
        """Higher thresholds → smaller expected clusters."""
        h = np.linspace(2.0, 6.0, 20)
        ec = expected_cluster_size(h, V=10_000, Rd=50.0)
        diffs = np.diff(ec)
        assert np.all(diffs <= 1e-10), "EC should be non-increasing in h"

    def test_scalar_input(self):
        ec = expected_cluster_size(np.array([3.0]), V=5000, Rd=30.0)
        assert ec.shape == (1,)
        assert ec[0] > 0

    def test_low_threshold_gives_large_ec(self):
        """h below 1.1 uses the simplified branch — EC should still be large."""
        ec = expected_cluster_size(np.array([0.5, 1.0]), V=10_000, Rd=50.0)
        assert np.all(ec > 100)


# ── cluster_pdf ────────────────────────────────────────────────────

class TestClusterPdf:

    @pytest.mark.parametrize("lam", [0.01, 0.05, 0.1])
    def test_integrates_to_one(self, lam):
        """The cluster-size PDF should integrate to ≈1 over (0, ∞).

        CDF is 1 - exp(-lam * c^{2/3}), so the integral of f over (0, ∞)
        is exactly 1.  We use small lambda values where the mass is spread
        wide enough for quad to handle the c^{-1/3} singularity at zero.
        """
        integral, _ = quad(cluster_pdf, 1e-10, np.inf, args=(lam,), limit=200)
        assert_allclose(integral, 1.0, atol=0.05,
                        err_msg=f"PDF with lam={lam} should integrate to ~1")

    @pytest.mark.parametrize("lam", [0.5, 1.0, 2.0])
    def test_cdf_consistency(self, lam):
        """Check that ∫_a^b f(c)dc matches the analytic CDF difference.

        CDF(c) = 1 - exp(-lam * c^{2/3}), avoiding the singularity at c=0.
        """
        a, b = 1.0, 100.0
        integral, _ = quad(cluster_pdf, a, b, args=(lam,), limit=200)
        cdf_diff = np.exp(-lam * a ** (2.0 / 3.0)) - np.exp(-lam * b ** (2.0 / 3.0))
        assert_allclose(integral, cdf_diff, rtol=0.01)

    def test_non_negative(self):
        for c in [0.1, 1.0, 10.0, 100.0]:
            assert cluster_pdf(c, lam=1.0) >= 0.0

    def test_zero_or_negative_c_returns_zero(self):
        assert cluster_pdf(0, lam=1.0) == 0.0
        assert cluster_pdf(-5.0, lam=1.0) == 0.0


# ── pvox_clust ─────────────────────────────────────────────────────

class TestPvoxClust:

    def test_output_in_unit_interval(self):
        p = pvox_clust(V=5000, Rd=30.0, c=50, h=3.0)
        assert 0 <= p <= 1

    def test_larger_cluster_lower_p(self):
        """Bigger cluster at the same threshold → lower (more significant) p."""
        p_small = pvox_clust(V=5000, Rd=30.0, c=10, h=3.0)
        p_large = pvox_clust(V=5000, Rd=30.0, c=200, h=3.0)
        assert p_large <= p_small

    def test_higher_threshold_lower_p(self):
        """Higher Z-threshold for same cluster → lower p."""
        p_low = pvox_clust(V=5000, Rd=30.0, c=50, h=2.5)
        p_high = pvox_clust(V=5000, Rd=30.0, c=50, h=4.0)
        assert p_high <= p_low

    def test_degenerate_c_zero(self):
        """c=0 should return 1.0 (no information from cluster)."""
        assert pvox_clust(V=5000, Rd=30.0, c=0, h=3.0) == 1.0

    def test_degenerate_h_nan(self):
        assert pvox_clust(V=5000, Rd=30.0, c=10, h=float("nan")) == 1.0


# ── aggregate_logpvals ─────────────────────────────────────────────

class TestAggregateLogpvals:

    def test_zero_input(self):
        assert aggregate_logpvals(0.0, delta=0.5) == 0.0

    def test_negative_input(self):
        assert aggregate_logpvals(-1.0, delta=0.5) == 0.0

    def test_positive_monotonic(self):
        """Larger accumulated -logp → larger output."""
        delta = 0.5
        vals = [aggregate_logpvals(s, delta) for s in range(1, 20)]
        diffs = np.diff(vals)
        assert np.all(np.array(diffs) > 0)

    def test_vec_matches_scalar(self):
        delta = 0.3
        s_vals = np.array([0.0, 0.5, 1.0, 3.0, 10.0])
        vec_result = aggregate_logpvals_vec(s_vals, delta)
        scalar_results = np.array([aggregate_logpvals(s, delta) for s in s_vals])
        assert_allclose(vec_result, scalar_results, atol=1e-12)

    def test_vec_output_shape(self):
        s = np.zeros(100)
        out = aggregate_logpvals_vec(s, delta=1.0)
        assert out.shape == s.shape
