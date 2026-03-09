"""Tests for smoothness estimation and FWER threshold computation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage

from pytfce import estimate_smoothness, fwer_z_threshold


# ── estimate_smoothness ───────────────────────────────────────────

class TestEstimateSmoothness:

    def test_known_smooth_volume(self):
        """Smooth white-noise volumes, build a Z-map, and verify
        estimated FWHM is in a plausible range.

        The estimator expects a unit-variance Z-score field, so we
        generate multiple subjects, smooth each, and compute a group Z.
        """
        rng = np.random.default_rng(0)
        shape = (32, 32, 32)
        sigma = 2.0
        n = 30

        vols = np.stack([
            ndimage.gaussian_filter(rng.standard_normal(shape), sigma=sigma)
            for _ in range(n)
        ], axis=-1)
        mean = vols.mean(axis=-1)
        std = vols.std(axis=-1, ddof=1)
        std[std < 1e-12] = 1e-12
        z_map = mean / (std / np.sqrt(n))

        mask = np.ones(shape, dtype=bool)
        mask[:2, :, :] = False
        mask[-2:, :, :] = False
        mask[:, :2, :] = False
        mask[:, -2:, :] = False
        mask[:, :, :2] = False
        mask[:, :, -2:] = False

        info = estimate_smoothness(z_map, mask)

        assert "V" in info
        assert "Rd" in info
        assert "dLh" in info
        assert "fwhm_voxels" in info
        assert info["V"] > 0
        assert info["Rd"] > 0

        fwhm = np.array(info["fwhm_voxels"])
        assert np.all(fwhm > 1.0), "Smoothed data should have FWHM > 1"
        assert np.all(fwhm < 30.0), "FWHM should be reasonable for 32^3 volume"

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(1)
        vol = rng.standard_normal((16, 16, 16))
        mask = np.ones((16, 16, 16), dtype=bool)
        info = estimate_smoothness(vol, mask)
        for key in ("V", "Rd", "dLh", "fwhm_voxels", "resel_size", "n_resels"):
            assert key in info, f"Missing key: {key}"

    def test_smoother_volume_larger_fwhm(self):
        """More smoothing → larger FWHM estimate."""
        rng = np.random.default_rng(2)
        shape = (32, 32, 32)
        mask = np.ones(shape, dtype=bool)
        base = rng.standard_normal(shape)

        info_low = estimate_smoothness(
            ndimage.gaussian_filter(base, sigma=1.0), mask)
        info_high = estimate_smoothness(
            ndimage.gaussian_filter(base, sigma=3.0), mask)

        assert np.mean(info_high["fwhm_voxels"]) > np.mean(info_low["fwhm_voxels"])


# ── fwer_z_threshold ──────────────────────────────────────────────

class TestFwerZThreshold:

    def test_reasonable_range(self):
        """For typical neuroimaging resel counts, FWER Z should be ~2–8."""
        for n_resels in [10.0, 100.0, 1000.0, 5000.0]:
            z = fwer_z_threshold(n_resels)
            assert 2.0 < z < 8.0, f"n_resels={n_resels} → z={z}"

    def test_more_resels_higher_threshold(self):
        z_few = fwer_z_threshold(10.0)
        z_many = fwer_z_threshold(1000.0)
        assert z_many > z_few

    def test_very_few_resels(self):
        """n_resels < 1 is clipped internally; should still return a finite value."""
        z = fwer_z_threshold(0.5)
        assert np.isfinite(z) and z > 0

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_stricter_alpha_higher_threshold(self, alpha):
        z_strict = fwer_z_threshold(500.0, alpha=0.01)
        z_liberal = fwer_z_threshold(500.0, alpha=0.10)
        assert z_strict > z_liberal
