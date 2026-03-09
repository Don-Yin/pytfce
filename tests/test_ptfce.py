"""Integration tests for pTFCE baseline and hybrid (exact) variants."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pytfce import ptfce_baseline, ptfce_exact

EXPECTED_KEYS = {"p", "logp", "Z_enhanced", "smoothness", "fwer_z_thresh", "diagnostics"}

# Shared kwargs to keep runs fast
_FAST = dict(Nh=30, lut_density=30, verbose=False)
_FAST_EXACT = dict(max_thresholds=60, lut_density=30, verbose=False)


# ── ptfce_baseline ─────────────────────────────────────────────────

class TestPtfceBaseline:

    def test_returns_expected_keys(self, z_map, brain_mask):
        result = ptfce_baseline(z_map, brain_mask, **_FAST)
        assert EXPECTED_KEYS <= set(result.keys())

    def test_output_shapes(self, z_map, brain_mask):
        result = ptfce_baseline(z_map, brain_mask, **_FAST)
        for key in ("p", "logp", "Z_enhanced"):
            assert result[key].shape == z_map.shape, f"Shape mismatch for {key}"

    def test_detects_signal(self, z_map, brain_mask, ground_truth):
        result = ptfce_baseline(z_map, brain_mask, **_FAST)
        sig = result["Z_enhanced"] > result["fwer_z_thresh"]
        dice = _dice(sig, ground_truth)
        assert dice > 0, "Baseline should detect at least some signal"

    def test_p_values_in_range(self, z_map, brain_mask):
        result = ptfce_baseline(z_map, brain_mask, **_FAST)
        assert np.all(result["p"] >= 0) and np.all(result["p"] <= 1)

    def test_logp_non_negative(self, z_map, brain_mask):
        result = ptfce_baseline(z_map, brain_mask, **_FAST)
        assert np.all(result["logp"] >= 0)


# ── ptfce_exact ────────────────────────────────────────────────────

class TestPtfceExact:

    def test_returns_expected_keys(self, z_map, brain_mask):
        result = ptfce_exact(z_map, brain_mask, **_FAST_EXACT)
        assert EXPECTED_KEYS <= set(result.keys())

    def test_output_shapes(self, z_map, brain_mask):
        result = ptfce_exact(z_map, brain_mask, **_FAST_EXACT)
        for key in ("p", "logp", "Z_enhanced"):
            assert result[key].shape == z_map.shape, f"Shape mismatch for {key}"

    def test_detects_signal(self, z_map, brain_mask, ground_truth):
        result = ptfce_exact(z_map, brain_mask, **_FAST_EXACT)
        sig = result["Z_enhanced"] > result["fwer_z_thresh"]
        dice = _dice(sig, ground_truth)
        assert dice > 0, "Exact variant should detect at least some signal"

    def test_p_values_in_range(self, z_map, brain_mask):
        result = ptfce_exact(z_map, brain_mask, **_FAST_EXACT)
        assert np.all(result["p"] >= 0) and np.all(result["p"] <= 1)


# ── cross-variant agreement ───────────────────────────────────────

class TestCrossVariantAgreement:

    @pytest.fixture(scope="class")
    def both_results(self, z_map, brain_mask):
        r_base = ptfce_baseline(z_map, brain_mask, **_FAST)
        r_exact = ptfce_exact(z_map, brain_mask, **_FAST_EXACT)
        return r_base, r_exact

    def test_enhanced_z_correlation(self, both_results, brain_mask):
        r_base, r_exact = both_results
        z_b = r_base["Z_enhanced"][brain_mask]
        z_e = r_exact["Z_enhanced"][brain_mask]
        active = (z_b > 0) | (z_e > 0)
        if active.sum() < 10:
            pytest.skip("too few active voxels for correlation")
        corr = np.corrcoef(z_b[active], z_e[active])[0, 1]
        assert corr > 0.8, f"Cross-variant correlation too low: {corr:.3f}"


# ── pure-noise volume ─────────────────────────────────────────────

class TestPureNoise:

    def test_baseline_few_significant(self, noise_z_map, noise_mask):
        result = ptfce_baseline(noise_z_map, noise_mask, **_FAST)
        fwer_z = result["fwer_z_thresh"]
        n_sig = int((result["Z_enhanced"] > fwer_z).sum())
        n_mask = int(noise_mask.sum())
        frac = n_sig / max(n_mask, 1)
        assert frac < 0.10, f"Too many false positives: {frac:.2%}"

    def test_exact_few_significant(self, noise_z_map, noise_mask):
        result = ptfce_exact(noise_z_map, noise_mask, **_FAST_EXACT)
        fwer_z = result["fwer_z_thresh"]
        n_sig = int((result["Z_enhanced"] > fwer_z).sum())
        n_mask = int(noise_mask.sum())
        frac = n_sig / max(n_mask, 1)
        assert frac < 0.10, f"Too many false positives: {frac:.2%}"


# ── helpers ────────────────────────────────────────────────────────

def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    intersection = (a & b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 0.0
    return 2.0 * intersection / total
