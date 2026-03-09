"""Tests for the standard TFCE transform."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from pytfce.core.tfce import tfce_transform


class TestTfceTransform:

    def test_non_negative_output(self, z_map, brain_mask):
        """TFCE output should be non-negative (input is clipped to >= 0)."""
        stat = np.clip(z_map, 0, None)
        out = tfce_transform(stat)
        assert np.all(out >= 0)

    def test_output_shape(self, z_map):
        stat = np.clip(z_map, 0, None)
        out = tfce_transform(stat)
        assert out.shape == z_map.shape

    def test_signal_higher_than_noise(self, z_map, brain_mask, ground_truth):
        """Voxels inside the ground-truth region should have higher mean
        TFCE scores than voxels outside (within the brain mask)."""
        stat = np.clip(z_map, 0, None)
        out = tfce_transform(stat)

        noise_mask = brain_mask & ~ground_truth
        mean_signal = out[ground_truth].mean()
        mean_noise = out[noise_mask].mean()
        assert mean_signal > mean_noise, (
            f"Signal TFCE ({mean_signal:.2f}) should exceed noise ({mean_noise:.2f})")

    def test_zero_input(self):
        """All-zero input → all-zero output."""
        zeros = np.zeros((10, 10, 10))
        out = tfce_transform(zeros)
        assert_allclose(out, 0.0)

    def test_single_voxel_signal(self):
        """A single hot voxel should produce non-zero TFCE at that location."""
        vol = np.zeros((16, 16, 16))
        vol[8, 8, 8] = 5.0
        out = tfce_transform(vol)
        assert out[8, 8, 8] > 0
        assert out[0, 0, 0] == 0.0

    def test_larger_cluster_higher_score(self):
        """A larger contiguous region should get a higher peak TFCE than a
        single-voxel signal at the same height, due to the extent exponent."""
        vol_single = np.zeros((16, 16, 16))
        vol_single[8, 8, 8] = 4.0

        vol_cluster = np.zeros((16, 16, 16))
        vol_cluster[7:10, 7:10, 7:10] = 4.0

        out_single = tfce_transform(vol_single)
        out_cluster = tfce_transform(vol_cluster)
        assert out_cluster[8, 8, 8] > out_single[8, 8, 8]
