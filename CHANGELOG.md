# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-11-02

### Added

- Expose `pytfce.__version__` from installed distribution metadata, with
  `0+unknown` as the fallback when importing a source checkout without metadata.
- Regression tests for package-version lookup, scale-invariant smoothness,
  large-cluster and extreme-tail probabilities, single-voxel LUT interpolation,
  and inclusion of peak voxels at the final threshold.
- Linux and macOS CI coverage for Python 3.12, 3.13, and 3.14, with Ruff linting,
  distribution validation, and an installed-wheel check for the bundled R script.
  Publishing reuses these checks.

### Changed

- Set minimum runtime dependencies to NumPy 1.26.0, SciPy 1.11.2, and
  connected-components-3d 3.12.4, with Python 3.12 wheel availability.
- Declare the license as an SPDX expression (`MIT`) and require setuptools 77
  or newer to build; list Python 3.13 and 3.14 classifiers.
- Increment the LUT cache version to 3 so tables built with the earlier numerical
  implementation are rebuilt automatically.

### Fixed

- Estimate Z-map smoothness using scale-invariant lag-1 autocorrelation over the
  common mask core, as in FSL and R pTFCE, instead of assuming unit mean-square
  Z-scores. Rescaling a map no longer changes its estimated smoothness. This can
  change FWHM, resel counts, GRF FWER thresholds, and enhanced results, especially
  for signal-dominated maps.
- Correct the `dLh` normalization to the FSL determinant-based convention. This
  changes `dLh` and `Rd` for both Z-map and residual-based smoothness estimates,
  and consequently changes the conditional GRF probabilities that use `Rd`.
- Remove the one-voxel floor on expected cluster size inside `pvox_clust` to match
  R pTFCE. Conditional probabilities can therefore differ from 0.1.1 even for
  clusters that do not encounter numerical underflow. The separate public
  `expected_cluster_size` helper retains its one-voxel floor.
- Evaluate conditional cluster probabilities with scaled integrals and log-space
  normalization. Very large clusters no longer revert to marginal voxel
  probabilities when integration underflows; they retain strong cluster evidence,
  down to the existing `1e-300` probability floor. Extreme-tail expected cluster
  sizes also use stable log-survival values rather than an additive epsilon,
  avoiding spurious non-monotone probabilities and overflow.
- Populate LUT nodes below one voxel with the one-voxel probability instead of
  leaving them at `p = 1`. Interpolated single-voxel probabilities are no longer
  biased toward weaker evidence.
- Keep the final threshold at or below the observed peak when an inverse-normal
  round trip rounds upward. Baseline, prebuilt-LUT, exact, and eTFCE sweeps no
  longer omit peak voxels from the last integration level, which can increase
  their enhanced significance.
- Document `logp` as the natural-log `-ln(p)`; the README previously said `-log10(p)`.
- Use absolute links in the README so images and links render on PyPI.

### Removed

- `pytfce.utils.metrics.soft_dice_torch`, which had no callers and referenced
  PyTorch without importing or declaring it.
- The paper draft under `paper/` and its draft-PDF workflow. The README figures
  moved to `docs/figures/`.

**Numerical compatibility:** enhanced p-values and Z-scores are not expected to
be identical to 0.1.1. Recompute affected analyses rather than mixing results
from the two versions. Old LUT caches are invalidated automatically.

## [0.1.1] - 2026-03-10

### Added

- Include `r_ptfce_pipeline.R` in built distributions; the script was present in
  the repository but missing from the 0.1.0 wheel.
- Add machine-readable software citation metadata and a code of conduct.

### Changed

- Require Python 3.12 or newer.
- Use `.results/.lut_cache` as the default local LUT cache directory.

### Fixed

- Interpolate LUT probabilities in log-p space rather than linear-p space, so
  interpolation respects probabilities spanning many orders of magnitude.
  Version-1 LUT caches are invalidated and rebuilt as version 2. LUT-accelerated
  enhanced results can differ from 0.1.0.

## [0.1.0] - 2026-03-09

### Added

- Initial Python implementation of probabilistic Threshold-Free Cluster
  Enhancement, with baseline pTFCE and hybrid eTFCE–GRF variants.
- Analytical GRF cluster probabilities, log-probability aggregation, and optional
  lookup-table acceleration without permutation testing.
- Z-map and GLM-residual smoothness estimation and GRF FWER Z-thresholds.
- Classical TFCE, synthetic phantom generation, evaluation metrics, and optional
  plotting and terminal progress utilities.
- Unit tests, CI and publishing workflows, and documentation.

[Unreleased]: https://github.com/Don-Yin/pytfce/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Don-Yin/pytfce/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Don-Yin/pytfce/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Don-Yin/pytfce/releases/tag/v0.1.0
