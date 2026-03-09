---
title: 'pytfce: Fast Probabilistic Threshold-Free Cluster Enhancement in Python'
tags:
  - Python
  - neuroimaging
  - TFCE
  - pTFCE
  - statistical inference
  - voxel-based morphometry
  - Gaussian random field
authors:
  - name: Don Yin
    orcid: 0000-0002-8971-1057
    corresponding: true
    email: dy323@cam.ac.uk
    affiliation: 1
  - name: Hao Chen
    orcid: 0009-0009-5281-4616
    email: hc666@cam.ac.uk
    affiliation: 1
affiliations:
  - name: University of Cambridge, United Kingdom
    index: 1
    ror: 013meh722
date: 9 March 2026
bibliography: paper.bib
---

# Summary

`pytfce` is a pure-Python package for probabilistic Threshold-Free Cluster
Enhancement (pTFCE), providing analytical inference on neuroimaging statistical
maps without permutation testing. The package implements two complementary
variants: a baseline pTFCE that faithfully reproduces the original R
implementation [@spisak2019], and a novel hybrid eTFCE--GRF that combines
union-find cluster retrieval from exact TFCE [@chen2026] with Gaussian Random
Field (GRF) analytical $p$-values [@worsley1992]. On real brain data (~2
million voxels), `pytfce` is 73$\times$ faster than the canonical R pTFCE
package while producing concordant results (voxel-wise $r = 0.997$ between
Python variants; $r > 0.86$ against R pTFCE on real brain data).

# Statement of need

Threshold-Free Cluster Enhancement [@smith2009] is widely used in neuroimaging
to detect spatially extended effects without choosing an arbitrary
cluster-forming threshold. However, standard TFCE requires permutation testing
[@winkler2014], typically 5,000 relabellings, to assign $p$-values, costing
35--87 hours per analysis on whole-brain voxel-based morphometry (VBM) data.
Probabilistic TFCE [@spisak2019] replaces permutations with analytical
$p$-values derived from GRF theory, reducing runtime to seconds, but until now
was only available in R.

Researchers using Python-based neuroimaging pipelines (e.g., Nilearn
[@nilearn2024], NiBabel, FSL's Python wrappers) had no native pTFCE option,
forcing either R subprocess calls or falling back to expensive permutation
TFCE. `pytfce` fills this gap with a `pip`-installable package requiring only
NumPy, SciPy, and `connected-components-3d` [@cc3d]. The target audience is
neuroimaging researchers using Python-based analysis pipelines who require fast,
reproducible whole-brain statistical inference without R interoperability
overhead.

# State of the field

| Tool | Language | Method | Inference | Runtime (2M vox) |
|------|----------|--------|-----------|-------------------|
| FSL `randomise` [@winkler2014] | C++ | TFCE | Permutation | hours |
| R `pTFCE` [@spisak2019] | R | pTFCE | GRF analytical | ~375 s |
| `eTFCE` [@chen2026] | R/C | eTFCE | Permutation | hours (exact) |
| **`pytfce` baseline** | Python | pTFCE | GRF analytical | **5.1 s** |
| **`pytfce` hybrid** | Python | eTFCE--GRF | GRF analytical | **83.8 s** |

: Runtime comparison on IXI VBM data (563 subjects, 182$\times$218$\times$182 voxels). \label{tab:runtime}

The R `pTFCE` package [@spisak2019] is the only existing analytical alternative
to permutation TFCE. The recent `eTFCE` [@chen2026] eliminates discretisation
error via union-find but still requires permutations. No existing tool combines
exact cluster retrieval with analytical inference. `pytfce` provides both a
faithful port of R pTFCE and a novel hybrid that fills this gap.

We chose to build `pytfce` as a standalone package rather than contributing
Python bindings to the R `pTFCE` for three reasons. First, the R
implementation's tight coupling to R's matrix representation and `oro.nifti`
I/O makes a Python wrapper infeasible without rewriting the core. Second,
Python's NumPy/SciPy ecosystem enables direct interoperability with Nilearn,
NiBabel, and other neuroimaging tools without R subprocess overhead. Third,
building from scratch allowed us to introduce the hybrid eTFCE--GRF variant,
which has no analogue in the R package.

# Software design

`pytfce` is structured around three core algorithms, each in a self-contained
module:

- **`ptfce_baseline`**: direct port of Spisák et al. [-@spisak2019]. Sweeps a
  threshold grid ($N_h = 100$), runs connected-component labelling at each
  level, and evaluates GRF $p$-values via a lookup table (LUT) with three-tier
  caching (in-memory, disk, build).
- **`ptfce_exact`** (hybrid eTFCE--GRF): replaces per-threshold
  connected-component labelling with a single union-find sweep that builds the
  cluster hierarchy in $O(N \cdot \alpha(N))$ time. Cluster sizes at any
  threshold are then queried in $O(\alpha(N))$. GRF $p$-values and aggregation
  are identical to the baseline.
- **`etfce`**: standalone exact TFCE via union-find [@chen2026], provided for
  benchmarking.

Supporting modules provide smoothness estimation matching FSL's `smoothest`
(`smoothness.py`), GRF cluster-size distributions (`grf.py`), and
FWER-corrected $Z$-thresholds via Euler characteristic density
(`fwer_z_threshold`). The user-facing API exposes two main
functions---`ptfce_baseline(Z, mask)` and `ptfce_exact(Z, mask)`---each
returning a dict with enhanced $p$-values, $Z$-scores, smoothness diagnostics,
and timing information.

The key design trade-off is between the baseline and the hybrid. The baseline
is faster on small volumes because the LUT amortises the GRF integral cost,
while the hybrid scales better on large volumes where repeated
connected-component labelling becomes the bottleneck.

# Research impact statement

`pytfce` has been validated against the reference R `pTFCE` package through a
Monte Carlo simulation study on synthetic phantoms: 200 null replications
confirm FWER control ($\hat{\alpha} = 0.000$, 95% CI $[0, 0.019]$), and 500
power-curve trials across 10 signal amplitudes show both variants achieve
identical spatial detection (Dice $= 1.0$) at sufficient signal strength
(\autoref{fig:detection}). Runtime benchmarks on both phantom and real brain
data demonstrate 64--73$\times$ speedup over R pTFCE (\autoref{fig:runtime}).
Reproducible benchmarking scripts and a synthetic phantom generator are
included in the package (`pytfce.utils.phantoms`). A companion methodology
paper with real-brain validation is in preparation for NeuroImage.

![Spatial detection on a synthetic phantom ($64^3$, 80 subjects, 3 embedded
signal blobs). Top: input Z-map and pTFCE-enhanced significance maps for each
variant. Bottom: detection overlay (green = true positive, red = false
positive, blue = false negative). All three pTFCE variants achieve Dice =
1.0.](figures/fig-spatial-detection.png){#fig:detection width="100%"}

![Log-scale runtime comparison across five methods on the emulated phantom.
Diamond markers show mean runtime; individual runs are shown as
dots.](figures/fig-runtime-log.png){#fig:runtime width="100%"}

# AI usage disclosure

GitHub Copilot (GPT-5.4, March 2026) was used for autocompletion during
development of test scaffolding and boilerplate I/O code. Core algorithm
modules (`ptfce_baseline`, `ptfce_exact`, `grf`, `smoothness`) were written
manually following the mathematical formulations in Spisák et al.
[-@spisak2019] and Chen et al. [-@chen2026]. All code was reviewed, tested
against the reference R implementation, and validated through Monte Carlo
simulation (200 null replications, 500 power-curve trials). No AI-generated
derivations were used.

# Acknowledgements

We thank Tamás Spisák for the original R pTFCE implementation that served as
the reference for validation. We acknowledge the use of the IXI dataset
(funded by EPSRC GR/S21533/02) and UK Biobank (application 87802).

# References
