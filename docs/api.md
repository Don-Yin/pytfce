# API reference

Everything below is importable from `pytfce`. Arrays are NumPy; 3-D maps share one shape.

- [`ptfce_baseline`](#ptfce_baseline) and [`ptfce_exact`](#ptfce_exact): enhance a Z-map
- [`estimate_smoothness`](#estimate_smoothness) and [`estimate_smoothness_from_residuals`](#estimate_smoothness_from_residuals): smoothness for the GRF model
- [`fwer_z_threshold`](#fwer_z_threshold), [`pvox_clust`](#pvox_clust) and [`aggregate_logpvals_vec`](#aggregate_logpvals_vec): the GRF building blocks

## Enhancement

### `ptfce_baseline`

```python
ptfce_baseline(Z, mask, Rd=None, V=None, Nh=100, z_est_thr=1.3, connectivity=6,
               use_lut=True, lut_density=120, prebuilt_lut=None, verbose=True,
               progress_queue=None, smooth_override=None) -> dict
```

pTFCE as in R pTFCE (Spisák et al. 2019): connected-component labelling at each of `Nh` thresholds.

| Parameter | Description |
|-----------|-------------|
| `Z` | voxel-wise Z-score map |
| `mask` | boolean brain mask |
| `Rd`, `V` | smoothness `V · dLh` and in-mask voxel count for the cluster p-values; taken from the smoothness estimate when `None`. They do not change `fwer_z_thresh` |
| `Nh` | number of equispaced thresholds on the −log(p) scale |
| `z_est_thr` | lower integration bound of the Bayesian posterior |
| `connectivity` | 6 (faces) or 26 (faces, edges and corners) |
| `use_lut`, `lut_density` | interpolate cluster p-values from a `lut_density × lut_density` table instead of integrating each one |
| `prebuilt_lut` | a table to reuse across maps with the same `V` and `Rd` |
| `verbose` | print progress and a summary |
| `progress_queue` | a `multiprocessing.Queue` that receives progress dicts (`name`, `current`, `total`, `phase`) instead of printed progress |
| `smooth_override` | a smoothness dict, for example from [`estimate_smoothness_from_residuals`](#estimate_smoothness_from_residuals) or FSL `smoothest`; skips estimation from `Z` |

Returns the [result dict](#result-dict).

### `ptfce_exact`

```python
ptfce_exact(Z, mask, z_est_thr=1.3, lut_density=120, use_lut=True, max_thresholds=500,
            verbose=True, progress_queue=None, smooth_override=None) -> dict
```

Hybrid eTFCE–GRF: one union-find sweep retrieves every cluster exactly, with the same GRF p-values and aggregation as the baseline. `max_thresholds` sets the −log(p) grid (default 500, five times the baseline's); the other parameters are as in [`ptfce_baseline`](#ptfce_baseline). Returns the [result dict](#result-dict).

### Result dict

| Key | Contents |
|-----|----------|
| `p` | enhanced p-values, shape of `Z` |
| `logp` | −ln(p), natural log; divide by ln(10) for −log10(p) |
| `Z_enhanced` | Z-scores of the enhanced p-values |
| `fwer_z_thresh` | GRF FWER Z-threshold (α = 0.05) for `Z_enhanced` |
| `smoothness` | the [smoothness dict](#smoothness-dict) used |
| `diagnostics` | timings, cache statistics and threshold-grid settings |

A map with no positive voxel in the mask returns all-ones `p`, an infinite `fwer_z_thresh`, and empty `smoothness` and `diagnostics`.

## Smoothness

### `estimate_smoothness`

```python
estimate_smoothness(Z, mask) -> dict
```

Smoothness of a Z-map from its scale-invariant lag-1 autocorrelation, as FSL `smoothest -z` and R pTFCE compute it. Signal in `Z` inflates the estimate; prefer residuals when they are available.

### `estimate_smoothness_from_residuals`

```python
estimate_smoothness_from_residuals(Y, X, mask, n_sample=None) -> dict
```

Smoothness from GLM residuals. It matches FSL `smoothest -r res4d -d dof` with `dof = n − rank(X)`.

| Parameter | Description |
|-----------|-------------|
| `Y` | `(n_subjects, n_voxels)` observations inside `mask`, in `mask`'s C order (`data[mask]` per subject) |
| `X` | `(n_subjects, p)` design matrix |
| `mask` | boolean brain mask |
| `n_sample` | use only the first `n_sample` residuals; default all, as FSL does |

Pass the result to `smooth_override`. It adds `source` and `n_sample` to the [smoothness dict](#smoothness-dict).

### Smoothness dict

| Key | Contents |
|-----|----------|
| `V` | in-mask voxel count |
| `dLh` | √det(Λ) per voxel, FSL's `DLH` |
| `Rd` | `V · dLh` |
| `fwhm_voxels` | FWHM along each axis, in voxels |
| `resel_size` | product of the three FWHMs |
| `n_resels` | `V / resel_size` |
| `var_deriv` | mean squared first difference along each axis |

## GRF building blocks

### `fwer_z_threshold`

```python
fwer_z_threshold(n_resels, alpha=0.05) -> float
```

Z-threshold whose expected Euler characteristic in a 3-D Gaussian field of `n_resels` resels equals `alpha`, as in R pTFCE.

### `pvox_clust`

```python
pvox_clust(V, Rd, c, h, z_est_thr=1.3) -> float
```

P(Z ≥ `h` | the voxel lies in a cluster of `c` voxels), from Bayes' rule with the GRF cluster-size likelihood, by numerical integration.

### `aggregate_logpvals_vec`

```python
aggregate_logpvals_vec(s, delta) -> ndarray
```

Combines accumulated −ln(p) sums `s` over a threshold grid of step `delta` into one −ln(p) per voxel (the pTFCE Q-function), element-wise.
