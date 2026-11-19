# Usage

## Installation

```bash
pip install pytfce
```

NumPy, SciPy and connected-components-3d install automatically. For a development install, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## From a NIfTI file

Reading NIfTI needs NiBabel (`pip install nibabel`); the array API does not.

```python
import nibabel as nib
from pytfce import ptfce_baseline, ptfce_exact

img = nib.load("zstat1.nii.gz")
Z = img.get_fdata()
mask = Z != 0  # or load a proper brain mask

result = ptfce_baseline(Z, mask)  # fast, matches R pTFCE
result = ptfce_exact(Z, mask)     # hybrid eTFCE–GRF, exact cluster retrieval
```

To choose between them, see [Variants](../README.md#variants). Every parameter and result key is in the [API reference](api.md).

## Smoothness from residuals

A Z-map's own smoothness is inflated by its signal. With the GLM at hand, estimate it from the residuals instead:

```python
from pytfce import estimate_smoothness_from_residuals

Y = data[mask].T  # data: (X, Y, Z, n_subjects) -> (n_subjects, n_voxels)
smooth = estimate_smoothness_from_residuals(Y, design, mask)
result = ptfce_baseline(Z, mask, smooth_override=smooth)
```

## Comparing backend timings

Time `ptfce_exact` with `uf_backend="python"` and `uf_backend="numba"` on the same map and hardware, using the same `max_thresholds`, `use_lut` and `lut_density`. A denser grid changes the numerical calculation, not just its speed.

Report the first call separately from repeated calls: building the probability lookup table takes time for either backend, and Numba must also compile the loops or load cached code on its first call. `result["diagnostics"]["uf_backend"]` identifies the backend used for a nonempty sweep.
